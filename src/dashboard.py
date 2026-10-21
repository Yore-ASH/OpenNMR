"""CNMR 训练看板

一个可以长期在后台运行的本地网页，用来：

- 实时查看训练进度（epoch / batch / loss / 耗时），数据来自 data/LogData/train_status.json；
- 查看 loss 曲线、逐 epoch 记录、历次运行汇总；
- 查看最近一次 F1 分类报告（每类 P/R/F1/AP）和阈值扫描结果；
- 直接在网页上“开始训练 / 停止训练”（停止 = 发 SIGINT，训练会先保存权重和日志再退出）；
- 在网页上触发一次 F1 评估，并在页面上看进度。

启动（建议用后台方式常驻）：
    cd src
    ../.venv/bin/python dashboard.py --host 0.0.0.0 --port 8090

然后浏览器打开 http://<服务器IP>:8090 。

看板本身退出不会影响已经启动的训练；训练进度都写在
data/LogData/train_status.json 里，重新打开看板还能接着看。
"""

import argparse
import csv
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time

from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from flask import Flask, Response, jsonify, request, send_file

from setup import SETTINGS


BASE = Path(__file__).resolve().parent
REPO = BASE.parent

LOG_DIR = Path(SETTINGS.paths.log_data_dir)
MODEL_DIR = Path(SETTINGS.paths.model_dir)
CONSOLE_DIR = REPO / "logs"

STATUS_PATH = LOG_DIR / "train_status.json"
HISTORY_PATH = LOG_DIR / "train_history.csv"
RUN_SUMMARY_PATH = LOG_DIR / "run_summary.jsonl"
F1_CLASS_PATH = LOG_DIR / "f1_by_class.csv"
F1_SUMMARY_PATH = LOG_DIR / "f1_summary.jsonl"
F1_SWEEP_PATH = LOG_DIR / "f1_threshold_sweep.csv"
TRAIN_CONSOLE = CONSOLE_DIR / "train_web.log"
EVAL_CONSOLE = CONSOLE_DIR / "f1_web.log"

DOWNLOAD_FILES = {
    "train_history.csv": HISTORY_PATH,
    "run_summary.jsonl": RUN_SUMMARY_PATH,
    "f1_by_class.csv": F1_CLASS_PATH,
    "f1_summary.jsonl": F1_SUMMARY_PATH,
    "f1_threshold_sweep.csv": F1_SWEEP_PATH,
}

app = Flask(__name__)

_plot_lock = threading.Lock()
_chart_cache = {}

_train_proc = None
_eval_proc = None


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def _read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _read_jsonl(path, limit=None):
    rows = []

    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []

    return rows[-limit:] if limit else rows


def _read_csv(path, limit=None):
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return []

    return rows[-limit:] if limit else rows


def _csv_columns(path):
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return next(csv.reader(fh), [])
    except OSError:
        return []


def _mtime(path):
    try:
        return Path(path).stat().st_mtime
    except OSError:
        return 0.0


def _proc_alive(pid):
    if not pid:
        return False

    try:
        os.kill(int(pid), 0)
    except (OSError, ValueError):
        return False

    return True


def _proc_args(pid):
    """读取进程的 argv 列表（而不是拼成整条命令字符串，避免误判）"""

    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
    except (OSError, ValueError):
        return []

    return [part.decode(errors="replace") for part in raw.split(b"\x00") if part]


def _is_script_process(pid, script_name):
    """判断 pid 是不是 `python xxx/train.py` 这类进程"""

    args = _proc_args(pid)

    if not args or ("python" not in args[0].lower()):
        return False

    return any(
        (arg == script_name) or arg.endswith("/" + script_name)
        for arg in args[1:]
    )


def _find_pid(script_name):
    """在 /proc 里找正在运行的脚本进程（用于识别外部启动的训练）"""

    try:
        entries = Path("/proc").iterdir()
    except OSError:
        return None

    for entry in entries:
        if entry.name.isdigit() and _is_script_process(int(entry.name), script_name):
            return int(entry.name)

    return None


def _tail_text(path, n_lines):
    path = Path(path)

    if not path.exists():
        return ""

    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(size - 300_000, 0))
        raw = fh.read().decode("utf-8", errors="replace")

    lines = [line for line in raw.replace("\r", "\n").splitlines() if line.strip()]

    return "\n".join(lines[-n_lines:])


# --------------------------------------------------------------------------- #
# 状态
# --------------------------------------------------------------------------- #
def _training_state():
    global _train_proc

    # 回收已经退出的训练子进程，避免留下僵尸进程
    if _train_proc is not None:
        _train_proc.poll()

    status = _read_json(STATUS_PATH, {}) or {}

    pid = status.get("pid")
    alive = bool(pid) and _proc_alive(pid) and _is_script_process(pid, "train.py")

    state = status.get("state") or "idle"

    # 状态文件还没写/被删时，退化为扫描进程
    if (not alive) and (not pid):
        found = _find_pid("train.py")

        if found:
            pid = found
            alive = True
            status["started_at"] = status.get("started_at") or datetime.fromtimestamp(
                Path(f"/proc/{found}").stat().st_mtime
            ).isoformat(timespec="seconds")
            state = "running"
            status["source"] = "process-scan"

    if alive and state not in ("running",):
        state = "running"

    if (not alive) and (state == "running"):
        state = "lost"

    status["pid"] = pid
    status["alive"] = bool(alive)
    status["state"] = state

    total_batches = status.get("total_batches") or 0
    batch = status.get("batch") or 0

    status["percent"] = round(min(batch / total_batches * 100, 100.0), 2) if total_batches else 0.0

    started_at = status.get("started_at")

    if started_at:
        try:
            elapsed = (datetime.now() - datetime.fromisoformat(started_at)).total_seconds()
            status["elapsed_seconds"] = max(0, int(elapsed))
        except (ValueError, TypeError):
            pass

    return status


def _eval_state():
    running = _eval_proc is not None and _eval_proc.poll() is None

    if not running:
        pid = _find_pid("f1_test.py")
        running = bool(pid)

    return {
        "running": bool(running),
        "log": _tail_text(EVAL_CONSOLE, 30),
    }


# --------------------------------------------------------------------------- #
# 图表（带 mtime 缓存，避免每 5 秒重画）
# --------------------------------------------------------------------------- #
def _cached_chart(name, deps, producer):
    key = tuple(_mtime(dep) for dep in deps)

    cached = _chart_cache.get(name)

    if cached and (cached[0] == key):
        return cached[1]

    with _plot_lock:
        cached = _chart_cache.get(name)

        if cached and (cached[0] == key):
            return cached[1]

        data = producer()

        _chart_cache[name] = (key, data)

        return data


def _loss_chart():
    rows = _read_csv(HISTORY_PATH)

    if not rows:
        return None

    frame = pd.DataFrame(rows)

    for column in ("epoch", "train_loss", "val_loss"):
        if column not in frame.columns:
            return None
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["epoch"])

    if frame.empty:
        return None

    runs = list(dict.fromkeys(frame["run_id"].astype(str)))[-3:]

    fig, ax = plt.subplots(figsize=(9.5, 4.0), dpi=110)

    for run in runs:
        sub = frame[frame["run_id"].astype(str) == run].sort_values("epoch")

        if sub.empty:
            continue

        loss_fn = ""

        if ("loss_fn" in sub.columns) and sub["loss_fn"].notna().any():
            loss_fn = str(sub["loss_fn"].iloc[-1]).strip()

        label = f"{run} ({loss_fn})" if loss_fn and (loss_fn != "nan") else run

        ax.plot(sub["epoch"], sub["train_loss"], marker="o", markersize=3, label=f"{label} train")
        ax.plot(
            sub["epoch"],
            sub["val_loss"],
            marker="s",
            markersize=3,
            linestyle="--",
            label=f"{label} val",
        )

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Train / validation loss by run")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    return buffer.getvalue()


def _latest_f1_classes():
    rows = _read_csv(F1_CLASS_PATH)

    if not rows:
        return []

    latest_run = rows[-1].get("run_id")

    return [row for row in rows if row.get("run_id") == latest_run]


def _f1_chart():
    rows = _latest_f1_classes()

    if not rows:
        return None

    frame = pd.DataFrame(rows).iloc[::-1]

    for column in ("f1", "best_f1"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=110)

    positions = np.arange(len(frame))

    if "f1" in frame.columns:
        ax.barh(positions - 0.2, frame["f1"], height=0.4, label="F1 at report threshold")

    if "best_f1" in frame.columns:
        ax.barh(positions + 0.2, frame["best_f1"], height=0.4, label="best per-class F1")

    ax.set_yticks(positions)
    ax.set_yticklabels(frame["class"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("F1 by functional group (latest evaluation)")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)

    return buffer.getvalue()


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #
@app.get("/api/state")
def api_state():
    training = _training_state()

    f1_summary = _read_jsonl(F1_SUMMARY_PATH, 20)
    latest_f1 = f1_summary[-1] if f1_summary else None

    class_rows = _latest_f1_classes() if latest_f1 else []

    return jsonify(
        {
            "now": datetime.now().isoformat(timespec="seconds"),
            "training": training,
            "eval": _eval_state(),
            "history": {
                "columns": _csv_columns(HISTORY_PATH),
                "rows": _read_csv(HISTORY_PATH, 60),
            },
            "runs": _read_jsonl(RUN_SUMMARY_PATH, 20),
            "f1": {
                "latest": latest_f1,
                "classes": class_rows,
                "history": f1_summary,
            },
            "charts": {
                "loss": _mtime(HISTORY_PATH),
                "f1": _mtime(F1_CLASS_PATH),
            },
            "paths": {
                "history": str(HISTORY_PATH),
                "runs": str(RUN_SUMMARY_PATH),
                "f1_classes": str(F1_CLASS_PATH),
                "f1_summary": str(F1_SUMMARY_PATH),
                "model": str(MODEL_DIR / "model_13c_v1.pth"),
                "status": str(STATUS_PATH),
            },
        }
    )


@app.get("/api/log")
def api_log():
    try:
        tail = int(request.args.get("tail", 120))
    except (TypeError, ValueError):
        tail = 120

    tail = max(1, min(tail, 1000))

    return jsonify({"text": _tail_text(TRAIN_CONSOLE, tail), "path": str(TRAIN_CONSOLE)})


@app.post("/api/train/start")
def api_train_start():
    global _train_proc

    current = _training_state()

    if current.get("alive"):
        return jsonify({"ok": False, "error": f"已有训练在运行 (pid={current.get('pid')})"}), 409

    payload = request.get_json(silent=True) or {}

    try:
        epochs = int(payload.get("epochs", 0))
        batch_size = int(payload.get("batch_size", 64))
        lr = float(payload.get("lr", 0.001))
    except (TypeError, ValueError):
        return jsonify({"ok": False, "error": "参数格式不正确"}), 400

    if (epochs < 0) or (batch_size < 1) or (lr <= 0):
        return jsonify({"ok": False, "error": "参数范围不正确"}), 400

    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)

    log_file = open(TRAIN_CONSOLE, "ab")

    command = [
        sys.executable,
        "train.py",
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--lr",
        str(lr),
    ]

    proc = subprocess.Popen(
        command,
        cwd=str(BASE),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    _train_proc = proc

    time.sleep(0.8)

    return jsonify({"ok": True, "pid": proc.pid, "command": command})


@app.post("/api/train/stop")
def api_train_stop():
    current = _training_state()
    pid = current.get("pid")

    if (not current.get("alive")) or (not pid):
        return jsonify({"ok": False, "error": "当前没有正在运行的训练"}), 409

    try:
        os.kill(int(pid), signal.SIGINT)
    except ProcessLookupError:
        return jsonify({"ok": True, "stopped": True, "note": "进程已经退出"})
    except PermissionError:
        return jsonify({"ok": False, "error": "没有权限停止该进程"}), 403

    deadline = time.time() + 30

    # 用 _is_script_process 判断：进程退出后成为僵尸时 cmdline 为空，
    # 不能再只用 os.kill(pid, 0) 判断存活。
    while (time.time() < deadline) and _is_script_process(pid, "train.py"):
        time.sleep(0.2)

    if (_train_proc is not None) and (_train_proc.pid == int(pid)):
        try:
            _train_proc.wait(timeout=5)  # 回收子进程
        except (subprocess.TimeoutExpired, OSError):
            pass

    stopped = not _is_script_process(pid, "train.py")

    return jsonify({"ok": True, "stopped": stopped, "pid": int(pid)})


@app.post("/api/eval/start")
def api_eval_start():
    global _eval_proc

    if _eval_state().get("running"):
        return jsonify({"ok": False, "error": "F1 评估正在运行"}), 409

    CONSOLE_DIR.mkdir(parents=True, exist_ok=True)

    log_file = open(EVAL_CONSOLE, "ab")

    proc = subprocess.Popen(
        [sys.executable, "f1_test.py"],
        cwd=str(BASE),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    _eval_proc = proc

    return jsonify({"ok": True, "pid": proc.pid})


@app.get("/plot/loss.png")
def plot_loss():
    data = _cached_chart("loss", (HISTORY_PATH,), _loss_chart)

    if not data:
        return Response("no data", status=404)

    return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})


@app.get("/plot/f1.png")
def plot_f1():
    data = _cached_chart("f1", (F1_CLASS_PATH,), _f1_chart)

    if not data:
        return Response("no data", status=404)

    return Response(data, mimetype="image/png", headers={"Cache-Control": "no-store"})


@app.get("/download/<name>")
def download(name):
    path = DOWNLOAD_FILES.get(name)

    if (path is None) or (not Path(path).exists()):
        return Response("not found", status=404)

    return send_file(path, as_attachment=True, download_name=name)


@app.get("/api/health")
def health():
    return jsonify({"ok": True, "now": datetime.now().isoformat(timespec="seconds")})


@app.get("/")
def index():
    return Response(INDEX_HTML, mimetype="text/html")


# --------------------------------------------------------------------------- #
# 页面
# --------------------------------------------------------------------------- #
INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CNMR 训练监控</title>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --line:#e3e8f0; --text:#1f2937; --muted:#6b7280;
          --blue:#2563eb; --green:#059669; --red:#dc2626; --amber:#d97706; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                      "Microsoft YaHei", Roboto, sans-serif; font-size:14px; }
  header { display:flex; align-items:center; justify-content:space-between; gap:12px;
           padding:14px 22px; background:#111827; color:#fff; position:sticky; top:0; z-index:10; }
  header h1 { font-size:17px; margin:0; font-weight:600; letter-spacing:.3px; }
  header .meta { font-size:12px; color:#9ca3af; }
  main { max-width:1180px; margin:0 auto; padding:18px 16px 40px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:16px 18px; margin-bottom:16px; box-shadow:0 1px 2px rgba(16,24,40,.04); }
  .card h2 { font-size:15px; margin:0 0 12px; display:flex; align-items:center; gap:8px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px 18px; }
  .kv { font-size:13px; color:var(--muted); }
  .kv b { display:block; font-size:16px; color:var(--text); margin-top:2px; font-weight:600; }
  .controls { display:flex; flex-wrap:wrap; gap:10px; align-items:flex-end; }
  .controls label { display:flex; flex-direction:column; gap:4px; font-size:12px; color:var(--muted); }
  .controls input { width:110px; padding:7px 9px; border:1px solid var(--line);
                    border-radius:6px; font-size:14px; }
  button { padding:8px 16px; border:0; border-radius:6px; font-size:14px; cursor:pointer;
           background:var(--blue); color:#fff; font-weight:500; }
  button.secondary { background:#eef2f7; color:#334155; }
  button.danger { background:var(--red); }
  button:disabled { opacity:.45; cursor:not-allowed; }
  .hint { color:var(--muted); font-size:12px; margin:10px 0 0; }
  .badge { display:inline-block; padding:2px 9px; border-radius:999px; font-size:12px;
           font-weight:600; color:#fff; background:var(--muted); }
  .badge.running { background:var(--green); }
  .badge.stopped, .badge.finished { background:#64748b; }
  .badge.lost { background:var(--red); }
  .bar { height:10px; background:#e8edf5; border-radius:999px; overflow:hidden; margin:12px 0 6px; }
  .bar > div { height:100%; width:0; background:linear-gradient(90deg,#3b82f6,#06b6d4);
               transition:width .4s ease; }
  table { width:100%; border-collapse:collapse; font-size:12.5px; }
  th, td { text-align:right; padding:6px 8px; border-bottom:1px solid var(--line); white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  th { color:var(--muted); font-weight:600; background:#fafbfe; }
  tbody tr:hover { background:#f8fafc; }
  pre { background:#0f172a; color:#d7e0ee; padding:12px; border-radius:8px; overflow:auto;
        max-height:230px; font-size:12px; line-height:1.55; margin:12px 0 0; }
  img { width:100%; max-width:100%; border:1px solid var(--line); border-radius:8px; background:#fff; }
  .msg { font-size:13px; margin-left:6px; }
  .msg.ok { color:var(--green); } .msg.err { color:var(--red); }
  .links { display:flex; flex-wrap:wrap; gap:12px; font-size:13px; }
  .links a { color:var(--blue); text-decoration:none; }
  .links a:hover { text-decoration:underline; }
  #console-wrap { margin-top:12px; }
  .small { font-size:12px; color:var(--muted); }
</style>
</head>
<body>
<header>
  <h1>CNMR 训练监控</h1>
  <div class="meta">服务时间 <span id="now">—</span> · <span id="conn">连接中…</span></div>
</header>

<main>
  <section class="card">
    <h2>训练控制 <span id="state-badge" class="badge">未知</span></h2>
    <div class="controls">
      <label>epochs（0=一直训练）<input id="epochs" type="number" min="0" step="1" value="0"></label>
      <label>batch size<input id="batch" type="number" min="1" step="1" value="64"></label>
      <label>learning rate<input id="lr" type="number" min="0.00001" step="0.0001" value="0.001"></label>
      <button id="btn-start">开始训练</button>
      <button id="btn-stop" class="danger">停止训练</button>
      <button id="btn-eval" class="secondary">跑一次 F1 评估</button>
      <span id="msg" class="msg"></span>
    </div>
    <p class="hint">停止训练 = 给训练进程发 SIGINT，会先保存当前权重和日志再退出。看板关闭不会影响训练。</p>
  </section>

  <section class="card">
    <h2>实时进度</h2>
    <div class="grid">
      <div class="kv">run_id<b id="run-id">—</b></div>
      <div class="kv">epoch<b id="epoch">—</b></div>
      <div class="kv">batch<b id="batch-now">—</b></div>
      <div class="kv">train loss<b id="train-loss">—</b></div>
      <div class="kv">val loss<b id="val-loss">—</b></div>
      <div class="kv">best val loss<b id="best-loss">—</b></div>
      <div class="kv">已运行<b id="elapsed">—</b></div>
      <div class="kv">损失函数<b id="loss-fn">—</b></div>
      <div class="kv">输入特征<b id="input-dims">—</b></div>
    </div>
    <div class="bar"><div id="bar"></div></div>
    <div class="small" id="bar-text">—</div>
    <div id="console-wrap"><pre id="console">（还没有控制台输出）</pre></div>
  </section>

  <section class="card">
    <h2>Loss 曲线</h2>
    <img id="loss-img" alt="loss chart" src="/plot/loss.png">
  </section>

  <section class="card">
    <h2>历次运行</h2>
    <div style="overflow:auto"><table id="runs-table"><thead></thead><tbody></tbody></table></div>
  </section>

  <section class="card">
    <h2>逐 epoch 记录（最近 60 条）</h2>
    <div style="overflow:auto"><table id="history-table"><thead></thead><tbody></tbody></table></div>
  </section>

  <section class="card">
    <h2>F1 分类报告（最近一次评估）</h2>
    <div class="grid" id="f1-overall"></div>
    <div style="margin-top:14px"><img id="f1-img" alt="f1 chart" src="/plot/f1.png"></div>
    <div style="overflow:auto;margin-top:14px"><table id="f1-table"><thead></thead><tbody></tbody></table></div>
    <p class="hint" id="f1-note"></p>
  </section>

  <section class="card">
    <h2>文件下载</h2>
    <div class="links">
      <a href="/download/train_history.csv">train_history.csv</a>
      <a href="/download/run_summary.jsonl">run_summary.jsonl</a>
      <a href="/download/f1_by_class.csv">f1_by_class.csv</a>
      <a href="/download/f1_threshold_sweep.csv">f1_threshold_sweep.csv</a>
      <a href="/download/f1_summary.jsonl">f1_summary.jsonl</a>
    </div>
    <p class="hint" id="paths"></p>
  </section>
</main>

<script>
const $ = (id) => document.getElementById(id);

let lossChartVersion = null;
let f1ChartVersion = null;

function fmt(value, digits = 4) {
  if (value === null || value === undefined || value === "" || value === "nan") return "—";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(digits) : String(value);
}

function fmtInt(value) {
  const num = Number(value);
  return Number.isFinite(num) ? num.toLocaleString("zh-CN") : "—";
}

function fmtDuration(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const total = Math.max(0, Math.round(Number(seconds)));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return (h ? h + "h " : "") + m + "m " + s + "s";
}

function setText(id, text) { $(id).textContent = text; }

function renderTable(table, columns, rows, formatters = {}) {
  const thead = table.querySelector("thead");
  const tbody = table.querySelector("tbody");
  thead.innerHTML = "";
  tbody.innerHTML = "";

  const headRow = document.createElement("tr");
  for (const column of columns) {
    const th = document.createElement("th");
    th.textContent = column.label;
    headRow.appendChild(th);
  }
  thead.appendChild(headRow);

  for (const row of rows) {
    const tr = document.createElement("tr");
    for (const column of columns) {
      const td = document.createElement("td");
      const formatter = formatters[column.key];
      const raw = row[column.key];
      td.textContent = formatter ? formatter(raw, row) : (raw === undefined || raw === null || raw === "" ? "—" : raw);
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }

  if (!rows.length) {
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = columns.length;
    td.textContent = "暂无数据";
    td.style.textAlign = "center";
    td.style.color = "#9ca3af";
    tr.appendChild(td);
    tbody.appendChild(tr);
  }
}

function badge(state) {
  const map = { running: "运行中", stopped: "已停止", finished: "已完成", idle: "空闲", lost: "进程丢失" };
  const el = $("state-badge");
  el.className = "badge " + (state || "idle");
  el.textContent = map[state] || state || "未知";
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || ("HTTP " + response.status));
  return data;
}

function showMessage(text, ok = true) {
  const el = $("msg");
  el.textContent = text;
  el.className = "msg " + (ok ? "ok" : "err");
  clearTimeout(showMessage.timer);
  showMessage.timer = setTimeout(() => { el.textContent = ""; }, 6000);
}

async function refresh() {
  try {
    const state = await (await fetch("/api/state", { cache: "no-store" })).json();
    $("conn").textContent = "已连接";
    setText("now", state.now || "—");

    const t = state.training || {};
    badge(t.state);
    $("btn-start").disabled = !!t.alive;
    $("btn-stop").disabled = !t.alive;

    setText("run-id", t.run_id || "—");
    setText("epoch", t.epoch ? `${t.epoch} / ${t.epochs_max > 0 ? t.epochs_max : "∞"}` : "—");
    setText("batch-now", t.total_batches ? `${fmtInt(t.batch || 0)} / ${fmtInt(t.total_batches)}` : "—");
    setText("train-loss", fmt(t.train_loss));
    setText("val-loss", fmt(t.val_loss));
    setText("best-loss", fmt(t.best_val_loss));
    setText("elapsed", fmtDuration(t.elapsed_seconds));
    setText("loss-fn", t.loss_fn ? `${t.loss_fn} (α=${t.focal_alpha ?? "—"}, γ=${t.focal_gamma ?? "—"})` : "—");
    setText(
      "input-dims",
      t.n_1h
        ? `13C ${t.n_13c ?? 880} 桶 + 1H ${t.n_1h} 维` +
          (t.h1_channels ? `（${t.h1_channels} 通道 + ${t.h1_scalars ?? 0} 标量）` : "")
        : (t.n_13c ? `13C ${t.n_13c} 桶` : "—")
    );
    $("bar").style.width = (t.percent || 0) + "%";
    setText("bar-text", t.total_batches
      ? `${fmt(t.percent, 1)}% · 每 epoch ${fmtInt(t.total_batches)} 个 batch · 更新于 ${t.updated_at || "—"}`
      : "—");

    if (state.charts && state.charts.loss !== lossChartVersion) {
      lossChartVersion = state.charts.loss;
      $("loss-img").src = "/plot/loss.png?v=" + lossChartVersion;
    }
    if (state.charts && state.charts.f1 !== f1ChartVersion) {
      f1ChartVersion = state.charts.f1;
      $("f1-img").src = "/plot/f1.png?v=" + f1ChartVersion;
    }

    renderTable(
      $("runs-table"),
      [
        { key: "run_id", label: "run_id" },
        { key: "loss_fn", label: "loss" },
        { key: "epochs_done", label: "epochs" },
        { key: "best_epoch", label: "best ep" },
        { key: "best_val_loss", label: "best val loss" },
        { key: "total_seconds", label: "耗时" },
        { key: "stopped_reason", label: "停止原因" },
        { key: "finished_at", label: "结束时间" },
      ],
      (state.runs || []).slice().reverse(),
      {
        best_val_loss: (v) => fmt(v),
        total_seconds: (v) => fmtDuration(v),
      }
    );

    renderTable(
      $("history-table"),
      [
        { key: "run_id", label: "run_id" },
        { key: "epoch", label: "epoch" },
        { key: "loss_fn", label: "loss" },
        { key: "train_loss", label: "train loss" },
        { key: "val_loss", label: "val loss" },
        { key: "best_val_loss", label: "best val loss" },
        { key: "epoch_seconds", label: "本 epoch 耗时" },
        { key: "interrupted", label: "中断" },
      ],
      (state.history && state.history.rows ? state.history.rows : []).slice().reverse(),
      {
        train_loss: (v) => fmt(v),
        val_loss: (v) => fmt(v),
        best_val_loss: (v) => fmt(v),
        epoch_seconds: (v) => fmtDuration(v),
        interrupted: (v) => (String(v) === "1" ? "是" : "否"),
      }
    );

    const f1 = state.f1 || {};
    const latest = f1.latest;

    if (latest) {
      const cards = [
        ["阈值", fmt(latest.threshold, 2)],
        ["macro-F1", fmt(latest.macro_f1)],
        ["micro-F1", fmt(latest.micro_f1)],
        ["weighted-F1", fmt(latest.weighted_f1)],
        ["子集准确率", fmt(latest.subset_accuracy)],
        ["Hamming loss", fmt(latest.hamming_loss)],
        ["macro AP", fmt(latest.macro_average_precision)],
        ["最佳阈值 macro-F1", `${fmt(latest.best_macro_f1)} @ ${fmt(latest.best_threshold, 2)}`],
      ];
      $("f1-overall").innerHTML = cards.map(
        ([k, v]) => `<div class="kv">${k}<b>${v}</b></div>`
      ).join("");

      renderTable(
        $("f1-table"),
        [
          { key: "class", label: "class" },
          { key: "support", label: "support" },
          { key: "precision", label: "precision" },
          { key: "recall", label: "recall" },
          { key: "f1", label: "F1" },
          { key: "average_precision", label: "AP" },
          { key: "best_threshold", label: "best thr" },
          { key: "best_f1", label: "best F1" },
        ],
        f1.classes || [],
        {
          precision: (v) => fmt(v),
          recall: (v) => fmt(v),
          f1: (v) => fmt(v),
          average_precision: (v) => fmt(v),
          best_threshold: (v) => fmt(v, 2),
          best_f1: (v) => fmt(v),
        }
      );

      $("f1-note").textContent =
        `评估 run_id: ${latest.run_id} · 模型: ${latest.model_path} · ` +
        (latest.n_1h ? `输入: 13C ${latest.n_13c} + 1H ${latest.n_1h} · ` : "") +
        `样本: ${fmtInt(latest.n_samples)} · 耗时: ${fmtDuration(latest.elapsed_seconds)}` +
        (state.eval && state.eval.running ? " · 新一轮评估进行中…" : "");
    } else {
      $("f1-overall").innerHTML = '<div class="kv">暂无评估结果<b>—</b></div>';
      renderTable($("f1-table"), [{ key: "class", label: "class" }], []);
      $("f1-note").textContent = "还没有 F1 评估结果，点上面的“跑一次 F1 评估”。";
    }

    $("paths").textContent =
      `训练记录: ${state.paths.history} · 运行汇总: ${state.paths.runs} · F1: ${state.paths.f1_classes} · 模型: ${state.paths.model}`;

    const log = await (await fetch("/api/log?tail=120", { cache: "no-store" })).json();
    $("console").textContent = log.text || "（还没有控制台输出；从网页启动训练后这里会实时显示）";
    $("console").scrollTop = $("console").scrollHeight;
  } catch (error) {
    $("conn").textContent = "连接失败：" + error.message;
  }
}

$("btn-start").addEventListener("click", async () => {
  const epochs = Number($("epochs").value || 0);
  const batch_size = Number($("batch").value || 64);
  const lr = Number($("lr").value || 0.001);

  if (epochs === 0 && !confirm("epochs=0 表示一直训练，直到你点“停止训练”。确定开始吗？")) return;

  try {
    const data = await postJSON("/api/train/start", { epochs, batch_size, lr });
    showMessage(`训练已启动 (pid=${data.pid})`);
    setTimeout(refresh, 1200);
  } catch (error) {
    showMessage(error.message, false);
  }
});

$("btn-stop").addEventListener("click", async () => {
  if (!confirm("确定停止训练吗？会在当前 epoch 的断点保存权重和日志后退出。")) return;
  try {
    const data = await postJSON("/api/train/stop");
    showMessage(data.stopped ? "训练已停止并保存" : "已发送停止信号，等待保存…");
    setTimeout(refresh, 1500);
  } catch (error) {
    showMessage(error.message, false);
  }
});

$("btn-eval").addEventListener("click", async () => {
  try {
    const data = await postJSON("/api/eval/start");
    showMessage(`F1 评估已启动 (pid=${data.pid})，完成后本页会自动更新`);
  } catch (error) {
    showMessage(error.message, false);
  }
});

refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="CNMR 训练看板")

    parser.add_argument("--host", default="127.0.0.1", help="监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8090, help="端口，默认 8090")
    parser.add_argument("--debug", action="store_true", help="Flask 调试模式")

    args = parser.parse_args()

    print(f"看板地址: http://{args.host}:{args.port}")
    print(f"日志目录: {LOG_DIR}")

    app.run(host=args.host, port=args.port, threaded=True, debug=args.debug, use_reloader=False)


if __name__ == "__main__":
    main()
