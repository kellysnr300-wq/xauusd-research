"""Local dashboard server (standard library only).

Usage:
    python -m src.server            # http://127.0.0.1:8000
    python -m src.server 8080

Security: listens on 127.0.0.1 only, checks the Host header, and every
API call needs a per-session token that is injected into the page.
Strategy code runs in separate processes, never inside the server.
"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)

from src import registry  # noqa: E402
from src.paths import OOS_START  # noqa: E402
from src.timeframes import BASE_TIMEFRAME, TIMEFRAMES  # noqa: E402

DASHBOARD = (ROOT / "dashboard").resolve()
RESULTS = ROOT / "results"
TOKEN = secrets.token_urlsafe(24)
JOB_TIMEOUT_SECONDS = 30 * 60
RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{6}_[a-z][a-z0-9_]*(_[0-9]+)?$")

PORT = 8000
ALLOWED_HOSTS: set[str] = set()
RUNNING: dict[str, subprocess.Popen] = {}
LOCK = threading.Lock()

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
    ".json": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        self.status = status
        self.message = message


# ----------------------------------------------------------------------
# Results helpers
# ----------------------------------------------------------------------

def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def run_status(run_dir: Path) -> dict:
    status = read_json(run_dir / "status.json")
    if status:
        return status
    if (run_dir / "summary.json").exists():  # legacy runs
        return {"stage": "done", "message": "Done"}
    return {"stage": "unknown", "message": ""}


def list_runs(strategy_id: str | None) -> list[dict]:
    runs = []

    if not RESULTS.exists():
        return runs

    for run_dir in sorted(RESULTS.iterdir(), reverse=True):
        summary = read_json(run_dir / "summary.json")
        status = run_status(run_dir)

        if summary is None and status["stage"] in ("unknown",):
            continue

        sid = None
        if summary:
            sid = summary.get("strategy_id") or summary.get(
                "config", {}).get("strategy")
        else:
            m = re.match(r"^\d{8}_\d{6}_(.+?)(_\d+)?$", run_dir.name)
            sid = m.group(1) if m else None

        if strategy_id and sid != strategy_id:
            continue

        runs.append({
            "run_id": run_dir.name,
            "strategy_id": sid,
            "stage": status["stage"],
            "message": status.get("message", ""),
            "total_pnl": summary.get("total_pnl") if summary else None,
            "profit_factor": summary.get("profit_factor") if summary else None,
            "trade_count": summary.get("trade_count") if summary else None,
            "first_year": (summary.get("first_year")
                           or summary.get("config", {}).get("first_year"))
            if summary else None,
            "last_year": (summary.get("last_year")
                          or summary.get("config", {}).get("last_year"))
            if summary else None,
            "created": summary.get("created") if summary else None,
            "timeframe": (summary.get("timeframe")
                          or summary.get("config", {}).get("timeframe")
                          or BASE_TIMEFRAME) if summary else None,
        })

    return runs[:100]


def downsample(frame: pd.DataFrame, limit: int = 600) -> pd.DataFrame:
    if len(frame) <= limit:
        return frame
    step = len(frame) / limit
    idx = sorted({int(i * step) for i in range(limit)} | {len(frame) - 1})
    return frame.iloc[idx]


def enrich_legacy(summary: dict, run_dir: Path) -> None:
    """Fill fields that results from the first runner version lack."""
    config = summary.get("config", {})

    summary.setdefault("strategy_id", config.get("strategy"))
    summary.setdefault("timeframe", config.get("timeframe", BASE_TIMEFRAME))
    summary.setdefault("first_year", config.get("first_year"))
    summary.setdefault("last_year", config.get("last_year"))
    summary.setdefault(
        "params", {k: config[k] for k in ("fast", "slow") if k in config})

    if "wins" in summary or not (run_dir / "trades.csv").exists():
        return

    try:
        pnl = pd.read_csv(run_dir / "trades.csv")["pnl"]
    except Exception:
        return

    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    summary["wins"] = int(len(wins))
    summary["losses"] = int(len(losses))
    summary["avg_win"] = float(wins.mean()) if len(wins) else 0.0
    summary["avg_loss"] = float(losses.mean()) if len(losses) else 0.0
    summary["payoff_ratio"] = (
        summary["avg_win"] / abs(summary["avg_loss"])
        if summary["avg_loss"] < 0 else None)


def run_detail(run_id: str) -> dict:
    if not RUN_ID_RE.match(run_id):
        raise ApiError(400, "Invalid run id.")

    run_dir = RESULTS / run_id
    if not run_dir.is_dir():
        raise ApiError(404, "Run not found.")

    detail = {"run_id": run_id, "status": run_status(run_dir)}
    summary = read_json(run_dir / "summary.json")
    if summary:
        enrich_legacy(summary, run_dir)
    detail["summary"] = summary

    if summary and (run_dir / "equity.csv").exists():
        try:
            equity = pd.read_csv(run_dir / "equity.csv")
            if len(equity):
                equity = downsample(equity)
                detail["equity"] = {
                    "equity": equity["equity"].round(2).tolist(),
                    "drawdown": equity["drawdown"].round(2).tolist(),
                    "start": float(summary.get("config", {}).get(
                        "capital", 10_000.0)),
                    "first": str(equity["timestamp"].iloc[0]),
                    "last": str(equity["timestamp"].iloc[-1]),
                }
        except Exception:
            pass

        if "yearly_pnl" not in summary and (run_dir / "trades.csv").exists():
            try:
                trades = pd.read_csv(run_dir / "trades.csv")
                years = pd.to_datetime(
                    trades["exit_time"], utc=True).dt.year
                summary["yearly_pnl"] = {
                    int(y): round(float(v), 2)
                    for y, v in trades.groupby(years)["pnl"].sum().items()
                }
            except Exception:
                pass

    log = run_dir / "log.txt"
    if detail["status"]["stage"] == "failed" and log.exists():
        detail["log_tail"] = log.read_text()[-600:]

    return detail


# ----------------------------------------------------------------------
# Actions
# ----------------------------------------------------------------------

def parse_params(value) -> dict:
    if value in (None, ""):
        return {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ApiError(400, f"Parameters are not valid JSON: {error}")
    if not isinstance(value, dict):
        raise ApiError(400, "Parameters must be a JSON object.")
    for key, item in value.items():
        if not isinstance(item, (int, float, str, bool, type(None))):
            raise ApiError(400, f"Parameter '{key}' must be a simple value.")
    return value


def number(body, key, default, low, high, cast=float):
    try:
        value = cast(body.get(key, default))
    except (TypeError, ValueError):
        raise ApiError(400, f"'{key}' must be a number.")
    if not (low <= value <= high):
        raise ApiError(400, f"'{key}' must be between {low} and {high}.")
    return value


def start_run(body: dict) -> dict:
    sid = body.get("strategy_id")

    if not registry.valid_id(sid):
        raise ApiError(400, "Invalid strategy.")
    try:
        registry.get_strategy(sid)
    except FileNotFoundError:
        raise ApiError(404, "Strategy not found.")

    params = parse_params(body.get("params"))
    last_allowed = OOS_START.year - 1
    first = number(body, "first_year", 2016, 2016, last_allowed, int)
    last = number(body, "last_year", last_allowed, 2016, last_allowed, int)
    if first > last:
        raise ApiError(400, "First year must not be after last year.")

    timeframe = body.get("timeframe", BASE_TIMEFRAME)
    if timeframe not in TIMEFRAMES:
        raise ApiError(400, f"Unknown timeframe. Choose from: {', '.join(TIMEFRAMES)}.")

    spread = number(body, "spread", 0.30, 0, 20)
    slippage = number(body, "slippage", 0.05, 0, 20)
    quantity = number(body, "quantity", 1.0, 0.0001, 1000)
    capital = number(body, "capital", 10_000.0, 1, 1e9)

    with LOCK:
        for run_id, proc in list(RUNNING.items()):
            if proc.poll() is None:
                raise ApiError(409, "A backtest is already running.")
            RUNNING.pop(run_id, None)

        base = f"{datetime.now():%Y%m%d_%H%M%S}_{sid}"
        run_id, n = base, 1
        while (RESULTS / run_id).exists():
            n += 1
            run_id = f"{base}_{n}"

        run_dir = RESULTS / run_id
        run_dir.mkdir(parents=True)
        (run_dir / "status.json").write_text(
            json.dumps({"stage": "queued", "message": "Queued"}))

        log = open(run_dir / "log.txt", "w")
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "src.run_backtest", sid,
                str(first), str(last),
                "--params", json.dumps(params),
                "--spread", str(spread), "--slippage", str(slippage),
                "--quantity", str(quantity), "--capital", str(capital),
                "--run-id", run_id, "--timeframe", timeframe,
            ],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        RUNNING[run_id] = proc

    def watchdog():
        try:
            proc.wait(timeout=JOB_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            proc.kill()
            (run_dir / "status.json").write_text(json.dumps(
                {"stage": "failed", "message": "Timed out"}))
        finally:
            log.close()

    threading.Thread(target=watchdog, daemon=True).start()
    return {"run_id": run_id}


def check_strategy(sid: str, body: dict) -> dict:
    if not registry.valid_id(sid):
        raise ApiError(400, "Invalid strategy.")
    try:
        meta = registry.get_strategy(sid)
    except FileNotFoundError:
        raise ApiError(404, "Save the strategy before checking it.")

    params = parse_params(body.get("params", meta.get("params")))

    try:
        proc = subprocess.run(
            [sys.executable, "-m", "src.check_strategy", sid,
             "--params", json.dumps(params)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "problems": ["Check timed out (90 s)."]}

    lines = [l for l in proc.stdout.strip().splitlines() if l.strip()]
    try:
        return json.loads(lines[-1])
    except Exception:
        return {"ok": False,
                "problems": [(proc.stderr or proc.stdout)[-500:] or "Check failed."]}


# ----------------------------------------------------------------------
# HTTP handler
# ----------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "XauLab"

    def log_message(self, fmt, *args):
        if args and str(args[1]).startswith(("4", "5")):
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing -------------------------------------------------------

    def send_bytes(self, status, body: bytes, content_type: str):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, status, payload):
        self.send_bytes(
            status, json.dumps(payload, default=str).encode(),
            "application/json")

    def host_ok(self) -> bool:
        return self.headers.get("Host", "") in ALLOWED_HOSTS

    def token_ok(self) -> bool:
        return secrets.compare_digest(
            self.headers.get("X-Token", ""), TOKEN)

    def read_body(self) -> dict:
        if not self.headers.get("Content-Type", "").startswith(
                "application/json"):
            raise ApiError(415, "Content-Type must be application/json.")
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length > 1_000_000:
            raise ApiError(413, "Request too large.")
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            raise ApiError(400, "Invalid JSON.")
        if not isinstance(body, dict):
            raise ApiError(400, "JSON object expected.")
        return body

    def dispatch(self, method):
        if not self.host_ok():
            return self.send_json(403, {"error": "Bad host."})

        url = urlparse(self.path)
        path = url.path

        if path.startswith("/api/"):
            if not self.token_ok():
                return self.send_json(403, {"error": "Bad token."})
            try:
                payload = self.route(method, path[4:], parse_qs(url.query))
                return self.send_json(200, payload)
            except ApiError as error:
                return self.send_json(error.status, {"error": error.message})
            except FileNotFoundError as error:
                return self.send_json(404, {"error": str(error)})
            except ValueError as error:
                return self.send_json(400, {"error": str(error)})
            except Exception as error:  # keep the server alive
                return self.send_json(
                    500, {"error": f"{type(error).__name__}: {error}"})

        if method == "GET":
            return self.serve_static(path)

        self.send_json(405, {"error": "Method not allowed."})

    def do_GET(self):
        self.dispatch("GET")

    def do_POST(self):
        self.dispatch("POST")

    # -- static ---------------------------------------------------------

    def serve_static(self, path: str):
        name = "index.html" if path in ("", "/") else path.lstrip("/")
        target = (DASHBOARD / name).resolve()

        if DASHBOARD not in target.parents or not target.is_file():
            return self.send_json(404, {"error": "Not found."})

        body = target.read_bytes()

        if target.name == "index.html":
            tag = f'<meta name="api-token" content="{TOKEN}">'
            body = body.decode().replace("</head>", tag + "\n</head>", 1).encode()

        self.send_bytes(
            200, body,
            CONTENT_TYPES.get(target.suffix, "application/octet-stream"))

    # -- API routes -----------------------------------------------------

    def route(self, method, path, query):
        if method == "GET" and path == "/timeframes":
            return {"timeframes": [
                {"id": k, "label": v["label"]} for k, v in TIMEFRAMES.items()]}

        if method == "GET" and path == "/strategies":
            return {"strategies": registry.list_strategies()}

        if method == "POST" and path == "/strategies":
            body = self.read_body()
            return registry.save_strategy(
                name=body.get("name", ""),
                strategy_type=body.get("type", "Other"),
                description=body.get("description", ""),
                params=parse_params(body.get("params")),
                code=body.get("code", ""),
                status=body.get("status", "saved"),
                strategy_id=body.get("id") or None,
            )

        m = re.fullmatch(r"/strategies/([a-z0-9_]+)", path)
        if method == "GET" and m:
            return registry.get_strategy(m.group(1))

        m = re.fullmatch(r"/strategies/([a-z0-9_]+)/check", path)
        if method == "POST" and m:
            return check_strategy(m.group(1), self.read_body())

        if method == "POST" and path == "/runs":
            return start_run(self.read_body())

        if method == "GET" and path == "/runs":
            return {"runs": list_runs((query.get("strategy") or [None])[0])}

        m = re.fullmatch(r"/runs/([0-9a-z_]+)", path)
        if method == "GET" and m:
            return run_detail(m.group(1))

        raise ApiError(404, "Unknown endpoint.")


def main():
    global PORT, ALLOWED_HOSTS

    if len(sys.argv) > 1:
        PORT = int(sys.argv[1])

    ALLOWED_HOSTS = {f"127.0.0.1:{PORT}", f"localhost:{PORT}"}
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)

    print(f"Dashboard running: http://127.0.0.1:{PORT}")
    print("Open it in your phone browser. Press Ctrl+C to stop.")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
