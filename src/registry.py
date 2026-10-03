"""Strategy storage and loading.

A strategy is two files in strategies/:
    <id>.py    the code (one class with generate_signals(data))
    <id>.json  metadata (name, type, description, default params)
"""

from datetime import datetime
import importlib.util
import inspect
import json
from pathlib import Path
import re

STRATEGY_DIR = Path("strategies")
RESULTS_ROOT = Path("results")

ID_RE = re.compile(r"^[a-z][a-z0-9_]{0,48}$")
MAX_CODE_BYTES = 100_000


def valid_id(strategy_id: str) -> bool:
    return isinstance(strategy_id, str) and bool(ID_RE.match(strategy_id))


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = "s_" + slug
    return slug[:49]


def _paths(strategy_id: str):
    return (
        STRATEGY_DIR / f"{strategy_id}.py",
        STRATEGY_DIR / f"{strategy_id}.json",
    )


def experiment_counts() -> dict:
    counts = {}
    if not RESULTS_ROOT.exists():
        return counts

    for summary in RESULTS_ROOT.glob("*/summary.json"):
        try:
            data = json.loads(summary.read_text())
        except Exception:
            continue
        sid = data.get("strategy_id") or data.get("config", {}).get("strategy")
        if sid:
            counts[sid] = counts.get(sid, 0) + 1

    return counts


def list_strategies() -> list[dict]:
    counts = experiment_counts()
    out = []

    for meta_path in sorted(STRATEGY_DIR.glob("*.json")):
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            continue
        meta["experiments"] = counts.get(meta.get("id"), 0)
        out.append(meta)

    return out


def get_strategy(strategy_id: str) -> dict:
    if not valid_id(strategy_id):
        raise ValueError("Invalid strategy id.")

    py_path, meta_path = _paths(strategy_id)

    if not py_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Strategy not found: {strategy_id}")

    meta = json.loads(meta_path.read_text())
    meta["code"] = py_path.read_text()
    return meta


def save_strategy(
    name: str,
    strategy_type: str,
    description: str,
    params: dict,
    code: str,
    status: str = "saved",
    strategy_id: str | None = None,
) -> dict:
    name = (name or "").strip()

    if not name or len(name) > 80:
        raise ValueError("Name is required (max 80 characters).")

    if not isinstance(params, dict):
        raise ValueError("Parameters must be a JSON object.")

    if len(code.encode()) > MAX_CODE_BYTES:
        raise ValueError("Code is too large.")

    try:
        compile(code, "<strategy>", "exec")
    except SyntaxError as error:
        raise ValueError(
            f"Syntax error on line {error.lineno}: {error.msg}"
        )

    is_new = strategy_id is None
    strategy_id = strategy_id or slugify(name)

    if not valid_id(strategy_id):
        raise ValueError("Invalid strategy id.")

    py_path, meta_path = _paths(strategy_id)

    if is_new:
        taken = meta_path.exists() or any(
            existing.get("name", "").strip().lower() == name.lower()
            for existing in list_strategies()
        )
        if taken:
            raise ValueError(
                f"A strategy named '{name}' already exists. "
                "Open it from the list to edit, or choose another name."
            )
    STRATEGY_DIR.mkdir(exist_ok=True)

    now = datetime.now().isoformat(timespec="seconds")
    created = now

    if meta_path.exists():
        try:
            created = json.loads(meta_path.read_text()).get("created", now)
        except Exception:
            pass

    meta = {
        "id": strategy_id,
        "name": name,
        "type": strategy_type or "Other",
        "description": (description or "").strip(),
        "params": params,
        "status": status if status in ("draft", "saved") else "saved",
        "created": created,
        "updated": now,
    }

    py_path.write_text(code)
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load_strategy_class(strategy_id: str):
    """Return the strategy class (kept for compatibility)."""
    classes = _find_classes(_load_module(strategy_id))
    if len(classes) != 1:
        raise ValueError(
            "Strategy file must define exactly one class with a "
            f"generate_signals method (found {len(classes)})."
        )
    return classes[0]


FUNCTION_NAMES = ("generate_signals", "strategy", "signals")


class FunctionStrategy:
    """Adapter so a plain function can be used as a strategy."""

    def __init__(self, func, **params):
        self._func = func
        self._params = params

    def generate_signals(self, data):
        return self._func(data, **self._params)


def _load_module(strategy_id: str):
    if not valid_id(strategy_id):
        raise ValueError("Invalid strategy id.")

    py_path, _ = _paths(strategy_id)

    if not py_path.exists():
        raise FileNotFoundError(f"Strategy not found: {strategy_id}")

    spec = importlib.util.spec_from_file_location(
        f"user_strategy_{strategy_id}", py_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _find_classes(module):
    return [
        cls
        for _, cls in inspect.getmembers(module, inspect.isclass)
        if cls.__module__ == module.__name__
        and callable(getattr(cls, "generate_signals", None))
    ]


def _find_function(module):
    functions = {
        name: func
        for name, func in inspect.getmembers(module, inspect.isfunction)
        if func.__module__ == module.__name__
    }

    for name in FUNCTION_NAMES:
        if name in functions:
            return functions[name]

    public = [f for n, f in functions.items() if not n.startswith("_")]
    if len(public) == 1:
        return public[0]

    return None


def build_strategy(strategy_id: str, params: dict | None = None):
    """Build a strategy from strategies/<id>.py.

    Accepts exactly one class with generate_signals(data), or a function
    named generate_signals / strategy / signals (or the only public
    function in the file) taking `data` plus keyword parameters.
    """

    module = _load_module(strategy_id)
    params = params or {}
    classes = _find_classes(module)

    if len(classes) == 1:
        return classes[0](**params)

    if len(classes) > 1:
        raise ValueError(
            "Strategy file defines several strategy classes "
            f"({[c.__name__ for c in classes]}). Keep exactly one."
        )

    func = _find_function(module)

    if func is None:
        raise ValueError(
            "No strategy found. Define one class with a generate_signals "
            "method, or a function named generate_signals(data)."
        )

    return FunctionStrategy(func, **params)
