"""Normalize whatever a strategy returns into the engine's canonical format.

Canonical output (what the Backtester understands)
--------------------------------------------------
Direction (exactly one of):
    signal   int  -1/0/1   position wanted after this candle closes
    entry    int  -1/0/1   one-off entry request (bracket-order mode)

Optional exit levels (float, NaN = not set), all resolved at the fill:
    stop_price       absolute stop price
    stop_distance    stop distance from the fill price
    target_price     absolute target price
    target_distance  target distance from the fill price
    target_r         target as a multiple of the risk (reward:risk)

Accepted shorthands (converted here, using only data up to that candle):
    stop_pct / target_pct   percent of the signal candle's close
    stop_atr / target_atr   multiple of ATR(atr_period)
    rr, rr_ratio, risk_reward, reward_risk   -> target_r
    position, positions, side, direction     -> signal
    entries                                  -> entry
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


class StrategyOutputError(ValueError):
    """The strategy output cannot be understood. Message is user-facing."""


DIRECTION_ALIASES = ("signal", "position", "positions", "side", "direction")
ENTRY_ALIASES = ("entry", "entries")
TARGET_R_ALIASES = ("target_r", "rr", "rr_ratio", "risk_reward", "reward_risk")
LEVEL_COLUMNS = (
    "stop_price", "stop_distance", "target_price", "target_distance",
    "target_r",
)
SHORTHAND_COLUMNS = ("stop_pct", "stop_atr", "target_pct", "target_atr")
CANONICAL_COLUMNS = ("signal", "entry") + LEVEL_COLUMNS

WORD_MAP = {
    "long": 1, "buy": 1, "up": 1, "bull": 1, "bullish": 1,
    "short": -1, "sell": -1, "down": -1, "bear": -1, "bearish": -1,
    "flat": 0, "none": 0, "hold": 0, "neutral": 0, "exit": 0, "": 0,
}


@dataclass
class Normalized:
    frame: pd.DataFrame
    mode: str  # "state" or "event"
    warnings: list[str] = field(default_factory=list)


def _as_frame(raw, n_rows: int, warnings: list[str], index) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        frame = raw.copy()
    elif isinstance(raw, pd.Series):
        frame = raw.to_frame("signal")
    elif isinstance(raw, (list, tuple, np.ndarray)):
        array = np.asarray(raw)
        if array.ndim != 1:
            raise StrategyOutputError(
                "A returned array must be 1-dimensional (one value per candle)."
            )
        frame = pd.DataFrame({"signal": array})
    else:
        raise StrategyOutputError(
            "generate_signals must return a DataFrame, Series or list "
            f"(got {type(raw).__name__})."
        )

    if len(frame) != n_rows:
        raise StrategyOutputError(
            f"Returned {len(frame)} rows for {n_rows} candles "
            "(need exactly one row per candle)."
        )

    if not frame.index.equals(index):
        warnings.append(
            "Output index differs from the data index; aligned by position."
        )
    frame.index = index

    frame.columns = [str(c).strip().lower() for c in frame.columns]
    if frame.columns.duplicated().any():
        dupes = sorted(set(frame.columns[frame.columns.duplicated()]))
        raise StrategyOutputError(f"Duplicate column names: {dupes}")

    return frame


def _to_direction(series: pd.Series, name: str, warnings: list[str]):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(int).to_numpy()

    if not pd.api.types.is_numeric_dtype(series):
        mapped = []
        unknown = set()
        for value in series:
            if pd.isna(value):
                mapped.append(0)
                continue
            key = str(value).strip().lower()
            if key in WORD_MAP:
                mapped.append(WORD_MAP[key])
            else:
                try:
                    mapped.append(float(key))
                except ValueError:
                    unknown.add(str(value))
                    mapped.append(0)
        if unknown:
            raise StrategyOutputError(
                f"Column '{name}' has unrecognised values: "
                f"{sorted(unknown)[:5]}. Use 1/-1/0 or long/short/flat."
            )
        values = np.asarray(mapped, dtype=float)
    else:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)

    nan = np.isnan(values)
    if np.isinf(values).any():
        raise StrategyOutputError(f"Column '{name}' contains infinite values.")
    values = np.where(nan, 0.0, values)

    if np.any((values != np.rint(values)) | (np.abs(values) > 1)):
        warnings.append(
            f"Column '{name}' had values other than -1/0/1; "
            "converted with sign()."
        )

    return np.sign(values).astype(int)


def _positive(series: pd.Series, name: str) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    values = values.replace([np.inf, -np.inf], np.nan)
    if (values.dropna() <= 0).any():
        raise StrategyOutputError(f"Column '{name}' must be positive.")
    return values


def _atr(data: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = data["high"], data["low"], data["close"]
    previous = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - previous).abs(), (low - previous).abs()], axis=1
    ).max(axis=1)
    return true_range.rolling(period).mean()


def normalize_output(raw, data: pd.DataFrame, atr_period: int = 14) -> Normalized:
    """Convert a strategy's raw output to the canonical frame."""

    warnings: list[str] = []
    n_rows = len(data)
    frame = _as_frame(raw, n_rows, warnings, data.index)
    columns = set(frame.columns)

    # ---- direction ------------------------------------------------------
    entry_cols = [c for c in ENTRY_ALIASES if c in columns]
    state_cols = [c for c in DIRECTION_ALIASES if c in columns]

    if entry_cols and state_cols:
        raise StrategyOutputError(
            "Return either state columns (signal/position) or event "
            "columns (entry), not both."
        )
    if not entry_cols and not state_cols:
        raise StrategyOutputError(
            "Output needs a 'signal' (or 'entry') column. "
            f"Found columns: {sorted(columns)}"
        )

    mode = "event" if entry_cols else "state"
    chosen = (entry_cols or state_cols)[0]
    used = {chosen}
    if len(entry_cols or state_cols) > 1:
        warnings.append(
            f"Several direction columns found; using '{chosen}'."
        )
        used.update(entry_cols or state_cols)

    direction = _to_direction(frame[chosen], chosen, warnings)

    # ---- exit levels ----------------------------------------------------
    out = {}
    nan = pd.Series(np.nan, index=frame.index)

    def column(name):
        if name in frame.columns:
            used.add(name)
            return frame[name]
        return None

    for name in ("stop_price", "target_price"):
        values = column(name)
        if values is not None:
            out[name] = _positive(values, name)

    stop_distance = column("stop_distance")
    target_distance = column("target_distance")
    stop_distance = _positive(stop_distance, "stop_distance") if stop_distance is not None else nan.copy()
    target_distance = _positive(target_distance, "target_distance") if target_distance is not None else nan.copy()
    has_stop_distance = "stop_distance" in frame.columns
    has_target_distance = "target_distance" in frame.columns

    close = data["close"].astype(float)
    atr = None
    warm = np.zeros(n_rows, dtype=bool)

    for prefix in ("stop", "target"):
        pct = column(f"{prefix}_pct")
        mult = column(f"{prefix}_atr")
        derived = nan.copy()

        if pct is not None:
            derived = derived.fillna(_positive(pct, f"{prefix}_pct") * close / 100.0)
        if mult is not None:
            if atr is None:
                atr = _atr(data, atr_period)
                warm = atr.isna().to_numpy()
            derived = derived.fillna(_positive(mult, f"{prefix}_atr") * atr)

        if pct is not None or mult is not None:
            if prefix == "stop":
                stop_distance = stop_distance.fillna(derived)
                has_stop_distance = True
            else:
                target_distance = target_distance.fillna(derived)
                has_target_distance = True

    if has_stop_distance:
        out["stop_distance"] = stop_distance
    if has_target_distance:
        out["target_distance"] = target_distance

    target_r = None
    for alias in TARGET_R_ALIASES:
        values = column(alias)
        if values is not None:
            candidate = _positive(values, alias)
            target_r = candidate if target_r is None else target_r.fillna(candidate)
    if target_r is not None:
        out["target_r"] = target_r

    # ATR warm-up: no usable risk yet, so ignore signals there.
    if warm.any():
        ignored = int(((direction != 0) & warm).sum())
        if ignored:
            warnings.append(
                f"Ignored {ignored} signals during ATR warm-up "
                f"({atr_period} candles)."
            )
        direction = np.where(warm, 0, direction)

    # ---- event mode needs a stop and a target on every entry ------------
    if mode == "event":
        active = direction != 0
        if active.any():
            def covered(names):
                have = np.zeros(n_rows, dtype=bool)
                for name in names:
                    if name in out:
                        have |= out[name].notna().to_numpy()
                return have

            no_stop = active & ~covered(("stop_price", "stop_distance"))
            no_target = active & ~covered(
                ("target_price", "target_distance", "target_r"))

            if no_stop.any():
                raise StrategyOutputError(
                    f"{int(no_stop.sum())} entry rows have no stop loss. "
                    "Provide stop_price, stop_distance, stop_pct or stop_atr."
                )
            if no_target.any():
                raise StrategyOutputError(
                    f"{int(no_target.sum())} entry rows have no target. "
                    "Provide target_r (or rr), target_distance, target_price, "
                    "target_pct or target_atr."
                )

    # ---- unknown columns --------------------------------------------------
    ignored_cols = sorted(columns - used)
    if ignored_cols:
        warnings.append(f"Ignored columns: {ignored_cols}")

    result = pd.DataFrame(index=data.index)
    result["entry" if mode == "event" else "signal"] = direction.astype(int)
    for name in LEVEL_COLUMNS:
        if name in out and out[name].notna().any():
            result[name] = out[name].astype(float)

    return Normalized(frame=result, mode=mode, warnings=warnings)
