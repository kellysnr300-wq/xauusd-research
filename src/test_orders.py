"""Tests for resting (limit / stop) orders.

Usage:
    python -m src.test_orders
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtester import Backtester
from src.execution import ExecutionModel
from src.intrabar import IntrabarResolver

N = float("nan")


def check(name, ok):
    print(("PASS  " if ok else "FAIL  ") + name)
    if not ok:
        raise SystemExit(1)


class Orders:
    def __init__(self, entry, price=None, stop=None, r=3.0, life=None, cancel=None):
        n = len(entry)
        self.cols = {"entry": entry}
        if price is not None: self.cols["entry_price"] = price
        if stop is not None: self.cols["stop_price"] = stop
        self.cols["target_r"] = [r if e else N for e in entry]
        if life is not None: self.cols["valid_for"] = life
        if cancel is not None: self.cols["cancel_beyond"] = cancel

    def generate_signals(self, data):
        return pd.DataFrame(self.cols, index=data.index)


def candles(rows, start="2025-01-07 10:00"):
    times = pd.date_range(start, periods=len(rows), freq="5min", tz="UTC")
    return pd.DataFrame({
        "timestamp": times,
        "open": [r[0] for r in rows], "high": [r[1] for r in rows],
        "low": [r[2] for r in rows], "close": [r[3] for r in rows],
    })


def run(rows, strategy, **kw):
    bt = Backtester(candles(rows), strategy, **kw)
    return bt, bt.simulate()


def row(first, size, **fill):
    """A column that is NaN except on candle 0."""
    out = [N] * size
    out[0] = first
    return out


# A buy limit at 98 below a market at 100: stop 96 (risk 2), target 3R = 104.
BASE = [
    (100, 101, 99, 100),        # 0 order placed on this close
    (100, 100.5, 99.2, 100),    # 1 low 99.2: not reached
    (100, 100.2, 97.5, 99),     # 2 low 97.5 <= 98: FILLED at 98
    (99, 105, 98.5, 104.5),     # 3 high 105 >= 104: target
    (104.5, 105, 104, 104.8),
]
buy = lambda **k: Orders([1, 0, 0, 0, 0], row(98.0, 5), row(96.0, 5), life=row(3, 5), **k)

bt, t = run(BASE, buy())
check("buy limit fills at the limit price, not the next open",
      len(t) == 1 and t[0].entry_price == 98.0 and t[0].entry_time == candles(BASE).timestamp[2])
check("exact 3R target from the limit price", t[0].exit_reason == "take_profit" and abs(t[0].pnl - 6.0) < 1e-9)

bt, t = run(BASE, Orders([1, 0, 0, 0, 0], row(98.0, 5), row(96.0, 5), life=row(1, 5)))
check("order expires after valid_for candles", len(t) == 0)

# ---- gap through the limit fills at the open --------------------------------
GAP = [(100, 101, 99, 100), (97, 100.5, 96.5, 99), (99, 99.5, 98.5, 99)]
bt, t = run(GAP, Orders([1, 0, 0], row(98.0, 3), row(96.0, 3), life=row(3, 3)))
check("candle opening below the limit fills at the open (better price)",
      t[0].entry_price == 97.0 and t[0].exit_reason == "take_profit" and abs(t[0].pnl - 3.0) < 1e-9)

# ---- sell limit, buy stop, sell stop -----------------------------------------------
SELL = [(100, 101, 99, 100), (100, 102.5, 99.5, 101), (101, 101.5, 95.5, 96), (96, 97, 95, 96)]
bt, t = run(SELL, Orders([-1, 0, 0, 0], row(102.0, 4), row(104.0, 4), life=row(3, 4)))
check("sell limit fills at its price when price rises to it",
      t[0].side == "short" and t[0].entry_price == 102.0 and abs(t[0].pnl - 6.0) < 1e-9)

BREAK = [(100, 101, 99, 100), (100, 102.5, 100.5, 102), (102, 107, 101.5, 106.5), (106.5, 107, 106, 106.8)]
bt, t = run(BREAK, Orders([1, 0, 0, 0], row(102.0, 4), row(100.0, 4), r=2.0, life=row(3, 4)))
check("buy stop (above the market) fills on the breakout at its price",
      t[0].entry_price == 102.0 and t[0].exit_reason == "take_profit" and abs(t[0].pnl - 4.0) < 1e-9)
bt, t = run(BREAK, Orders([1, 0, 0, 0], row(102.0, 4), row(100.0, 4), r=2.0, life=row(3, 4)),
            execution=ExecutionModel(0.0, 0.5))
check("stop entries pay slippage, limit entries do not",
      t[0].entry_price == 102.5
      and run(BASE, buy(), execution=ExecutionModel(0.0, 0.5))[1][0].entry_price == 98.0)

SELL_STOP = [(100, 101, 99, 100), (99.6, 99.8, 97.5, 98), (98, 98.5, 93.5, 94), (94, 95, 93, 94)]
bt, t = run(SELL_STOP, Orders([-1, 0, 0, 0], row(98.0, 4), row(100.0, 4), r=2.0, life=row(3, 4)))
check("sell stop (below the market) fills on the break down",
      t[0].side == "short" and t[0].entry_price == 98.0 and abs(t[0].pnl - 4.0) < 1e-9)

# ---- spread and fill margin -------------------------------------------------------
NEAR = [(100, 101, 99, 100), (100, 100.5, 97.9, 99), (99, 99.5, 98.5, 99), (99, 99.5, 98.5, 99)]
mk = lambda: Orders([1, 0, 0, 0], row(98.0, 4), row(96.0, 4), life=row(3, 4))
check("zero spread: a low of 97.9 fills the 98 limit", len(run(NEAR, mk())[1]) == 1)
bt, t = run(NEAR, mk(), execution=ExecutionModel(0.4, 0.0))
check("0.4 spread: the ask must reach 98, so mid 97.8; a low of 97.9 does not fill", len(t) == 0)
touch = [(100, 101, 99, 100), (100, 100.5, 97.7, 99), (99, 99.5, 98.5, 99), (99, 99.5, 98.5, 99)]
bt, t = run(touch, mk(), execution=ExecutionModel(0.4, 0.0))
check("0.4 spread: a low of 97.7 fills, paying exactly the limit price", len(t) == 1 and abs(t[0].entry_price - 98.0) < 1e-9)
check("fill_margin 0.3 needs a deeper dip", len(run(NEAR, mk(), fill_margin=0.3)[1]) == 0
      and len(run([(100, 101, 99, 100), (100, 100.5, 97.6, 99), (99, 99.5, 98.5, 99), (99, 99.5, 98.5, 99)], mk(), fill_margin=0.3)[1]) == 1)
check("negative fill_margin fills on a touch of the chart price",
      len(run(touch, mk(), execution=ExecutionModel(0.4, 0.0), fill_margin=-0.2)[1]) == 1)

# ---- cancel_beyond ------------------------------------------------------------------
CLOSE_THROUGH = [(100, 100.4, 99.5, 98.8), (98.8, 99, 97.5, 98.5), (98.5, 99, 98, 98.5)]
mk2 = lambda cancel: Orders([1, 0, 0], row(98.0, 3), row(96.0, 3), life=row(3, 3), cancel=cancel)
check("without a cancel level the order fills", len(run(CLOSE_THROUGH, mk2(None))[1]) == 1)
CT = [(100, 101, 99, 100), (100, 100.2, 99.5, 98.8), (98.8, 99, 97.5, 98.5), (98.5, 99, 98, 98.5)]
check("a close beyond cancel_beyond cancels the order before it can fill",
      len(run(CT, Orders([1, 0, 0, 0], row(98.0, 4), row(96.0, 4), life=row(3, 4), cancel=row(99.0, 4)))[1]) == 0)

# ---- one position at a time, cancel_on_fill ---------------------------------------
TWO = [(100, 101, 99, 100), (100, 100.5, 99.5, 100), (100, 100.5, 96.5, 97), (97, 100, 96.8, 99), (99, 99.5, 98.5, 99)]
two = lambda: Orders([1, 1, 0, 0, 0], [98.0, 97.0, N, N, N], [96.0, 95.0, N, N, N], life=[4, 4, N, N, N])
bt, t = run(TWO, two())
check("two live orders: the older fills, the other is cancelled", len(t) == 1 and t[0].entry_price == 98.0)

# ---- invalid order and kind detection ----------------------------------------------
bt, t = run(BASE, Orders([1, 0, 0, 0, 0], row(98.0, 5), row(99.0, 5), life=row(3, 5)))
check("a stop above the entry makes the order invalid (skipped, counted)", len(t) == 0 and bt.skipped_entries == 1)

# ---- same-candle ordering, no 1-minute data -----------------------------------------
CERTAIN_STOP = [(100, 101, 99, 100), (100, 100.5, 95.5, 97), (97, 98, 96, 97)]
mk3 = lambda: Orders([1, 0, 0], row(98.0, 3), row(96.0, 3), life=row(3, 3))
for policy in ("stop_first", "target_first"):
    bt, t = run(CERTAIN_STOP, mk3(), ambiguity=policy)
    check(f"limit buy then stop in the same candle is a loss ({policy})", t[0].exit_reason == "stop_loss")

FILL_AND_TARGET = [(100, 101, 99, 100), (100, 105, 97.5, 99), (99, 99.5, 98.5, 99)]
bt, t = run(FILL_AND_TARGET, mk3(), ambiguity="stop_first")
check("fill + target in one candle, conservative: the target may have come first, trade stays open",
      t[0].exit_reason == "end_of_data" and bt.trade_meta[0]["ambiguous"])
bt, t = run(FILL_AND_TARGET, mk3(), ambiguity="target_first")
check("fill + target in one candle, optimistic: the target counts",
      t[0].exit_reason == "take_profit" and abs(t[0].pnl - 6.0) < 1e-9)

STOP_ENTRY = [(100, 101, 99, 100), (100, 102.5, 99.8, 102), (102, 102.5, 101.5, 102)]
mk4 = lambda: Orders([1, 0, 0], row(102.0, 3), row(100.0, 3), r=2.0, life=row(3, 3))
bt, t = run(STOP_ENTRY, mk4(), ambiguity="stop_first")
check("stop entry, stop level also reached in the candle: conservative = stopped out",
      t[0].exit_reason == "stop_loss")
bt, t = run(STOP_ENTRY, mk4(), ambiguity="target_first")
check("same candle, optimistic: the dip came before the fill, trade survives",
      t[0].exit_reason == "end_of_data")

# ---- same-candle ordering WITH 1-minute data -----------------------------------------
TMP = Path(tempfile.mkdtemp())


def write_minutes(day, rows):
    lines = ["Etc/UTC,Open,High,Low,Close,Volume"]
    for hhmm, o, h, l, c in rows:
        lines.append(f"{day}T{hhmm}:00+00:00,{o},{h},{l},{c},1000")
    (TMP / f"{day}.csv").write_text("\n".join(lines) + "\n")


def resolver():
    return IntrabarResolver(path_for_day=lambda day: TMP / f"{day}.csv")


# candle 1 = 10:05-10:10. Limit 98, stop 96, target 104.
write_minutes("2025-01-07", [
    ("10:05", 100, 100.2, 99.5, 99.8), ("10:06", 99.8, 99.9, 97.5, 98.2), ("10:07", 98.2, 104.6, 98.0, 104.4),
    ("10:08", 104.4, 104.5, 104, 104.2), ("10:09", 104.2, 104.4, 104, 104.2)])
bt, t = run(FILL_AND_TARGET, mk3(), resolver=resolver(), ambiguity="stop_first")
check("1-minute data: filled first, then the target -> a win inside the candle",
      t[0].exit_reason == "take_profit" and bt.trade_meta[0] == {"ambiguous": True, "resolved": True})

write_minutes("2025-01-07", [
    ("10:05", 100, 105, 99.8, 104.5), ("10:06", 104.5, 104.6, 103, 103.5), ("10:07", 103.5, 103.6, 97.5, 98.2),
    ("10:08", 98.2, 99, 98, 98.5), ("10:09", 98.5, 99, 98, 98.5)])
bt, t = run(FILL_AND_TARGET, mk3(), resolver=resolver(), ambiguity="target_first")
check("1-minute data: the high came BEFORE the fill, so no win even when optimistic",
      t[0].exit_reason == "end_of_data" and bt.trade_meta[0]["resolved"])

write_minutes("2025-01-07", [
    ("10:05", 100, 100.2, 99.5, 99.8), ("10:06", 99.8, 104.6, 97.5, 104.0), ("10:07", 104, 104.2, 103.5, 104),
    ("10:08", 104, 104.2, 103.5, 104), ("10:09", 104, 104.2, 103.5, 104)])
bt, t = run(FILL_AND_TARGET, mk3(), resolver=resolver(), ambiguity="stop_first")
check("fill and target inside one minute stay unresolved (fallback applies)",
      t[0].exit_reason == "end_of_data" and bt.trade_meta[0] == {"ambiguous": True, "resolved": False})

# ======================================================================
# Ground truth at scale: resting orders vs an independent minute-by-minute
# simulator (spread 0, no slippage, same conventions).
# ======================================================================
rng = np.random.default_rng(23)
minutes = pd.date_range("2025-03-03 00:00", periods=6 * 1440, freq="1min", tz="UTC")
step = rng.normal(0, 0.5, len(minutes))
c1 = 2000 + step.cumsum()
o1 = np.concatenate([[c1[0]], c1[:-1]])
h1 = np.maximum(o1, c1) + rng.uniform(0, 0.12, len(minutes))
l1 = np.minimum(o1, c1) - rng.uniform(0, 0.12, len(minutes))
for day, idx in pd.Series(range(len(minutes))).groupby(minutes.date.astype(str)):
    write_minutes(day, [(f"{minutes[i]:%H:%M}", o1[i], h1[i], l1[i], c1[i]) for i in idx])

group = np.arange(len(minutes)) // 5
five = pd.DataFrame({
    "timestamp": minutes[::5],
    "open": pd.Series(o1).groupby(group).first().to_numpy(),
    "high": pd.Series(h1).groupby(group).max().to_numpy(),
    "low": pd.Series(l1).groupby(group).min().to_numpy(),
    "close": pd.Series(c1).groupby(group).last().to_numpy(),
})
nc = len(five)
R = 1.5
plan = {}
j = 3
while j < nc - 5:
    side = int(rng.choice([-1, 1]))
    off = float(rng.uniform(0.2, 1.4)) * (1 if rng.random() < 0.7 else -1)  # mostly limits, some stops
    price = five.close[j] - off if side == 1 else five.close[j] + off
    risk = float(rng.uniform(0.5, 1.2))
    stop = price - risk if side == 1 else price + risk
    plan[j] = (side, round(price, 4), round(stop, 4), int(rng.integers(1, 4)))
    j += int(rng.integers(5, 9))

entry = np.zeros(nc, int); eprice = np.full(nc, N); estop = np.full(nc, N); elife = np.full(nc, N)
for j, (side, price, stop, life) in plan.items():
    entry[j], eprice[j], estop[j], elife[j] = side, price, stop, life


class Plan:
    def generate_signals(self, data):
        return pd.DataFrame({"entry": entry, "entry_price": eprice, "stop_price": estop,
                             "target_r": np.where(entry != 0, R, N), "valid_for": elife}, index=data.index)


def reference():
    out, pos, orders = [], None, []
    for j in range(nc):
        o5, h5, l5, c5 = five.open[j], five.high[j], five.low[j], five.close[j]
        mins = [(o1[5 * j + k], h1[5 * j + k], l1[5 * j + k]) for k in range(5)]
        orders = [od for od in orders if j <= od["last"]]
        filled = None
        if pos is None:
            for od in list(orders):
                side, P, kind = od["side"], od["price"], od["kind"]
                if side == 1 and kind == "limit":   trig, gap, down = l5 <= P, o5 <= P, True
                elif side == 1:                     trig, gap, down = h5 >= P, o5 >= P, False
                elif kind == "limit":               trig, gap, down = h5 >= P, o5 >= P, False
                else:                               trig, gap, down = l5 <= P, o5 <= P, True
                if not trig:
                    continue
                orders.remove(od)
                fill = o5 if gap else P
                risk = (fill - od["stop"]) if side == 1 else (od["stop"] - fill)
                if risk <= 0:
                    continue
                take = fill + R * risk if side == 1 else fill - R * risk
                pos = dict(side=side, stop=od["stop"], take=take, born=j, fill=fill, kind=kind, P=P, down=down)
                filled = "gap" if gap else "touch"
                orders = []
                break

        if pos is not None:
            side, stop, take = pos["side"], pos["stop"], pos["take"]
            hits = lambda o, h, l: ((l <= stop, h >= take, o <= stop, o >= take) if side == 1
                                    else (h >= stop, l <= take, o >= stop, o <= take))
            reason = None
            if filled == "touch":
                seen = False
                for o, h, l in mins:
                    s_hit, t_hit, g_s, g_t = hits(o, h, l)
                    if not seen:
                        touched = (l <= pos["P"]) if pos["down"] else (h >= pos["P"])
                        if not touched:
                            continue
                        seen = True
                        if s_hit or t_hit:  # fill and exit in one minute: documented fallback
                            c_s, c_t = hits(o5, h5, l5)[:2]
                            if c_s: reason = "stop_loss"
                            elif c_t and pos["kind"] == "stop": reason = "take_profit"
                            break
                        continue
                    if s_hit and t_hit:
                        reason = "take_profit" if (g_t and not g_s) else "stop_loss"
                        break
                    if s_hit: reason = "stop_loss"; break
                    if t_hit: reason = "take_profit"; break
            else:
                for o, h, l in mins:
                    s_hit, t_hit, g_s, g_t = hits(o, h, l)
                    if s_hit and t_hit:
                        reason = "take_profit" if (g_t and not g_s) else "stop_loss"
                        break
                    if s_hit: reason = "stop_loss"; break
                    if t_hit: reason = "take_profit"; break
            if reason:
                out.append((pos["born"], j, reason, round(pos["fill"], 6)))
                pos = None

        if j in plan:
            side, price, stop, life = plan[j]
            kind = ("limit" if price <= c5 else "stop") if side == 1 else ("limit" if price >= c5 else "stop")
            orders.append(dict(side=side, price=price, stop=stop, kind=kind, last=j + life))
    return out


def trades_of(bt):
    index = {t: i for i, t in enumerate(five.timestamp)}
    return [(index[t.entry_time], index[t.exit_time], t.exit_reason, round(t.entry_price, 6))
            for t in bt.simulate() if t.exit_reason != "end_of_data"]


truth = reference()
engine_bt = Backtester(five, Plan(), resolver=resolver(), ambiguity="stop_first")
engine = trades_of(engine_bt)
amb = sum(m["ambiguous"] for m in engine_bt.trade_meta)
res = sum(m["ambiguous"] and m["resolved"] for m in engine_bt.trade_meta)
reasons = pd.Series([r[2] for r in truth]).value_counts().to_dict()
check(f"scale test is rich: {len(truth)} trades {reasons}, {amb} ambiguous, {res} settled by 1-minute data",
      len(truth) >= 60 and res >= 5)
check(f"resting-order engine matches the minute-by-minute simulator on all {len(truth)} trades (entry, exit, reason, price)",
      engine == truth)
blind = trades_of(Backtester(five, Plan(), ambiguity="stop_first"))
check("without 1-minute data the engine DOES differ (test is discriminating)", blind != truth)

print("\nAll order tests passed.")
