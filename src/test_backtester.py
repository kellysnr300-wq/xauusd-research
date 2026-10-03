import pandas as pd

from src.backtester import Backtester
from src.execution import ExecutionModel
from src.test_strategy import DummyStrategy
from src.trades import Trade


data = pd.DataFrame({
    "timestamp": pd.to_datetime([
        "2025-01-01 00:00:00",
        "2025-01-01 00:05:00",
        "2025-01-01 00:10:00",
    ]),
    "open": [2600.0, 2605.0, 2610.0],
    "high": [2606.0, 2611.0, 2615.0],
    "low": [2598.0, 2603.0, 2608.0],
    "close": [2605.0, 2610.0, 2614.0],
})


strategy = DummyStrategy()

execution = ExecutionModel(
    spread=1.0,
    slippage=0.2,
)

backtester = Backtester(
    data=data,
    strategy=strategy,
    initial_capital=10_000.0,
    execution=execution,
)

signals = backtester.run()

entry_price = execution.buy_price(data.iloc[1]["open"])
exit_price = execution.exit_long_price(data.iloc[2]["open"])

trade = Trade(
    side="long",
    entry_time=data.iloc[1]["timestamp"],
    exit_time=data.iloc[2]["timestamp"],
    entry_price=entry_price,
    exit_price=exit_price,
    quantity=1.0,
    pnl=exit_price - entry_price,
    exit_reason="test_exit",
)

backtester.record_trade(trade)

print("Signals:")
print(signals)

print("\nTrade:")
print(backtester.trades[0])

print("\nEquity:", backtester.portfolio.equity)
print("Max drawdown:", backtester.portfolio.max_drawdown)

from src.metrics import calculate_metrics

metrics = calculate_metrics(
    backtester.trades,
    initial_capital=10_000.0,
)

print("\nMetrics:")
for name, value in metrics.items():
    print(f"{name}: {value}")

from src.equity import build_equity_curve

equity_curve = build_equity_curve(
    initial_capital=10_000.0,
    trades=backtester.trades,
)

print("\nEquity curve:")
print(equity_curve)
