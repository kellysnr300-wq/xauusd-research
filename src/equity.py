import pandas as pd


def build_equity_curve(initial_capital: float, trades) -> pd.DataFrame:
    """Build an equity curve from completed trades."""

    equity = initial_capital
    peak = initial_capital
    rows = []

    for trade in trades:
        equity += float(trade.pnl)

        if equity > peak:
            peak = equity

        drawdown = peak - equity

        rows.append({
            "timestamp": trade.exit_time,
            "equity": equity,
            "peak_equity": peak,
            "drawdown": drawdown,
        })

    return pd.DataFrame(rows)
