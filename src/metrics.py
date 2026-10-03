import pandas as pd


def calculate_metrics(trades, initial_capital: float) -> dict:
    """Calculate basic performance statistics from completed trades."""

    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "return_pct": 0.0,
            "average_pnl": 0.0,
        }

    pnls = pd.Series([float(trade.pnl) for trade in trades])

    wins = (pnls > 0).sum()

    total_pnl = float(pnls.sum())
    return_pct = (total_pnl / initial_capital) * 100.0
    average_pnl = float(pnls.mean())
    win_rate = (wins / len(pnls)) * 100.0

    return {
        "trade_count": len(trades),
        "win_rate": win_rate,
        "total_pnl": total_pnl,
        "return_pct": return_pct,
        "average_pnl": average_pnl,
    }
