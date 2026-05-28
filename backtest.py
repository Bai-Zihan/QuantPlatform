from __future__ import annotations

import pandas as pd


def run_ma_cross_backtest(df: pd.DataFrame, fast_window: int, slow_window: int) -> tuple[pd.DataFrame, dict[str, float]]:
    if fast_window >= slow_window:
        raise ValueError("短均线周期必须小于长均线周期")

    result = df.copy()
    result["fast_ma"] = result["close"].rolling(fast_window).mean()
    result["slow_ma"] = result["close"].rolling(slow_window).mean()
    result["signal"] = (result["fast_ma"] > result["slow_ma"]).astype(int)
    result["position"] = result["signal"].shift(1).fillna(0)
    result["asset_return"] = result["close"].pct_change().fillna(0)
    result["strategy_return"] = result["position"] * result["asset_return"]
    result["asset_curve"] = (1 + result["asset_return"]).cumprod()
    result["strategy_curve"] = (1 + result["strategy_return"]).cumprod()

    trades = result["signal"].diff().fillna(0)
    buy_count = int((trades == 1).sum())
    sell_count = int((trades == -1).sum())

    drawdown = result["strategy_curve"] / result["strategy_curve"].cummax() - 1

    metrics = {
        "asset_total_return": float(result["asset_curve"].iloc[-1] - 1),
        "strategy_total_return": float(result["strategy_curve"].iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "buy_count": float(buy_count),
        "sell_count": float(sell_count),
    }
    return result, metrics
