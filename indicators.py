from __future__ import annotations

import pandas as pd


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["ma5"] = result["close"].rolling(5).mean()
    result["ma20"] = result["close"].rolling(20).mean()
    result["ma60"] = result["close"].rolling(60).mean()

    middle = result["close"].rolling(20).mean()
    std = result["close"].rolling(20).std()
    result["boll_mid"] = middle
    result["boll_up"] = middle + 2 * std
    result["boll_low"] = middle - 2 * std

    result["rsi14"] = calculate_rsi(result["close"], period=14)

    ema12 = result["close"].ewm(span=12, adjust=False).mean()
    ema26 = result["close"].ewm(span=26, adjust=False).mean()
    result["macd_dif"] = ema12 - ema26
    result["macd_dea"] = result["macd_dif"].ewm(span=9, adjust=False).mean()
    result["macd_hist"] = 2 * (result["macd_dif"] - result["macd_dea"])

    result["daily_return"] = result["close"].pct_change()
    result["volatility20"] = result["daily_return"].rolling(20).std() * (252**0.5)
    result["drawdown"] = result["close"] / result["close"].cummax() - 1

    return result


def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.where(avg_loss != 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50)
    return rsi.fillna(50)
