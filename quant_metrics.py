from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import pandas as pd


TRADING_DAYS = 252


@dataclass(frozen=True)
class QuantMetrics:
    cagr: float
    annual_volatility: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float
    calmar: float | None
    var_95: float
    cvar_95: float
    win_rate: float
    atr_14: float
    atr_pct: float
    kelly_fraction: float
    volatility_target_position: float
    sample_days: int


def calculate_quant_metrics(df: pd.DataFrame, risk_free_rate: float = 0.02, target_volatility: float = 0.20) -> QuantMetrics:
    if len(df) < 2:
        raise ValueError("至少需要两个交易日才能计算量化指标")

    close = df["close"].astype(float)
    returns = close.pct_change().dropna()
    sample_days = len(returns)
    total_return = close.iloc[-1] / close.iloc[0] - 1
    cagr = (1 + total_return) ** (TRADING_DAYS / max(sample_days, 1)) - 1

    annual_volatility = returns.std() * sqrt(TRADING_DAYS)
    annual_return = returns.mean() * TRADING_DAYS
    sharpe = safe_divide(annual_return - risk_free_rate, annual_volatility)

    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * sqrt(TRADING_DAYS)
    sortino = safe_divide(annual_return - risk_free_rate, downside_deviation)

    curve = (1 + returns).cumprod()
    drawdown = curve / curve.cummax() - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    calmar = safe_divide(cagr, abs(max_drawdown))

    var_threshold = returns.quantile(0.05)
    tail_returns = returns[returns <= var_threshold]
    var_95 = max(0.0, -float(var_threshold))
    cvar_95 = max(0.0, -float(tail_returns.mean())) if not tail_returns.empty else 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    win_rate = len(wins) / len(returns) if len(returns) else 0.0
    avg_win = wins.mean() if not wins.empty else 0.0
    avg_loss = abs(losses.mean()) if not losses.empty else 0.0
    payoff_ratio = safe_divide(avg_win, avg_loss) or 0.0
    raw_kelly = win_rate - (1 - win_rate) / payoff_ratio if payoff_ratio > 0 else 0.0
    kelly_fraction = min(0.25, max(0.0, float(raw_kelly)))

    atr_14 = calculate_atr(df, 14).iloc[-1]
    latest_close = close.iloc[-1]
    atr_pct = safe_divide(float(atr_14), float(latest_close)) or 0.0

    volatility_target_position = safe_divide(target_volatility, annual_volatility) or 0.0
    volatility_target_position = min(1.0, max(0.0, volatility_target_position))

    return QuantMetrics(
        cagr=float(cagr),
        annual_volatility=float(annual_volatility),
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        calmar=calmar,
        var_95=var_95,
        cvar_95=cvar_95,
        win_rate=float(win_rate),
        atr_14=float(atr_14),
        atr_pct=float(atr_pct),
        kelly_fraction=kelly_fraction,
        volatility_target_position=volatility_target_position,
        sample_days=sample_days,
    )


def calculate_atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window, min_periods=1).mean()


def safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0 or pd.isna(denominator):
        return None
    value = numerator / denominator
    if pd.isna(value):
        return None
    return float(value)
