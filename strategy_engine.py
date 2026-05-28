from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class StrategySignal:
    action: str
    position_hint: str
    score: float
    confidence: str
    risk_level: str
    reasons: list[str]
    warnings: list[str]


def evaluate_signal(df: pd.DataFrame) -> StrategySignal:
    if len(df) < 80:
        return StrategySignal(
            action="观望",
            position_hint="0%-20%",
            score=0,
            confidence="低",
            risk_level="未知",
            reasons=["样本不足，至少建议读取 80 个交易日以上的数据"],
            warnings=["数据太少时，指标容易失真"],
        )

    latest = df.iloc[-1]
    previous = df.iloc[-2]
    recent = df.tail(20)
    score = 0.0
    reasons: list[str] = []
    warnings: list[str] = []

    close = float(latest["close"])
    ma20 = float(latest["ma20"])
    ma60 = float(latest["ma60"])
    rsi = float(latest["rsi14"])
    macd_dif = float(latest["macd_dif"])
    macd_dea = float(latest["macd_dea"])
    prev_macd_dif = float(previous["macd_dif"])
    prev_macd_dea = float(previous["macd_dea"])
    drawdown = float(latest["drawdown"])
    volatility = latest.get("volatility20")

    if close > ma20 > ma60:
        score += 28
        reasons.append("价格站上 MA20 且 MA20 高于 MA60，趋势结构偏强")
    elif close < ma20 < ma60:
        score -= 28
        reasons.append("价格跌破 MA20 且 MA20 低于 MA60，趋势结构偏弱")
    elif close > ma60:
        score += 10
        reasons.append("价格仍在 MA60 上方，中期趋势尚未破坏")
    else:
        score -= 10
        reasons.append("价格在 MA60 下方，中期趋势需要谨慎")

    if macd_dif > macd_dea and prev_macd_dif <= prev_macd_dea:
        score += 18
        reasons.append("MACD 出现金叉，动量有改善迹象")
    elif macd_dif < macd_dea and prev_macd_dif >= prev_macd_dea:
        score -= 18
        reasons.append("MACD 出现死叉，短期动量转弱")
    elif macd_dif > macd_dea:
        score += 8
        reasons.append("MACD 维持多头排列")
    else:
        score -= 8
        reasons.append("MACD 维持空头排列")

    if 45 <= rsi <= 68:
        score += 14
        reasons.append("RSI 处于健康强势区间，未明显过热")
    elif rsi > 78:
        score -= 18
        reasons.append("RSI 明显过热，短线追高风险较高")
    elif rsi < 30:
        score += 8
        reasons.append("RSI 进入超卖区，可能有反弹需求")
        warnings.append("超卖不等于立刻反转，需要趋势确认")
    elif rsi < 42:
        score -= 8
        reasons.append("RSI 偏弱，买盘动能不足")

    volume_signal = evaluate_volume(df)
    score += volume_signal[0]
    reasons.append(volume_signal[1])

    if drawdown < -0.25:
        score -= 12
        reasons.append("当前离阶段高点回撤较深，说明趋势修复仍需要时间")
    elif drawdown > -0.06:
        score += 8
        reasons.append("当前回撤较浅，走势保持相对强势")

    if pd.notna(volatility):
        volatility = float(volatility)
        if volatility > 0.55:
            score -= 16
            warnings.append("20 日年化波动率较高，仓位应降低")
        elif volatility < 0.28:
            score += 6
            reasons.append("波动率处于较温和区间")

    breakout_level = float(recent["high"].iloc[:-1].max())
    breakdown_level = float(recent["low"].iloc[:-1].min())
    if close > breakout_level:
        score += 16
        reasons.append("收盘价突破近 20 日高点，存在趋势突破信号")
    elif close < breakdown_level:
        score -= 16
        reasons.append("收盘价跌破近 20 日低点，风险释放尚未结束")

    score = max(-100, min(100, score))
    action, position_hint = action_from_score(score)
    confidence = confidence_from_score(score)
    risk_level = risk_from_context(drawdown, volatility if pd.notna(volatility) else None)

    if action in {"买入", "加仓"}:
        warnings.append("建议分批执行，并设置跌破 MA20 或固定比例止损")
    elif action in {"减仓", "卖出"}:
        warnings.append("如果已有仓位，可优先控制回撤而不是一次性押方向")

    return StrategySignal(
        action=action,
        position_hint=position_hint,
        score=round(score, 1),
        confidence=confidence,
        risk_level=risk_level,
        reasons=reasons[:6],
        warnings=warnings[:4],
    )


def evaluate_volume(df: pd.DataFrame) -> tuple[float, str]:
    latest = df.iloc[-1]
    volume_ma20 = df["volume"].tail(20).mean()
    if pd.isna(volume_ma20) or volume_ma20 <= 0:
        return 0, "成交量数据不足，未纳入量能判断"

    volume_ratio = float(latest["volume"] / volume_ma20)
    daily_return = float(latest["daily_return"]) if pd.notna(latest["daily_return"]) else 0

    if daily_return > 0.02 and volume_ratio > 1.25:
        return 12, "放量上涨，资金参与度改善"
    if daily_return < -0.02 and volume_ratio > 1.25:
        return -14, "放量下跌，短线抛压较重"
    if volume_ratio < 0.65:
        return -4, "成交量低于近 20 日均量，信号可靠性一般"
    return 4, "成交量相对平稳，未出现异常抛压"


def action_from_score(score: float) -> tuple[str, str]:
    if score >= 55:
        return "买入", "50%-70%"
    if score >= 25:
        return "加仓", "30%-50%"
    if score > -20:
        return "观望", "0%-30%"
    if score > -55:
        return "减仓", "0%-20%"
    return "卖出", "0%"


def confidence_from_score(score: float) -> str:
    absolute = abs(score)
    if absolute >= 65:
        return "高"
    if absolute >= 35:
        return "中"
    return "低"


def risk_from_context(drawdown: float, volatility: float | None) -> str:
    if drawdown < -0.25 or (volatility is not None and volatility > 0.55):
        return "高"
    if drawdown < -0.12 or (volatility is not None and volatility > 0.35):
        return "中"
    return "低"
