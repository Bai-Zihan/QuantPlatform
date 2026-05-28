from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from io import BytesIO, StringIO
import codecs
from typing import BinaryIO

import pandas as pd


REQUIRED_COLUMNS = ["date", "open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class DataRequest:
    symbol: str
    start_date: date
    end_date: date
    adjust: str = "qfq"


@dataclass(frozen=True)
class StockProfile:
    symbol: str
    name: str
    source: str


@dataclass(frozen=True)
class ResolvedStock:
    symbol: str
    name: str
    source: str


def resolve_stock(query: str) -> ResolvedStock:
    value = query.strip()
    if not value:
        raise ValueError("请输入股票名称或代码")
    if value.isdigit() and len(value) == 6:
        profile = load_stock_profile(value)
        return ResolvedStock(symbol=profile.symbol, name=profile.name, source=profile.source)

    matches = search_stock_by_name(value)
    if not matches:
        raise ValueError(f"没有找到名称包含“{value}”的股票，请换一个关键词或输入 6 位代码")
    if len(matches) > 1:
        preview = "、".join(f"{item.name}({item.symbol})" for item in matches[:8])
        raise ValueError(f"找到多个匹配股票：{preview}。请输入更完整名称或直接输入代码")
    return matches[0]


def search_stock_by_name(keyword: str) -> list[ResolvedStock]:
    tencent_matches = search_stock_by_tencent(keyword)
    if tencent_matches:
        return tencent_matches

    try:
        import akshare as ak

        spot = ak.stock_zh_a_spot_em()
        code_column = first_existing_column(spot, ["代码", "code", "证券代码"])
        name_column = first_existing_column(spot, ["名称", "name", "证券简称"])
        if code_column is None or name_column is None:
            return []
        name_text = spot[name_column].astype(str)
        matched = spot[name_text.str.contains(keyword, case=False, na=False)].copy()
        exact = matched[matched[name_column].astype(str) == keyword]
        if not exact.empty:
            matched = exact
        results = []
        for _, row in matched.head(20).iterrows():
            results.append(
                ResolvedStock(
                    symbol=str(row[code_column]).zfill(6),
                    name=str(row[name_column]),
                    source="东方财富 A 股列表",
                )
            )
        return results
    except Exception:
        return []


def search_stock_by_tencent(keyword: str) -> list[ResolvedStock]:
    try:
        import requests

        response = requests.get(
            "https://smartbox.gtimg.cn/s3/",
            params={"q": keyword, "t": "all"},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        text = response.text.split('"', 1)[1].split('"', 1)[0]
        results = []
        for item in text.split("^"):
            parts = item.split("~")
            if len(parts) >= 3 and parts[0] in {"sh", "sz", "bj"}:
                results.append(ResolvedStock(symbol=parts[1].zfill(6), name=decode_escaped_text(parts[2]), source="腾讯股票搜索"))
        exact = [item for item in results if item.name == keyword]
        return exact or results[:10]
    except Exception:
        return []


def decode_escaped_text(value: str) -> str:
    if "\\u" not in value:
        return value
    try:
        return codecs.decode(value, "unicode_escape")
    except Exception:
        return value


def load_a_share_daily(request: DataRequest) -> pd.DataFrame:
    errors: list[str] = []
    for loader in (load_tencent_daily, load_eastmoney_daily):
        try:
            df = loader(request)
            if not df.empty:
                return df
        except Exception as exc:
            errors.append(f"{loader.__name__}: {exc}")
    raise RuntimeError("网络行情读取失败：" + "；".join(errors))


def load_tencent_daily(request: DataRequest) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，请先运行 pip install -r requirements.txt") from exc

    raw = ak.stock_zh_index_daily_tx(
        symbol=to_prefixed_symbol(request.symbol),
        start_date=request.start_date.strftime("%Y%m%d"),
        end_date=request.end_date.strftime("%Y%m%d"),
    )
    return normalize_market_frame(raw)


def load_eastmoney_daily(request: DataRequest) -> pd.DataFrame:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("缺少 akshare，请先运行 pip install -r requirements.txt") from exc

    start = request.start_date.strftime("%Y%m%d")
    end = request.end_date.strftime("%Y%m%d")
    raw = ak.stock_zh_a_hist(
        symbol=request.symbol,
        period="daily",
        start_date=start,
        end_date=end,
        adjust=request.adjust,
    )
    return normalize_market_frame(raw)


def load_csv(uploaded_file: BinaryIO | BytesIO | StringIO) -> pd.DataFrame:
    raw = pd.read_csv(uploaded_file)
    return normalize_market_frame(raw)


def load_stock_profile(symbol: str) -> StockProfile:
    clean_symbol = symbol.strip()
    prefixed_symbol = to_prefixed_symbol(clean_symbol)
    try:
        import requests

        response = requests.get(
            "https://qt.gtimg.cn/q=" + prefixed_symbol,
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        payload = response.text.split('"', 1)[1].split('"', 1)[0]
        parts = payload.split("~")
        if len(parts) > 2 and parts[1].strip():
            return StockProfile(symbol=parts[2].strip() or clean_symbol, name=parts[1].strip(), source="腾讯实时行情")
    except Exception:
        pass

    try:
        import akshare as ak

        info = ak.stock_individual_info_em(symbol=clean_symbol)
        if not info.empty and {"item", "value"}.issubset(info.columns):
            rows = dict(zip(info["item"], info["value"]))
            name = str(rows.get("股票简称") or rows.get("证券简称") or "").strip()
            code = str(rows.get("股票代码") or clean_symbol).strip()
            if name:
                return StockProfile(symbol=code, name=name, source="东方财富个股资料")
    except Exception:
        pass

    return StockProfile(symbol=clean_symbol, name=clean_symbol, source="代码")


def normalize_market_frame(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    column_map = {
        "date": ["date", "日期", "交易日期", "Date"],
        "open": ["open", "开盘", "开盘价", "Open"],
        "high": ["high", "最高", "最高价", "High"],
        "low": ["low", "最低", "最低价", "Low"],
        "close": ["close", "收盘", "收盘价", "Close"],
        "volume": ["volume", "成交量", "Volume", "vol"],
        "amount": ["amount", "成交额", "Amount"],
        "pct_change": ["pct_change", "涨跌幅", "涨跌幅%"],
    }

    renamed: dict[str, str] = {}
    for standard_name, candidates in column_map.items():
        for candidate in candidates:
            if candidate in raw.columns:
                renamed[candidate] = standard_name
                break

    df = raw.rename(columns=renamed).copy()
    if "volume" not in df.columns and "amount" in df.columns:
        df["volume"] = df["amount"]

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"行情数据缺少必要字段: {', '.join(missing)}")

    df = df[[column for column in ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"] if column in df.columns]]
    df["date"] = pd.to_datetime(df["date"])

    numeric_columns = [column for column in df.columns if column != "date"]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    return df


def to_prefixed_symbol(symbol: str) -> str:
    value = symbol.strip().lower()
    if value.startswith(("sh", "sz", "bj")):
        return value
    if value.startswith(("6", "9")):
        return f"sh{value}"
    if value.startswith(("0", "2", "3")):
        return f"sz{value}"
    if value.startswith(("4", "8")):
        return f"bj{value}"
    return value


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in df.columns:
            return column
    return None
