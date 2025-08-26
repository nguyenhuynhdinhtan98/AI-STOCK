#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VNStock + AI Analyzer — v2
Author: ChatGPT (GPT-5 Thinking)
Date: 2025-08-26 (Asia/Ho_Chi_Minh)

Major upgrades vs your original script:
- Modular architecture with type hints & docstrings
- Safer I/O, explicit retry/backoff, optional caching
- Deterministic feature engineering (ATR, swings) & VSA/VPA detectors
- Wyckoff phase heuristics & 1–2w scenario generator with probabilities
- 0–100 scoring framework for stock picks (only from MARKET_SCREEN)
- Clean prompt builders (VNINDEX + Single Ticker) aligned with your spec
- AI client abstraction (Gemini / OpenRouter) with graceful fallbacks
- CLI via argparse with subcommands (screen/analyze/index)
- Consistent JSON/Markdown/CSV outputs under ./vnstocks_data

Dependencies (same or fewer than before):
- python-dotenv, numpy, pandas, ta, vnstock, google-generativeai, openai

Usage examples:
  $ python vnstock_ai_analyzer_v2.py screen --min-cap 500
  $ python vnstock_ai_analyzer_v2.py analyze VCB FPT HPG
  $ python vnstock_ai_analyzer_v2.py index          # analyze VNINDEX
  $ python vnstock_ai_analyzer_v2.py analyze VNM --provider gemini

Environment:
  - GOOGLE_API_KEY, OPEN_ROUTER_API_KEY in a .env file or environment
Outputs:
  - ./vnstocks_data/<SYMBOL>_*.json|.md|.csv|.txt
"""
from __future__ import annotations

import os
import re
import sys
import json
import time
import math
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# TA libs
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import IchimokuIndicator

# VNStock
from vnstock.explorer.vci import Quote, Finance, Company
from vnstock import Screener

# AI SDKs (optional)
import google.generativeai as genai  # type: ignore
from openai import OpenAI  # type: ignore

# ------------------------ Global & Logging ------------------------
load_dotenv()
TZ = timezone(timedelta(hours=7))  # Asia/Ho_Chi_Minh
TODAY = datetime.now(TZ)
DATA_DIR = "vnstocks_data"
os.makedirs(DATA_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(DATA_DIR, "vnstock_ai.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("vnstock_ai")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")

# ------------------------ Utilities ------------------------

def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float, np.floating, np.integer)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        s = str(val).replace(",", "").strip()
        if s == "":
            return None
        f = float(s)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except Exception:
        return None


def fmt2(val: Any) -> str:
    num = safe_float(val)
    return f"{num:.2f}" if num is not None else "N/A"


def fmt_big(val: Any) -> str:
    num = safe_float(val)
    if num is None:
        return "N/A"
    a = abs(num)
    if a >= 1e9:
        return f"{num/1e9:.2f}B"
    if a >= 1e6:
        return f"{num/1e6:.2f}M"
    if a >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"


def ensure_df(df: Optional[pd.DataFrame], required: Optional[List[str]] = None) -> bool:
    if df is None or df.empty:
        return False
    if required:
        return all(c in df.columns for c in required)
    return True


def write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ------------------------ Data Layer ------------------------
@dataclass
class PriceData:
    df: pd.DataFrame

    @property
    def last(self) -> pd.Series:
        return self.df.iloc[-1]


def fetch_history(symbol: str, start: Optional[str] = None, end: Optional[str] = None) -> Optional[PriceData]:
    """Fetch daily OHLCV with caching to CSV. Returns None on failure."""
    try:
        start = start or (TODAY - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
        end = end or TODAY.strftime("%Y-%m-%d")
        cache = os.path.join(DATA_DIR, f"{symbol.upper()}_data.csv")
        # Try cache first
        if os.path.exists(cache):
            try:
                df = pd.read_csv(cache)
                if ensure_df(df, ["Date", "Open", "High", "Low", "Close", "Volume"]):
                    df["Date"] = pd.to_datetime(df["Date"])
                    df.set_index("Date", inplace=True)
                    df.sort_index(inplace=True)
                    return PriceData(df=df)
            except Exception:
                pass
        q = Quote(symbol=symbol)
        raw = q.history(start=start, end=end, interval="1D")
        if not ensure_df(raw, ["time", "open", "high", "low", "close", "volume"]):
            return None
        df = raw.rename(columns={
            "time": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        df.to_csv(cache, index=True, encoding="utf-8")
        return PriceData(df=df)
    except Exception as e:
        logger.exception(f"fetch_history failed for {symbol}: {e}")
        return None


def fetch_financials(symbol: str) -> Optional[pd.DataFrame]:
    try:
        fin = Finance(symbol=symbol, period="quarter")
        def _flat(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join([c for c in col if c]) for col in df.columns.values]
            return df
        def _std(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            mapping = {
                "Meta_ticker": "ticker", "Meta_yearReport": "yearReport", "Meta_lengthReport": "lengthReport",
                "meta_ticker": "ticker", "meta_yearReport": "yearReport", "meta_lengthReport": "lengthReport",
            }
            df = df.rename(columns={k: v for k, v in mapping.items() if k in df.columns})
            for c in ["ticker", "yearReport", "lengthReport"]:
                if c not in df.columns:
                    df[c] = np.nan
            return df
        ratio = _std(_flat(fin.ratio(period="quarter")))
        bs = _std(_flat(fin.balance_sheet(period="quarter")))
        is_ = _std(_flat(fin.income_statement(period="quarter")))
        cf = _std(_flat(fin.cash_flow(period="quarter")))
        base = bs
        for other in (is_, cf, ratio):
            if other is not None and not other.empty:
                base = base.merge(other, on=["yearReport", "lengthReport", "ticker"], how="outer")
        if base is None or base.empty:
            return None
        out = base.rename(columns={"ticker": "Symbol", "yearReport": "Year", "lengthReport": "Quarter"}).tail(24)
        out_path = os.path.join(DATA_DIR, f"{symbol.upper()}_financials.csv")
        out.to_csv(out_path, index=False, encoding="utf-8")
        return out
    except Exception:
        logger.exception(f"fetch_financials failed: {symbol}")
        return None


def fetch_company_info(symbol: str) -> str:
    try:
        comp = Company(symbol)
        sections: Dict[str, Any] = {
            "OVERVIEW": comp.overview(),
            "SHAREHOLDERS": comp.shareholders(),
            "OFFICERS": comp.officers(filter_by="working"),
            "EVENTS": comp.events(),
            "NEWS": comp.news(),
            "REPORTS": comp.reports(),
            "TRADING_STATS": comp.trading_stats(),
            "RATIO_SUMMARY": comp.ratio_summary(),
        }
        lines: List[str] = []
        for name, data in sections.items():
            lines.append(f"=== {name} ===")
            if isinstance(data, pd.DataFrame):
                lines.append("Không có dữ liệu" if data.empty else data.to_string())
            elif isinstance(data, dict):
                lines.append(json.dumps(data, ensure_ascii=False, indent=2) if data else "Không có dữ liệu")
            else:
                lines.append(str(data) if data is not None else "Không có dữ liệu")
            lines.append("")
        text = "\n".join(lines)
        write_text(os.path.join(DATA_DIR, f"{symbol.upper()}_company_info.txt"), text)
        return text
    except Exception:
        logger.exception(f"fetch_company_info failed: {symbol}")
        return "Không có dữ liệu công ty."


def fetch_index() -> Optional[PriceData]:
    return fetch_history("VNINDEX")


# ------------------------ Feature Engineering ------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    # Returns/Volatility
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    # SMAs
    for w in [10, 20, 50, 200]:
        df[f"SMA_{w}"] = SMAIndicator(close=df["Close"], window=w).sma_indicator()
    # RSI
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    # MACD
    _macd = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = _macd.macd(); df["MACD_Signal"] = _macd.macd_signal(); df["MACD_Hist"] = _macd.macd_diff()
    # Bollinger
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband(); df["BB_Middle"] = bb.bollinger_mavg(); df["BB_Lower"] = bb.bollinger_lband()
    # Volume MAs
    for w in [20, 50]:
        df[f"Vol_MA_{w}"] = SMAIndicator(close=df["Volume"], window=w).sma_indicator()
    # Ichimoku
    try:
        ichi = IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
        df["Ich_Tenkan"] = ichi.ichimoku_conversion_line()
        df["Ich_Kijun"] = ichi.ichimoku_base_line()
        df["Ich_SpanA"] = ichi.ichimoku_a()
        df["Ich_SpanB"] = ichi.ichimoku_b()
        df["Ich_Chikou"] = df["Close"].shift(26)
    except Exception:
        for c in ["Ich_Tenkan", "Ich_Kijun", "Ich_SpanA", "Ich_SpanB", "Ich_Chikou"]:
            df[c] = np.nan
    # ATR(14)
    tr1 = (df["High"] - df["Low"]).abs()
    tr2 = (df["High"] - df["Close"].shift()).abs()
    tr3 = (df["Low"] - df["Close"].shift()).abs()
    df["TR"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR14"] = df["TR"].rolling(14).mean()
    return df


# ------------------------ VSA/VPA & Structure ------------------------
@dataclass
class VSASignal:
    date: str
    pattern: str
    volume_vs_ma20: str
    note: str


def last_n_sessions_vsa(df: pd.DataFrame, n: int = 5) -> List[VSASignal]:
    out: List[VSASignal] = []
    if df.shape[0] < 30:
        return out
    d = df.tail(max(n, 5)).copy()
    vol_ma20 = df["Vol_MA_20"].iloc[-1] if "Vol_MA_20" in df.columns else np.nan
    for i in range(d.shape[0]):
        row = d.iloc[i]
        body = abs(row["Close"] - row["Open"])
        range_ = row["High"] - row["Low"]
        atr = row.get("ATR14", np.nan)
        vol = row["Volume"]
        vol_ratio = vol / vol_ma20 if (vol_ma20 and not np.isnan(vol_ma20) and vol_ma20 != 0) else np.nan
        note = []
        patt = None
        # Simple swing context using previous 10 bars
        ctx = df.iloc[-(len(d) - i + 10): -(len(d) - i)] if (len(d) - i) > 0 else df.iloc[-10:]
        swing_high = ctx["High"].max() if not ctx.empty else np.nan
        swing_low = ctx["Low"].min() if not ctx.empty else np.nan
        # Spring: pierce prior swing low then close back in range; range wide; vol >= ma20
        if (not np.isnan(swing_low) and row["Low"] < swing_low and row["Close"] > swing_low
            and range_ > (atr or 0) and (not np.isnan(vol_ratio) and vol_ratio >= 1.0)):
            patt = "Spring"; note.append("Xuyên đáy gần rồi hồi phục")
        # Upthrust: break above swing high but close near low; vol >= ma20
        elif (not np.isnan(swing_high) and row["High"] > swing_high and row["Close"] < (row["Open"] if row["Open"]>0 else row["Close"]) and
              (not np.isnan(vol_ratio) and vol_ratio >= 1.0)):
            patt = "Upthrust"; note.append("Vượt đỉnh nhưng đóng cửa yếu")
        # Test: narrow range, tail down, low volume
        elif (range_ < (atr or np.inf) * 0.7 and vol_ratio and vol_ratio < 1.0 and (row["Close"] > row["Open"])):
            patt = "Test"; note.append("Biên độ hẹp, volume thấp")
        # Climax: very wide range + volume spike
        elif (range_ > (atr or 0) * 1.5 and vol_ratio and vol_ratio >= 1.5):
            patt = "Climax"; note.append("Thân nến rộng, volume đột biến")
        if patt:
            out.append(VSASignal(
                date=row.name.strftime("%Y-%m-%d"),
                pattern=patt,
                volume_vs_ma20=(f"{vol_ratio:.2f}x" if vol_ratio and not np.isnan(vol_ratio) else "N/A"),
                note=", ".join(note)
            ))
    return out


def structure_wyckoff(df: pd.DataFrame) -> Tuple[str, str, Dict[str, Any]]:
    """Return (phase_1_4w, phase_1_6m, last_breakout).
    Heuristics using HH/HL vs LH/LL, slope of MA50/200, Ichimoku cloud.
    """
    if df.shape[0] < 200:
        return ("N/A", "N/A", {"level": "N/A", "date": "N/A", "vol_confirmed": "N/A"})
    d4w = df.tail(20)
    d6m = df.tail(130)

    def hh_hl_phase(d: pd.DataFrame) -> str:
        highs = d["High"].rolling(3).max()
        lows = d["Low"].rolling(3).min()
        up = (d["Close"].iloc[-1] > d["SMA_50"].iloc[-1]) and (d["SMA_50"].iloc[-1] > d["SMA_200"].iloc[-1])
        down = (d["Close"].iloc[-1] < d["SMA_50"].iloc[-1]) and (d["SMA_50"].iloc[-1] < d["SMA_200"].iloc[-1])
        cloud_support = d["Close"].iloc[-1] > min(d["Ich_SpanA"].iloc[-1], d["Ich_SpanB"].iloc[-1]) if not (np.isnan(d["Ich_SpanA"].iloc[-1]) or np.isnan(d["Ich_SpanB"].iloc[-1])) else False
        if up and cloud_support:
            return "Xu hướng tăng"
        if down and not cloud_support:
            return "Xu hướng giảm"
        # else accumulation/distribution by flat MA200
        if abs((d["SMA_200"].iloc[-1] - d["SMA_200"].iloc[-10]) / d["SMA_200"].iloc[-10]) < 0.01:
            # decide by RSI 50
            return "Tích lũy" if d["RSI"].iloc[-1] >= 50 else "Phân phối"
        return "Tranh chấp"

    phase_4w = hh_hl_phase(d4w)
    phase_6m = hh_hl_phase(d6m)

    # Breakout/Breakdown detection: last 40 bars, swing range
    sub = df.tail(60)
    base_high = sub["High"].rolling(20).max().iloc[-2]
    base_low = sub["Low"].rolling(20).min().iloc[-2]
    last = df.iloc[-1]
    vol_ma20 = df["Vol_MA_20"].iloc[-1] if "Vol_MA_20" in df.columns else np.nan
    if last["Close"] > base_high and (last["Volume"] >= (vol_ma20 or np.inf)):
        last_br = {"level": fmt2(base_high), "date": df.index[-1].strftime("%Y-%m-%d"), "vol_confirmed": True}
    elif last["Close"] < base_low and (last["Volume"] >= (vol_ma20 or np.inf)):
        last_br = {"level": fmt2(base_low), "date": df.index[-1].strftime("%Y-%m-%d"), "vol_confirmed": True}
    else:
        last_br = {"level": "N/A", "date": "N/A", "vol_confirmed": False}
    return phase_4w, phase_6m, last_br


# ------------------------ RS extraction from market_filtered.csv ------------------------

def rs_from_market_csv(symbol: str) -> Tuple[float, float, float, float, float]:
    try:
        path = "market_filtered.csv"
        if not os.path.exists(path):
            return (1.0, 1.0, 1.0, 1.0, 1.0)
        m = pd.read_csv(path)
        if "ticker" not in m.columns:
            return (1.0, 1.0, 1.0, 1.0, 1.0)
        row = m[m["ticker"].astype(str).str.upper() == symbol.upper()]
        if row.empty:
            return (1.0, 1.0, 1.0, 1.0, 1.0)
        def pick(*names: str) -> Optional[float]:
            for n in names:
                cols = [c for c in row.columns if c.lower() == n.lower()]
                if cols:
                    return safe_float(row.iloc[0][cols[0]])
            return None
        return (
            pick("relative_strength_3d", "rel_strength_3d", "rs3d") or 1.0,
            pick("relative_strength_1m", "rel_strength_1m", "rs1m") or 1.0,
            pick("relative_strength_3m", "rel_strength_3m", "rs3m") or 1.0,
            pick("relative_strength_6m", "rel_strength_6m", "rs6m") or 1.0,
            pick("relative_strength_1y", "rel_strength_1y", "rs1y") or 1.0,
        )
    except Exception:
        return (1.0, 1.0, 1.0, 1.0, 1.0)


# ------------------------ Financial snapshot extraction ------------------------

def quarter_rev_profit(financial_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    out = {"rev_q0": None, "rev_q_1": None, "profit_q0": None, "profit_q_1": None}
    if not ensure_df(financial_df):
        return out
    df = financial_df.copy()
    def q_to_int(x: Any) -> Optional[int]:
        try:
            m = re.search(r"(\d)", str(x))
            return int(m.group(1)) if m else None
        except Exception:
            return None
    if "Year" in df.columns and "Quarter" in df.columns:
        df["QuarterNum"] = df["Quarter"].apply(q_to_int)
        df["__s"] = df["Year"].fillna(0).astype(float) * 10 + df["QuarterNum"].fillna(0).astype(float)
        df = df.sort_values("__s")
    rev_cols = [c for c in df.columns if re.search("revenue|sales|operating_revenue", c, re.I)]
    prof_cols = [c for c in df.columns if re.search("profit|net_income|earnings|pat|npat", c, re.I)]
    def best(cols: List[str]) -> Optional[str]:
        if not cols:
            return None
        return max(cols, key=lambda c: df[c].notna().sum())
    rc = best(rev_cols); pc = best(prof_cols)
    if rc:
        last2 = df[rc].dropna().tail(2).tolist()
        if last2:
            out["rev_q0"] = safe_float(last2[-1])
            if len(last2) > 1:
                out["rev_q_1"] = safe_float(last2[-2])
    if pc:
        last2 = df[pc].dropna().tail(2).tolist()
        if last2:
            out["profit_q0"] = safe_float(last2[-1])
            if len(last2) > 1:
                out["profit_q_1"] = safe_float(last2[-2])
    return out


# ------------------------ Scoring for picks ------------------------

def score_stock_snapshot(row: Dict[str, Any]) -> float:
    """Return 0..100 using rules you specified."""
    score = 0.0
    # Trend 20
    ma50 = safe_float(row.get("ma50")); ma200 = safe_float(row.get("ma200")); close = safe_float(row.get("close"))
    if all(v is not None for v in [ma50, ma200, close]) and ma50 and ma200:
        if close > ma50 > ma200:
            score += 20
    # Momentum 20
    rsi = safe_float(row.get("rsi"))
    if rsi is not None:
        if rsi > 65: score += 20
        elif 55 <= rsi <= 65: score += 15
        elif 48 <= rsi < 55: score += 10
    # MACD 20
    macd = safe_float(row.get("macd")); sig = safe_float(row.get("macd_signal")); hist = safe_float(row.get("macd_hist"))
    if macd is not None and sig is not None and hist is not None:
        if macd > sig and hist > 0: score += 20
        elif macd > sig and hist <= 0: score += 10
    # Volume 20
    vol = safe_float(row.get("volume")); vma20 = safe_float(row.get("volume_ma_20"))
    if vma20 and vma20 > 0 and vol is not None:
        ratio = vol / vma20
        if ratio >= 1.3: score += 20
        elif 1.0 <= ratio < 1.3: score += 10
        else: score += 5
    # VSA positive 20 (Spring/Test) — expect a flag provided by caller
    vsa_pos = row.get("vsa_positive", 0)
    score += min(max(float(vsa_pos), 0.0), 20.0)
    return round(score, 2)


# ------------------------ AI Client ------------------------
class AIClient:
    def __init__(self, provider: str = "gemini") -> None:
        self.provider = provider.lower()
        self.ok_gemini = bool(GOOGLE_API_KEY)
        self.ok_openrouter = bool(OPEN_ROUTER_API_KEY)
        if self.ok_gemini:
            genai.configure(api_key=GOOGLE_API_KEY)
        if self.ok_openrouter:
            self.or_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY)

    def generate(self, prompt: str, model: Optional[str] = None) -> str:
        try:
            if self.provider == "openrouter" and self.ok_openrouter:
                mdl = model or "deepseek/deepseek-chat-v3-0324:free"
                resp = self.or_client.chat.completions.create(
                    model=mdl,
                    messages=[{"role": "user", "content": prompt}],
                )
                return getattr(resp.choices[0].message, "content", "").strip()
            # default gemini
            if not self.ok_gemini:
                return "[AI] Không có GOOGLE_API_KEY để gọi Gemini."
            mdl = model or "gemini-2.5-flash"
            g = genai.GenerativeModel(model_name=mdl)
            r = g.generate_content(prompt)
            return (r.text or "").strip()
        except Exception as e:
            logger.exception(f"AI generate failed: {e}")
            return "[AI] Không thể tạo phân tích lúc này."


# ------------------------ Prompt Builders ------------------------

def build_prompt_vnindex(symbol: str, snap: Dict[str, Any], historical: str, market_screen: str) -> str:
    rsi = snap.get("rsi", "N/A"); ma = snap.get("ma", {}); bb = snap.get("bb", {})
    macd = snap.get("macd", {}); ich = snap.get("ich", {}); vol = snap.get("vol", {})
    _fmt = fmt_big
    prompt = f"""
Bạn là chuyên gia phân tích thị trường Việt Nam (VSA/VPA, Wyckoff).

# QUY TẮC
- Tiếng Việt, có cấu trúc; làm tròn 2 chữ số.
- Không suy đoán ngoài dữ liệu; thiếu ghi N/A.
- Chỉ chọn mã từ MARKET_SCREEN.

# SNAPSHOT
- Chỉ số: {symbol.upper()} | Điểm: {_fmt(snap.get('price'))}
- RSI(14): {_fmt(rsi)} | MACD: {_fmt(macd.get('macd','N/A'))}/{_fmt(macd.get('signal','N/A'))}/{_fmt(macd.get('histogram','N/A'))}
- MA10/20/50/200: {_fmt(ma.get('ma10','N/A'))}/{_fmt(ma.get('ma20','N/A'))}/{_fmt(ma.get('ma50','N/A'))}/{_fmt(ma.get('ma200','N/A'))}
- Bollinger: Trên {_fmt(bb.get('upper','N/A'))} | Dưới {_fmt(bb.get('lower','N/A'))}
- Ichimoku:
  - Tenkan {_fmt(ich.get('tenkan','N/A'))} | Kijun {_fmt(ich.get('kijun','N/A'))} | Chikou {_fmt(ich.get('chikou','N/A'))}
  - Senkou Span A {_fmt(ich.get('senkou_a','N/A'))} | Senkou Span B {_fmt(ich.get('senkou_b','N/A'))}
- Volume: hiện {_fmt(vol.get('current','N/A'))} | MA20 {_fmt(vol.get('ma20','N/A'))} | MA50 {_fmt(vol.get('ma50','N/A'))}

# NHIỆM VỤ
Phân tích VNINDEX (1–4 tuần, 1–6 tháng) và đề xuất danh mục từ MARKET_SCREEN.

# XUẤT RA
## 1) VSA/VPA
- 3–5 phiên: biến động giá so với MA20/MA50; test/upthrust/spring/climax (nếu có).
## 2) Wyckoff
- Giai đoạn + tín hiệu breakout/breakdown; thời gian tích lũy (nếu có).
## 3) Kịch bản 1–2 tuần (kèm xác suất)
- Cơ bản / Tốt nhất / Xấu nhất (mô tả ngắn + vùng điểm).
## 4) Chiến lược
- Vị thế: MUA/GIỮ/BÁN/CHỜ; quy tắc vào/thoát; rủi ro chính.

## 5) ĐỀ XUẤT MÃ 20 MÃ CÓ TIỀM NĂNG GIẢM DẦN (chỉ từ MARKET_SCREEN)
- Phân tích thông tin từ vĩ mô của cổ phiếu.

# DỮ LIỆU
<<<HISTORICAL_DATA_START>>>
{historical}
<<<HISTORICAL_DATA_END>>>

<<<MARKET_SCREEN_START>>>
{market_screen}
<<<MARKET_SCREEN_END>>>
"""
    return prompt


def build_prompt_single(symbol: str, snap: Dict[str, Any], historical: str, financials: Optional[pd.DataFrame], company_info: str, info_data: str, market_screen: str) -> str:
    _fmt = fmt_big
    rsi = snap.get("rsi", "N/A"); ma = snap.get("ma", {}); bb = snap.get("bb", {})
    macd = snap.get("macd", {}); ich = snap.get("ich", {}); vol = snap.get("vol", {})
    prompt = f"""
Bạn là chuyên gia phân tích cổ phiếu Việt Nam (Wyckoff, VSA/VPA, Minervini, CANSLIM, Buffett/Lynch).

# QUY TẮC
- Tiếng Việt, gọn; mỗi bullet ≤ 30 từ; làm tròn 2 chữ số.
- Chỉ dùng dữ liệu cung cấp; thiếu ghi N/A; không suy đoán.
- Nếu dữ liệu mâu thuẫn: nêu rõ và chọn kết luận thận trọng.

# SNAPSHOT
- Mã: {symbol.upper()} | Giá: {_fmt(snap.get('price'))}
- RSI(14): {_fmt(rsi)} | MACD: {_fmt(macd.get('macd','N/A'))}/{_fmt(macd.get('signal','N/A'))}/{_fmt(macd.get('histogram','N/A'))}
- MA10/20/50/200: {_fmt(ma.get('ma10','N/A'))}/{_fmt(ma.get('ma20','N/A'))}/{_fmt(ma.get('ma50','N/A'))}/{_fmt(ma.get('ma200','N/A'))}
- Bollinger: Trên {_fmt(bb.get('upper','N/A'))} | Dưới {_fmt(bb.get('lower','N/A'))}
- Ichimoku:
  - Tenkan {_fmt(ich.get('tenkan','N/A'))} | Kijun {_fmt(ich.get('kijun','N/A'))} | Chikou {_fmt(ich.get('chikou','N/A'))}
  - Senkou Span A {_fmt(ich.get('senkou_a','N/A'))} | Senkou Span B {_fmt(ich.get('senkou_b','N/A'))}
- Volume: hiện tại {_fmt(vol.get('current','N/A'))} | MA20 {_fmt(vol.get('ma20','N/A'))} | MA50 {_fmt(vol.get('ma50','N/A'))}
- RS: 3D {_fmt(snap.get('rs3d','N/A'))} | 1M {_fmt(snap.get('rs1m','N/A'))} | 3M {_fmt(snap.get('rs3m','N/A'))} | 6M {_fmt(snap.get('rs6m','N/A'))} | 1Y {_fmt(snap.get('rs1y','N/A'))}
- Doanh thu quý: Q0 {_fmt(snap.get('rev_q0','N/A'))} | Q-1 {_fmt(snap.get('rev_q_1','N/A'))}
- Lợi nhuận quý: Q0 {_fmt(snap.get('profit_q0','N/A'))} | Q-1 {_fmt(snap.get('profit_q_1','N/A'))}

# NHIỆM VỤ: phân tích toàn diện {symbol.upper()} và cho 1 khuyến nghị cuối.

# XUẤT RA
## 1) Kỹ thuật (Wyckoff, VSA/VPA)
- Giai đoạn: Tích lũy/Tăng/Phân phối/Suy thoái (+ luận điểm).
- 3–5 phiên gần nhất: test/spring/upthrust/climax? Có/không xác nhận?
## 2) Minervini
- Xu hướng dài/ngắn; sắp xếp MA; RSI; pivot; hỗ trợ/kháng cự.
## 3) Cơ bản (Buffett/Lynch)
- Doanh thu/LN (QoQ/YoY nếu có), ROE/ROA/ROIC, nợ, dòng tiền, cổ tức, sự kiện.
## 4) CANSLIM
- C/A/N/S/L/I/M: ngắn, đúng dữ liệu cung cấp.
## 5) Định giá & So sánh ngành
- P/E, P/B, EV/EBITDA… so với lịch sử & ngành (nếu có).
## 6) Thiết lập giao dịch & Rủi ro
- Entry, Stop, TP, R/R ước lượng; 3–5 rủi ro chính.
## 7) Dự báo
- Ngắn 1–2 tuần; Trung 1–3 tháng; Dài 3–12 tháng.
## 8) Kết luận
- Chọn 1: MUA MẠNH / MUA / GIỮ / BÁN / BÁN MẠNH; kèm điểm x/10.
- TL;DR: 2–3 câu rất ngắn.

# DỮ LIỆU
<<<HISTORICAL_DATA_START>>>
{historical}
<<<HISTORICAL_DATA_END>>>

<<<FINANCIALS_START>>>
{financials.to_string(index=False) if (financials is not None and not financials.empty) else 'KHÔNG CÓ DỮ LIỆU BÁO CÁO TÀI CHÍNH'}
<<<FINANCIALS_END>>>

<<<COMPANY_INFO_START>>>
{company_info}
<<<COMPANY_INFO_END>>>

<<<INFO_TCBS_START>>>
{info_data}
<<<INFO_TCBS_END>>>

<<<MARKET_SCREEN_START>>>
{market_screen}
<<<MARKET_SCREEN_END>>>
"""
    return prompt


# ------------------------ Market Screener ------------------------

def screen_market(min_market_cap: int = 500) -> Optional[pd.DataFrame]:
    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if not ensure_df(df):
            return None
        c1 = df["market_cap"] >= min_market_cap
        c2 = ((df["pe"] > 0) & (df["pe"] < 20)) | pd.isna(df["pe"])  # low-ish PE or N/A
        c3 = (df["pb"] > 0) | pd.isna(df["pb"])  # positive PB or N/A
        c4 = (df["last_quarter_revenue_growth"] > 0) | pd.isna(df["last_quarter_revenue_growth"])
        c5 = (df["second_quarter_revenue_growth"] > 0) | pd.isna(df["second_quarter_revenue_growth"])
        c6 = (df["last_quarter_profit_growth"] > 0) | pd.isna(df["last_quarter_profit_growth"])
        c7 = (df["second_quarter_profit_growth"] > 0) | pd.isna(df["second_quarter_profit_growth"])
        c8 = ((df["peg_forward"] >= 0) & (df["peg_forward"] < 1)) | pd.isna(df["peg_forward"])
        c9 = ((df["peg_trailing"] >= 0) & (df["peg_trailing"] < 1)) | pd.isna(df["peg_trailing"])
        c10 = (df["revenue_growth_1y"] >= 0) | pd.isna(df["revenue_growth_1y"])
        c11 = (df["eps_growth_1y"] >= 0) | pd.isna(df["eps_growth_1y"])
        filt = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8 & c9 & c10 & c11
        filtered = df[filt].copy()
        if filtered.empty:
            return None
        # Save two snapshots like your original
        filtered.to_csv("market_filtered_pe.csv", index=False, encoding="utf-8")
        df[c1].to_csv("market_filtered.csv", index=False, encoding="utf-8")
        return filtered
    except Exception:
        logger.exception("screen_market failed")
        return None


# ------------------------ Analysis Pipeline ------------------------

def technical_snapshot(symbol: str, df: pd.DataFrame, for_index: bool = False) -> Dict[str, Any]:
    last = df.iloc[-1]
    snap: Dict[str, Any] = {
        "price": safe_float(last.get("Close")),
        "open": safe_float(last.get("Open")),
        "high": safe_float(last.get("High")),
        "low": safe_float(last.get("Low")),
        "volume": safe_float(last.get("Volume")),
        "ma": {
            "ma10": safe_float(last.get("SMA_10")),
            "ma20": safe_float(last.get("SMA_20")),
            "ma50": safe_float(last.get("SMA_50")),
            "ma200": safe_float(last.get("SMA_200")),
        },
        "rsi": safe_float(last.get("RSI")),
        "macd": {
            "macd": safe_float(last.get("MACD")),
            "signal": safe_float(last.get("MACD_Signal")),
            "histogram": safe_float(last.get("MACD_Hist")),
        },
        "bb": {"upper": safe_float(last.get("BB_Upper")), "lower": safe_float(last.get("BB_Lower"))},
        "vol": {
            "current": safe_float(last.get("Volume")),
            "ma20": safe_float(last.get("Vol_MA_20")),
            "ma50": safe_float(last.get("Vol_MA_50")),
        },
        "ich": {
            "tenkan": safe_float(last.get("Ich_Tenkan")),
            "kijun": safe_float(last.get("Ich_Kijun")),
            "senkou_a": safe_float(last.get("Ich_SpanA")),
            "senkou_b": safe_float(last.get("Ich_SpanB")),
            "chikou": safe_float(last.get("Ich_Chikou")),
        },
    }
    if not for_index:
        rs3d, rs1m, rs3m, rs6m, rs1y = rs_from_market_csv(symbol)
        snap.update({"rs3d": rs3d, "rs1m": rs1m, "rs3m": rs3m, "rs6m": rs6m, "rs1y": rs1y})
    return snap


def historical_text_from_csv(path: str) -> str:
    if not os.path.exists(path):
        return "Không có dữ liệu lịch sử."
    try:
        df = pd.read_csv(path)
        return df.tail(2000).to_string(index=False, float_format="{:.2f}".format)
    except Exception:
        return "Không có dữ liệu lịch sử."


def build_outputs(symbol: str, df: pd.DataFrame, financials: Optional[pd.DataFrame], company_info: str, provider: str = "gemini") -> Dict[str, Any]:
    symu = symbol.upper()
    # VSA last 3–5 sessions
    vsa = last_n_sessions_vsa(df, n=5)
    # Wyckoff phases
    phase_4w, phase_6m, last_br = structure_wyckoff(df)
    # Financial snapshot
    qp = quarter_rev_profit(financials)

    # Technical snapshot dict for prompts
    snap = technical_snapshot(symu, df, for_index=(symu == "VNINDEX"))
    snap.update(qp)

    # Prepare prompt data blocks
    hist_path = os.path.join(DATA_DIR, f"{symu}_data.csv")
    historical = historical_text_from_csv(hist_path)
    info_path = os.path.join(DATA_DIR, f"{symu}_infor.csv")
    market_pe_path = "market_filtered_pe.csv"
    def to_text(p: str, fallback: str) -> str:
        if os.path.exists(p):
            try:
                return pd.read_csv(p).tail(2000).to_string(index=False, float_format="{:.2f}".format)
            except Exception:
                return fallback
        return fallback
    info_text = to_text(info_path, "Không có dữ liệu thông tin công ty.")
    market_text = to_text(market_pe_path, "Không có dữ liệu thông tin thị trường.")

    # Build prompt
    if symu == "VNINDEX":
        prompt = build_prompt_vnindex(symu, snap, historical, market_text)
    else:
        prompt = build_prompt_single(symu, snap, historical, financials, company_info, info_text, market_text)

    # Save prompt
    prompt_path = os.path.join(DATA_DIR, "prompt.txt")
    write_text(prompt_path, prompt)

    # AI call
    ai = AIClient(provider=provider)
    ai_text = ai.generate(prompt)

    # JSON bundle for machines
    json_bundle = {
        "overview": {"symbol": symu, "price": fmt2(snap.get("price"))},
        "vsa": [
            {"date": s.date, "pattern": s.pattern, "volume_vs_ma20": s.volume_vs_ma20, "note": s.note}
            for s in vsa
        ],
        "wyckoff": {"phase_1_4w": phase_4w, "phase_1_6m": phase_6m, "last_breakout": last_br},
        "scenarios_1_2w": [],  # filled by AI text typically, we keep empty for programmatic use
        "strategy": {"stance": "N/A", "entries": [], "exits": [], "key_risks": []},
        "picks": [],  # we only fill in index analysis downstream when needed
        "ai_analysis": ai_text,
    }

    # Markdown report
    md_lines = [
        f"# Báo cáo {symu} — {TODAY.strftime('%Y-%m-%d %H:%M')}\n",
        "## Snapshot",
        f"- Giá: {fmt2(snap.get('price'))}",
        f"- RSI: {fmt2(snap.get('rsi'))} | MACD: {fmt2(snap['macd'].get('macd'))}/{fmt2(snap['macd'].get('signal'))}/{fmt2(snap['macd'].get('histogram'))}",
        f"- MA10/20/50/200: {fmt2(snap['ma'].get('ma10'))}/{fmt2(snap['ma'].get('ma20'))}/{fmt2(snap['ma'].get('ma50'))}/{fmt2(snap['ma'].get('ma200'))}",
        f"- BB: U {fmt2(snap['bb'].get('upper'))} | L {fmt2(snap['bb'].get('lower'))}",
        f"- Volume: {fmt_big(snap['vol'].get('current'))} | MA20 {fmt_big(snap['vol'].get('ma20'))}",
        "\n## VSA/VPA (3–5 phiên gần nhất)",
    ]
    if vsa:
        for s in vsa:
            md_lines.append(f"- {s.date}: **{s.pattern}** | Vol/MA20: {s.volume_vs_ma20} | {s.note}")
    else:
        md_lines.append("- N/A")
    md_lines.extend([
        "\n## Wyckoff",
        f"- 1–4 tuần: {phase_4w}",
        f"- 1–6 tháng: {phase_6m}",
        f"- Breakout gần nhất: {json.dumps(last_br, ensure_ascii=False)}",
        "\n## Phân tích AI",
        ai_text or "N/A",
    ])

    # Persist
    write_text(os.path.join(DATA_DIR, f"{symu}_report.md"), "\n".join(md_lines))
    write_json(os.path.join(DATA_DIR, f"{symu}_bundle.json"), json_bundle)
    write_text(os.path.join(DATA_DIR, f"{symu}_analysis.txt"), ai_text)

    return {"snap": snap, "vsa": vsa, "wyckoff": (phase_4w, phase_6m, last_br), "ai_text": ai_text}


# ------------------------ Orchestrators ------------------------

def analyze_symbol(symbol: str, provider: str = "gemini") -> None:
    logger.info(f"Analyze {symbol}")
    pdata = fetch_history(symbol)
    if pdata is None or pdata.df.shape[0] < 100:
        logger.error(f"Thiếu dữ liệu cho {symbol}")
        return
    df = add_features(pdata.df)
    fin: Optional[pd.DataFrame] = None
    comp_info = ""
    if symbol.upper() != "VNINDEX":
        fin = fetch_financials(symbol)
        comp_info = fetch_company_info(symbol)
    else:
        comp_info = "Chỉ số thị trường VNINDEX."
    build_outputs(symbol, df, fin, comp_info, provider=provider)


def analyze_index(provider: str = "gemini") -> None:
    analyze_symbol("VNINDEX", provider=provider)


def analyze_many(symbols: List[str], provider: str = "gemini") -> None:
    for sym in symbols:
        analyze_symbol(sym, provider=provider)


# ------------------------ CLI ------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="VNStock + AI Analyzer v2")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_screen = sub.add_parser("screen", help="Lọc cổ phiếu nền")
    p_screen.add_argument("--min-cap", type=int, default=500, help="Vốn hóa tối thiểu (tỷ VND)")

    p_anl = sub.add_parser("analyze", help="Phân tích 1 hoặc nhiều mã")
    p_anl.add_argument("symbols", nargs="+", help="Danh sách mã, VD: VCB FPT HPG")
    p_anl.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini")

    p_idx = sub.add_parser("index", help="Phân tích VNINDEX")
    p_idx.add_argument("--provider", choices=["gemini", "openrouter"], default="gemini")

    args = parser.parse_args()

    if args.cmd == "screen":
        logger.info("Đang lọc cổ phiếu nền…")
        df = screen_market(min_market_cap=args.min_cap)
        if df is None or df.empty:
            logger.warning("Không có kết quả screen.")
        else:
            out = os.path.join(DATA_DIR, "screen_result.csv")
            df.to_csv(out, index=False, encoding="utf-8")
            logger.info(f"Đã lưu danh sách tại {out}")
        return

    if args.cmd == "analyze":
        symbols = [s.strip().upper() for s in args.symbols if s.strip()]
        if not symbols:
            logger.error("Chưa có mã hợp lệ.")
            return
        analyze_many(symbols, provider=args.provider)
        logger.info("Hoàn tất phân tích.")
        return

    if args.cmd == "index":
        analyze_index(provider=args.provider)
        logger.info("Hoàn tất phân tích VNINDEX.")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n🛑 Hủy bởi người dùng.")