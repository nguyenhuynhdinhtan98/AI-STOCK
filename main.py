#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VN-Stock AI Analyzer — Refactored
---------------------------------
Điểm chính:
- Tách rõ cấu hình, xử lý lỗi, logging.
- Chuẩn hoá tính chỉ báo (ta) dùng API lớp để tránh lệch phiên bản.
- Cache dữ liệu VNINDEX để không gọi lặp lại.
- Sửa lỗi .fillna không gán, kiểm tra None an toàn, ép kiểu an toàn.
- Chấm điểm (scoring) minh bạch với rubric 100 điểm (chi tiết dưới).
- Prompt builder tối ưu, ngắn gọn, có **SCORING_RUBRIC** cho AI.
- Lưu mọi artefacts vào thư mục `vnstocks_data/`.

Yêu cầu thư viện: vnstock, ta, pandas, numpy, openpyxl, python-dotenv, google-generativeai, openrouter-compatible openai
"""

from __future__ import annotations
import os
import json
import traceback
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Matplotlib chỉ dùng nếu cần vẽ; giữ import tối thiểu
import matplotlib.pyplot as plt

from dotenv import load_dotenv

# Indicators
import ta
from ta.trend import SMAIndicator, MACD, IchimokuIndicator
from ta.volatility import BollingerBands

# Data sources
from vnstock.explorer.vci import Quote, Finance, Company
from vnstock import Screener

# LLM clients
from openai import OpenAI as OpenRouter
import google.generativeai as genai

# -------------------------------
# Config & Globals
# -------------------------------

@dataclass
class AppConfig:
    data_dir: Path = Path("vnstocks_data")
    log_file: Path = Path("stock_analysis.log")
    days_back: int = 365 * 15

    # API keys
    google_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None

    @property
    def start_date(self) -> str:
        return (datetime.today() - timedelta(days=self.days_back)).strftime("%Y-%m-%d")

    @property
    def end_date(self) -> str:
        return datetime.today().strftime("%Y-%m-%d")


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger("vnstock_ai")


CFG = AppConfig()
CFG.data_dir.mkdir(parents=True, exist_ok=True)
logger = setup_logging(CFG.log_file)

# Load .env
load_dotenv()
CFG.google_api_key = os.getenv("GOOGLE_API_KEY")
CFG.openrouter_api_key = os.getenv("OPEN_ROUTER_API_KEY")

if not CFG.google_api_key or not CFG.openrouter_api_key:
    raise ValueError("Vui lòng đặt GOOGLE_API_KEY và OPEN_ROUTER_API_KEY trong file .env")

# Init LLM clients
try:
    genai.configure(api_key=CFG.google_api_key)
except Exception as e:
    logger.warning(f"Không thể cấu hình Gemini: {e}")

openrouter_client = OpenRouter(base_url="https://openrouter.ai/api/v1", api_key=CFG.openrouter_api_key)

# -------------------------------
# Utilities
# -------------------------------

def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        return float(str(val).replace(',', ''))
    except (TypeError, ValueError):
        return None


def fmt_num(value: Any) -> str:
    num = safe_float(value)
    if num is None:
        return "N/A"
    abs_value = abs(num)
    if abs_value >= 1e9:
        return f"{num / 1e9:.2f}B"
    if abs_value >= 1e6:
        return f"{num / 1e6:.2f}M"
    if abs_value >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.2f}"


def validate_df(df: Optional[pd.DataFrame], required: Optional[List[str]] = None) -> bool:
    if df is None or df.empty:
        return False
    if required:
        return all(c in df.columns for c in required)
    return True


# -------------------------------
# Data Access (cached)
# -------------------------------

_VNI_CACHE: Optional[pd.DataFrame] = None

def get_vnindex() -> Optional[pd.DataFrame]:
    global _VNI_CACHE
    if _VNI_CACHE is not None:
        return _VNI_CACHE
    try:
        logger.info("Lấy dữ liệu VNINDEX...")
        q = Quote(symbol="VNINDEX")
        df = q.history(start=CFG.start_date, end=CFG.end_date, interval="1D")
        if not validate_df(df, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning("Không lấy được dữ liệu VNINDEX")
            return None
        df = df.rename(columns={"time":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        (CFG.data_dir / "VNINDEX_data.csv").write_text(df.to_csv(encoding='utf-8-sig'))
        _VNI_CACHE = df
        return df
    except Exception as e:
        logger.error(f"Lỗi VNINDEX: {e}")
        return None


def get_stock_history(symbol: str) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Lấy dữ liệu {symbol}...")
        stock = Quote(symbol=symbol)
        df = stock.history(start=CFG.start_date, end=CFG.end_date, interval="1D")
        if not validate_df(df, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning(f"Không lấy được dữ liệu cho {symbol}")
            return None
        df = df.rename(columns={"time":"Date","open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"})
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        path = CFG.data_dir / f"{symbol}_data.csv"
        path.write_text(df.to_csv(encoding='utf-8-sig'))
        # Đồng thời ghi ra data.csv cho pipeline khác (nếu cần)
        Path("data.csv").write_text(df.to_csv(encoding='utf-8-sig'))
        return df
    except Exception as e:
        logger.error(f"Lỗi dữ liệu {symbol}: {e}")
        return None


def get_financials(symbol: str) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Lấy BCTC {symbol} (quarter)...")
        fin = Finance(symbol=symbol, period="quarter")
        ratio = fin.ratio(period="quarter")
        bs = fin.balance_sheet(period="quarter")
        is_ = fin.income_statement(period="quarter")
        cf = fin.cash_flow(period="quarter")

        def flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join([c for c in col if c]).strip() for col in df.columns.values]
            return df

        def standardize(df: pd.DataFrame) -> pd.DataFrame:
            return df.rename(columns={"Meta_ticker":"ticker","Meta_yearReport":"yearReport","Meta_lengthReport":"lengthReport"})

        ratio = standardize(flatten_cols(ratio))
        df_fin = bs.merge(is_, on=["yearReport","lengthReport","ticker"], how="outer")\
                   .merge(cf, on=["yearReport","lengthReport","ticker"], how="outer")\
                   .merge(ratio, on=["yearReport","lengthReport","ticker"], how="outer")
        df_fin = df_fin.rename(columns={"ticker":"Symbol","yearReport":"Year","lengthReport":"Quarter"}).tail(20)
        (CFG.data_dir / f"{symbol}_financial_statements.csv").write_text(df_fin.to_csv(index=False, encoding='utf-8-sig'))
        return df_fin
    except Exception as e:
        logger.error(f"Lỗi BCTC {symbol}: {e}")
        return None


def get_company_blob(symbol: str) -> str:
    try:
        logger.info(f"Lấy thông tin công ty {symbol}...")
        c = Company(symbol)
        blocks = {
            "OVERVIEW": c.overview(),
            "SHAREHOLDERS": c.shareholders(),
            "OFFICERS": c.officers(filter_by='working'),
            "EVENTS": c.events(),
            "NEWS": c.news(),
            "REPORTS": c.reports(),
            "TRADING STATS": c.trading_stats(),
            "RATIO SUMMARY": c.ratio_summary(),
        }
        out = []
        for name, data in blocks.items():
            out.append(f"=== {name} ===")
            if isinstance(data, pd.DataFrame):
                out.append(data.to_string() if not data.empty else "Không có dữ liệu")
            elif isinstance(data, dict):
                out.append(json.dumps(data, ensure_ascii=False, indent=2) if data else "Không có dữ liệu")
            else:
                out.append(str(data) if data is not None else "Không có dữ liệu")
            out.append("")
        blob = "\n".join(out)
        (CFG.data_dir / f"{symbol}_company_info.txt").write_text(blob, encoding="utf-8-sig")
        return blob
    except Exception as e:
        logger.error(f"Lỗi company info {symbol}: {e}")
        return "Không thể lấy thông tin công ty."\


# -------------------------------
# Feature Engineering
# -------------------------------

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    if not validate_df(df):
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    # Fill numeric cols
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].ffill().bfill()
    df['returns'] = df['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if not validate_df(df, ['Close','High','Low','Volume']):
        return df
    out = df.copy()

    # SMA
    for w in [10, 20, 50, 200]:
        out[f'SMA_{w}'] = SMAIndicator(out['Close'], window=w).sma_indicator()

    # RSI 14 (ta.momentum.rsi)
    out['RSI'] = ta.momentum.rsi(out['Close'], window=14)

    # MACD (class-based tránh khác phiên bản)
    macd = MACD(out['Close'], window_slow=26, window_fast=12, window_sign=9)
    out['MACD'] = macd.macd()
    out['MACD_Signal'] = macd.macd_signal()
    out['MACD_Hist'] = macd.macd_diff()

    # Bollinger
    bb = BollingerBands(out['Close'], window=20, window_dev=2)
    out['BB_Upper'] = bb.bollinger_hband()
    out['BB_Middle'] = bb.bollinger_mavg()
    out['BB_Lower'] = bb.bollinger_lband()

    # Volume MAs
    out['Volume_MA_20'] = SMAIndicator(out['Volume'], window=20).sma_indicator()
    out['Volume_MA_50'] = SMAIndicator(out['Volume'], window=50).sma_indicator()

    # Ichimoku
    ichi = IchimokuIndicator(high=out['High'], low=out['Low'], window1=9, window2=26, window3=52)
    out['ichimoku_tenkan_sen'] = ichi.ichimoku_conversion_line()
    out['ichimoku_kijun_sen'] = ichi.ichimoku_base_line()
    out['ichimoku_senkou_span_a'] = ichi.ichimoku_a()
    out['ichimoku_senkou_span_b'] = ichi.ichimoku_b()
    out['ichimoku_chikou_span'] = out['Close'].shift(26)

    return out


def add_relative_strength(df_stock: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    if not (validate_df(df_stock) and validate_df(df_index)):
        return df_stock
    try:
        merged = df_stock[['Close']].join(df_index[['Close']].rename(columns={'Close':'Index_Close'}), how='inner')
        if merged.empty or merged['Index_Close'].isna().all():
            logger.warning("Thiếu dữ liệu Index để tính RS. Dùng mặc định.")
            defaults = {
                'RS': 1.0, 'RS_Point': 0.0,
                'RS_SMA_10': 1.0, 'RS_SMA_20': 1.0, 'RS_SMA_50': 1.0, 'RS_SMA_200': 1.0,
                'RS_Point_SMA_10': 0.0, 'RS_Point_SMA_20': 0.0, 'RS_Point_SMA_50': 0.0, 'RS_Point_SMA_200': 0.0,
            }
            for k, v in defaults.items():
                df_stock[k] = v
            return df_stock

        merged['RS'] = merged['Close'] / merged['Index_Close']

        # RS Point (IBD-style) dùng ROC của giá cổ phiếu
        roc_63 = ta.momentum.roc(merged['Close'], window=63)
        roc_126 = ta.momentum.roc(merged['Close'], window=126)
        roc_189 = ta.momentum.roc(merged['Close'], window=189)
        roc_252 = ta.momentum.roc(merged['Close'], window=252)
        merged['RS_Point'] = (roc_63*0.4 + roc_126*0.2 + roc_189*0.2 + roc_252*0.2) * 100

        for w in [10, 20, 50, 200]:
            merged[f'RS_SMA_{w}'] = SMAIndicator(merged['RS'], window=w).sma_indicator()
            merged[f'RS_Point_SMA_{w}'] = SMAIndicator(merged['RS_Point'], window=w).sma_indicator()

        join_cols = [
            'RS','RS_Point', 'RS_SMA_10','RS_SMA_20','RS_SMA_50','RS_SMA_200',
            'RS_Point_SMA_10','RS_Point_SMA_20','RS_Point_SMA_50','RS_Point_SMA_200'
        ]
        df_out = df_stock.join(merged[join_cols], how='left')
        # Fill NA (phải gán lại!)
        for c in join_cols:
            if 'RS_Point' in c:
                df_out[c] = df_out[c].fillna(0.0)
            else:
                df_out[c] = df_out[c].fillna(1.0)
        return df_out
    except Exception as e:
        logger.error(f"Lỗi RS: {e}")
        return df_stock


def get_rs_from_csv(symbol: str) -> Tuple[float, float, float, float]:
    try:
        path = Path("market_filtered.csv")
        if not path.exists():
            return 1.0, 1.0, 1.0, 1.0
        mkt = pd.read_csv(path)
        if 'ticker' not in mkt.columns:
            return 1.0, 1.0, 1.0, 1.0
        row = mkt[mkt['ticker'].str.upper() == symbol.upper()]
        if row.empty:
            return 1.0, 1.0, 1.0, 1.0
        (CFG.data_dir / f"{symbol}_infor.csv").write_text(row.to_csv(index=False, encoding='utf-8-sig'))
        r3d = row.get('relative_strength_3d', pd.Series([1.0])).iloc[0]
        r1m = row.get('rel_strength_1m', pd.Series([1.0])).iloc[0]
        r3m = row.get('rel_strength_3m', pd.Series([1.0])).iloc[0]
        r1y = row.get('rel_strength_1y', pd.Series([1.0])).iloc[0]
        return float(r3d), float(r1m), float(r3m), float(r1y)
    except Exception as e:
        logger.error(f"Lỗi đọc RS csv: {e}")
        return 1.0, 1.0, 1.0, 1.0


# -------------------------------
# Scoring
# -------------------------------

SCORING_RUBRIC = {
    # Tổng trọng số 100. Điểm khởi tạo 50, dao động ±50
    "MA": 28,           # 8 điều kiện xấp xỉ 3.5đ
    "RSI": 14,
    "MACD": 14,
    "Ichimoku": 14,
    "Volume": 14,
    "RelativeStrength": 14,  # chỉ áp dụng cho cổ phiếu (không phải Index)
    "Bollinger": 6,
}


def _add_score(score: float, delta: float, cap: float) -> float:
    # mỗi mục không vượt quá trọng số mục
    if delta > 0:
        return score + min(delta, cap)
    return score + max(delta, -cap)


def calc_score(df: pd.DataFrame, symbol: str) -> Tuple[float, Dict[str, Any]]:
    if not validate_df(df):
        return 50.0, empty_signal()

    last = df.iloc[-1]
    price = safe_float(last.get('Close'))
    if price is None:
        return 50.0, empty_signal()

    # indicator snapshot
    snap = {
        'rsi': safe_float(last.get('RSI')),
        'ma10': safe_float(last.get('SMA_10', price)),
        'ma20': safe_float(last.get('SMA_20', price)),
        'ma50': safe_float(last.get('SMA_50', price)),
        'ma200': safe_float(last.get('SMA_200', price)),
        'macd': safe_float(last.get('MACD')),
        'macd_sig': safe_float(last.get('MACD_Signal')),
        'macd_hist': safe_float(last.get('MACD_Hist')),
        'bb_up': safe_float(last.get('BB_Upper')),
        'bb_lo': safe_float(last.get('BB_Lower')),
        'vol': safe_float(last.get('Volume')),
        'vol_ma20': safe_float(last.get('Volume_MA_20')),
        'vol_ma50': safe_float(last.get('Volume_MA_50')),
        'tenkan': safe_float(last.get('ichimoku_tenkan_sen')),
        'kijun': safe_float(last.get('ichimoku_kijun_sen')),
        'sa': safe_float(last.get('ichimoku_senkou_span_a')),
        'sb': safe_float(last.get('ichimoku_senkou_span_b')),
        'chikou': safe_float(last.get('ichimoku_chikou_span')),
        'rs': safe_float(last.get('RS', 1.0)) if symbol.upper() != 'VNINDEX' else 1.0,
        'rs_point': safe_float(last.get('RS_Point', 0.0)) if symbol.upper() != 'VNINDEX' else 0.0,
        'rs_sma10': safe_float(last.get('RS_SMA_10')),
        'rs_sma50': safe_float(last.get('RS_SMA_50')),
        'rs_point_sma20': safe_float(last.get('RS_Point_SMA_20')),
    }

    r3d, r1m, r3m, r1y = get_rs_from_csv(symbol)

    score = 50.0

    # --- MA (28)
    delta = 0.0
    for m in ['ma10','ma20','ma50','ma200']:
        if snap[m] is not None:
            delta += 3.5 if price > snap[m] else -3.5
    # layout
    ma_vals = [snap['ma10'], snap['ma20'], snap['ma50'], snap['ma200']]
    if all(v is not None for v in ma_vals):
        if all(ma_vals[i] >= ma_vals[i+1] for i in range(3)):
            delta += 3.5  # đúng thứ tự tăng
        elif all(ma_vals[i] <= ma_vals[i+1] for i in range(3)):
            delta -= 3.5
        elif (ma_vals[0] > ma_vals[1] and ma_vals[2] > ma_vals[3]):
            delta += 1.75
        elif (ma_vals[0] < ma_vals[1] and ma_vals[2] < ma_vals[3]):
            delta -= 1.75
    score = _add_score(score, delta, SCORING_RUBRIC['MA'])

    # --- RSI (14)
    delta = 0.0
    rsi = snap['rsi']
    if rsi is not None:
        if rsi < 30: delta += 14
        elif 30 <= rsi < 40: delta += 10
        elif 40 <= rsi < 50: delta += 7
        elif 50 <= rsi < 60: delta += 3.5
        elif 60 <= rsi < 70: delta -= 3.5
        elif 70 <= rsi < 80: delta -= 7
        else: delta -= 14
    score = _add_score(score, delta, SCORING_RUBRIC['RSI'])

    # --- MACD (14)
    delta = 0.0
    if snap['macd'] is not None and snap['macd_sig'] is not None and snap['macd_hist'] is not None:
        if snap['macd'] > snap['macd_sig'] and snap['macd_hist'] > 0: delta += 7
        elif snap['macd'] < snap['macd_sig'] and snap['macd_hist'] < 0: delta -= 7
        if len(df) > 1:
            mh_prev = safe_float(df['MACD_Hist'].iloc[-2])
            if mh_prev is not None:
                if snap['macd_hist'] > mh_prev: delta += 3.5
                elif snap['macd_hist'] < mh_prev: delta -= 3.5
            m_prev = safe_float(df['MACD'].iloc[-2])
            s_prev = safe_float(df['MACD_Signal'].iloc[-2])
            if None not in (m_prev, s_prev):
                if snap['macd'] > snap['macd_sig'] and m_prev <= s_prev: delta += 3.5
                elif snap['macd'] < snap['macd_sig'] and m_prev >= s_prev: delta -= 3.5
    score = _add_score(score, delta, SCORING_RUBRIC['MACD'])

    # --- Ichimoku (14)
    delta = 0.0
    if None not in (snap['sa'], snap['sb']):
        top, bot = max(snap['sa'], snap['sb']), min(snap['sa'], snap['sb'])
        if price > top: delta += 14
        elif price < bot: delta -= 14
    score = _add_score(score, delta, SCORING_RUBRIC['Ichimoku'])

    # --- Volume (14)
    delta = 0.0
    if snap['vol'] is not None:
        if snap['vol_ma20'] and snap['vol_ma20'] > 0:
            ratio = snap['vol'] / snap['vol_ma20']
            if ratio > 2.0: delta += 4
            elif ratio > 1.5: delta += 3
            elif ratio > 1.0: delta += 1
            elif ratio < 0.5: delta -= 2
        if snap['vol_ma50'] and snap['vol_ma50'] > 0:
            ratio50 = snap['vol'] / snap['vol_ma50']
            if ratio50 > 2.0: delta += 3
            elif ratio50 > 1.5: delta += 2
            elif ratio50 > 1.0: delta += 1
            elif ratio50 < 0.5: delta -= 1
        if len(df) > 2:
            v1 = safe_float(df['Volume'].iloc[-2])
            v2 = safe_float(df['Volume'].iloc[-3])
            if None not in (v1, v2):
                if snap['vol'] > v1 > v2:
                    delta += 4 if (v2 and snap['vol']/v2 > 1.5) else 2
                elif snap['vol'] < v1 < v2:
                    delta -= 4 if (v2 and snap['vol']/v2 < 0.7) else 2
        if len(df) > 40 and snap['vol_ma20'] and snap['vol_ma20'] > 0:
            prev_ma20 = df['Volume'].iloc[-21:-1].mean()
            if prev_ma20 > 0:
                acc = snap['vol_ma20'] / prev_ma20
                if acc > 2.0: delta += 3
                elif acc > 1.5: delta += 1.5
                elif acc < 0.5: delta -= 2
    score = _add_score(score, delta, SCORING_RUBRIC['Volume'])

    # --- RS (14, cổ phiếu mới tính)
    if symbol.upper() != 'VNINDEX':
        delta = 0.0
        if snap['rs'] is not None:
            if snap['rs_sma10'] is not None:
                delta += 3.5 if snap['rs'] > snap['rs_sma10'] else -3.5
            if snap['rs_sma50'] is not None:
                delta += 3.5 if snap['rs'] > snap['rs_sma50'] else -3.5
        if snap['rs_point'] is not None and snap['rs_point_sma20'] is not None:
            delta += 3.5 if snap['rs_point'] > snap['rs_point_sma20'] else -3.5
        if snap['rs_point'] is not None:
            if snap['rs_point'] > 1.0: delta += 3.5
            elif snap['rs_point'] < -1.0: delta -= 3.5
        score = _add_score(score, delta, SCORING_RUBRIC['RelativeStrength'])

    # --- Bollinger (6)
    delta = 0.0
    if None not in (snap['bb_up'], snap['bb_lo']) and snap['bb_up'] > snap['bb_lo']:
        width = snap['bb_up'] - snap['bb_lo']
        to_up = (snap['bb_up'] - price) / width
        to_lo = (price - snap['bb_lo']) / width
        if to_lo < 0.15: delta += 6
        elif to_lo < 0.30: delta += 3
        if to_up < 0.15: delta -= 6
        elif to_up < 0.30: delta -= 3
        # co/giãn băng so với phiên trước
        if len(df) > 1:
            u_prev, l_prev = safe_float(df['BB_Upper'].iloc[-2]), safe_float(df['BB_Lower'].iloc[-2])
            if None not in (u_prev, l_prev) and (u_prev - l_prev) > 0:
                w_prev = u_prev - l_prev
                if width > 1.1 * w_prev: delta -= 1.5
                elif width < 0.9 * w_prev: delta += 1.5
    score = _add_score(score, delta, SCORING_RUBRIC['Bollinger'])

    score = max(0.0, min(100.0, score))

    # Label & Recommendation
    if score >= 80:
        signal, rec = "MUA MẠNH", "MUA MẠNH"
    elif score >= 65:
        signal, rec = "MUA", "MUA"
    elif score >= 55:
        signal, rec = "TĂNG MẠNH", "GIỮ - TĂNG"
    elif score >= 45:
        signal, rec = "TRUNG LẬP", "GIỮ"
    elif score >= 35:
        signal, rec = "GIẢM MẠNH", "GIỮ - GIẢM"
    elif score >= 20:
        signal, rec = "BÁN", "BÁN"
    else:
        signal, rec = "BÁN MẠNH", "BÁN MẠNH"

    out = {
        "signal": signal,
        "score": float(score),
        "current_price": price,
        "rsi_value": snap['rsi'],
        "ma10": snap['ma10'], "ma20": snap['ma20'], "ma50": snap['ma50'], "ma200": snap['ma200'],
        "rs": snap['rs'], "rs_point": snap['rs_point'],
        "recommendation": rec,
        "open": safe_float(last.get('Open')),
        "high": safe_float(last.get('High')),
        "low": safe_float(last.get('Low')),
        "volume": snap['vol'],
        "volume_ma_20": snap['vol_ma20'],
        "volume_ma_50": snap['vol_ma50'],
        "macd": snap['macd'], "macd_signal": snap['macd_sig'], "macd_hist": snap['macd_hist'],
        "bb_upper": snap['bb_up'], "bb_lower": snap['bb_lo'],
        "ichimoku_tenkan_sen": snap['tenkan'], "ichimoku_kijun_sen": snap['kijun'],
        "ichimoku_senkou_span_a": snap['sa'], "ichimoku_senkou_span_b": snap['sb'],
        "ichimoku_chikou_span": snap['chikou'],
        "rs_sma_10": snap['rs_sma10'], "rs_sma_50": snap['rs_sma50'],
        "rs_point_sma_20": snap['rs_point_sma20'],
        "relative_strength_3d": r3d, "relative_strength_1m": r1m, "relative_strength_3m": r3m, "relative_strength_1y": r1y,
        "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
    }
    return score, out


def empty_signal() -> Dict[str, Any]:
    return {
        "signal":"LỖI","score":50,"current_price":0.0,"rsi_value":0.0,
        "ma10":0,"ma20":0,"ma50":0,"ma200":0,"rs":1.0,"rs_point":0.0,
        "recommendation":"KHÔNG XÁC ĐỊNH","open":None,"high":None,"low":None,
        "volume":None,"macd":None,"macd_signal":None,"macd_hist":None,
        "bb_upper":None,"bb_lower":None,"volume_ma_20":None,"volume_ma_50":None,
        "ichimoku_tenkan_sen":None,"ichimoku_kijun_sen":None,"ichimoku_senkou_span_a":None,
        "ichimoku_senkou_span_b":None,"ichimoku_chikou_span":None,
        "rs_sma_10":None,"rs_sma_50":None,"rs_point_sma_20":None,
        "relative_strength_3d":None,"relative_strength_1m":None,
        "relative_strength_3m":None,"relative_strength_1y":None,
        "forecast_dates":[],"forecast_prices":[],"forecast_plot_path":"",
    }


# -------------------------------
# Prompt Builders (Optimized)
# -------------------------------

RUBRIC_TEXT = f"""
SCORING_RUBRIC (0-100, base 50, ±50):
- MA Alignment & Price vs MA: {SCORING_RUBRIC['MA']} pts
- RSI(14): {SCORING_RUBRIC['RSI']} pts
- MACD (trend + cross + histogram slope): {SCORING_RUBRIC['MACD']} pts
- Ichimoku (price vs cloud): {SCORING_RUBRIC['Ichimoku']} pts
- Volume (relative to MA20/50, trend, acceleration): {SCORING_RUBRIC['Volume']} pts
- Relative Strength (RS, RS Point vs SMA): {SCORING_RUBRIC['RelativeStrength']} pts (stocks only)
- Bollinger position & band width change: {SCORING_RUBRIC['Bollinger']} pts
Hãy dùng rubric này để cân nhắc, giải thích rõ điểm cộng/trừ ở từng mục khi kết luận.
"""


def build_stock_prompt(symbol: str, price: float, tech: Dict[str, Any], sig: Dict[str, Any],
                       fin: Optional[pd.DataFrame], company_blob: str,
                       hist_text: str, infor_text: str, market_text: str) -> str:
    def v(x: Any) -> str: return fmt_num(x)
    rsi = tech.get('rsi')
    ma = tech.get('ma', {})
    bb = tech.get('bollinger_bands', {})
    macd = tech.get('macd', {})
    ich = tech.get('ichimoku', {})
    vol = tech.get('volume', {})

    parts = [
        f"PHÂN TÍCH TỔNG HỢP MÃ {symbol.upper()} — GIÁ HIỆN TẠI: {v(price)} VND",
        RUBRIC_TEXT,
        "DỮ LIỆU KỸ THUẬT TÓM TẮT:",
        f"- RSI(14): {v(rsi)}",
        f"- MA10/20/50/200: {v(ma.get('ma10'))} | {v(ma.get('ma20'))} | {v(ma.get('ma50'))} | {v(ma.get('ma200'))}",
        f"- MACD/Signal/Hist: {v(macd.get('macd'))} / {v(macd.get('signal'))} / {v(macd.get('histogram'))}",
        f"- BB(Upper/Lower): {v(bb.get('upper'))} / {v(bb.get('lower'))}",
        f"- Ichi Tenkan/Kijun/SenkouA/SenkouB/Chikou: {v(ich.get('tenkan'))} / {v(ich.get('kijun'))} / {v(ich.get('senkou_a'))} / {v(ich.get('senkou_b'))} / {v(ich.get('chikou'))}",
        f"- Volume/MA20/MA50: {v(vol.get('current'))} / {v(vol.get('ma20'))} / {v(vol.get('ma50'))}",
        "SỨC MẠNH TƯƠNG ĐỐI (RS):",
        f"- RS(3D/1M/3M/1Y): {v(sig.get('relative_strength_3d'))}/{v(sig.get('relative_strength_1m'))}/{v(sig.get('relative_strength_3m'))}/{v(sig.get('relative_strength_1y'))}",
        "BÁO CÁO TÀI CHÍNH (nếu có):",
        fin.to_string(index=False) if (fin is not None and not fin.empty) else "Không có dữ liệu BCTC",
        "\nTHÔNG TIN LỊCH SỬ GIÁ (rút gọn):\n" + hist_text,
        "\nTHÔNG TIN CÔNG TY:\n" + company_blob,
        "\nTHÔNG TIN CỔ PHIẾU TĂNG TRƯỞNG THỊ TRƯỜNG VÀ PE DƯỚI 20:\n" + market_text,
        "\nYÊU CẦU TRẢ LỜI THEO CẤU TRÚC RÕ RÀNG:",
        "1) Wyckoff & VSA/VPA (3 phiên gần nhất, xác nhận khối lượng, cung/cầu)",
        "2) Minervini (xu hướng, MA alignment, pivot, hỗ trợ/kháng cự, RS)",
        "3) Cơ bản (doanh thu/lợi nhuận, ROE/ROIC, nợ, dòng tiền, cổ tức, tin tức)",
        "4) Định giá & So sánh ngành (P/E, P/B, EV/EBITDA...)",
        "5) Chiến lược (điểm vào/SL/TP, RR)",
        "6) Dự báo: ngắn (1-2w) / trung (1-3m) / dài (3-12m)",
        "7) Rủi ro chính & mức cần theo dõi",
        "8) KẾT LUẬN: chọn 1 trong [MUA MẠNH/MUA/GIỮ/BÁN/BÁN MẠNH] + Điểm 1-10 + Tóm tắt 2-3 câu",
    ]
    return "\n".join(parts)


def build_index_prompt(symbol: str, price: float, tech: Dict[str, Any], hist_text: str, market_text: str) -> str:
    def v(x: Any) -> str: return fmt_num(x)
    ma = tech.get('ma', {})
    macd = tech.get('macd', {})
    bb = tech.get('bollinger_bands', {})
    ich = tech.get('ichimoku', {})
    vol = tech.get('volume', {})

    parts = [
        f"PHÂN TÍCH VNINDEX ({symbol}) — ĐIỂM HIỆN TẠI: {v(price)}",
        RUBRIC_TEXT,
        "TÓM TẮT KỸ THUẬT:",
        f"- MA10/20/50/200: {v(ma.get('ma10'))} | {v(ma.get('ma20'))} | {v(ma.get('ma50'))} | {v(ma.get('ma200'))}",
        f"- MACD/Signal/Hist: {v(macd.get('macd'))} / {v(macd.get('signal'))} / {v(macd.get('histogram'))}",
        f"- BB(Upper/Lower): {v(bb.get('upper'))} / {v(bb.get('lower'))}",
        f"- Ichi Tenkan/Kijun/SenkouA/SenkouB/Chikou: {v(ich.get('tenkan'))} / {v(ich.get('kijun'))} / {v(ich.get('senkou_a'))} / {v(ich.get('senkou_b'))} / {v(ich.get('chikou'))}",
        f"- Volume/MA20/MA50: {v(vol.get('current'))} / {v(vol.get('ma20'))} / {v(vol.get('ma50'))}",
        "\nDỮ LIỆU LỊCH SỬ (rút gọn):\n" + hist_text,
        "\nTHÔNG TIN CỔ PHIẾU TĂNG TRƯỞNG THỊ TRƯỜNG VÀ PE DƯỚI 20:\n" + market_text,
        "\nTRẢ LỜI THEO CẤU TRÚC: VSA/VPA (3 phiên), Wyckoff, Kịch bản 1-2w/1-3m, Chiến lược, Rủi ro, Khuyến nghị cuối." ,
    ]
    return "\n".join(parts)


# -------------------------------
# Orchestration
# -------------------------------

def analyze_symbol(symbol: str) -> Optional[Dict[str, Any]]:
    logger.info("="*60)
    logger.info(f"PHÂN TÍCH MÃ {symbol}")
    logger.info("="*60)

    df = get_stock_history(symbol)
    if not validate_df(df):
        logger.error(f"Thiếu dữ liệu {symbol}")
        return None

    fin = get_financials(symbol)
    company_blob = get_company_blob(symbol)

    dfp = preprocess(df)
    if not validate_df(dfp) or len(dfp) < 100:
        logger.error(f"Dữ liệu {symbol} chưa đủ dài để phân tích")
        return None

    if symbol.upper() != 'VNINDEX':
        vni = get_vnindex()
        if validate_df(vni):
            dfp = add_relative_strength(add_indicators(dfp), vni)
        else:
            dfp = add_indicators(dfp)
    else:
        dfp = add_indicators(dfp)

    score, sig = calc_score(dfp, symbol)
    as_of = dfp.index[-1].strftime('%d/%m/%Y')

    logger.info(f"TÍN HIỆU CUỐI CHO {symbol} ({as_of})")
    logger.info(f"Giá hiện tại: {fmt_num(sig['current_price'])} | Điểm: {score:.1f}/100 | Tín hiệu: {sig['signal']} | Đề xuất: {sig['recommendation']}")

    # Chuẩn bị dữ liệu văn bản cho Prompt
    hist_path = CFG.data_dir / f"{symbol}_data.csv"
    infor_path = CFG.data_dir / f"{symbol}_infor.csv"
    market_path = Path("market_filtered_pe.csv")

    hist_text = "Không có dữ liệu lịch sử."
    infor_text = "Không có dữ liệu thông tin công ty."
    market_text = "Không có dữ liệu thị trường."

    try:
        if hist_path.exists():
            hist_df = pd.read_csv(hist_path).tail(2000)
            hist_text = hist_df.to_string(index=False, float_format="{:.2f}".format)
        if infor_path.exists():
            inf_df = pd.read_csv(infor_path)
            infor_text = inf_df.to_string(index=False, float_format="{:.2f}".format)
        if market_path.exists():
            mk_df = pd.read_csv(market_path)
            market_text = mk_df.to_string(index=False, float_format="{:.2f}".format)
    except Exception as e:
        logger.warning(f"Read text tables lỗi: {e}")

    tech = {
        "rsi": sig.get("rsi_value"),
        "ma": {"ma10":sig.get("ma10"),"ma20":sig.get("ma20"),"ma50":sig.get("ma50"),"ma200":sig.get("ma200")},
        "bollinger_bands": {"upper":sig.get("bb_upper"),"lower":sig.get("bb_lower")},
        "macd": {"macd":sig.get("macd"),"signal":sig.get("macd_signal"),"histogram":sig.get("macd_hist")},
        "ichimoku": {"tenkan":sig.get("ichimoku_tenkan_sen"),"kijun":sig.get("ichimoku_kijun_sen"),
                       "senkou_a":sig.get("ichimoku_senkou_span_a"),"senkou_b":sig.get("ichimoku_senkou_span_b"),
                       "chikou":sig.get("ichimoku_chikou_span")},
        "volume": {"current":sig.get("volume"),"ma20":sig.get("volume_ma_20"),"ma50":sig.get("volume_ma_50")},
    }

    if symbol.upper() == 'VNINDEX':
        prompt = build_index_prompt(symbol, sig['current_price'], tech, hist_text, market_text)
    else:
        prompt = build_stock_prompt(symbol, sig['current_price'], tech, sig, fin, company_blob, hist_text, infor_text, market_text)

    # Lưu prompt
    prompt_path = Path("prompt.txt")
    prompt_path.write_text(prompt, encoding='utf-8-sig')
    logger.info(f"Đã ghi prompt vào {prompt_path.resolve()}")

    # Gọi LLMs
    gem = run_gemini(prompt)
    oai = run_openrouter(prompt)

    # Gộp kết quả
    report = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": sig.get("current_price"),
        "signal": sig.get("signal"),
        "recommendation": sig.get("recommendation"),
        "score": sig.get("score"),
        "indicators": tech,
        "rs": {"rs":sig.get("rs"),"rs_point":sig.get("rs_point"),
                "r3d":sig.get("relative_strength_3d"),"r1m":sig.get("relative_strength_1m"),
                "r3m":sig.get("relative_strength_3m"),"r1y":sig.get("relative_strength_1y")},
        "gemini_analysis": gem,
        "openrouter_analysis": oai,
    }
    report_path = CFG.data_dir / f"{symbol}_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8-sig')
    logger.info(f"Đã lưu báo cáo {report_path}")
    return report


# -------------------------------
# LLM Drivers
# -------------------------------

def run_openrouter(prompt_text: str) -> str:
    try:
        logger.info("Gửi prompt tới OpenRouter...")
        resp = openrouter_client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role":"user","content":prompt_text}],
        )
        if resp and getattr(resp, 'choices', None):
            return resp.choices[0].message.content
        return "Không nhận được phản hồi từ OpenRouter."
    except Exception as e:
        logger.error(f"OpenRouter error: {e}")
        return "Không thể tạo phân tích bằng OpenRouter tại thời điểm này."


def run_gemini(prompt_text: str) -> str:
    try:
        logger.info("Gửi prompt tới Gemini...")
        model = genai.GenerativeModel(model_name="gemini-2.5-pro")
        resp = model.generate_content(prompt_text)
        if resp and getattr(resp, 'text', None):
            return resp.text.strip()
        return "Không nhận được phản hồi từ Gemini."
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."


# -------------------------------
# Screener
# -------------------------------

def screen_low_pe_high_cap(min_market_cap: int = 500) -> Optional[pd.DataFrame]:
    try:
        logger.info("Lọc cổ phiếu P/E thấp & vốn hoá cao...")
        df = Screener().stock(params={"exchangeName":"HOSE,HNX,UPCOM"}, limit=5000)
        if not validate_df(df):
            logger.error("Không thể lấy danh sách công ty.")
            return None
        c1 = df["market_cap"] >= min_market_cap
        c2 = ((df["pe"] > 0) & (df["pe"] < 20)) | pd.isna(df["pe"])
        c3 = (df["pb"] > 0) | pd.isna(df["pb"])
        c4 = (df["last_quarter_revenue_growth"] > 0) | pd.isna(df["last_quarter_revenue_growth"])
        c5 = (df["second_quarter_revenue_growth"] > 0) | pd.isna(df["second_quarter_revenue_growth"])
        c6 = (df["last_quarter_profit_growth"] > 0) | pd.isna(df["last_quarter_profit_growth"])
        c7 = (df["second_quarter_profit_growth"] > 0) | pd.isna(df["second_quarter_profit_growth"])
        c8 = ((df["peg_forward"] >= 0) & (df["peg_forward"] < 1)) | pd.isna(df["peg_forward"])
        c9 = ((df["peg_trailing"] >= 0) & (df["peg_trailing"] < 1)) | pd.isna(df["peg_trailing"])
        c10 = (df["revenue_growth_1y"] >= 0) | pd.isna(df["revenue_growth_1y"])
        c11 = (df["eps_growth_1y"] >= 0) | pd.isna(df["eps_growth_1y"])

        cond = c1 & c2 & c3 & c4 & c5 & c6 & c7 & c8 & c9 & c10 & c11
        filtered = df[cond]
        if filtered.empty:
            logger.warning("Không có mã thoả điều kiện.")
            return None
        (Path("market_filtered_pe.csv")).write_text(filtered.to_csv(index=False, encoding='utf-8-sig'))
        (Path("market_filtered.csv")).write_text(df[c1].to_csv(index=False, encoding='utf-8-sig'))
        logger.info(f"Đã lưu {len(filtered)} mã vào market_filtered_pe.csv")
        return filtered
    except Exception as e:
        logger.error(f"Screener lỗi: {e}")
        return None


# -------------------------------
# CLI
# -------------------------------

def main():
    print("="*60)
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM — VNSTOCK + AI")
    print("="*60)

    print("🔍 Lọc cổ phiếu P/E thấp...")
    screen_low_pe_high_cap()

    user = input("\nNhập mã để phân tích (ví dụ: VCB, FPT) hoặc 'exit': ").strip().upper()
    if user and user.lower() != 'exit':
        for ticker in [t.strip().upper() for t in user.split(',') if t.strip()]:
            print(f"\nPhân tích: {ticker}")
            analyze_symbol(ticker)
        print("\n✅ Hoàn tất. Xem thư mục 'vnstocks_data/'.")
    else:
        print("👋 Thoát.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nĐã dừng bởi người dùng.")
    except Exception:
        logger.error(traceback.format_exc())
