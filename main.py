import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated", category=UserWarning)

import os
import json
import logging
import sys
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from dotenv import load_dotenv

# TA libs
from ta.trend import SMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from ta.trend import IchimokuIndicator

# VNStock
from vnstock.explorer.vci import Quote, Finance, Company
from vnstock import Screener

# AI SDKs (yêu cầu API key trong .env)
import google.generativeai as genai
from openai import OpenAI

# ======================= LOGGING =======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======================= GLOBAL CONFIG =======================
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if not GOOGLE_API_KEY or not OPEN_ROUTER_API_KEY:
    raise ValueError("Vui lòng đặt GOOGLE_API_KEY và OPEN_ROUTER_API_KEY trong file .env")

genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPEN_ROUTER_API_KEY)

GLOBAL_START_DATE = (datetime.today() - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
GLOBAL_END_DATE = datetime.today().strftime("%Y-%m-%d")
DATA_DIR = "vnstocks_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ======================= HELPERS =======================
def safe_float(val: Any) -> Optional[float]:
    try:
        if val is None:
            return None
        if isinstance(val, (int, float, np.floating, np.integer)):
            if np.isnan(val) or np.isinf(val):
                return None
            return float(val)
        s = str(val).replace(',', '').strip()
        if s == "":
            return None
        num = float(s)
        return None if (np.isnan(num) or np.isinf(num)) else num
    except Exception:
        return None

def safe_format(val: Any, fmt: str = ".2f") -> str:
    num = safe_float(val)
    return f"{num:{fmt}}" if num is not None else "N/A"

def format_large_value(value: Any) -> str:
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

def validate_dataframe(df: Optional[pd.DataFrame], required_columns: List[str] = None) -> bool:
    if df is None or df.empty:
        return False
    if required_columns:
        return all(col in df.columns for col in required_columns)
    return True

# ======================= FETCHERS =======================
def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Lấy dữ liệu giá {symbol}")
        stock = Quote(symbol=symbol)
        df = stock.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if not validate_dataframe(df, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning(f"Không lấy được dữ liệu cho {symbol}")
            return None
        df = df.rename(columns={
            "time": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        csv_path = f"{DATA_DIR}/{symbol}_data.csv"
        df.to_csv(csv_path, index=True, encoding="utf-8-sig")
        df.to_csv("data.csv", index=True, encoding="utf-8-sig")
        logger.info(f"Đã lưu {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Lỗi lấy dữ liệu {symbol}: {e}")
        return None

def _flatten(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([c for c in col if c]).strip() for col in df.columns.values]
    return df

def _std_cols(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    if df is None:
        return df
    mapping = {
        "Meta_ticker": "ticker",
        "Meta_yearReport": "yearReport",
        "Meta_lengthReport": "lengthReport",
        "meta_ticker": "ticker",
        "meta_yearReport": "yearReport",
        "meta_lengthReport": "lengthReport",
    }
    cols = {k: v for k, v in mapping.items() if k in df.columns}
    if cols:
        df = df.rename(columns=cols)
    for col in ["ticker", "yearReport", "lengthReport"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def get_company_info(symbol: str) -> str:
    try:
        logger.info(f"Lấy thông tin công ty {symbol}")
        company = Company(symbol)
        info_sections = {
            "OVERVIEW": company.overview(),
            "SHAREHOLDERS": company.shareholders(),
            "OFFICERS": company.officers(filter_by='working'),
            "EVENTS": company.events(),
            "NEWS": company.news(),
            "REPORTS": company.reports(),
            "TRADING STATS": company.trading_stats(),
            "RATIO SUMMARY": company.ratio_summary()
        }
        result = []
        for section_name, data in info_sections.items():
            result.append(f"=== {section_name} ===")
            if isinstance(data, pd.DataFrame):
                result.append(data.to_string() if not data.empty else "Không có dữ liệu")
            elif isinstance(data, dict):
                result.append(json.dumps(data, ensure_ascii=False, indent=2) if data else "Không có dữ liệu")
            elif data is not None:
                result.append(str(data))
            else:
                result.append("Không có dữ liệu")
            result.append("")
        text = "\n".join(result)
        file_path = f"{DATA_DIR}/{symbol}_company_info.txt"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(text)
        return text
    except Exception as e:
        msg = f"Lỗi khi lấy thông tin công ty {symbol}: {e}"
        logger.error(msg)
        return msg

def get_financial_data(symbol: str) -> Optional[pd.DataFrame]:
    try:
        logger.info(f"Lấy BCTC {symbol}")
        stock = Finance(symbol=symbol, period="quarter")
        df_ratio = _std_cols(_flatten(stock.ratio(period="quarter")))
        df_bs = _std_cols(_flatten(stock.balance_sheet(period="quarter")))
        df_is = _std_cols(_flatten(stock.income_statement(period="quarter")))
        df_cf = _std_cols(_flatten(stock.cash_flow(period="quarter")))
        for d in (df_ratio, df_bs, df_is, df_cf):
            if d is not None:
                d['ticker'] = d['ticker'].fillna(symbol)
        base = df_bs
        for other in (df_is, df_cf, df_ratio):
            if other is not None and not other.empty:
                base = base.merge(
                    other, on=["yearReport", "lengthReport", "ticker"],
                    how="outer", suffixes=(None, None)
                )
        if base is None or base.empty:
            logger.warning(f"Không có BCTC hợp lệ cho {symbol}")
            return None
        financial_data = base.rename(columns={
            "ticker": "Symbol", "yearReport": "Year", "lengthReport": "Quarter"
        }).tail(20)
        csv_path = f"{DATA_DIR}/{symbol}_financial_statements.csv"
        financial_data.to_csv(csv_path, index=False, encoding='utf-8-sig')
        return financial_data
    except Exception as e:
        logger.error(f"Lỗi lấy BCTC {symbol}: {e}")
        return None

def get_market_data() -> Optional[pd.DataFrame]:
    try:
        logger.info("Lấy VNINDEX")
        quote = Quote(symbol="VNINDEX")
        vnindex = quote.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if not validate_dataframe(vnindex, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning("Không lấy được VNINDEX")
            return None
        vnindex = vnindex.rename(columns={
            "time": "Date", "open": "Open", "high": "High", "low": "Low",
            "close": "Close", "volume": "Volume"
        })
        vnindex["Date"] = pd.to_datetime(vnindex["Date"])
        vnindex.set_index("Date", inplace=True)
        vnindex.sort_index(inplace=True)
        csv_path = f"{DATA_DIR}/VNINDEX_data.csv"
        vnindex.to_csv(csv_path, index=True, encoding='utf-8-sig')
        return vnindex
    except Exception as e:
        logger.error(f"Lỗi lấy VNINDEX: {e}")
        return None

# ======================= PREPROCESS & FEATURES =======================
def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    if not validate_dataframe(df):
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].ffill().bfill()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    if not validate_dataframe(df, ['Close', 'High', 'Low', 'Volume']):
        return df
    df = df.copy()
    for window in [10, 20, 50, 200]:
        df[f"SMA_{window}"] = SMAIndicator(close=df["Close"], window=window).sma_indicator()
    df["RSI"] = RSIIndicator(close=df["Close"], window=14).rsi()
    macd_ind = MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df["MACD"] = macd_ind.macd()
    df["MACD_Signal"] = macd_ind.macd_signal()
    df["MACD_Hist"] = macd_ind.macd_diff()
    bb = BollingerBands(close=df["Close"], window=20, window_dev=2)
    df["BB_Upper"] = bb.bollinger_hband()
    df["BB_Middle"] = bb.bollinger_mavg()
    df["BB_Lower"] = bb.bollinger_lband()
    for window in [20, 50]:
        df[f"Volume_MA_{window}"] = SMAIndicator(close=df["Volume"], window=window).sma_indicator()
    try:
        ich = IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
        df["ichimoku_tenkan_sen"] = ich.ichimoku_conversion_line()
        df["ichimoku_kijun_sen"] = ich.ichimoku_base_line()
        df["ichimoku_senkou_span_a"] = ich.ichimoku_a()
        df["ichimoku_senkou_span_b"] = ich.ichimoku_b()
        df["ichimoku_chikou_span"] = df["Close"].shift(26)
    except Exception as e:
        logger.warning(f"Lỗi tính Ichimoku: {e}")
        for k in ["ichimoku_tenkan_sen","ichimoku_kijun_sen","ichimoku_senkou_span_a","ichimoku_senkou_span_b","ichimoku_chikou_span"]:
            df[k] = np.nan
    return df

# ======================= RS HANDLING =======================
def get_rs_from_market_data(symbol: str) -> Tuple[float, float, float, float]:
    try:
        file_path = "market_filtered.csv"
        if not os.path.exists(file_path):
            return 1.0, 1.0, 1.0, 1.0
        market_df = pd.read_csv(file_path)
        if "ticker" not in market_df.columns:
            return 1.0, 1.0, 1.0, 1.0
        filtered_df = market_df[market_df["ticker"].str.upper() == symbol.upper()]
        if filtered_df.empty:
            return 1.0, 1.0, 1.0, 1.0
        output_csv_file = f"{DATA_DIR}/{symbol}_infor.csv"
        filtered_df.to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        def getc(col: str) -> float:
            return safe_float(filtered_df[col].iloc[0]) if col in filtered_df.columns else 1.0
        rs3d = getc("relative_strength_3d")
        rs1m = getc("rel_strength_1m")
        rs3m = getc("rel_strength_3m")
        rs1y = getc("rel_strength_1y")
        return (rs3d or 1.0), (rs1m or 1.0), (rs3m or 1.0), (rs1y or 1.0)
    except Exception:
        return 1.0, 1.0, 1.0, 1.0

# ======================= TECHNICAL SNAPSHOT =======================
def create_empty_trading_signal() -> Dict[str, Any]:
    return {
        "signal": "LỖI", "score": 0, "current_price": 0,
        "rsi_value": 0, "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0,
        "open": None, "high": None, "low": None, "volume": None,
        "macd": None, "macd_signal": None, "macd_hist": None,
        "bb_upper": None, "bb_lower": None, "volume_ma_20": None, "volume_ma_50": None,
        "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
        "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
        "ichimoku_chikou_span": None,
        "relative_strength_3d": None, "relative_strength_1m": None,
        "relative_strength_3m": None, "relative_strength_1y": None
    }

def calculate_technical(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    if not validate_dataframe(df):
        return create_empty_trading_signal()
    try:
        last = df.iloc[-1]
        current_price = safe_float(last["Close"]) or 0.0
        indicators = {
            'rsi_value': safe_float(last.get("RSI", 50)),
            'ma10_value': safe_float(last.get("SMA_10", current_price)),
            'ma20_value': safe_float(last.get("SMA_20", current_price)),
            'ma50_value': safe_float(last.get("SMA_50", current_price)),
            'ma200_value': safe_float(last.get("SMA_200", current_price)),
            'macd_value': safe_float(last.get("MACD")),
            'macd_signal': safe_float(last.get("MACD_Signal")),
            'macd_hist': safe_float(last.get("MACD_Hist")),
            'bb_upper': safe_float(last.get("BB_Upper")),
            'bb_lower': safe_float(last.get("BB_Lower")),
            'volume_ma_20': safe_float(df["Volume"].rolling(20).mean().iloc[-1]) if "Volume" in df else None,
            'volume_ma_50': safe_float(df["Volume"].rolling(50).mean().iloc[-1]) if "Volume" in df else None,
        }
        ich_vals = {
            'tenkan_sen': safe_float(last.get("ichimoku_tenkan_sen")),
            'kijun_sen': safe_float(last.get("ichimoku_kijun_sen")),
            'senkou_span_a': safe_float(last.get("ichimoku_senkou_span_a")),
            'senkou_span_b': safe_float(last.get("ichimoku_senkou_span_b")),
            'chikou_span': safe_float(last.get("ichimoku_chikou_span")),
        }
        if symbol.upper() != "VNINDEX":
            rs3d, rs1m, rs3m, rs1y = get_rs_from_market_data(symbol)
        else:
            rs3d = rs1m = rs3m = rs1y = None
        return {
            "current_price": current_price,
            "rsi_value": indicators['rsi_value'],
            "ma10": indicators['ma10_value'],
            "ma20": indicators['ma20_value'],
            "ma50": indicators['ma50_value'],
            "ma200": indicators['ma200_value'],
            "open": safe_float(last.get("Open")),
            "high": safe_float(last.get("High")),
            "low": safe_float(last.get("Low")),
            "volume": safe_float(last.get("Volume")),
            "volume_ma_20": indicators['volume_ma_20'],
            "volume_ma_50": indicators['volume_ma_50'],
            "macd": indicators['macd_value'],
            "macd_signal": indicators['macd_signal'],
            "macd_hist": indicators['macd_hist'],
            "bb_upper": indicators['bb_upper'],
            "bb_lower": indicators['bb_lower'],
            "ichimoku_tenkan_sen": ich_vals['tenkan_sen'],
            "ichimoku_kijun_sen": ich_vals['kijun_sen'],
            "ichimoku_senkou_span_a": ich_vals['senkou_span_a'],
            "ichimoku_senkou_span_b": ich_vals['senkou_span_b'],
            "ichimoku_chikou_span": ich_vals['chikou_span'],
            "relative_strength_3d": rs3d,
            "relative_strength_1m": rs1m,
            "relative_strength_3m": rs3m,
            "relative_strength_1y": rs1y,
            "signal": None,
            "score": 0,
            "recommendation": None
        }
    except Exception as e:
        logger.error(f"Lỗi snapshot kỹ thuật {symbol}: {e}")
        return create_empty_trading_signal()

def plot_stock_analysis(symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
    if not validate_dataframe(df):
        return create_empty_trading_signal()
    try:
        df = df.sort_index()
        df = create_features(df)
        return calculate_technical(df, symbol)
    except Exception as e:
        logger.error(f"Lỗi phân tích {symbol}: {e}")
        return create_empty_trading_signal()

# ======================= AI ANALYSIS =======================
def analyze_with_openrouter(symbol: str) -> str:
    try:
        if not os.path.exists("prompt.txt"):
            return "Không tìm thấy prompt để phân tích."
        with open("prompt.txt", "r", encoding="utf-8-sig") as f:
            prompt_text = f.read()
        resp = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt_text}],
        )
        result = getattr(resp.choices[0].message, "content", "").strip() if getattr(resp, 'choices', None) else ""
        result = result or "Không nhận được phản hồi từ OpenRouter."
        out = f"{DATA_DIR}/openrouter_analysis_{symbol}.txt"
        with open(out, "w", encoding="utf-8-sig") as f:
            f.write(result)
        return result
    except Exception as e:
        logger.error(f"OpenRouter lỗi {symbol}: {e}")
        return "Không thể tạo phân tích bằng OpenRouter tại thời điểm này."

def analyze_with_gemini(symbol: str) -> str:
    try:
        if not os.path.exists("prompt.txt"):
            return "Không tìm thấy prompt để phân tích."
        with open("prompt.txt", "r", encoding="utf-8-sig") as f:
            prompt_text = f.read()
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt_text)
        result = (response.text or "").strip() if getattr(response, "text", None) else "Không nhận được phản hồi từ Gemini."
        out = f"{DATA_DIR}/gemini_analysis_{symbol}.txt"
        with open(out, "w", encoding="utf-8-sig") as f:
            f.write(result)
        return result
    except Exception as e:
        logger.error(f"Gemini lỗi {symbol}: {e}")
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."

# ======================= PROMPT BUILDERS =======================
def _fmt(v: Any) -> str:
    return format_large_value(v)

def generate_advanced_stock_analysis_prompt(
    symbol: str,
    current_price: float,
    technical_indicators: Dict[str, Any],
    trading_signal: Dict[str, Any],
    financial_data: Optional[pd.DataFrame],
    company_info: str,
    historical_data: str,
    info_data: str,
    market_data_str: str
) -> str:
    rsi = technical_indicators.get("rsi", "N/A")
    ma = technical_indicators.get("ma", {}) or {}
    bb = technical_indicators.get("bollinger_bands", {}) or {}
    macd = technical_indicators.get("macd", {}) or {}
    ich = technical_indicators.get("ichimoku", {}) or {}
    vol = technical_indicators.get("volume", {}) or {}

    prompt = f"""
Bạn là chuyên gia phân tích cổ phiếu Việt Nam (Wyckoff, VSA/VPA, Minervini, CANSLIM, Buffett/Lynch).

# QUY TẮC
- Tiếng Việt, gọn; mỗi bullet ≤ 30 từ; làm tròn 2 chữ số.
- Chỉ dùng dữ liệu cung cấp; thiếu ghi N/A; không suy đoán.
- Nếu dữ liệu mâu thuẫn: nêu rõ và chọn kết luận thận trọng.

# SNAPSHOT
- Mã: {symbol.upper()} | Giá: {_fmt(current_price)}
- RSI(14): {_fmt(rsi)} | MACD: {_fmt(macd.get('macd','N/A'))}/{_fmt(macd.get('signal','N/A'))}/{_fmt(macd.get('histogram','N/A'))}
- MA10/20/50/200: {_fmt(ma.get('ma10','N/A'))}/{_fmt(ma.get('ma20','N/A'))}/{_fmt(ma.get('ma50','N/A'))}/{_fmt(ma.get('ma200','N/A'))}
- Bollinger: Trên {_fmt(bb.get('upper','N/A'))} | Dưới {_fmt(bb.get('lower','N/A'))}
- Ichimoku:
  - Tenkan {_fmt(ich.get('tenkan','N/A'))} | Kijun {_fmt(ich.get('kijun','N/A'))} | Chikou {_fmt(ich.get('chikou','N/A'))}
  - Senkou Span A {_fmt(ich.get('senkou_a','N/A'))} | Senkou Span B {_fmt(ich.get('senkou_b','N/A'))}
- Volume: hiện tại {_fmt(vol.get('current','N/A'))} | MA20 {_fmt(vol.get('ma20','N/A'))} | MA50 {_fmt(vol.get('ma50','N/A'))}
- RS: 3D {_fmt(trading_signal.get('relative_strength_3d','N/A'))} | 1M {_fmt(trading_signal.get('relative_strength_1m','N/A'))} | 3M {_fmt(trading_signal.get('relative_strength_3m','N/A'))} | 1Y {_fmt(trading_signal.get('relative_strength_1y','N/A'))}

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
{historical_data}
<<<HISTORICAL_DATA_END>>>

<<<FINANCIALS_START>>>
{financial_data.to_string(index=False) if (financial_data is not None and not financial_data.empty) else 'KHÔNG CÓ DỮ LIỆU BÁO CÁO TÀI CHÍNH'}
<<<FINANCIALS_END>>>

<<<COMPANY_INFO_START>>>
{company_info}
<<<COMPANY_INFO_END>>>

<<<INFO_TCBS_START>>>
{info_data}
<<<INFO_TCBS_END>>>

<<<MARKET_SCREEN_START>>>
{market_data_str}
<<<MARKET_SCREEN_END>>>
"""
    return prompt

def generate_vnindex_analysis_prompt(
    symbol: str,
    current_price: float,
    technical_indicators: Dict[str, Any],
    historical_data: str,
    market_data_str: str
) -> str:
    rsi = technical_indicators.get("rsi", "N/A")
    ma = technical_indicators.get("ma", {}) or {}
    bb = technical_indicators.get("bollinger_bands", {}) or {}
    macd = technical_indicators.get("macd", {}) or {}
    ich = technical_indicators.get("ichimoku", {}) or {}
    vol = technical_indicators.get("volume", {}) or {}

    prompt = f"""
Bạn là chuyên gia phân tích thị trường Việt Nam (VSA/VPA, Wyckoff, CANSLIM, Minervini).

# QUY TẮC
- Tiếng Việt, có cấu trúc; làm tròn 2 chữ số.
- Không suy đoán ngoài dữ liệu; thiếu ghi N/A.
- Chỉ chọn mã từ MARKET_SCREEN.

# SNAPSHOT
- Chỉ số: {symbol.upper()} | Điểm: {_fmt(current_price)}
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
## 3) Minervini
- Xu hướng dài/ngắn; sắp xếp MA; hỗ trợ/kháng cự.
## 4) CANSLIM (M)
- On nếu (VNINDEX > MA50 và MACD>Signal); ngược lại Off.
## 5) Kịch bản 1–2 tuần (kèm xác suất)
- Cơ bản / Tốt nhất / Xấu nhất (mô tả ngắn + vùng điểm).
## 6) Chiến lược
- Vị thế: MUA/GIỮ/BÁN/CHỜ; quy tắc vào/thoát; rủi ro chính.

## 7) ĐỀ XUẤT MÃ (chỉ từ MARKET_SCREEN)
### 7.1) Xếp hạng (không dùng điểm tổng)
- Ưu tiên: RS1M cao → RS3M cao → (nếu có) RS6M cao → P/E thấp → PEG thấp.
- Bỏ qua tiêu chí nếu cột tương ứng thiếu toàn bộ.
### 7.2) Ràng buộc
- Tối đa 2 mã/nhóm ngành (sector/industry/icb_name/industry_name; nếu thiếu hết → bỏ ràng buộc).
- Loại đáy 20% thanh khoản nếu có volume/avg_volume_20d/turnover/value_traded.
- Tie-break: market_cap lớn hơn.
### 7.3) Bảng Top 20 (theo 7.1)
| Mã | Ngành | P/E | PEG | Rev 1Y | EPS 1Y | RS1M | RS3M | RS6M | Luận điểm (ngắn gọn) | Entry | SL | TP | RR | Trạng thái |
- Entry/SL/TP:
  - On: Entry="Mua từng phần"; SL=-7%; TP=+15% (RR≈2).
  - Off hoặc thiếu kỹ thuật: Entry="Theo dõi"; SL=N/A; TP=N/A; RR=N/A.

## 8) Danh mục & Phân bổ
- Chọn 20 mã mạnh nhất, tối đa 2 mã/ngành.
- On: tổng 40–60% NAV; Off: 0–20% NAV (theo dõi).

# DỮ LIỆU
<<<HISTORICAL_DATA_START>>>
{historical_data}
<<<HISTORICAL_DATA_END>>>

<<<MARKET_SCREEN_START>>>
{market_data_str}
<<<MARKET_SCREEN_END>>>
"""
    return prompt

# ======================= MAIN PIPELINE =======================
def analyze_stock(symbol: str) -> Optional[Dict[str, Any]]:
    logger.info("=" * 60)
    logger.info(f"PHÂN TÍCH MÃ {symbol}")
    logger.info("=" * 60)

    is_index = symbol.upper() == "VNINDEX"

    df = get_stock_data(symbol)
    if not validate_dataframe(df):
        logger.error(f"Thiếu dữ liệu {symbol}")
        return None

    financial_data_statement = None
    company_info_data = ""
    if not is_index:
        financial_data_statement = get_financial_data(symbol)
        company_info_data = get_company_info(symbol)
    else:
        company_info_data = "Chỉ số thị trường VNINDEX."

    df_processed = preprocess_stock_data(df)
    if not validate_dataframe(df_processed) or len(df_processed) < 100:
        logger.error(f"Dữ liệu {symbol} không đủ để phân tích")
        return None

    trading_signal = plot_stock_analysis(symbol, df_processed)

    csv_file_path = f"{DATA_DIR}/{symbol}_data.csv"
    infor_csv_file_path = f"{DATA_DIR}/{symbol}_infor.csv"
    market_file_path = f"market_filtered_pe.csv"

    def to_text_if_exists(path: str, fallback: str) -> str:
        if os.path.exists(path):
            try:
                df_ = pd.read_csv(path)
                return df_.tail(2000).to_string(index=False, float_format="{:.2f}".format)
            except Exception as e:
                logger.warning(f"Không đọc được '{path}': {e}")
        return fallback

    historical_data_str = to_text_if_exists(csv_file_path, "Không có dữ liệu lịch sử.")
    infor_data_str = to_text_if_exists(infor_csv_file_path, "Không có dữ liệu thông tin công ty.")
    market_data_str = to_text_if_exists(market_file_path, "Không có dữ liệu thông tin thị trường.")

    technical_indicators = {
        "rsi": trading_signal.get("rsi_value"),
        "ma": {
            "ma10": trading_signal.get("ma10"),
            "ma20": trading_signal.get("ma20"),
            "ma50": trading_signal.get("ma50"),
            "ma200": trading_signal.get("ma200"),
        },
        "bollinger_bands": {
            "upper": trading_signal.get("bb_upper"),
            "lower": trading_signal.get("bb_lower"),
        },
        "macd": {
            "macd": trading_signal.get("macd"),
            "signal": trading_signal.get("macd_signal"),
            "histogram": trading_signal.get("macd_hist"),
        },
        "ichimoku": {
            "tenkan": trading_signal.get("ichimoku_tenkan_sen"),
            "kijun": trading_signal.get("ichimoku_kijun_sen"),
            "senkou_a": trading_signal.get("ichimoku_senkou_span_a"),
            "senkou_b": trading_signal.get("ichimoku_senkou_span_b"),
            "chikou": trading_signal.get("ichimoku_chikou_span"),
        },
        "volume": {
            "current": trading_signal.get("volume"),
            "ma20": trading_signal.get("volume_ma_20"),
            "ma50": trading_signal.get("volume_ma_50"),
        },
    }

    if is_index:
        prompt = generate_vnindex_analysis_prompt(
            symbol=symbol,
            current_price=trading_signal.get("current_price"),
            technical_indicators=technical_indicators,
            historical_data=historical_data_str,
            market_data_str=market_data_str,
        )
    else:
        prompt = generate_advanced_stock_analysis_prompt(
            symbol=symbol,
            current_price=trading_signal.get("current_price"),
            technical_indicators=technical_indicators,
            trading_signal=trading_signal,
            financial_data=financial_data_statement,
            company_info=company_info_data,
            historical_data=historical_data_str,
            info_data=infor_data_str,
            market_data_str=market_data_str,
        )

    with open("prompt.txt", "w", encoding="utf-8-sig") as f:
        f.write(prompt)
    logger.info("Đã lưu prompt.txt")

    gemini_analysis = analyze_with_gemini(symbol)
    openrouter_analysis = analyze_with_openrouter(symbol)

    report = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": safe_float(trading_signal.get("current_price")),
        "signal": trading_signal.get("signal"),
        "recommendation": trading_signal.get("recommendation"),
        "score": safe_float(trading_signal.get("score")),
        "rsi_value": safe_float(trading_signal.get("rsi_value")),
        "ma10": safe_float(trading_signal.get("ma10")),
        "ma20": safe_float(trading_signal.get("ma20")),
        "ma50": safe_float(trading_signal.get("ma50")),
        "ma200": safe_float(trading_signal.get("ma200")),
        "open": safe_float(trading_signal.get("open")),
        "high": safe_float(trading_signal.get("high")),
        "low": safe_float(trading_signal.get("low")),
        "volume": safe_float(trading_signal.get("volume")),
        "macd": safe_float(trading_signal.get("macd")),
        "macd_signal": safe_float(trading_signal.get("macd_signal")),
        "macd_hist": safe_float(trading_signal.get("macd_hist")),
        "bb_upper": safe_float(trading_signal.get("bb_upper")),
        "bb_lower": safe_float(trading_signal.get("bb_lower")),
        "volume_ma_20": safe_float(trading_signal.get("volume_ma_20")),
        "volume_ma_50": safe_float(trading_signal.get("volume_ma_50")),
        "ichimoku_tenkan_sen": safe_float(trading_signal.get("ichimoku_tenkan_sen")),
        "ichimoku_kijun_sen": safe_float(trading_signal.get("ichimoku_kijun_sen")),
        "ichimoku_senkou_span_a": safe_float(trading_signal.get("ichimoku_senkou_span_a")),
        "ichimoku_senkou_span_b": safe_float(trading_signal.get("ichimoku_senkou_span_b")),
        "ichimoku_chikou_span": safe_float(trading_signal.get("ichimoku_chikou_span")),
        "relative_strength_3d": safe_float(trading_signal.get("relative_strength_3d")) if not is_index else None,
        "relative_strength_1m": safe_float(trading_signal.get("relative_strength_1m")) if not is_index else None,
        "relative_strength_3m": safe_float(trading_signal.get("relative_strength_3m")) if not is_index else None,
        "relative_strength_1y": safe_float(trading_signal.get("relative_strength_1y")) if not is_index else None,
        "gemini_analysis": gemini_analysis,
        "openrouter_analysis": openrouter_analysis,
    }

    report_path = f"{DATA_DIR}/{symbol}_report.json"
    with open(report_path, "w", encoding='utf-8-sig') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    logger.info(f"Đã lưu {report_path}")
    return report

# ======================= SCREENER =======================
def filter_stocks_low_pe_high_cap(min_market_cap: int = 500) -> Optional[pd.DataFrame]:
    try:
        logger.info("Lọc cổ phiếu nền (P/E thấp, vốn hóa đủ lớn)")
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if not validate_dataframe(df):
            logger.error("Không lấy được danh sách niêm yết")
            return None
        condition1 = df["market_cap"] >= min_market_cap
        condition2_pe = ((df["pe"] > 0) & (df["pe"] < 20)) | pd.isna(df["pe"])
        condition3_pb = (df["pb"] > 0) | pd.isna(df["pb"])
        condition4_rev_growth_last = (df["last_quarter_revenue_growth"] > 0) | pd.isna(df["last_quarter_revenue_growth"])
        condition5_rev_growth_second = (df["second_quarter_revenue_growth"] > 0) | pd.isna(df["second_quarter_revenue_growth"])
        condition6_profit_growth_last = (df["last_quarter_profit_growth"] > 0) | pd.isna(df["last_quarter_profit_growth"])
        condition7_profit_growth_second = (df["second_quarter_profit_growth"] > 0) | pd.isna(df["second_quarter_profit_growth"])
        condition8_peg_forward = ((df["peg_forward"] >= 0) & (df["peg_forward"] < 1)) | pd.isna(df["peg_forward"])
        condition9_peg_trailing = ((df["peg_trailing"] >= 0) & (df["peg_trailing"] < 1)) | pd.isna(df["peg_trailing"])
        condition10_revenue_growth_1y = (df["revenue_growth_1y"] >= 0) | pd.isna(df["revenue_growth_1y"])
        condition11_eps_growth_1y = (df["eps_growth_1y"] >= 0) | pd.isna(df["eps_growth_1y"])

        final_condition = (
            condition1 & condition2_pe & condition3_pb &
            condition4_rev_growth_last & condition5_rev_growth_second &
            condition6_profit_growth_last & condition7_profit_growth_second &
            condition8_peg_forward & condition9_peg_trailing &
            condition10_revenue_growth_1y & condition11_eps_growth_1y
        )

        filtered_df = df[final_condition]
        if filtered_df.empty:
            logger.warning("Không tìm thấy mã phù hợp.")
            return None

        output_csv_file = "market_filtered.csv"
        output_csv_file_pe = "market_filtered_pe.csv"
        filtered_df.to_csv(output_csv_file_pe, index=False, encoding='utf-8-sig')
        df[condition1].to_csv(output_csv_file, index=False, encoding='utf-8-sig')
        logger.info(f"Đã lưu {output_csv_file_pe} và {output_csv_file}")
        return filtered_df
    except Exception as e:
        logger.error(f"Lỗi screener: {e}")
        return None

# ======================= CLI =======================
def main():
    print("=" * 60)
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM — VNSTOCK + AI")
    print("=" * 60)

    tickers_from_args = [arg.upper() for arg in sys.argv[1:] if arg and not arg.startswith('-')]

    print("🔍 Đang lọc cổ phiếu nền")
    filter_stocks_low_pe_high_cap()

    any_ran = False
    if tickers_from_args:
        print(f"\nPhân tích: {', '.join(tickers_from_args)}")
        for ticker in tickers_from_args:
            if ticker:
                any_ran = True
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
    else:
        user_input = input("\nNhập mã (VD: VCB, FPT) hoặc 'exit': ").strip()
        if user_input and user_input.lower() != "exit":
            for ticker in [t.strip().upper() for t in user_input.split(",") if t.strip()]:
                any_ran = True
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
        else:
            print("👋 Thoát.")

    if any_ran:
        print("\n✅ Hoàn thành. Kết quả trong thư mục 'vnstocks_data/'.")

if __name__ == "__main__":
    main()