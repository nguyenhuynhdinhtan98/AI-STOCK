import warnings
warnings.filterwarnings("ignore", message=".*pkg_resources.*deprecated", category=UserWarning)
import os
import time
import json
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from openai import OpenAI
from openpyxl import load_workbook
from openpyxl.worksheet.table import Table, TableStyleInfo
import ta
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
from vnstock.explorer.vci import Quote, Finance, Company
from vnstock import Screener
import matplotlib.dates as mdates
import mplfinance as mpf
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import sys # Thêm để xử lý đối số dòng lệnh

# --- Cấu hình logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("stock_analysis.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Cấu hình toàn cục ---
GLOBAL_START_DATE = (datetime.today() - timedelta(days=365 * 15)).strftime("%Y-%m-%d")
GLOBAL_END_DATE = datetime.today().strftime("%Y-%m-%d")
DATA_DIR = "vnstocks_data"
os.makedirs(DATA_DIR, exist_ok=True)

# --- Cấu hình API ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")
if not GOOGLE_API_KEY or not OPEN_ROUTER_API_KEY:
    raise ValueError("Vui lòng đặt API keys trong file .env")
genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)

# --- Constants ---
TECHNICAL_INDICATORS = [
    'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200', 'RSI', 'MACD',
    'MACD_Signal', 'MACD_Hist', 'BB_Upper', 'BB_Middle', 'BB_Lower',
    'Volume_MA_20', 'Volume_MA_50'
]

# --- Hàm tiện ích ---
def safe_float(val: Any) -> Optional[float]:
    """Chuyển đổi giá trị sang float an toàn, trả về None nếu không hợp lệ."""
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

def safe_format(val: Any, fmt: str = ".2f") -> str:
    """Định dạng giá trị float an toàn, trả về 'N/A' nếu không hợp lệ."""
    num = safe_float(val)
    if num is None:
        return "N/A"
    return f"{num:{fmt}}"

def format_large_value(value: Any) -> str:
    """Định dạng giá trị lớn cho dễ đọc (K, M, B)"""
    num = safe_float(value)
    if num is None:
        return "N/A"
    abs_value = abs(num)
    if abs_value >= 1e9:
        return f"{num / 1e9:.2f}B"
    elif abs_value >= 1e6:
        return f"{num / 1e6:.2f}M"
    elif abs_value >= 1e3:
        return f"{num / 1e3:.2f}K"
    return f"{num:.2f}"

def validate_dataframe(df: pd.DataFrame, required_columns: list = None) -> bool:
    """Kiểm tra DataFrame có hợp lệ không"""
    if df is None or df.empty:
        return False
    if required_columns:
        return all(col in df.columns for col in required_columns)
    return True

# --- Hàm lấy dữ liệu ---
def get_stock_data(symbol: str) -> Optional[pd.DataFrame]:
    """Lấy dữ liệu lịch sử giá cổ phiếu từ VCI và lưu vào file CSV."""
    try:
        logger.info(f"Đang lấy dữ liệu cho mã {symbol}")
        stock = Quote(symbol=symbol)
        df = stock.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if not validate_dataframe(df, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning(f"Không lấy được dữ liệu cho mã {symbol}")
            return None
        column_mapping = {
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        df = df.rename(columns=column_mapping)
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        csv_path = f"{DATA_DIR}/{symbol}_data.csv"
        data_path = f"data.csv"
        df.to_csv(csv_path, index=True, encoding="utf-8-sig")
        df.to_csv(data_path, index=True, encoding="utf-8-sig")
        logger.info(f"Đã lưu dữ liệu cho mã {symbol} vào file {csv_path}")
        return df
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None

def get_company_info(symbol: str) -> str:
    """Lấy toàn bộ thông tin công ty từ vnstock và trả về chuỗi văn bản"""
    try:
        logger.info(f"Đang lấy thông tin công ty cho {symbol}")
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
        result = ""
        for section_name, data in info_sections.items():
            result += f"=== {section_name} ===\n"
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    result += data.to_string() + "\n"
                else:
                    result += "Không có dữ liệu\n"
            elif isinstance(data, dict):
                if data:
                    result += json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                else:
                    result += "Không có dữ liệu\n"
            elif data is not None:
                result += str(data) + "\n"
            else:
                result += "Không có dữ liệu\n"
            result += "\n"
        file_path = f"{DATA_DIR}/{symbol}_company_info.txt"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(result)
        logger.info(f"Đã lấy thông tin công ty {symbol} thành công")
        return result
    except Exception as e:
        error_msg = f"Lỗi khi lấy thông tin công ty {symbol}: {str(e)}"
        logger.error(error_msg)
        return error_msg

def get_financial_data(symbol: str) -> Optional[pd.DataFrame]:
    """Lấy dữ liệu báo cáo tài chính từ VCI và lưu vào file CSV."""
    try:
        logger.info(f"Đang lấy dữ liệu tài chính cho {symbol}")
        stock = Finance(symbol=symbol, period="quarter")
        df_ratio = stock.ratio(period="quarter")
        df_bs = stock.balance_sheet(period="quarter")
        df_is = stock.income_statement(period="quarter")
        df_cf = stock.cash_flow(period="quarter")
        def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df.columns.values]
            return df
        def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
            column_mapping = {
                "Meta_ticker": "ticker",
                "Meta_yearReport": "yearReport",
                "Meta_lengthReport": "lengthReport",
            }
            return df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        df_ratio = standardize_columns(flatten_columns(df_ratio))
        # Merge tất cả dữ liệu tài chính
        financial_data = df_bs.merge(
            df_is, on=["yearReport", "lengthReport", "ticker"], how="outer"
        ).merge(
            df_cf, on=["yearReport", "lengthReport", "ticker"], how="outer"
        ).merge(
            df_ratio, on=["yearReport", "lengthReport", "ticker"], how="outer"
        )
        # Đổi tên cột và lấy 20 bản ghi gần nhất
        rename_mapping = {
            "ticker": "Symbol",
            "yearReport": "Year",
            "lengthReport": "Quarter"
        }
        financial_data = financial_data.rename(
            columns={k: v for k, v in rename_mapping.items() if k in financial_data.columns}
        ).tail(20)
        csv_path = f"{DATA_DIR}/{symbol}_financial_statements.csv"
        financial_data.to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu dữ liệu tài chính của mã {symbol} vào file {csv_path}")
        return financial_data
    except Exception as e:
        logger.error(f"Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None

def get_market_data() -> Optional[pd.DataFrame]:
    """Lấy dữ liệu lịch sử của VNINDEX từ VCI và lưu vào file CSV."""
    try:
        logger.info("Đang lấy dữ liệu VNINDEX")
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if not validate_dataframe(vnindex, ['time', 'open', 'high', 'low', 'close', 'volume']):
            logger.warning("Không lấy được dữ liệu VNINDEX")
            return None
        column_mapping = {
            "time": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
        vnindex = vnindex.rename(columns=column_mapping)
        vnindex["Date"] = pd.to_datetime(vnindex["Date"])
        vnindex.set_index("Date", inplace=True)
        vnindex.sort_index(inplace=True)
        csv_path = f"{DATA_DIR}/VNINDEX_data.csv"
        vnindex.to_csv(csv_path, index=True, encoding="utf-8-sig")
        logger.info(f"Đã lưu dữ liệu VNINDEX vào file {csv_path}")
        return vnindex
    except Exception as e:
        logger.error(f"Lỗi khi lấy dữ liệu thị trường (VNINDEX): {str(e)}")
        return None

# --- Tiền xử lý dữ liệu ---
def preprocess_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """Tiền xử lý dữ liệu giá cổ phiếu cơ bản."""
    if not validate_dataframe(df):
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = df[col].ffill().bfill()
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=10).std()
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các chỉ báo kỹ thuật sử dụng thư viện 'ta'."""
    if not validate_dataframe(df, ['Close', 'High', 'Low', 'Volume']):
        return df
    df = df.copy()
    # Moving Averages
    for window in [10, 20, 50, 200]:
        df[f"SMA_{window}"] = ta.trend.sma_indicator(df["Close"], window=window)
    # RSI
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    # MACD
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_Hist"] = ta.trend.macd_diff(df["Close"])
    # Bollinger Bands
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"])
    # Volume Moving Averages
    for window in [20, 50]:
        df[f"Volume_MA_{window}"] = ta.trend.sma_indicator(df["Volume"], window=window)
    # Ichimoku
    ichimoku_indicator = ta.trend.IchimokuIndicator(
        high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
    )
    df["ichimoku_tenkan_sen"] = ichimoku_indicator.ichimoku_conversion_line()
    df["ichimoku_kijun_sen"] = ichimoku_indicator.ichimoku_base_line()
    df["ichimoku_senkou_span_a"] = ichimoku_indicator.ichimoku_a()
    df["ichimoku_senkou_span_b"] = ichimoku_indicator.ichimoku_b()
    df["ichimoku_chikou_span"] = df["Close"].shift(26)
    return df

# --- Tính toán Relative Strength (Đơn giản hóa) ---
def calculate_relative_strength(df_stock: pd.DataFrame, df_index: pd.DataFrame) -> pd.DataFrame:
    """Hàm được đơn giản hóa, không tính toán RS nội bộ."""
    logger.info("calculate_relative_strength được gọi nhưng không tính toán RS nội bộ.")
    return df_stock

# --- Lấy dữ liệu RS từ market_filtered.csv ---
def get_rs_from_market_data(symbol: str) -> Tuple[float, float, float, float]:
    """Lấy dữ liệu RS từ file market_filtered.csv"""
    try:
        file_path = "market_filtered.csv"
        if not os.path.exists(file_path):
            logger.warning(f"File {file_path} không tồn tại. Trả về giá trị mặc định.")
            return 1.0, 1.0, 1.0, 1.0
        market_df = pd.read_csv(file_path)
        if "ticker" not in market_df.columns:
            logger.error(f"Không tìm thấy cột 'ticker' trong file {file_path}")
            return 1.0, 1.0, 1.0, 1.0
        filtered_df = market_df[market_df["ticker"].str.upper() == symbol.upper()]
        if filtered_df.empty:
            logger.warning(f"Không tìm thấy dữ liệu cho mã cổ phiếu '{symbol}' trong file.")
            return 1.0, 1.0, 1.0, 1.0

        output_csv_file = f"{DATA_DIR}/{symbol}_infor.csv"
        filtered_df.to_csv(output_csv_file, index=False, encoding="utf-8-sig")

        # Lấy các giá trị RS, trả về giá trị mặc định nếu không có
        rs_value_3d = safe_float(filtered_df["relative_strength_3d"].iloc[0]) if "relative_strength_3d" in filtered_df.columns else 1.0
        rs_value_1m = safe_float(filtered_df["rel_strength_1m"].iloc[0]) if "rel_strength_1m" in filtered_df.columns else 1.0
        rs_value_3m = safe_float(filtered_df["rel_strength_3m"].iloc[0]) if "rel_strength_3m" in filtered_df.columns else 1.0
        rs_value_1y = safe_float(filtered_df["rel_strength_1y"].iloc[0]) if "rel_strength_1y" in filtered_df.columns else 1.0

        logger.info(f"Đã tìm thấy dữ liệu RS cho mã '{symbol}' trong file market_filtered.csv")
        return rs_value_3d, rs_value_1m, rs_value_3m, rs_value_1y
    except Exception as e:
        logger.error(f"Lỗi khi đọc hoặc lọc file market_filtered.csv: {e}")
        return 1.0, 1.0, 1.0, 1.0

# --- Phân tích kỹ thuật (Đơn giản hóa, không chấm điểm) ---
def calculate_technical(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """Lấy các chỉ báo kỹ thuật cơ bản, không tính điểm."""
    if not validate_dataframe(df):
        return 0.0, create_empty_trading_signal() # Trả về điểm 0 và tín hiệu rỗng
    try:
        last_row = df.iloc[-1]
        current_price = safe_float(last_row["Close"])
        if current_price is None:
            logger.error("Không thể lấy giá hiện tại")
            return create_empty_trading_signal()

        # Lấy các giá trị chỉ báo cơ bản
        indicators = {
            'rsi_value': safe_float(last_row.get("RSI", 50)),
            'ma10_value': safe_float(last_row.get("SMA_10", current_price)),
            'ma20_value': safe_float(last_row.get("SMA_20", current_price)),
            'ma50_value': safe_float(last_row.get("SMA_50", current_price)),
            'ma200_value': safe_float(last_row.get("SMA_200", current_price)),
            'macd_value': safe_float(last_row.get("MACD")),
            'macd_signal': safe_float(last_row.get("MACD_Signal")),
            'macd_hist': safe_float(last_row.get("MACD_Hist")),
            'bb_upper': safe_float(last_row.get("BB_Upper")),
            'bb_lower': safe_float(last_row.get("BB_Lower")),
            'volume_ma_20': safe_float(last_row.get("Volume_MA_20", df["Volume"].rolling(20).mean().iloc[-1])),
            'volume_ma_50': safe_float(last_row.get("Volume_MA_50", df["Volume"].rolling(50).mean().iloc[-1])),
        }

        # Ichimoku
        ichimoku_values = {}
        try:
            ichimoku_indicator = ta.trend.IchimokuIndicator(
                high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
            )
            ichimoku_values = {
                'tenkan_sen': safe_float(ichimoku_indicator.ichimoku_conversion_line().iloc[-1]),
                'kijun_sen': safe_float(ichimoku_indicator.ichimoku_base_line().iloc[-1]),
                'senkou_span_a': safe_float(ichimoku_indicator.ichimoku_a().iloc[-1]),
                'senkou_span_b': safe_float(ichimoku_indicator.ichimoku_b().iloc[-1]),
                'chikou_span': safe_float(df["Close"].shift(26).iloc[-1]),
            }
        except Exception as e:
            logger.warning(f"Lỗi khi tính Ichimoku: {e}")
            ichimoku_values = {k: None for k in ['tenkan_sen', 'kijun_sen', 'senkou_span_a', 'senkou_span_b', 'chikou_span']}

        # RS values từ market_filtered.csv
        rs_value_3d, rs_value_1m, rs_value_3m, rs_value_1y = get_rs_from_market_data(symbol)

        # Tạo kết quả (không có điểm số và tín hiệu MUA/BÁN)
        result = {
            "current_price": current_price,
            "rsi_value": indicators['rsi_value'],
            "ma10": indicators['ma10_value'],
            "ma20": indicators['ma20_value'],
            "ma50": indicators['ma50_value'],
            "ma200": indicators['ma200_value'],
            "open": safe_float(last_row.get("Open")),
            "high": safe_float(last_row.get("High")),
            "low": safe_float(last_row.get("Low")),
            "volume": safe_float(last_row.get("Volume")), # Volume hiện tại
            "volume_ma_20": indicators['volume_ma_20'],
            "volume_ma_50": indicators['volume_ma_50'],
            "macd": indicators['macd_value'],
            "macd_signal": indicators['macd_signal'],
            "macd_hist": indicators['macd_hist'],
            "bb_upper": indicators['bb_upper'],
            "bb_lower": indicators['bb_lower'],
            "ichimoku_tenkan_sen": ichimoku_values['tenkan_sen'],
            "ichimoku_kijun_sen": ichimoku_values['kijun_sen'],
            "ichimoku_senkou_span_a": ichimoku_values['senkou_span_a'],
            "ichimoku_senkou_span_b": ichimoku_values['senkou_span_b'],
            "ichimoku_chikou_span": ichimoku_values['chikou_span'],
            "relative_strength_3d": rs_value_3d,
            "relative_strength_1m": rs_value_1m,
            "relative_strength_3m": rs_value_3m,
            "relative_strength_1y": rs_value_1y,
        }
        return result # Trả về điểm 0 và kết quả
    except Exception as e:
        logger.error(f"Lỗi khi lấy chỉ báo kỹ thuật cho {symbol}: {str(e)}")
        return create_empty_trading_signal() # Trả về điểm 0 và tín hiệu rỗng

def create_empty_trading_signal() -> Dict[str, Any]:
    """Tạo tín hiệu giao dịch mặc định khi có lỗi"""
    return {
        "signal": "LỖI", "score": 0, "current_price": 0, "rsi_value": 0,
        "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0,
        "open": None, "high": None, "low": None,
        "volume": None, "macd": None, "macd_signal": None, "macd_hist": None,
        "bb_upper": None, "bb_lower": None, "volume_ma_20": None, "volume_ma_50": None,
        "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
        "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
        "ichimoku_chikou_span": None,
        "relative_strength_3d": None, "relative_strength_1m": None,
        "relative_strength_3m": None, "relative_strength_1y": None,
        "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
    }

def plot_stock_analysis(symbol: str, df: pd.DataFrame, show_volume: bool = True) -> Dict[str, Any]:
    """Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán."""
    if not validate_dataframe(df):
        logger.error("Dữ liệu phân tích rỗng")
        return create_empty_trading_signal()
    try:
        df = df.sort_index()
        df = create_features(df)
        # Không còn tính Relative Strength nội bộ nữa

        # Gọi hàm lấy chỉ báo, không chấm điểm
        trading_signal = calculate_technical(df, symbol)
        analysis_date = df.index[-1].strftime("%d/%m/%Y")

        # Hiển thị kết quả (chỉ hiển thị các chỉ báo kỹ thuật)
        logger.info(f"PHÂN TÍCH KỸ THUẬT CHO {symbol} ({analysis_date}):")
        logger.info(f" - Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
        logger.info(f" - Đường trung bình:")
        logger.info(f" * MA10: {trading_signal['ma10']:,.2f}| MA20: {trading_signal['ma20']:,.2f}| MA50: {trading_signal['ma50']:,.2f}| MA200: {trading_signal['ma200']:,.2f}")
        logger.info(f" - Chỉ báo dao động:")
        logger.info(f" * RSI (14): {trading_signal['rsi_value']:.2f}")
        logger.info(f" * MACD: {trading_signal['macd']:.2f}| Signal: {trading_signal['macd_signal']:.2f}| Histogram: {trading_signal['macd_hist']:.2f}")
        logger.info(f" * Bollinger Bands: Trên: {trading_signal['bb_upper']:,.2f}| Dưới: {trading_signal['bb_lower']:,.2f}")
        # Hiển thị RS từ market_filtered.csv nếu có
        if symbol.upper() != "VNINDEX":
             logger.info(f" - Sức mạnh tương đối (RS từ dữ liệu thị trường):")
             logger.info(f" * RS3D: {trading_signal['relative_strength_3d']}")
             logger.info(f" * RS1M: {trading_signal['relative_strength_1m']}")
             logger.info(f" * RS3M: {trading_signal['relative_strength_3m']}")
             logger.info(f" * RS1Y: {trading_signal['relative_strength_1y']}")
        # Ichimoku
        try:
            logger.info(f" - Mô hình Ichimoku:")
            logger.info(f" * Tenkan-sen (Chuyển đổi): {trading_signal['ichimoku_tenkan_sen']:.2f}")
            logger.info(f" * Kijun-sen (Cơ sở): {trading_signal['ichimoku_kijun_sen']:.2f}")
            logger.info(f" * Senkou Span A (Leading Span A): {trading_signal['ichimoku_senkou_span_a']:.2f}")
            logger.info(f" * Senkou Span B (Leading Span B): {trading_signal['ichimoku_senkou_span_b']:.2f}")
            logger.info(f" * Chikou Span (Trễ): {trading_signal['ichimoku_chikou_span']:.2f}")
        except Exception as e:
            logger.info(f" - Ichimoku: Không có đủ dữ liệu hoặc lỗi. {e}")
        logger.info(f" - Khối lượng:")
        logger.info(f" * Khối lượng hiện tại: {trading_signal.get('volume', 'N/A')}")
        logger.info(f" * MA Khối lượng (20): {trading_signal['volume_ma_20']:,.2f}")
        logger.info(f" * MA Khối lượng (50): {trading_signal['volume_ma_50']:,.2f}")

        return trading_signal
    except Exception as e:
        logger.error(f"Lỗi nghiêm trọng khi phân tích {symbol}: {str(e)}")
        traceback.print_exc() # In traceback để dễ debug
        return create_empty_trading_signal()

# --- Phân tích bằng AI ---
def analyze_with_openrouter(symbol: str) -> str:
    """Phân tích tổng hợp với OpenRouter"""
    try:
        prompt_path = "prompt.txt"
        if not os.path.exists(prompt_path):
            logger.error("File prompt.txt không tồn tại.")
            return "Không tìm thấy prompt để phân tích."
        with open(prompt_path, "r", encoding="utf-8-sig") as file:
            prompt_text = file.read()
        logger.info("Đang gửi prompt tới OpenRouter...")
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt_text}],
        )
        if response and response.choices:
            result = response.choices[0].message.content
            output_path = f"{DATA_DIR}/openrouter_analysis_{symbol}.txt"
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(result)
            logger.info(f"Đã lưu phân tích OpenRouter vào {output_path}")
            return result
        else:
            return "Không nhận được phản hồi từ OpenRouter."
    except Exception as e:
        logger.error(f"Lỗi khi phân tích bằng OpenRouter cho {symbol}: {str(e)}")
        return "Không thể tạo phân tích bằng OpenRouter tại thời điểm này."

def analyze_with_gemini(symbol: str) -> str:
    """Phân tích tổng hợp với AI Gemini, đọc prompt từ file"""
    try:
        prompt_path = "prompt.txt"
        if not os.path.exists(prompt_path):
            logger.error("File prompt.txt không tồn tại.")
            return "Không tìm thấy prompt để phân tích."
        with open(prompt_path, "r", encoding="utf-8-sig") as file:
            prompt_text = file.read()
        logger.info("Đang gửi prompt tới Gemini...")
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt_text)
        if response and response.text:
            result = response.text.strip()
            output_path = f"{DATA_DIR}/gemini_analysis_{symbol}.txt"
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(result)
            logger.info(f"Đã lưu phân tích Gemini vào {output_path}")
            return result
        else:
            return "Không nhận được phản hồi từ Gemini."
    except Exception as e:
        logger.error(f"Lỗi khi phân tích bằng Gemini cho {symbol}: {str(e)}")
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."

# --- Hàm tạo Prompt ---
def generate_advanced_stock_analysis_prompt(
    symbol: str, current_price: float, technical_indicators: Dict[str, Any],
    trading_signal: Dict[str, Any], financial_data: Optional[pd.DataFrame],
    company_info: str, historical_data: str, info_data: str, market_data_str: str
) -> str:
    """Tạo prompt phân tích chứng khoán nâng cao với đầy đủ thông tin kỹ thuật và cơ bản + CANSLIM"""
    def format_value(value: Any) -> str:
        num = safe_float(value)
        if num is None:
            return "N/A"
        if abs(num) >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"{num / 1e3:.2f}K"
        return f"{num:.2f}"
    # Extract technical indicators
    rsi = technical_indicators.get("rsi", "N/A")
    ma_values = technical_indicators.get("ma", {})
    bb = technical_indicators.get("bollinger_bands", {})
    macd = technical_indicators.get("macd", {})
    ichimoku = technical_indicators.get("ichimoku", {})
    volume_data = technical_indicators.get("volume", {})
    # Tạo prompt
    prompt = f"""
YÊU CẦU PHÂN TÍCH CHUYÊN SÂU:
Bạn hãy đóng vai một chuyên gia phân tích đầu tư chứng khoán hàng đầu, am hiểu cả phân tích kỹ thuật (Wyckoff, Minervini, VSA/VPA) và phân tích cơ bản (Buffett, Lynch). 
Hãy phân tích mã {symbol} một cách toàn diện, logic và có dẫn chứng cụ thể từ dữ liệu được cung cấp, sau đó đưa ra khuyến nghị cuối cùng.
MÃ PHÂN TÍCH: {symbol.upper()}
GIÁ HIỆN TẠI: {format_value(current_price)} VND
DỮ LIỆU KỸ THUẬT CHI TIẾT:
1. CHỈ BÁO XUNG LƯỢNG:
- RSI (14): {format_value(rsi)} {"(Quá mua)" if isinstance(rsi, (int, float)) and rsi > 70 else "(Quá bán)" if isinstance(rsi, (int, float)) and rsi < 30 else ""}
- MACD: {format_value(macd.get("macd", "N/A"))} | Signal: {format_value(macd.get("signal", "N/A"))} | Histogram: {format_value(macd.get("histogram", "N/A"))}
2. ĐƯỜNG TRUNG BÌNH (MA):
- MA10: {format_value(ma_values.get("ma10", "N/A"))}
- MA20: {format_value(ma_values.get("ma20", "N/A"))}
- MA50: {format_value(ma_values.get("ma50", "N/A"))}
- MA200: {format_value(ma_values.get("ma200", "N/A"))}
3. DẢI BOLLINGER:
- Band trên: {format_value(bb.get("upper", "N/A"))}
- Band dưới: {format_value(bb.get("lower", "N/A"))}
4. ICHIMOKU CLOUD:
- Tenkan-sen: {format_value(ichimoku.get("tenkan", "N/A"))}
- Kijun-sen: {format_value(ichimoku.get("kijun", "N/A"))}
- Senkou Span A: {format_value(ichimoku.get("senkou_a", "N/A"))}
- Senkou Span B: {format_value(ichimoku.get("senkou_b", "N/A"))}
- Chikou Span: {format_value(ichimoku.get("chikou", "N/A"))}
5. KHỐI LƯỢNG GIAO DỊCH:
- Khối lượng hiện tại: {format_value(volume_data.get("current", "N/A"))}
- Khối lượng trung bình 20 ngày: {format_value(volume_data.get("ma20", "N/A"))}
6. SỨC MẠNH TƯƠNG ĐỐI (RS):
- RS 3 ngày: {format_value(trading_signal.get("relative_strength_3d", "N/A"))}
- RS 1 tháng: {format_value(trading_signal.get("relative_strength_1m", "N/A"))}
- RS 3 tháng: {format_value(trading_signal.get("relative_strength_3m", "N/A"))}
- RS 1 năm: {format_value(trading_signal.get("relative_strength_1y", "N/A"))}

**PHÂN TÍCH THEO CÁC KHÚC CHÍNH SAU:**
**1. Phân tích kỹ thuật (Wyckoff, VSA & VPA):**
- **Giai đoạn thị trường:** Xác định mã đang ở giai đoạn nào (Tích lũy, Tăng trưởng, Phân phối, Suy thoái) theo Wyckoff. Giải thích tại sao.
- **Phân tích Giá & Khối lượng (VSA/VPA):** Phân tích mối quan hệ giữa biến động giá và khối lượng giao dịch gần đây.
  Có dấu hiệu tích lũy hay phân phối mạnh không? Khối lượng có xác nhận (hoặc không xác nhận) xu hướng giá không?
**2. Phân tích theo phương pháp Mark Minervini:**
- **Xu hướng:** Nhận định xu hướng chính (dài hạn) và xu hướng phụ (ngắn hạn).
- **Cấu trúc thị trường:** Phân tích các đỉnh/đáy để xác định xu hướng (đỉnh/đáy cao hơn hay thấp hơn).
- **Pivot & Hỗ trợ/Kháng cự:** Xác định các điểm pivot quan trọng và các vùng hỗ trợ/kháng cự gần đây.
- **Sức mạnh tương đối (RS):** Đánh giá sức mạnh tương đối của mã so với thị trường (VNINDEX).
**3. Phân tích cơ bản (Buffett, Lynch, dữ liệu TCBS):**
- **Chất lượng Doanh thu & Lợi nhuận:** Đánh giá tính ổn định và xu hướng tăng trưởng của doanh thu và lợi nhuận.
- **Hiệu quả Sử dụng Vốn:** Phân tích các chỉ số ROE, ROA, ROIC để đánh giá năng lực sử dụng vốn.
- **Tình hình Tài chính:** Đánh giá cơ cấu nợ, khả năng thanh khoản và chất lượng dòng tiền tự do (FCF).
- **Ban lãnh đạo & Nội bộ:** Đánh giá chất lượng ban lãnh đạo và hoạt động nội bộ.
- **Chia cổ tức:** Nhận xét về lịch sử và xu hướng chia cổ tức.
- **Tin tức & Internet:** Tổng hợp những tin tức quan trọng gần đây ảnh hưởng đến mã.
**4. Phân tích CANSLIM:**
- **C - Current Earnings (Lợi nhuận hiện tại):** Tốc độ tăng trưởng EPS quý gần nhất là bao nhiêu? Có ổn định không?
- **A - Annual Earnings (Lợi nhuận hàng năm):** Lịch sử tăng trưởng EPS hàng năm trong 3-5 năm qua như thế nào? Có tăng trưởng ổn định không?
- **N - New Products, New Management, New Highs (Sản phẩm mới, Ban lãnh đạo mới, Giá mới):** Có sản phẩm/dịch vụ mới, thay đổi ban lãnh đạo, hoặc phá mức cao mới gần đây không?
- **S - Supply and Demand (Cung và Cầu - Khối lượng):** Khối lượng giao dịch hiện tại so với trung bình như thế nào? Có dấu hiệu tích lũy hay phân phối lớn từ khối ngoại/NĐT lớn không?
- **L - Leader or Laggard (Người dẫn đầu hay kẻ chậm chân):** Mã này thuộc nhóm ngành nào? Hiệu suất so với ngành/ngành chính là tốt hay kém?
- **I - Institutional Sponsorship (Sự bảo trợ của tổ chức):** Có bao nhiêu quỹ tổ chức nắm giữ? Tỷ lệ thay đổi sở hữu của NĐT tổ chức gần đây như thế nào?
- **M - Market Direction (Xu hướng thị trường):** Môi trường thị trường hiện tại (VNINDEX) thuận lợi cho việc mua cổ phiếu tăng trưởng không? (Tích cực, Tiêu cực, Trung tính?)
**5. Định giá & So sánh ngành:**
- **Chỉ số Định giá:** Phân tích các chỉ số P/E, P/B, P/S, EV/EBITDA... ở hiện tại và so sánh với lịch sử.
- **So sánh Ngành:** So sánh các chỉ số định giá và tăng trưởng của mã với trung bình ngành.
**6. Nhận định vị thế mua ngắn hạn:**
- **Khả năng bật tăng ngắn hạn:** Đánh giá khả năng tăng giá trong ngắn hạn (1-4 tuần).
- **Các tín hiệu mua/bán gần đây:** Liệt kê và phân tích các tín hiệu mua/bán kỹ thuật gần đây.
- **Tâm lý thị trường ngắn hạn:** Nhận định tâm lý chung của NĐT với mã này trong ngắn hạn.
**7. Chiến lược giao dịch & Quản lý rủi ro:**
- **Điểm vào:** Đề xuất các điểm vào lệnh tiềm năng.
- **Stop-loss & Take-profit:** Đề xuất mức dừng lỗ và chốt lời hợp lý.
- **Risk/Reward:** Ước lượng tỷ lệ lợi nhuận trên rủi ro.
**8. Dự báo xu hướng:**
- **Ngắn hạn (1-2 tuần):** Dự báo ngắn hạn dựa trên phân tích kỹ thuật.
- **Trung hạn (1-3 tháng):** Dự báo trung hạn kết hợp kỹ thuật và cơ bản.
- **Dài hạn (3-12 tháng):** Dự báo dài hạn dựa trên triển vọng ngành.
**9. Kết luận & Khuyến nghị cuối cùng:**
Dựa trên toàn bộ phân tích ở trên, hãy đưa ra khuyến nghị cuối cùng cho mã {symbol}.
Bạn phải chọn MỘT trong 5 khuyến nghị sau và giải thích rõ lý do:
- **MUA MẠNH:** Tín hiệu kỹ thuật và cơ bản rất tích cực, điểm vào tốt, rủi ro thấp.
- **MUA:** Tín hiệu kỹ thuật và cơ bản tích cực, điểm vào hợp lý.
- **GIỮ:** Xu hướng đi ngang hoặc đang chờ xác nhận tín hiệu tiếp theo.
- **BÁN:** Tín hiệu kỹ thuật và cơ bản tiêu cực, điểm vào rủi ro cao.
- **BÁN MẠNH:** Tín hiệu kỹ thuật và cơ bản rất tiêu cực, điểm vào rủi ro rất cao.
**Yêu cầu cụ thể:**
- **Khuyến nghị:** Chọn một trong năm và giải thích rõ lý do chính.
- **Điểm số đánh giá (1-10):** Đánh giá mã trên thang điểm 10.
- **Tóm tắt ngắn gọn:** Tóm tắt lý do chính cho khuyến nghị trong 2-3 câu.
- **Rủi ro chính:** Liệt kê những rủi ro lớn nhất cần lưu ý.
**Yêu cầu về định dạng:**
- Trình bày rõ ràng, logic theo từng phần.
- Luôn đưa ra dẫn chứng cụ thể từ dữ liệu đã cung cấp.
- Kết hợp cả phân tích định lượng và định tính.
- Ưu tiên chất lượng, độ sâu và tính chính xác.
- Phân tích thêm thông tin vĩ mô cổ phiếu đảm bảo lợi nhuận đột biến tương lai.
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
    symbol: str, current_price: float, technical_indicators: Dict[str, Any],
    historical_data: str, market_data_str: str
) -> str:
    """Tạo prompt phân tích thị trường VNINDEX"""
    def format_value(value: Any) -> str:
        num = safe_float(value)
        if num is None:
            return "N/A"
        if abs(num) >= 1e9:
            return f"{num / 1e9:.2f}B"
        elif abs(num) >= 1e6:
            return f"{num / 1e6:.2f}M"
        elif abs(num) >= 1e3:
            return f"{num / 1e3:.2f}K"
        return f"{num:.2f}"
    # Extract technical indicators
    rsi = technical_indicators.get("rsi", "N/A")
    ma_values = technical_indicators.get("ma", {})
    bb = technical_indicators.get("bollinger_bands", {})
    macd = technical_indicators.get("macd", {})
    ichimoku = technical_indicators.get("ichimoku", {})
    volume_data = technical_indicators.get("volume", {})
    # Tạo prompt cho phân tích VNINDEX
    prompt = f"""
BẠN LÀ CHUYÊN GIA PHÂN TÍCH THỊ TRƯỜNG HÀNG ĐẦU VỚI CHUYÊN MÔN VSA/VPA & WYCKOFF
Kinh nghiệm: 20+ năm phân tích thị trường chứng khoán
Chuyên môn: Volume Spread Analysis, Volume Price Analysis, Wyckoff Method, Canslim, Minervini
🎯 **NHIỆM VỤ:** Phân tích VNINDEX toàn diện + Dự báo chính xác + Chiến lược thực tế
**DỮ LIỆU THỰC TẾ:**
CHỈ SỐ PHÂN TÍCH: {symbol.upper()}
ĐIỂM HIỆN TẠI: {format_value(current_price)}
DỮ LIỆU KỸ THUẬT CHI TIẾT:
1. CHỈ BÁO XUNG LƯỢNG:
- RSI (14): {format_value(rsi)} {"(Quá mua)" if isinstance(rsi, (int, float)) and rsi > 70 else "(Quá bán)" if isinstance(rsi, (int, float)) and rsi < 30 else ""}
- MACD: {format_value(macd.get("macd", "N/A"))} | Signal: {format_value(macd.get("signal", "N/A"))} | Histogram: {format_value(macd.get("histogram", "N/A"))}
2. ĐƯỜNG TRUNG BÌNH (MA):
- MA10: {format_value(ma_values.get("ma10", "N/A"))}
- MA20: {format_value(ma_values.get("ma20", "N/A"))}
- MA50: {format_value(ma_values.get("ma50", "N/A"))}
- MA200: {format_value(ma_values.get("ma200", "N/A"))}
3. DẢI BOLLINGER:
- Band trên: {format_value(bb.get("upper", "N/A"))}
- Band dưới: {format_value(bb.get("lower", "N/A"))}
4. ICHIMOKU CLOUD:
- Tenkan-sen: {format_value(ichimoku.get("tenkan", "N/A"))}
- Kijun-sen: {format_value(ichimoku.get("kijun", "N/A"))}
- Senkou Span A: {format_value(ichimoku.get("senkou_a", "N/A"))}
- Senkou Span B: {format_value(ichimoku.get("senkou_b", "N/A"))}
- Chikou Span: {format_value(ichimoku.get("chikou", "N/A"))}
5. KHỐI LƯỢNG GIAO DỊCH:
- Khối lượng hiện tại: {format_value(volume_data.get("current", "N/A"))}
- Khối lượng trung bình 20 ngày: {format_value(volume_data.get("ma20", "N/A"))}
**YÊU CẦU CỤ THỂ - TRẢ LỜI THEO CẤU TRÚC SAU:**
🔍 **1. PHÂN TÍCH VSA/VPA CHI TIẾT (Volume Spread Analysis):**
- **3 phiên gần nhất - Phân tích từng phiên:**
  * Phiên 1: Giá thay đổi? Volume so với trung bình? Mô hình VSA nào? (Test/Stop/Climax/Upthrust)
  * Phiên 2: Giá thay đổi? Volume so với trung bình? Mô hình VSA nào?
  * Phiên 3: Giá thay đổi? Volume so với trung bình? Mô hình VSA nào?
- **Volume Confirmation:** Volume đang xác nhận/XÁC NHẬN YẾU/KHÔNG XÁC NHẬN xu hướng giá?
- **Supply/Demand Analysis:** Dấu hiệu tích lũy (Demand) hay phân phối (Supply)?
📊 **2. PHÂN TÍCH WYCKOFF - Giai đoạn thị trường:**
- **Giai đoạn hiện tại:** TÍCH LŨY/TĂNG TRƯỞNG/PHÂN PHỐI/SUY THOÁI?
- **Dẫn chứng Wyckoff:**
  * Spring/Upthrust gần nhất?
  * Volume tại các điểm quan trọng?
  * Thời gian tích lũy (nếu có)?
- **Wyckoff Signal:** Có dấu hiệu breakout/breakdown không?
🔮 **3. DỰ BÁO CỤ THỂ (1-2 tuần) - Xác suất:**
- **Kịch bản CƠ BẢN :** VNINDEX sẽ... trong range...
- **Kịch bản TỐT NHẤT :** VNINDEX sẽ...
- **Kịch bản XẤU NHẤT :** VNINDEX sẽ...
💰 **4. CHIẾN LƯỢC ĐẦU TƯ THỰC TẾ:**
- **Vị thế hiện tại:** MUA/BÁN/GIỮ/CHỜ/GIẢM TỶ TRỌNG/TĂNG TỶ TRỌNG.
- **Entry Point:** Mức giá vào lệnh cụ thể?
- **Stop Loss:** Mức cắt lỗ?
- **Take Profit:** Mức chốt lời?
- **Risk/Reward:** Tỷ lệ thưởng/trừng phạt?
⭐ **7. TOP 20 MÃ CỔ PHIẾU TIỀM NĂNG (Dựa trên VSA/VPA & WYCKOFF & CANSLIM & MINERVINI) ĐƯỢC SẮP XẾP THEO TIỀM NĂNG GIẢM DẦN:**
- **Bắt buộc lấy từ MARKET_SCREEN_START**
- **Phân tích thêm thông tin vĩ mô cổ phiếu đảm bảo lợi nhuận đột biến tương lai**
| Mã | Lý do chọn (VSA/VPA & WYCKOFF) (ngắn gọn dưới 30 từ) | Entry | SL | TP | RR |
|----|---------------------|-------|----|----|----|
|    |                     |       |    |    |    |
⚠️  **8. RỦI RO & ĐIỂM CẦN THEO DÕI:**
- **Rủi ro kỹ thuật:** ...
- **Rủi ro vĩ mô:** ...
- **Rủi ro tâm lý:** ...
- **Các mức quan trọng cần theo dõi:** ...
🎯 **9. KHUYẾN NGHỊ CUỐI CÙNG:**
- **KHUYẾN NGHỊ:** THAM GIA/KHÔNG THAM GIA/GIẢM TỶ TRỌNG
- **LÝ DO CHÍNH:** (2-3 câu ngắn gọn, súc tích)
- **ĐIỂM SỐ ĐÁNH GIÁ:** .../10
**QUY TẮC BẮT BUỘC:**
✅ Dẫn chứng cụ thể cho mọi nhận định
✅ Ưu tiên chất lượng hơn số lượng
✅ Trả lời ngắn gọn, thực tế, có thể áp dụng
✅ Dùng bảng biểu khi liệt kê danh sách
✅ Tập trung vào VSA/VPA & WYCKOFF

# DỮ LIỆU
<<<HISTORICAL_DATA_START>>>
{historical_data}
<<<HISTORICAL_DATA_END>>>

<<<MARKET_SCREEN_START>>>
{market_data_str}
<<<MARKET_SCREEN_END>>>
"""
    return prompt

# --- Phân tích một mã cổ phiếu ---
def analyze_stock(symbol: str) -> Optional[Dict[str, Any]]:
    """Phân tích toàn diện một mã chứng khoán."""
    logger.info(f"{'=' * 60}")
    logger.info(f"PHÂN TÍCH TOÀN DIỆN MÃ {symbol}")
    logger.info(f"{'=' * 60}")
    # Lấy dữ liệu
    df = get_stock_data(symbol)
    if not validate_dataframe(df):
        logger.error(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    financial_data_statement = get_financial_data(symbol)
    company_info_data = get_company_info(symbol)
    # Tiền xử lý dữ liệu
    df_processed = preprocess_stock_data(df)
    if not validate_dataframe(df_processed) or len(df_processed) < 100:
        logger.error(f"Dữ liệu cho mã {symbol} không đủ để phân tích")
        return None
    # Phân tích kỹ thuật
    logger.info(f"Đang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_processed)
    # Chuẩn bị dữ liệu cho prompt
    csv_file_path = f"{DATA_DIR}/{symbol}_data.csv"
    infor_csv_file_path = f"{DATA_DIR}/{symbol}_infor.csv"
    market_file_path = f"market_filtered_pe.csv"
    historical_data_str = "Không có dữ liệu lịch sử."
    infor_data_str = "Không có dữ liệu thông tin công ty."
    market_data_str = "Không có dữ liệu thông tin thị trường."
    # Đọc dữ liệu lịch sử
    if os.path.exists(csv_file_path):
        try:
            df_history = pd.read_csv(csv_file_path).tail(2000)
            historical_data_str = df_history.to_string(index=False, float_format="{:.2f}".format)
            logger.info(f"Đã đọc dữ liệu lịch sử từ '{csv_file_path}'")
        except Exception as e:
            logger.warning(f"Không thể đọc file '{csv_file_path}': {e}")
    # Đọc dữ liệu thông tin
    if os.path.exists(infor_csv_file_path):
        try:
            df_infor = pd.read_csv(infor_csv_file_path)
            infor_data_str = df_infor.to_string(index=False, float_format="{:.2f}".format)
            logger.info(f"Đã đọc dữ liệu thông tin từ '{infor_csv_file_path}'")
        except Exception as e:
            logger.warning(f"Không thể đọc file '{infor_csv_file_path}': {e}")
    # Đọc dữ liệu thị trường
    if os.path.exists(market_file_path):
        try:
            df_market = pd.read_csv(market_file_path)
            market_data_str = df_market.to_string(index=False, float_format="{:.2f}".format)
            logger.info(f"Đã đọc dữ liệu thông tin từ '{market_file_path}'")
        except Exception as e:
            logger.warning(f"Không thể đọc file '{market_file_path}': {e}")
    # Chuẩn bị technical indicators
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
    if symbol == "VNINDEX":
        prompt = generate_vnindex_analysis_prompt(
            symbol=symbol,
            current_price=trading_signal.get("current_price"),
            technical_indicators=technical_indicators,
            historical_data=historical_data_str,
            market_data_str=market_data_str
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
            market_data_str=market_data_str
        )
    with open("prompt.txt", "w", encoding="utf-8-sig") as file:
        file.write(prompt)
    logger.info("Đã lưu nội dung prompt vào file prompt.txt")
    # Phân tích AI
    logger.info("Đang phân tích bằng Gemini...")
    gemini_analysis = analyze_with_gemini(symbol)
    logger.info("Đang phân tích bằng OpenRouter...")
    openrouter_analysis = analyze_with_openrouter(symbol)
    # Hiển thị kết quả
    logger.info(f"\n{'=' * 20} KẾT QUẢ PHÂN TÍCH CHO Mã {symbol} {'=' * 20}")
    logger.info(f"💰 Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    logger.info(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ GEMINI ---")
    logger.info(gemini_analysis)
    logger.info(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ OPENROUTER ---")
    logger.info(openrouter_analysis)
    logger.info(f"{'=' * 60}\n")
    # Tạo báo cáo
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
        "relative_strength_3d": safe_float(trading_signal.get("relative_strength_3d")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_1m": safe_float(trading_signal.get("relative_strength_1m")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_3m": safe_float(trading_signal.get("relative_strength_3m")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_1y": safe_float(trading_signal.get("relative_strength_1y")) if symbol.upper() != "VNINDEX" else None,
        "gemini_analysis": gemini_analysis,
        "openrouter_analysis": openrouter_analysis
    }
    report_path = f"{DATA_DIR}/{symbol}_report.json"
    with open(report_path, "w", encoding="utf-8-sig") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    logger.info(f"Đã lưu báo cáo phân tích vào file '{report_path}'")
    return report

# --- Lọc cổ phiếu ---
def filter_stocks_low_pe_high_cap(min_market_cap: int = 500) -> Optional[pd.DataFrame]:
    """Lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao."""
    try:
        logger.info("Đang lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao")
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if not validate_dataframe(df):
            logger.error("Không thể lấy dữ liệu danh sách công ty niêm yết.")
            return None
        # Điều kiện lọc
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
        # Tổng hợp tất cả điều kiện
        final_condition = (
            condition1 &
            condition2_pe &
            condition3_pb &
            condition4_rev_growth_last &
            condition5_rev_growth_second &
            condition6_profit_growth_last &
            condition7_profit_growth_second &
            condition8_peg_forward &
            condition9_peg_trailing &
            condition10_revenue_growth_1y &
            condition11_eps_growth_1y
        )
        # Lọc dữ liệu
        filtered_df = df[final_condition]
        if filtered_df.empty:
            logger.warning("Không tìm thấy cổ phiếu nào đáp ứng tất cả các tiêu chí lọc.")
            return None
        # Lưu kết quả
        output_csv_file = "market_filtered.csv"
        output_csv_file_pe = "market_filtered_pe.csv"
        filtered_df.to_csv(output_csv_file_pe, index=False, encoding="utf-8-sig")
        df[condition1].to_csv(output_csv_file, index=False, encoding="utf-8-sig")
        logger.info(f"Đã lưu danh sách cổ phiếu được lọc ({len(filtered_df)} mã) vào '{output_csv_file_pe}'")
        return filtered_df
    except Exception as e:
        logger.error(f"Đã xảy ra lỗi trong quá trình lọc cổ phiếu: {e}")
        return None

# --- Hàm chính ---
def main():
    """Hàm chính để chạy chương trình."""
    print("=" * 60)
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print("TÍCH HỢP VNSTOCK & AI")
    print("=" * 60)

    # Kiểm tra đối số dòng lệnh
    tickers_from_args = [arg.upper() for arg in sys.argv[1:] if arg and not arg.startswith('-')]

    # Lọc cổ phiếu
    print("🔍 Đang lọc cổ phiếu có P/E thấp")
    filter_stocks_low_pe_high_cap()

    if tickers_from_args:
        # Phân tích các mã từ dòng lệnh
        print(f"\nPhân tích các mã từ dòng lệnh: {', '.join(tickers_from_args)}")
        for ticker in tickers_from_args:
            if ticker:
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
    else:
        # Nhập mã cổ phiếu để phân tích
        print("\nNhập mã cổ phiếu để phân tích riêng lẻ (ví dụ: VCB, FPT) hoặc 'exit' để thoát")
        user_input = input("Nhập mã cổ phiếu để phân tích: ").strip().upper()
        if user_input and user_input.lower() != "exit":
            tickers = [ticker.strip() for ticker in user_input.split(",")]
            for ticker in tickers:
                if ticker:
                    print(f"\nPhân tích mã: {ticker}")
                    analyze_stock(ticker)
        else:
            print("👋 Thoát chương trình.")

    if tickers_from_args or (not tickers_from_args and user_input.lower() != "exit"):
        print("\n✅ Hoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")

if __name__ == "__main__":
    main()
