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
# --- Cấu hình toàn cục ---
GLOBAL_START_DATE = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
GLOBAL_END_DATE = datetime.today().strftime("%Y-%m-%d")
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
# Tạo thư mục lưu trữ dữ liệu
os.makedirs("vnstocks_data", exist_ok=True)
# --- Hàm tiện ích ---
def safe_float(val):
    """Chuyển đổi giá trị sang float an toàn, trả về None nếu không hợp lệ."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None
def safe_format(val, fmt=".2f"):
    """Định dạng giá trị float an toàn, trả về 'N/A' nếu không hợp lệ."""
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
            return "N/A"
        return f"{val:{fmt}}"
    except (TypeError, ValueError):
        return "N/A"
def format_large_value(value):
    """Định dạng giá trị lớn cho dễ đọc (K, M, B)"""
    if value is None or not isinstance(value, (int, float)):
        return "N/A"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{value / 1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{value / 1e3:.2f}K"
    return f"{value:.2f}"
# --- Hàm lấy dữ liệu ---
def get_stock_data(symbol):
    """Lấy dữ liệu lịch sử giá cổ phiếu từ VCI và lưu vào file CSV."""
    try:
        stock = Quote(symbol=symbol)
        df = stock.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if df is None or df.empty:
            print(f"⚠️ Không lấy được dữ liệu cho mã {symbol}")
            return None
        df.rename(
            columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        df["Date"] = pd.to_datetime(df["Date"])
        df.set_index("Date", inplace=True)
        df.sort_index(inplace=True)
        csv_path = f"vnstocks_data/{symbol}_data.csv"
        df.to_csv(csv_path, index=True, encoding="utf-8-sig")
        print(f"✅ Đã lưu dữ liệu cho mã {symbol} vào file {csv_path}")
        return df
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None
def get_company_info(symbol):
    """Lấy toàn bộ thông tin công ty từ vnstock và trả về chuỗi văn bản"""
    try:
        company = Company(symbol)
        overview_info = company.overview()
        shareholders_info = company.shareholders()
        officers_info = company.officers(filter_by='working')
        event_info = company.events()
        news_info = company.news()
        reports_info = company.reports()
        trading_stats_info = company.trading_stats()
        ratio_summary_info = company.ratio_summary()
        def convert_to_string(data, section_name):
            section_result = f"=== {section_name} ===\n"
            if isinstance(data, pd.DataFrame):
                if not data.empty:
                    section_result += data.to_string() + "\n"
                else:
                    section_result += "Không có dữ liệu\n"
            elif isinstance(data, dict):
                if data:
                    section_result += json.dumps(data, ensure_ascii=False, indent=2) + "\n"
                else:
                    section_result += "Không có dữ liệu\n"
            elif data is not None:
                section_result += str(data) + "\n"
            else:
                section_result += "Không có dữ liệu\n"
            section_result += "\n"
            return section_result
        result = ""
        result += convert_to_string(overview_info, "OVERVIEW")
        result += convert_to_string(shareholders_info, "SHAREHOLDERS")
        result += convert_to_string(officers_info, "OFFICERS")
        result += convert_to_string(event_info, "EVENTS")
        result += convert_to_string(news_info, "NEWS")
        result += convert_to_string(reports_info, "REPORTS")
        result += convert_to_string(trading_stats_info, "TRADING STATS")
        result += convert_to_string(ratio_summary_info, "RATIO SUMMARY")
        file_path = f"vnstocks_data/{symbol}_company_info.txt"
        with open(file_path, 'w', encoding='utf-8-sig') as f:
            f.write(result)
        print(f"✅ Đã lấy thông tin công ty {symbol} thành công")
        return result # Trả về chuỗi
    except Exception as e:
        error_msg = f"❌ Lỗi khi lấy thông tin công ty {symbol}: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        return error_msg # Trả về chuỗi lỗi
def safe_rename(df, mapping):
    """Đổi tên cột an toàn, chỉ đổi tên các cột tồn tại"""
    valid_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=valid_mapping)
def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính từ VCI và lưu vào file CSV."""
    try:
        stock = Finance(symbol=symbol, period="quarter")
        df_ratio = stock.ratio(period="quarter")
        df_bs = stock.balance_sheet(period="quarter")
        df_is = stock.income_statement(period="quarter")
        df_cf = stock.cash_flow(period="quarter")
        def flatten_columns(df):
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [
                    "_".join(col).strip() if col[1] else col[0]
                    for col in df.columns.values
                ]
            return df
        def standardize_columns(df):
            column_mapping = {
                "Meta_ticker": "ticker",
                "Meta_yearReport": "yearReport",
                "Meta_lengthReport": "lengthReport",
            }
            return safe_rename(df, column_mapping)
        df_ratio = standardize_columns(flatten_columns(df_ratio))
        financial_data = (
            df_bs.merge(df_is, on=["yearReport", "lengthReport", "ticker"], how="outer")
            .merge(df_cf, on=["yearReport", "lengthReport", "ticker"], how="outer")
            .merge(df_ratio, on=["yearReport", "lengthReport", "ticker"], how="outer")
        )
        renameFinance =  safe_rename(financial_data, {
            "ticker": "Symbol",
            "yearReport": "Year",
            "lengthReport": "Quarter"
        }).tail(20)
        csv_path = f"vnstocks_data/{symbol}_financial_statements.csv"
        renameFinance.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu dữ liệu tài chính của mã {symbol} vào file {csv_path}")
        return renameFinance
    except Exception as e:
        print(f"❌ Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None
def get_market_data():
    """Lấy dữ liệu lịch sử của VNINDEX từ VCI và lưu vào file CSV."""
    try:
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(
            start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D"
        )
        if vnindex is None or vnindex.empty:
            print("⚠️ Không lấy được dữ liệu VNINDEX")
            return None
        vnindex.rename(
            columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )
        vnindex["Date"] = pd.to_datetime(vnindex["Date"])
        vnindex.set_index("Date", inplace=True)
        vnindex.sort_index(inplace=True)
        csv_path = "vnstocks_data/VNINDEX_data.csv"
        vnindex.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu dữ liệu VNINDEX vào file {csv_path}")
        return vnindex
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu thị trường (VNINDEX): {str(e)}")
        return None
# --- Tiền xử lý dữ liệu ---
def preprocess_stock_data(df):
    """Tiền xử lý dữ liệu giá cổ phiếu cơ bản."""
    if df is None or df.empty:
        return df
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(window=10).std()
    return df
def create_features(df):
    """Tạo các chỉ báo kỹ thuật sử dụng thư viện 'ta'."""
    if df is None or df.empty:
        return df
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = ta.trend.macd(df["Close"])
    df["MACD_Signal"] = ta.trend.macd_signal(df["Close"])
    df["MACD_Hist"] = ta.trend.macd_diff(df["Close"])
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"])
    df["Volume_MA_20"] = ta.trend.sma_indicator(df["Volume"], window=20)
    df["Volume_MA_50"] = ta.trend.sma_indicator(df["Volume"], window=50)
    ichimoku_indicator = ta.trend.IchimokuIndicator(
        high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
    )
    df["ichimoku_tenkan_sen"] = ichimoku_indicator.ichimoku_conversion_line()
    df["ichimoku_kijun_sen"] = ichimoku_indicator.ichimoku_base_line()
    df["ichimoku_senkou_span_a"] = ichimoku_indicator.ichimoku_a()
    df["ichimoku_senkou_span_b"] = ichimoku_indicator.ichimoku_b()
    df["ichimoku_chikou_span"] = df["Close"].shift(26)
    return df
# --- Tính toán Relative Strength ---
def calculate_relative_strength(df_stock, df_index):
    """Tính Relative Strength (RS) và các chỉ báo RS Point theo công thức tiêu chuẩn."""
    if df_stock is None or df_index is None:
        return df_stock
    df_merged = df_stock[["Close"]].join(
        df_index[["Close"]].rename(columns={"Close": "Index_Close"}), how="inner"
    )
    if df_merged.empty or df_merged["Index_Close"].isna().all():
        print("⚠️ Cảnh báo: Không có dữ liệu chỉ số thị trường để tính RS. Gán giá trị mặc định.")
        rs_columns = [
            "RS", "RS_Point", "RS_SMA_10", "RS_SMA_20", "RS_SMA_50", "RS_SMA_200",
            "RS_Point_SMA_10", "RS_Point_SMA_20", "RS_Point_SMA_50", "RS_Point_SMA_200",
        ]
        for col in rs_columns:
            if "RS_Point" in col:
                df_stock[col] = 0.0
            else:
                df_stock[col] = 1.0
        return df_stock
    df_merged["RS"] = df_merged["Close"] / df_merged["Index_Close"]
    roc_63 = ta.momentum.roc(df_merged["Close"], window=63)
    roc_126 = ta.momentum.roc(df_merged["Close"], window=126)
    roc_189 = ta.momentum.roc(df_merged["Close"], window=189)
    roc_252 = ta.momentum.roc(df_merged["Close"], window=252)
    df_merged["RS_Point"] = (roc_63 * 0.4 + roc_126 * 0.2 + roc_189 * 0.2 + roc_252 * 0.2) * 100
    df_merged["RS_SMA_10"] = ta.trend.sma_indicator(df_merged["RS"], window=10)
    df_merged["RS_SMA_20"] = ta.trend.sma_indicator(df_merged["RS"], window=20)
    df_merged["RS_SMA_50"] = ta.trend.sma_indicator(df_merged["RS"], window=50)
    df_merged["RS_SMA_200"] = ta.trend.sma_indicator(df_merged["RS"], window=200)
    df_merged["RS_Point_SMA_10"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=10)
    df_merged["RS_Point_SMA_20"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=20)
    df_merged["RS_Point_SMA_50"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=50)
    df_merged["RS_Point_SMA_200"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=200)
    cols_to_join = [
        "RS", "RS_Point", "RS_SMA_10", "RS_SMA_20", "RS_SMA_50", "RS_SMA_200",
        "RS_Point_SMA_10", "RS_Point_SMA_20", "RS_Point_SMA_50", "RS_Point_SMA_200",
    ]
    df_stock = df_stock.join(df_merged[cols_to_join], how="left")
    for col in cols_to_join:
        if "RS_Point" in col:
            df_stock[col].fillna(0.0, inplace=True)
        else:
            df_stock[col].fillna(1.0, inplace=True)
    return df_stock
# --- Phân tích kỹ thuật và vẽ biểu đồ ---
def get_rs_from_market_data(symbol):
    """Lấy dữ liệu RS từ file market_filtered.csv"""
    try:
        file_path = "market_filtered.csv"
        if not os.path.exists(file_path):
            return 1.0, 1.0, 1.0, 1.0
        market_df = pd.read_csv(file_path)
        if "ticker" not in market_df.columns:
            print(f"Lỗi: Không tìm thấy cột 'ticker' trong file {file_path}")
            return 1.0, 1.0, 1.0, 1.0
        filtered_df = market_df[market_df["ticker"].str.upper() == symbol.upper()]
        if filtered_df.empty:
            print(f"Không tìm thấy dữ liệu cho mã cổ phiếu '{symbol}' trong file.")
            return 1.0, 1.0, 1.0, 1.0
        output_csv_file = f"vnstocks_data/{symbol}_infor.csv"
        filtered_df.to_csv(output_csv_file, index=False, encoding="utf-8-sig")
        rs_value_3d = filtered_df["relative_strength_3d"].iloc[0] if "relative_strength_3d" in filtered_df.columns else 1.0
        rs_value_1m = filtered_df["rel_strength_1m"].iloc[0] if "rel_strength_1m" in filtered_df.columns else 1.0
        rs_value_3m = filtered_df["rel_strength_3m"].iloc[0] if "rel_strength_3m" in filtered_df.columns else 1.0
        rs_value_1y = filtered_df["rel_strength_1y"].iloc[0] if "rel_strength_1y" in filtered_df.columns else 1.0
        print(f"Đã tìm thấy dữ liệu RS cho mã '{symbol}' trong file market_filtered.csv")
        return rs_value_3d, rs_value_1m, rs_value_3m, rs_value_1y
    except Exception as e:
        print(f"Lỗi khi đọc hoặc lọc file market_filtered.csv: {e}")
        return 1.0, 1.0, 1.0, 1.0
def calculate_technical_score(df, symbol):
    """Tính điểm kỹ thuật dựa trên các chỉ báo"""
    if df is None or df.empty:
        return 50, {}
    try:
        last_row = df.iloc[-1]
        current_price = last_row["Close"]
        rsi_value = last_row["RSI"] if not pd.isna(last_row["RSI"]) else 50
        ma10_value = last_row["SMA_10"] if not pd.isna(last_row["SMA_10"]) else current_price
        ma20_value = last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else current_price
        ma50_value = last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else current_price
        ma200_value = last_row["SMA_200"] if not pd.isna(last_row["SMA_200"]) else current_price
        macd_value = last_row["MACD"]
        macd_signal = last_row["MACD_Signal"]
        macd_hist = last_row["MACD_Hist"]
        bb_upper = last_row["BB_Upper"]
        bb_lower = last_row["BB_Lower"]
        volume_ma_20 = last_row["Volume_MA_20"] if "Volume_MA_20" in last_row else last_row["Volume"].rolling(20).mean().iloc[-1]
        volume_ma_50 = last_row["Volume_MA_50"] if "Volume_MA_50" in last_row else last_row["Volume"].rolling(50).mean().iloc[-1]
        ichimoku_indicator = ta.trend.IchimokuIndicator(high=df["High"], low=df["Low"], window1=9, window2=26, window3=52)
        tenkan_sen_series = ichimoku_indicator.ichimoku_conversion_line()
        kijun_sen_series = ichimoku_indicator.ichimoku_base_line()
        senkou_span_a_series = ichimoku_indicator.ichimoku_a()
        senkou_span_b_series = ichimoku_indicator.ichimoku_b()
        chikou_span_series = df["Close"].shift(26)
        tenkan_sen = tenkan_sen_series.iloc[-1] if len(tenkan_sen_series) > 0 and not pd.isna(tenkan_sen_series.iloc[-1]) else np.nan
        kijun_sen = kijun_sen_series.iloc[-1] if len(kijun_sen_series) > 0 and not pd.isna(kijun_sen_series.iloc[-1]) else np.nan
        senkou_span_a = senkou_span_a_series.iloc[-1] if len(senkou_span_a_series) > 0 and not pd.isna(senkou_span_a_series.iloc[-1]) else np.nan
        senkou_span_b = senkou_span_b_series.iloc[-1] if len(senkou_span_b_series) > 0 and not pd.isna(senkou_span_b_series.iloc[-1]) else np.nan
        chikou_span = chikou_span_series.iloc[-1] if len(chikou_span_series) > 26 and not pd.isna(chikou_span_series.iloc[-1]) else np.nan
        rs_value = last_row["RS"] if symbol.upper() != "VNINDEX" else 1.0
        rs_point_value = last_row["RS_Point"] if symbol.upper() != "VNINDEX" else 0.0
        rs_value_3d, rs_value_1m, rs_value_3m, rs_value_1y = get_rs_from_market_data(symbol)
        score = 50
        # MA Score
        ma_score = 0
        if current_price > ma10_value: ma_score += 3.5
        if current_price > ma20_value: ma_score += 3.5
        if current_price > ma50_value: ma_score += 3.5
        if current_price > ma200_value: ma_score += 3.5
        if ma10_value > ma20_value > ma50_value > ma200_value: ma_score += 3.5
        elif ma10_value < ma20_value < ma50_value < ma200_value: ma_score -= 3.5
        elif ma10_value > ma20_value and ma50_value > ma200_value: ma_score += 1.75
        elif ma10_value < ma20_value and ma50_value < ma200_value: ma_score -= 1.75
        score += ma_score
        # RSI Score
        rsi_score = 0
        if rsi_value < 30: rsi_score += 14
        elif 30 <= rsi_value < 40: rsi_score += 10
        elif 40 <= rsi_value < 50: rsi_score += 7
        elif 50 <= rsi_value < 60: rsi_score += 3.5
        elif 60 <= rsi_value < 70: rsi_score -= 3.5
        elif 70 <= rsi_value < 80: rsi_score -= 7
        else: rsi_score -= 14
        score += rsi_score
        # MACD Score
        macd_score = 0
        if macd_value > macd_signal and macd_hist > 0: macd_score += 7
        elif macd_value < macd_signal and macd_hist < 0: macd_score -= 7
        if len(df) > 1:
            macd_hist_prev = df["MACD_Hist"].iloc[-2]
            if macd_hist > macd_hist_prev: macd_score += 3.5
            elif macd_hist < macd_hist_prev: macd_score -= 3.5
        if len(df) > 1:
            macd_prev = df["MACD"].iloc[-2]
            signal_prev = df["MACD_Signal"].iloc[-2]
            if macd_value > macd_signal and macd_prev <= signal_prev: macd_score += 3.5
            elif macd_value < macd_signal and macd_prev >= signal_prev: macd_score -= 3.5
        score += macd_score
        # Ichimoku Score
        ichimoku_score = 0
        if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
            kumo_top = max(senkou_span_a, senkou_span_b)
            kumo_bottom = min(senkou_span_a, senkou_span_b)
            if current_price > kumo_top: ichimoku_score += 14
            elif current_price >= kumo_bottom and current_price <= kumo_top: ichimoku_score += 0
            elif current_price < kumo_bottom: ichimoku_score -= 14
        score += ichimoku_score
        # Volume Score
        volume_score = 0
        if "Volume" in last_row and not pd.isna(last_row["Volume"]):
            current_volume = last_row["Volume"]
            vol_ratio_to_ma20 = current_volume / volume_ma_20 if volume_ma_20 and volume_ma_20 > 0 else 0
            if vol_ratio_to_ma20 > 2.0: volume_score += 4
            elif vol_ratio_to_ma20 > 1.5: volume_score += 3
            elif vol_ratio_to_ma20 > 1.0: volume_score += 1
            elif vol_ratio_to_ma20 < 0.5: volume_score -= 2
            vol_ratio_to_ma50 = current_volume / volume_ma_50 if volume_ma_50 and volume_ma_50 > 0 else 0
            if vol_ratio_to_ma50 > 2.0: volume_score += 3
            elif vol_ratio_to_ma50 > 1.5: volume_score += 2
            elif vol_ratio_to_ma50 > 1.0: volume_score += 1
            elif vol_ratio_to_ma50 < 0.5: volume_score -= 1
            if len(df) > 2:
                vol_prev = df["Volume"].iloc[-2]
                vol_prev2 = df["Volume"].iloc[-3]
                if current_volume > vol_prev > vol_prev2:
                    if current_volume / vol_prev2 > 1.5: volume_score += 4
                    else: volume_score += 2
                elif current_volume < vol_prev < vol_prev2:
                    if current_volume / vol_prev2 < 0.7: volume_score -= 4
                    else: volume_score -= 2
            if len(df) > 40:
                vol_ma20_prev = df["Volume"].iloc[-21:-1].mean()
                if vol_ma20_prev > 0 and volume_ma_20 > 0:
                    vol_acc_ratio = volume_ma_20 / vol_ma20_prev
                    if vol_acc_ratio > 2.0: volume_score += 3
                    elif vol_acc_ratio > 1.5: volume_score += 1.5
                    elif vol_acc_ratio < 0.5: volume_score -= 2
            volume_score = np.clip(volume_score, -14, 14)
        score += volume_score
        # RS Score
        if symbol.upper() != "VNINDEX":
            rs_score = 0
            if rs_value > last_row.get("RS_SMA_10", rs_value): rs_score += 3.5
            elif rs_value < last_row.get("RS_SMA_10", rs_value): rs_score -= 3.5
            if rs_value > last_row.get("RS_SMA_50", rs_value): rs_score += 3.5
            elif rs_value < last_row.get("RS_SMA_50", rs_value): rs_score -= 3.5
            rs_point_sma20 = last_row.get("RS_Point_SMA_20", 0)
            if rs_point_value > rs_point_sma20: rs_score += 3.5
            elif rs_point_value < rs_point_sma20: rs_score -= 3.5
            if rs_point_value > 1.0: rs_score += 3.5
            elif rs_point_value < -1.0: rs_score -= 3.5
            score += rs_score
        # BB Score
        bb_score = 0
        if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper > bb_lower:
            bb_width = bb_upper - bb_lower
            price_to_upper = (bb_upper - current_price) / bb_width
            price_to_lower = (current_price - bb_lower) / bb_width
            if price_to_lower < 0.15: bb_score += 7
            elif price_to_lower < 0.3: bb_score += 3.5
            if price_to_upper < 0.15: bb_score -= 7
            elif price_to_upper < 0.3: bb_score -= 3.5
            if len(df) > 1 and not pd.isna(df["BB_Upper"].iloc[-2]) and not pd.isna(df["BB_Lower"].iloc[-2]):
                bb_width_prev = df["BB_Upper"].iloc[-2] - df["BB_Lower"].iloc[-2]
                if bb_width > bb_width_prev * 1.1: bb_score -= 1.75
                elif bb_width < bb_width_prev * 0.9: bb_score += 1.75
        score += bb_score
        score = np.clip(score, 0, 100)
        if score >= 80: signal, recommendation = "MUA MẠNH", "MUA MẠNH"
        elif score >= 65: signal, recommendation = "MUA", "MUA"
        elif score >= 55: signal, recommendation = "TĂNG MẠNH", "GIỮ - TĂNG"
        elif score >= 45: signal, recommendation = "TRUNG LẬP", "GIỮ"
        elif score >= 35: signal, recommendation = "GIẢM MẠNH", "GIỮ - GIẢM"
        elif score >= 20: signal, recommendation = "BÁN", "BÁN"
        else: signal, recommendation = "BÁN MẠNH", "BÁN MẠNH"
        result = {
            "signal": signal, "score": float(score), "current_price": float(current_price),
            "rsi_value": float(rsi_value), "ma10": float(ma10_value), "ma20": float(ma20_value),
            "ma50": float(ma50_value), "ma200": float(ma200_value), "rs": float(rs_value),
            "rs_point": float(rs_point_value), "recommendation": recommendation,
            "open": safe_float(last_row.get("Open")), "high": safe_float(last_row.get("High")),
            "low": safe_float(last_row.get("Low")), "volume": safe_float(last_row.get("Volume")),
            "volume_ma_20": safe_float(volume_ma_20), "volume_ma_50": safe_float(volume_ma_50),
            "macd": safe_float(macd_value), "macd_signal": safe_float(macd_signal),
            "macd_hist": safe_float(macd_hist), "bb_upper": safe_float(bb_upper),
            "bb_lower": safe_float(bb_lower), "ichimoku_tenkan_sen": safe_float(tenkan_sen),
            "ichimoku_kijun_sen": safe_float(kijun_sen), "ichimoku_senkou_span_a": safe_float(senkou_span_a),
            "ichimoku_senkou_span_b": safe_float(senkou_span_b), "ichimoku_chikou_span": safe_float(chikou_span),
            "rs_sma_10": safe_float(last_row.get("RS_SMA_10")), "rs_sma_20": safe_float(last_row.get("RS_SMA_20")),
            "rs_sma_50": safe_float(last_row.get("RS_SMA_50")), "rs_sma_200": safe_float(last_row.get("RS_SMA_200")),
            "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10")), "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20")),
            "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50")), "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200")),
            "relative_strength_3d": safe_float(rs_value_3d), "relative_strength_1m": safe_float(rs_value_1m),
            "relative_strength_3m": safe_float(rs_value_3m), "relative_strength_1y": safe_float(rs_value_1y),
            "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
        }
        return score, result
    except Exception as e:
        print(f"❌ Lỗi khi tính điểm kỹ thuật cho {symbol}: {str(e)}")
        traceback.print_exc()
        return 50, {}
def plot_stock_analysis(symbol, df, show_volume=True):
    """Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán."""
    if df is None or len(df) == 0:
        print("❌ Dữ liệu phân tích rỗng")
        return create_empty_trading_signal()
    try:
        df = df.sort_index()
        df = create_features(df)
        if symbol.upper() != "VNINDEX":
            try:
                vnindex = get_market_data()
                if vnindex is not None and not vnindex.empty:
                    df = calculate_relative_strength(df, vnindex)
                else:
                    print("⚠️ Không lấy được dữ liệu VNINDEX")
            except Exception as e:
                print(f"❌ Lỗi khi lấy dữ liệu thị trường (VNINDEX): {str(e)}")
        score, trading_signal = calculate_technical_score(df, symbol)
        analysis_date = df.index[-1].strftime("%d/%m/%Y")
        print(f"📊 TÍN HIỆU GIAO DỊCH CUỐI CÙNG CHO {symbol} ({analysis_date}):")
        print(f" - Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
        print(f" - Đường trung bình:")
        print(f" * MA10: {trading_signal['ma10']:,.2f}| MA20: {trading_signal['ma20']:,.2f}| MA50: {trading_signal['ma50']:,.2f}| MA200: {trading_signal['ma200']:,.2f}")
        print(f" - Chỉ báo dao động:")
        print(f" * RSI (14): {trading_signal['rsi_value']:.2f}")
        print(f" * MACD: {trading_signal['macd']:.2f}| Signal: {trading_signal['macd_signal']:.2f}| Histogram: {trading_signal['macd_hist']:.2f}")
        print(f" * Bollinger Bands: Trên: {trading_signal['bb_upper']:,.2f}| Dưới: {trading_signal['bb_lower']:,.2f}")
        if symbol.upper() != "VNINDEX":
            print(f" - Sức mạnh tương đối (RS):")
            print(f" * RS: {trading_signal['rs']}")
            print(f" * RS_Point: {trading_signal['rs_point']:.2f}")
            print(f" * RS3D: {trading_signal['relative_strength_3d']}")
            print(f" * RS1M: {trading_signal['relative_strength_1m']}")
            print(f" * RS3M: {trading_signal['relative_strength_3m']}")
            print(f" * RS1y: {trading_signal['relative_strength_1y']}")
        try:
            print(f" - Mô hình Ichimoku:")
            print(f" * Tenkan-sen (Chuyển đổi): {trading_signal['ichimoku_tenkan_sen']:.2f}")
            print(f" * Kijun-sen (Cơ sở): {trading_signal['ichimoku_kijun_sen']:.2f}")
            print(f" * Senkou Span A (Leading Span A): {trading_signal['ichimoku_senkou_span_a']:.2f}")
            print(f" * Senkou Span B (Leading Span B): {trading_signal['ichimoku_senkou_span_b']:.2f}")
            print(f" * Chikou Span (Trễ): {trading_signal['ichimoku_chikou_span']:.2f}")
        except:
            print(f" - Ichimoku: Không có đủ dữ liệu.")
        print(f" - Khối lượng:")
        print(f" * Khối lượng hiện tại: {trading_signal.get('volume', 'N/A')}")
        print(f" * MA Khối lượng (20): {trading_signal['volume_ma_20']:,.2f}")
        print(f" * MA Khối lượng (50): {trading_signal['volume_ma_50']:,.2f}")
        print(f" 🎯 ĐỀ XUẤT CUỐI CÙNG: {trading_signal['recommendation']}")
        print(f" 📊 TỔNG ĐIỂM PHÂN TÍCH: {score:.1f}/100")
        print(f" 📈 TÍN HIỆU: {trading_signal['signal']}")
        return trading_signal
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi phân tích {symbol}: {str(e)}")
        traceback.print_exc()
        return create_empty_trading_signal()
def create_empty_trading_signal():
    """Tạo tín hiệu giao dịch mặc định khi có lỗi"""
    return {
        "signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0,
        "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0,
        "recommendation": "KHÔNG XÁC ĐỊNH", "open": None, "high": None, "low": None,
        "volume": None, "macd": None, "macd_signal": None, "macd_hist": None,
        "bb_upper": None, "bb_lower": None, "volume_ma_20": None, "volume_ma_50": None,
        "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
        "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
        "ichimoku_chikou_span": None, "rs_sma_10": None, "rs_sma_20": None,
        "rs_sma_50": None, "rs_sma_200": None, "rs_point_sma_10": None,
        "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
        "relative_strength_3d": None, "relative_strength_1m": None,
        "relative_strength_3m": None, "relative_strength_1y": None,
        "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": "",
    }
# --- Phân tích bằng AI ---
def analyze_with_openrouter(symbol):
    """Phân tích tổng hợp với OpenRouter """
    try:
        prompt_path = "prompt.txt"
        if not os.path.exists(prompt_path):
            print(f"❌ File prompt.txt không tồn tại.")
            return "Không tìm thấy prompt để phân tích."
        with open(prompt_path, "r", encoding="utf-8-sig") as file:
            prompt_text = file.read()
        print(f"📤 Đang gửi prompt tới OpenRouter...")
        response = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=[{"role": "user", "content": prompt_text}],
        )
        if response and response.choices:
            result = response.choices[0].message.content
            output_path = f"vnstocks_data/openrouter_analysis_{symbol}.txt"
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(result)
            print(f"✅ Đã lưu phân tích OpenRouter vào {output_path}")
            return result
        else:
            return "Không nhận được phản hồi từ OpenRouter."
    except Exception as e:
        print(f"❌ Lỗi khi phân tích bằng OpenRouter cho {symbol}: {str(e)}")
        return "Không thể tạo phân tích bằng OpenRouter tại thời điểm này."
def analyze_with_gemini(symbol):
    """Phân tích tổng hợp với AI Gemini, đọc prompt từ file"""
    try:
        prompt_path = "prompt.txt"
        if not os.path.exists(prompt_path):
            print(f"❌ File prompt.txt không tồn tại.")
            return "Không tìm thấy prompt để phân tích."
        with open(prompt_path, "r", encoding="utf-8-sig") as file:
            prompt_text = file.read()
        print(f"📤 Đang gửi prompt tới Gemini...")
        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(prompt_text) # Gửi trực tiếp nội dung prompt
        if response and response.text:
            result = response.text.strip()
            output_path = f"vnstocks_data/gemini_analysis_{symbol}.txt"
            with open(output_path, "w", encoding="utf-8-sig") as f:
                f.write(result)
            print(f"✅ Đã lưu phân tích Gemini vào {output_path}")
            return result
        else:
            return "Không nhận được phản hồi từ Gemini."
    except Exception as e:
        print(f"❌ Lỗi khi phân tích bằng Gemini cho {symbol}: {str(e)}")
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."
# --- Hàm tạo Prompt (Prompt Engineering) ---
def generate_advanced_stock_analysis_prompt(
    symbol, current_price, technical_indicators, trading_signal,
    financial_data, company_info, historical_data, info_data, market_data_str
):
    """Tạo prompt phân tích chứng khoán nâng cao với đầy đủ thông tin kỹ thuật và cơ bản"""
    def format_value(value):
        if isinstance(value, (int, float)):
            if abs(value) >= 1e9: return f"{value / 1e9:.2f}B"
            elif abs(value) >= 1e6: return f"{value / 1e6:.2f}M"
            elif abs(value) >= 1e3: return f"{value / 1e3:.2f}K"
            return f"{value:.2f}"
        return str(value)
    rsi = technical_indicators.get("rsi", "N/A")
    ma_values = technical_indicators.get("ma", {})
    bb = technical_indicators.get("bollinger_bands", {})
    macd = technical_indicators.get("macd", {})
    ichimoku = technical_indicators.get("ichimoku", {})
    volume_data = technical_indicators.get("volume", {})
    company_info_str = company_info if company_info else "Không có thông tin công ty"
    prompt = f"""
Yêu cầu phân tích chuyên sâu:
Bạn hãy đóng vai một chuyên gia phân tích đầu tư chứng khoán hàng đầu, am hiểu cả phân tích kỹ thuật (Wyckoff, Minervini, VSA/VPA) và phân tích cơ bản (Buffett, Lynch). Hãy phân tích mã {symbol} một cách toàn diện, logic và có dẫn chứng cụ thể từ dữ liệu được cung cấp, sau đó đưa ra khuyến nghị cuối cùng.
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
- Vị trí giá so với MA: {"Trên tất cả MA - Xu hướng tăng mạnh" if all(current_price > ma for ma in [ma_values.get("ma10", 0), ma_values.get("ma20", 0), ma_values.get("ma50", 0), ma_values.get("ma200", 0)]) else "Dưới tất cả MA - Xu hướng giảm mạnh" if all(current_price < ma for ma in [ma_values.get("ma10", 0), ma_values.get("ma20", 0), ma_values.get("ma50", 0), ma_values.get("ma200", 0)]) else "Hỗn hợp - Xu hướng đi ngang/thiếu định hướng"}
3. DẢI BOLLINGER:
- Band trên: {format_value(bb.get("upper", "N/A"))}
- Band dưới: {format_value(bb.get("lower", "N/A"))}
- Độ rộng dải: {format_value((bb.get("upper", 0) - bb.get("lower", 0)) if all(k in bb for k in ["upper", "lower"]) else "N/A")}
- Vị trí giá: {"Gần band trên - Có thể quá mua" if isinstance(current_price, (int, float)) and isinstance(bb.get("upper", None), (int, float)) and current_price > bb["upper"] * 0.9 else "Gần band dưới - Có thể quá bán" if isinstance(current_price, (int, float)) and isinstance(bb.get("lower", None), (int, float)) and current_price < bb["lower"] * 1.1 else "Trong dải - Trạng thái bình thường"}
4. ICHIMOKU CLOUD:
- Tenkan-sen: {format_value(ichimoku.get("tenkan", "N/A"))}
- Kijun-sen: {format_value(ichimoku.get("kijun", "N/A"))}
- Senkou Span A: {format_value(ichimoku.get("senkou_a", "N/A"))}
- Senkou Span B: {format_value(ichimoku.get("senkou_b", "N/A"))}
- Chikou Span: {format_value(ichimoku.get("chikou", "N/A"))}
- Vị trí giá so với đám mây: {"Trên đám mây - Tăng giá" if isinstance(current_price, (int, float)) and isinstance(ichimoku.get("senkou_a", None), (int, float)) and isinstance(ichimoku.get("senkou_b", None), (int, float)) and current_price > max(ichimoku["senkou_a"], ichimoku["senkou_b"]) else "Dưới đám mây - Giảm giá" if isinstance(current_price, (int, float)) and isinstance(ichimoku.get("senkou_a", None), (int, float)) and isinstance(ichimoku.get("senkou_b", None), (int, float)) and current_price < min(ichimoku["senkou_a"], ichimoku["senkou_b"]) else "Trong đám mây - Thiếu xu hướng rõ ràng"}
5. KHỐI LƯỢNG GIAO DỊCH:
- Khối lượng hiện tại: {format_value(volume_data.get("current", "N/A"))}
- Khối lượng trung bình 20 ngày: {format_value(volume_data.get("ma20", "N/A"))}
- Tỷ lệ khối lượng: {format_value(volume_data.get("current", 0) / volume_data.get("ma20", 1) if volume_data.get("ma20", 0) != 0 else "N/A")} {"(Cao hơn trung bình - Khối lượng tăng mạnh)" if isinstance(volume_data.get("current", None), (int, float)) and isinstance(volume_data.get("ma20", None), (int, float)) and volume_data["current"] > volume_data["ma20"] * 1.5 else "(Thấp hơn trung bình - Khối lượng yếu)"}
6. SỨC MẠNH TƯƠNG ĐỐI (RS):
- RS so với VNINDEX: {format_value(trading_signal.get("rs", "N/A"))}
- RS Point (IBD): {format_value(trading_signal.get("rs_point", "N/A"))}
- RS 3 ngày: {format_value(trading_signal.get("relative_strength_3d", "N/A"))}
- RS 1 tháng: {format_value(trading_signal.get("relative_strength_1m", "N/A"))}
- RS 3 tháng: {format_value(trading_signal.get("relative_strength_3m", "N/A"))}
- RS 1 năm: {format_value(trading_signal.get("relative_strength_1y", "N/A"))}
"""
    if financial_data is not None and not financial_data.empty:
        prompt += f"""
BÁO CÁO TÀI CHÍNH:
{financial_data.to_string(index=False)}
"""
    else:
        prompt += "\nKHÔNG CÓ DỮ LIỆU BÁO CÁO TÀI CHÍNH\n"
    prompt += f"""
THÔNG TIN DỮ LIỆU LỊCH SỬ GIÁ:
{historical_data}
THÔNG TIN CÔNG TY:
{company_info_str}
THÔNG TIN CHUNG TỪ TCBS:
{info_data}
THÔNG TIN TOÀN BỘ CỔ PHIẾU THỊ TRƯỜNG:
{market_data_str}
**PHÂN TÍCH THEO CÁC KHÚC CHÍNH SAU:**
**1. Phân tích kỹ thuật (Wyckoff, VSA & VPA):**
- **Giai đoạn thị trường:** Xác định mã đang ở giai đoạn nào (Tích lũy, Tăng trưởng, Phân phối, Suy thoái) theo Wyckoff. Giải thích tại sao.
- **Phân tích Giá & Khối lượng (VSA/VPA):** Phân tích mối quan hệ giữa biến động giá và khối lượng giao dịch gần đây. Có dấu hiệu tích lũy hay phân phối mạnh không? Khối lượng có xác nhận (hoặc không xác nhận) xu hướng giá không? (Ví dụ: Khối lượng lớn khi giá tăng = xác nhận; Khối lượng lớn khi giá giảm = không xác nhận).
- **Mô hình & Dấu hiệu Wyckoff:** Tìm kiếm và bình luận về các dấu hiệu Wyckoff như Spring, Upthrust, Selling Climax, Buying Climax.
**2. Phân tích theo phương pháp Mark Minervini:**
- **Xu hướng:** Nhận định xu hướng chính (dài hạn) và xu hướng phụ (ngắn hạn).
- **Cấu trúc thị trường:** Phân tích các đỉnh/đáy để xác định xu hướng (đỉnh/đáy cao hơn hay thấp hơn).
- **Pivot & Hỗ trợ/Kháng cự:** Xác định các điểm pivot quan trọng và các vùng hỗ trợ/kháng cự gần đây.
- **Sức mạnh tương đối (RS):** Đánh giá sức mạnh tương đối của mã so với thị trường (VNINDEX) dựa trên dữ liệu RS đã cung cấp.
**3. Phân tích cơ bản (Buffett, Lynch, dữ liệu TCBS):**
- **Chất lượng Doanh thu & Lợi nhuận:** Đánh giá tính ổn định và xu hướng tăng trưởng của doanh thu và lợi nhuận từ dữ liệu BCTC.
- **Hiệu quả Sử dụng Vốn:** Phân tích các chỉ số ROE, ROA, ROIC để đánh giá năng lực sử dụng vốn.
- **Tình hình Tài chính:** Đánh giá cơ cấu nợ, khả năng thanh khoản và chất lượng dòng tiền tự do (FCF).
- **Ban lãnh đạo & Nội bộ:** Dựa trên thông tin công ty và tin tức, đánh giá chất lượng ban lãnh đạo và hoạt động nội bộ.
- **Chia cổ tức:** Nhận xét về lịch sử và xu hướng chia cổ tức.
- **Tin tức & Internet:** Tổng hợp những tin tức quan trọng gần đây ảnh hưởng đến mã và tìm kiếm thông tin từ internet (nếu có) để bổ sung góc nhìn.
**4. Định giá & So sánh ngành:**
- **Chỉ số Định giá:** Phân tích các chỉ số P/E, P/B, P/S, EV/EBITDA... ở hiện tại và so sánh với lịch sử.
- **So sánh Ngành:** So sánh các chỉ số định giá và tăng trưởng của mã với trung bình ngành và các đối thủ cạnh tranh chính.
**5. Nhận định vị thế mua ngắn hạn:**
- **Khả năng bật tăng ngắn hạn:** Dựa trên phân tích kỹ thuật (RSI, MACD, MA, Volume, Ichimoku, Bollinger, Sức mạnh Giá, RS...) và tin tức gần đây, đánh giá khả năng tăng giá trong ngắn hạn (1-4 tuần) là cao, trung bình hay thấp.
- **Các tín hiệu mua/bán gần đây:** Liệt kê và phân tích các tín hiệu mua/bán kỹ thuật gần đây (nếu có).
- **Tâm lý thị trường ngắn hạn:** Nhận định tâm lý chung của NĐT với mã này trong ngắn hạn (lạc quan, bi quan, thận trọng).
**6. Chiến lược giao dịch & Quản lý rủi ro:**
- **Điểm vào:** Đề xuất các điểm vào lệnh tiềm năng dựa trên phân tích kỹ thuật và cơ bản.
- **Stop-loss & Take-profit:** Đề xuất mức dừng lỗ và chốt lời hợp lý cho từng kịch bản.
- **Risk/Reward:** Ước lượng tỷ lệ lợi nhuận trên rủi ro cho các phương án đề xuất.
**7. Dự báo xu hướng:**
- **Ngắn hạn (1-2 tuần):** Dự báo ngắn hạn dựa trên phân tích kỹ thuật.
- **Trung hạn (1-3 tháng):** Dự báo trung hạn kết hợp kỹ thuật và cơ bản.
- **Dài hạn (3-12 tháng):** Dự báo dài hạn dựa trên triển vọng ngành và phân tích cơ bản.
**8. Kết luận & Khuyến nghị cuối cùng:**
Dựa trên toàn bộ phân tích ở trên, hãy đưa ra khuyến nghị cuối cùng cho mã {symbol}. Bạn **BẮT BUỘC** phải chọn **MỘT** trong 5 khuyến nghị sau và giải thích rõ lý do:
- **MUA MẠNH:** Khi có tín hiệu kỹ thuật và cơ bản rất tích cực, điểm vào tốt, rủi ro thấp, tiềm năng tăng giá mạnh trong ngắn hạn. (Ví dụ: Vượt breakout khỏi vùng tích lũy, volume bùng nổ, RS tăng mạnh, fundamentals tốt).
- **MUA:** Khi có tín hiệu kỹ thuật và cơ bản tích cực, điểm vào hợp lý, rủi ro chấp nhận được, tiềm năng tăng giá tốt. (Ví dụ: Đáy tăng, MA hỗ trợ, RSI phục hồi, fundamentals ổn định).
- **GIỮ:** Khi xu hướng đi ngang hoặc đang chờ xác nhận tín hiệu tiếp theo, không có điểm vào rõ ràng hoặc rủi ro/ng reward không hấp dẫn. (Ví dụ: Trong đám mây Ichimoku, volume yếu, RS trung lập).
- **BÁN:** Khi có tín hiệu kỹ thuật và cơ bản tiêu cực, điểm vào rủi ro cao, hoặc đang ở vùng kháng cự mạnh. (Ví dụ: Vỡ đáy, cắt xuống MA, volume lớn khi giảm, RS yếu).
- **BÁN MẠNH:** Khi có tín hiệu kỹ thuật và cơ bản rất tiêu cực, điểm vào rủi ro rất cao, hoặc đang trong giai đoạn phân phối rõ ràng. (Ví dụ: Vỡ đáy quan trọng, volume selling climax, RS giảm mạnh, fundamentals xấu đi).
**Yêu cầu cụ thể cho phần này:**
- **Khuyến nghị MUA/MUA MẠNH/GIỮ/BÁN/BÁN MẠNH:** Chọn một trong năm và giải thích rõ lý do chính dựa trên phân tích đã trình bày.
- **Điểm số đánh giá (1-10):** Đánh giá mã trên thang điểm 10 (1: Rất xấu, 10: Rất tốt).
- **Tóm tắt ngắn gọn:** Tóm tắt lý do chính cho khuyến nghị trong 2-3 câu.
- **Rủi ro chính:** Liệt kê những rủi ro lớn nhất cần lưu ý đối với mã này.
**Yêu cầu về định dạng:**
- Trình bày rõ ràng, logic theo từng phần như trên.
- Luôn đưa ra dẫn chứng cụ thể từ dữ liệu đã cung cấp (giá, chỉ báo, BCTC, tin tức...).
- Kết hợp cả phân tích định lượng (số liệu) và định tính (giải thích, nhận định).
- Ưu tiên chất lượng, độ sâu và tính chính xác của phân tích hơn là liệt kê dài dòng.
"""
    return prompt.upper()
# --- Phân tích một mã cổ phiếu ---
def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán."""
    print(f"\n{'=' * 60}")
    print(f"PHÂN TÍCH TOÀN DIỆN MÃ {symbol}")
    print(f"{'=' * 60}")
    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"❌ Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    financial_data_statement = get_financial_data(symbol)
    company_info_data = get_company_info(symbol) # Luôn trả về chuỗi
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"❌ Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    if len(df_processed) < 100:
        print(f"❌ Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_processed)} điểm)")
        return None
    print(f"📈 Đang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_processed)
    # --- Chuẩn bị dữ liệu cho Prompt ---
    csv_file_path = f"vnstocks_data/{symbol}_data.csv"
    infor_csv_file_path = f"vnstocks_data/{symbol}_infor.csv"
    market_file_path = f"market_filtered_pe.csv"
    historical_data_str = "Không có dữ liệu lịch sử."
    infor_data_str = "Không có dữ liệu thông tin công ty."
    market_data_str = "Không có dữ liệu thông tin thị trường."
    if os.path.exists(csv_file_path):
        try:
            df_history = pd.read_csv(csv_file_path).tail(2000)
            historical_data_str = df_history.to_string(index=False, float_format="{:.2f}".format)
            print(f"✅ Đã đọc dữ liệu lịch sử từ '{csv_file_path}'")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể đọc file '{csv_file_path}': {e}")
    if os.path.exists(infor_csv_file_path):
        try:
            df_infor = pd.read_csv(infor_csv_file_path)
            infor_data_str = df_infor.to_string(index=False, float_format="{:.2f}".format)
            print(f"✅ Đã đọc dữ liệu thông tin từ '{infor_csv_file_path}'")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể đọc file '{infor_csv_file_path}': {e}")
    if os.path.exists(market_file_path):
        try:
            df_market = pd.read_csv(market_file_path)
            market_data_str = df_market.to_string(index=False, float_format="{:.2f}".format)
            print(f"✅ Đã đọc dữ liệu thông tin từ '{market_file_path}'")
        except Exception as e:
            print(f"⚠️ Cảnh báo: Không thể đọc file '{market_file_path}': {e}")
    technical_indicators = {
        "rsi": trading_signal.get("rsi_value"),
        "ma": {
            "ma10": trading_signal.get("ma10"), "ma20": trading_signal.get("ma20"),
            "ma50": trading_signal.get("ma50"), "ma200": trading_signal.get("ma200"),
        },
        "bollinger_bands": {
            "upper": trading_signal.get("bb_upper"), "lower": trading_signal.get("bb_lower"),
        },
        "macd": {
            "macd": trading_signal.get("macd"), "signal": trading_signal.get("macd_signal"),
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
            "ma20": trading_signal.get("volume_ma_20"), "ma50": trading_signal.get("volume_ma_50"),
        },
    }
    # --- Tạo và lưu Prompt ---
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
    print(f"✅ Đã lưu nội dung prompt vào file prompt.txt")
    # --- Phân tích AI ---
    print(f"🤖 Đang phân tích bằng Gemini ...")
    gemini_analysis = analyze_with_gemini(symbol) # Đã đọc prompt từ file
    print(f"🤖 Đang phân tích bằng OpenRouter ...")
    openrouter_analysis = analyze_with_openrouter(symbol) # Đã đọc prompt từ file
    # --- Hiển thị kết quả ---
    print(f"\n{'=' * 20} KẾT QUẢ PHÂN TÍCH CHO Mã {symbol} {'=' * 20}")
    print(f"💰 Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"📈 Tín hiệu: {trading_signal['signal']}")
    print(f"🎯 Đề xuất: {trading_signal['recommendation']}")
    print(f"📊 Điểm phân tích: {trading_signal['score']:.2f}/100")
    if symbol.upper() != "VNINDEX":
        print(f"📊 RS (so với VNINDEX): {trading_signal['rs']:.4f}")
        print(f"📊 RS_Point: {trading_signal['rs_point']:.2f}")
    print(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ GEMINI ---")
    print(gemini_analysis)
    print(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ OPENROUTER ---")
    print(openrouter_analysis)
    print(f"{'=' * 60}\n")
    # --- Tạo báo cáo ---
    report = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": safe_float(trading_signal.get("current_price")),
        "signal": trading_signal.get("signal"), "recommendation": trading_signal.get("recommendation"),
        "score": safe_float(trading_signal.get("score")), "rsi_value": safe_float(trading_signal.get("rsi_value")),
        "ma10": safe_float(trading_signal.get("ma10")), "ma20": safe_float(trading_signal.get("ma20")),
        "ma50": safe_float(trading_signal.get("ma50")), "ma200": safe_float(trading_signal.get("ma200")),
        "rs": safe_float(trading_signal.get("rs")) if symbol.upper() != "VNINDEX" else None,
        "rs_point": safe_float(trading_signal.get("rs_point")) if symbol.upper() != "VNINDEX" else None,
        "open": safe_float(trading_signal.get("open")), "high": safe_float(trading_signal.get("high")),
        "low": safe_float(trading_signal.get("low")), "volume": safe_float(trading_signal.get("volume")),
        "macd": safe_float(trading_signal.get("macd")), "macd_signal": safe_float(trading_signal.get("macd_signal")),
        "macd_hist": safe_float(trading_signal.get("macd_hist")), "bb_upper": safe_float(trading_signal.get("bb_upper")),
        "bb_lower": safe_float(trading_signal.get("bb_lower")), "volume_ma_20": safe_float(trading_signal.get("volume_ma_20")),
        "volume_ma_50": safe_float(trading_signal.get("volume_ma_50")),
        "ichimoku_tenkan_sen": safe_float(trading_signal.get("ichimoku_tenkan_sen")),
        "ichimoku_kijun_sen": safe_float(trading_signal.get("ichimoku_kijun_sen")),
        "ichimoku_senkou_span_a": safe_float(trading_signal.get("ichimoku_senkou_span_a")),
        "ichimoku_senkou_span_b": safe_float(trading_signal.get("ichimoku_senkou_span_b")),
        "ichimoku_chikou_span": safe_float(trading_signal.get("ichimoku_chikou_span")),
        "rs_sma_10": safe_float(trading_signal.get("rs_sma_10")) if symbol.upper() != "VNINDEX" else None,
        "rs_sma_20": safe_float(trading_signal.get("rs_sma_20")) if symbol.upper() != "VNINDEX" else None,
        "rs_sma_50": safe_float(trading_signal.get("rs_sma_50")) if symbol.upper() != "VNINDEX" else None,
        "rs_sma_200": safe_float(trading_signal.get("rs_sma_200")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_sma_10": safe_float(trading_signal.get("rs_point_sma_10")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_sma_20": safe_float(trading_signal.get("rs_point_sma_20")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_sma_50": safe_float(trading_signal.get("rs_point_sma_50")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_sma_200": safe_float(trading_signal.get("rs_point_sma_200")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_3d": safe_float(trading_signal.get("relative_strength_3d")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_1m": safe_float(trading_signal.get("relative_strength_1m")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_3m": safe_float(trading_signal.get("relative_strength_3m")) if symbol.upper() != "VNINDEX" else None,
        "relative_strength_1y": safe_float(trading_signal.get("relative_strength_1y")) if symbol.upper() != "VNINDEX" else None,
        "gemini_analysis": gemini_analysis, "openrouter_analysis": openrouter_analysis
    }
    report_path = f"vnstocks_data/{symbol}_report.json"
    with open(report_path, "w", encoding="utf-8-sig") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file '{report_path}'")
    return report
# --- Lọc cổ phiếu ---
def filter_stocks_low_pe_high_cap(min_market_cap=500):
    """Lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao."""
    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if df is None or df.empty:
            print("❌ Không thể lấy dữ liệu danh sách công ty niêm yết.")
            return None
        condition1 = df["market_cap"] >= min_market_cap
        condition2_pe = (df["pe"] > 0) & (df["pe"] < 20)
        condition3_pb = df["pb"] > 0
        condition4_rev_growth_last = df["last_quarter_revenue_growth"] > 0
        condition5_rev_growth_second = df["second_quarter_revenue_growth"] > 0
        condition6_profit_growth_last = df["last_quarter_profit_growth"] > 0
        condition7_profit_growth_second = df["second_quarter_profit_growth"] > 0
        condition8_peg_forward = ((df["peg_forward"] < 1) & (df["peg_forward"] >= 0)) | pd.isna(df["peg_forward"])
        condition9_peg_trailing = ((df["peg_trailing"] < 1) & (df["peg_trailing"] >= 0)) | pd.isna(df["peg_trailing"])
        filtered_conditions = condition1 & condition2_pe & condition3_pb & condition4_rev_growth_last & \
                              condition5_rev_growth_second & condition6_profit_growth_last & \
                              condition7_profit_growth_second & condition8_peg_forward & condition9_peg_trailing
        filtered_df = df[filtered_conditions]
        if filtered_df.empty:
            print("⚠️ Không tìm thấy cổ phiếu nào đáp ứng tất cả các tiêu chí lọc.")
            return None
        output_csv_file = "market_filtered.csv"
        output_csv_file_pe = "market_filtered_pe.csv"
        filtered_df.to_csv(output_csv_file_pe, index=False, encoding="utf-8-sig")
        df[condition1].to_csv(output_csv_file, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu danh sách cổ phiếu được lọc ({len(filtered_df)} mã) vào '{output_csv_file_pe}'")
        return filtered_df
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình lọc cổ phiếu: {e}")
        return None
# --- Hàm chính ---
def main():
    """Hàm chính để chạy chương trình."""
    print("=" * 60)
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print("TÍCH HỢP VNSTOCK & AI")
    print("=" * 60)
    print(f"🔍 Đang lọc cổ phiếu có P/E thấp")
    filter_stocks_low_pe_high_cap()
    print("\nNhập mã cổ phiếu để phân tích riêng lẻ (ví dụ: VCB, FPT) hoặc 'exit' để thoát")
    user_input = input("Nhập mã cổ phiếu để phân tích: ").strip().upper()
    if user_input and user_input.lower() != "exit":
        tickers = [ticker.strip() for ticker in user_input.split(",")]
        for ticker in tickers:
            if ticker:
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
        print("\n✅ Hoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")
    else:
        print("👋 Thoát chương trình.")
if __name__ == "__main__":
    main()