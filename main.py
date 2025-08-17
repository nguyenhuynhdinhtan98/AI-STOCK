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
import ta
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
from vnstock import *
import traceback
from vnstock.explorer.vci import Quote, Finance
import matplotlib.dates as mdates
import mplfinance as mpf
warnings.filterwarnings("ignore")

# --- Cấu hình toàn cục cho phân tích dữ liệu ---
# Thời gian lấy dữ liệu (ĐÃ THAY ĐỔI THÀNH 10 NĂM)
GLOBAL_START_DATE = (datetime.today() - timedelta(days=365 * 10)).strftime("%Y-%m-%d") # Lấy dữ liệu 10 năm gần nhất
GLOBAL_END_DATE = datetime.today().strftime("%Y-%m-%d")

# --- Cấu hình API và thư mục lưu trữ ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Khóa API cho Google Gemini
if not GOOGLE_API_KEY:
    raise ValueError("Vui lòng đặt GOOGLE_API_KEY trong file .env")
genai.configure(api_key=GOOGLE_API_KEY)
os.makedirs("vnstocks_data", exist_ok=True) # Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại

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

# --- Hàm lấy dữ liệu ---
def get_stock_data(symbol):
    """Lấy dữ liệu lịch sử giá cổ phiếu từ VCI và lưu vào file CSV."""
    try:
        quote = Quote(symbol=symbol)
        df = quote.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if df is not None and not df.empty:
            df.rename(columns={
                "time": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume"
            }, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(f"vnstocks_data/{symbol}_data.csv")
            print(f"✅ Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'")
            return df
        else:
            print(f"⚠️ Không lấy được dữ liệu cho mã {symbol}")
            return None
    except Exception as e:
        print(f"❌ Exception khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None

def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính (12 quý gần nhất) từ VCI và lưu vào file CSV."""
    try:
        financial_obj = Finance(symbol=symbol)
        financial_data = financial_obj.ratio(period="quarter", lang="en", flatten_columns=True).head(13)
        if financial_data is not None and not financial_data.empty:
            financial_data.to_csv(f"vnstocks_data/{symbol}_financial.csv", index=False)
            print(f"✅ Đã lưu BCTC cho mã {symbol} vào file 'vnstocks_data/{symbol}_financial.csv'")
            return financial_data
        else:
            print(f"⚠️ Không lấy được BCTC cho mã {symbol}")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None

def get_market_data():
    """Lấy dữ liệu lịch sử của VNINDEX từ VCI và lưu vào file CSV."""
    try:
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if vnindex is not None and not vnindex.empty:
            vnindex.rename(columns={
                "time": "Date", "open": "Open", "high": "High",
                "low": "Low", "close": "Close", "volume": "Volume"
            }, inplace=True)
            vnindex["Date"] = pd.to_datetime(vnindex["Date"])
            vnindex.set_index("Date", inplace=True)
            vnindex.sort_index(inplace=True)
            vnindex.to_csv("vnstocks_data/VNINDEX_data.csv")
            print(f"✅ Đã lưu dữ liệu VNINDEX vào file 'vnstocks_data/VNINDEX_data.csv'")
            return vnindex
        else:
            print("⚠️ Không lấy được dữ liệu VNINDEX")
            return None
    except Exception as e:
        print(f"❌ Lỗi khi lấy dữ liệu thị trường (VNINDEX): {str(e)}")
        return None

# --- Tiền xử lý dữ liệu ---
def preprocess_stock_data(df):
    """Tiền xử lý dữ liệu giá cổ phiếu cơ bản (sắp xếp, xử lý NaN, tính returns, MA)."""
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["returns"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["volatility"] = df["returns"].rolling(window=10).std()
    return df

def create_features(df):
    """Tạo các chỉ báo kỹ thuật sử dụng thư viện 'ta'."""
    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
    df["EMA_12"] = ta.trend.ema_indicator(df["Close"], window=12)
    df["EMA_26"] = ta.trend.ema_indicator(df["Close"], window=26)
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = ta.trend.ema_indicator(df["MACD"], window=9)
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    df["BB_Upper"], df["BB_Middle"], df["BB_Lower"] = ta.volatility.bollinger_hband_indicator(df["Close"]), ta.volatility.bollinger_mavg(df["Close"]), ta.volatility.bollinger_lband_indicator(df["Close"])
    df["Volume_MA"] = df["Volume"].rolling(window=20).mean()
    return df

# --- Tính toán Relative Strength ---
def calculate_relative_strength(df_stock, df_index):
    """Tính Relative Strength (RS) và các chỉ báo RS Point theo công thức tiêu chuẩn."""
    df_merged = df_stock[["Close"]].join(df_index[["Close"]].rename(columns={"Close": "Index_Close"}), how="inner")
    if df_merged.empty or df_merged["Index_Close"].isna().all():
        print("⚠️ Cảnh báo: Không có dữ liệu chỉ số thị trường để tính RS. Gán giá trị mặc định.")
        # Gán giá trị mặc định cho tất cả các chỉ báo RS
        df_stock["RS"] = 1.0
        df_stock["RS_Point"] = 0.0
        df_stock["RS_Point_252"] = 0.0
        df_stock["RS_SMA_10"] = 1.0
        df_stock["RS_SMA_20"] = 1.0
        df_stock["RS_SMA_50"] = 1.0
        df_stock["RS_SMA_200"] = 1.0
        df_stock["RS_Point_SMA_10"] = 0.0
        df_stock["RS_Point_SMA_20"] = 0.0
        df_stock["RS_Point_SMA_50"] = 0.0
        df_stock["RS_Point_SMA_200"] = 0.0
        df_stock["RS_Point_252_SMA_10"] = 0.0
        df_stock["RS_Point_252_SMA_20"] = 0.0
        df_stock["RS_Point_252_SMA_50"] = 0.0
        df_stock["RS_Point_252_SMA_200"] = 0.0
        return df_stock
    df_merged["Index_Close"] = df_merged["Index_Close"].ffill().bfill()
    # Tính RS
    df_merged["RS"] = df_merged["Close"] / df_merged["Index_Close"]
    # Tính các thành phần ROC cho RS_Point
    roc_63 = (df_merged["Close"] / df_merged["Close"].shift(63) - 1) * 100
    roc_126 = (df_merged["Close"] / df_merged["Close"].shift(126) - 1) * 100
    roc_189 = (df_merged["Close"] / df_merged["Close"].shift(189) - 1) * 100
    roc_252_for_rs_point = (df_merged["Close"] / df_merged["Close"].shift(252) - 1) * 100
    # Tính RS_Point theo công thức: (ROC(63)*0.4 + ROC(126)*0.2 + ROC(189)*0.2 + ROC(252)*0.2)
    # Vì ROC đã được nhân 100, kết quả không cần nhân thêm.
    df_merged["RS_Point"] = (
        roc_63.fillna(0) * 0.4 +
        roc_126.fillna(0) * 0.2 +
        roc_189.fillna(0) * 0.2 +
        roc_252_for_rs_point.fillna(0) * 0.2
    )
    # Tính RS_Point_252 = ((C / Ref(C, -252)) - 1) * 100
    df_merged["RS_Point_252"] = ((df_merged["Close"] / df_merged["Close"].shift(252)) - 1) * 100
    # Tính các đường trung bình cho RS, RS_Point, RS_Point_252
    df_merged["RS_SMA_10"] = ta.trend.sma_indicator(df_merged["RS"], window=10)
    df_merged["RS_SMA_20"] = ta.trend.sma_indicator(df_merged["RS"], window=20)
    df_merged["RS_SMA_50"] = ta.trend.sma_indicator(df_merged["RS"], window=50)
    df_merged["RS_SMA_200"] = ta.trend.sma_indicator(df_merged["RS"], window=200)
    df_merged["RS_Point_SMA_10"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=10)
    df_merged["RS_Point_SMA_20"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=20)
    df_merged["RS_Point_SMA_50"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=50)
    df_merged["RS_Point_SMA_200"] = ta.trend.sma_indicator(df_merged["RS_Point"], window=200)
    df_merged["RS_Point_252_SMA_10"] = ta.trend.sma_indicator(df_merged["RS_Point_252"], window=10)
    df_merged["RS_Point_252_SMA_20"] = ta.trend.sma_indicator(df_merged["RS_Point_252"], window=20)
    df_merged["RS_Point_252_SMA_50"] = ta.trend.sma_indicator(df_merged["RS_Point_252"], window=50)
    df_merged["RS_Point_252_SMA_200"] = ta.trend.sma_indicator(df_merged["RS_Point_252"], window=200)
    # Gán các chỉ báo trở lại dataframe gốc
    cols_to_join = [
        "RS", "RS_Point", "RS_Point_252",
        "RS_SMA_10", "RS_SMA_20", "RS_SMA_50", "RS_SMA_200",
        "RS_Point_SMA_10", "RS_Point_SMA_20", "RS_Point_SMA_50", "RS_Point_SMA_200",
        "RS_Point_252_SMA_10", "RS_Point_252_SMA_20", "RS_Point_252_SMA_50", "RS_Point_252_SMA_200"
    ]
    df_stock = df_stock.join(df_merged[cols_to_join], how="left")
    # Xử lý giá trị NaN
    df_stock["RS"].fillna(1.0, inplace=True)
    df_stock["RS_Point"].fillna(0.0, inplace=True)
    df_stock["RS_Point_252"].fillna(0.0, inplace=True)
    df_stock["RS_SMA_10"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_20"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_50"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_200"].fillna(1.0, inplace=True)
    df_stock["RS_Point_SMA_10"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_20"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_50"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_200"].fillna(0.0, inplace=True)
    df_stock["RS_Point_252_SMA_10"].fillna(0.0, inplace=True)
    df_stock["RS_Point_252_SMA_20"].fillna(0.0, inplace=True)
    df_stock["RS_Point_252_SMA_50"].fillna(0.0, inplace=True)
    df_stock["RS_Point_252_SMA_200"].fillna(0.0, inplace=True)
    return df_stock

# --- Phân tích kỹ thuật và vẽ biểu đồ ---
def plot_stock_analysis(symbol, df, show_volume=True):
    """Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán."""
    try:
        if df is None or len(df) == 0:
            print("❌ Dữ liệu phân tích rỗng")
            return {
                "signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0,
                "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0,
                "rs_point_252": 0.0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "open": None, "high": None, "low": None, "volume": None,
                "macd": None, "macd_signal": None, "macd_hist": None,
                "bb_upper": None, "bb_lower": None,
                "volume_ma": None,
                "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, "ichimoku_chikou_span": None,
                "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
                "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
                "rs_point_252_sma_10": None, "rs_point_252_sma_20": None,
                "rs_point_252_sma_50": None, "rs_point_252_sma_200": None,
                "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
            }
        df = df.sort_index()
        df = create_features(df)
        # Tính RS (Relative Strength so với VNINDEX)
        if symbol.upper() != "VNINDEX":
            try:
                quoteVNI = Quote(symbol="VNINDEX")
                vnindex_df = quoteVNI.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
                if len(vnindex_df) == 0:
                    raise ValueError("Không lấy được dữ liệu VNINDEX")
                vnindex_df.rename(columns={
                    "time": "Date", "open": "Open", "high": "High",
                    "low": "Low", "close": "Close", "volume": "Volume"
                }, inplace=True)
                vnindex_df["Date"] = pd.to_datetime(vnindex_df["Date"])
                vnindex_df.set_index("Date", inplace=True)
                vnindex_df.sort_index(inplace=True)
                df = calculate_relative_strength(df, vnindex_df)
            except Exception as e:
                print(f"⚠️ Cảnh báo: Không thể tính RS cho {symbol} do lỗi: {e}")
                # Gán giá trị mặc định nếu lỗi tính RS
                df["RS"] = 1.0
                df["RS_Point"] = 0.0
                df["RS_Point_252"] = 0.0
                df["RS_SMA_10"] = 1.0
                df["RS_SMA_20"] = 1.0
                df["RS_SMA_50"] = 1.0
                df["RS_SMA_200"] = 1.0
                df["RS_Point_SMA_10"] = 0.0
                df["RS_Point_SMA_20"] = 0.0
                df["RS_Point_SMA_50"] = 0.0
                df["RS_Point_SMA_200"] = 0.0
                df["RS_Point_252_SMA_10"] = 0.0
                df["RS_Point_252_SMA_20"] = 0.0
                df["RS_Point_252_SMA_50"] = 0.0
                df["RS_Point_252_SMA_200"] = 0.0
        else:
            # Nếu là VNINDEX, gán các giá trị mặc định
            df["RS"] = 1.0
            df["RS_Point"] = 0.0
            df["RS_Point_252"] = 0.0
            df["RS_SMA_10"] = 1.0
            df["RS_SMA_20"] = 1.0
            df["RS_SMA_50"] = 1.0
            df["RS_SMA_200"] = 1.0
            df["RS_Point_SMA_10"] = 0.0
            df["RS_Point_SMA_20"] = 0.0
            df["RS_Point_SMA_50"] = 0.0
            df["RS_Point_SMA_200"] = 0.0
            df["RS_Point_252_SMA_10"] = 0.0
            df["RS_Point_252_SMA_20"] = 0.0
            df["RS_Point_252_SMA_50"] = 0.0
            df["RS_Point_252_SMA_200"] = 0.0
        df = df.dropna(subset=["Close", "SMA_10", "SMA_20", "SMA_50"], how="all")
        if len(df) < 20:
            print("❌ Không đủ dữ liệu hợp lệ để phân tích")
            return {
                "signal": "LỖI", "score": 50,
                "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rsi_value": 50,
                "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rs": 1.0, "rs_point": 0, "rs_point_252": 0.0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
                "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
                "rs_point_252_sma_10": None, "rs_point_252_sma_20": None,
                "rs_point_252_sma_50": None, "rs_point_252_sma_200": None,
                "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
            }
        # Vẽ biểu đồ phân tích kỹ thuật

        try:
            plot_configs = ["price_sma", "ichimoku", "rsi", "macd", "rs", "rs_point", "rs_point_252", "volume"]
            num_subplots = len(plot_configs)
            height_per_subplot = 3
            width = 18
            height = num_subplots * height_per_subplot
            plt.figure(figsize=(width, height), constrained_layout=True)

            grid = plt.GridSpec(8, 1, hspace=0.3, height_ratios=[3, 3, 2, 2, 2, 2, 2, 2])

            # === Biểu đồ 1: Giá và các đường trung bình ===
            ax1 = plt.subplot(grid[0])
            plt.plot(df.index, df["Close"], label=f"Giá đóng cửa {df['Close'].iloc[-1]:,.2f}", color="#1f77b4",
                     linewidth=1.5)
            plt.plot(df.index, df["SMA_10"], label=f"SMA 10 {df['SMA_10'].iloc[-1]:,.2f}", color="blue", alpha=0.7,
                     linewidth=1)
            plt.plot(df.index, df["SMA_20"], label=f"SMA 20 {df['SMA_20'].iloc[-1]:,.2f}", color="orange", alpha=0.8,
                     linewidth=1.5)
            plt.plot(df.index, df["SMA_50"], label=f"SMA 50 {df['SMA_50'].iloc[-1]:,.2f}", color="green", alpha=0.8,
                     linewidth=1.5)
            plt.plot(df.index, df["SMA_200"], label=f"SMA 200 {df['SMA_200'].iloc[-1]:,.2f}", color="purple", alpha=0.8,
                     linewidth=1.5)
            plt.plot(df.index, df["BB_upper"], label=f"BB Upper {df['BB_upper'].iloc[-1]:,.2f}", color="red", alpha=0.5,
                     linestyle="--")
            plt.plot(df.index, df["BB_lower"], label=f"BB Lower {df['BB_lower'].iloc[-1]:,.2f}", color="green", alpha=0.5,
                     linestyle="--")
            plt.fill_between(df.index, df["BB_lower"], df["BB_upper"], color="gray", alpha=0.1)

            cross_10_20_above = (df["SMA_10"] > df["SMA_20"]) & (df["SMA_10"].shift(1) <= df["SMA_20"].shift(1))
            cross_10_20_below = (df["SMA_10"] < df["SMA_20"]) & (df["SMA_10"].shift(1) >= df["SMA_20"].shift(1))
            if cross_10_20_above.any():
                plt.scatter(df.index[cross_10_20_above], df.loc[cross_10_20_above, "SMA_10"], marker="^", color="lime",
                            s=60, label="SMA10 > SMA20", zorder=5)
            if cross_10_20_below.any():
                plt.scatter(df.index[cross_10_20_below], df.loc[cross_10_20_below, "SMA_10"], marker="v", color="magenta",
                            s=60, label="SMA10 < SMA20", zorder=5)

            plt.suptitle(f"Phân tích kỹ thuật {symbol} - Giá và Chỉ báo", fontsize=16, fontweight="bold", y=0.98)
            plt.ylabel("Giá (VND)", fontsize=12)
            plt.legend(loc="upper left", fontsize=10)
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 2: Ichimoku Cloud ===
            ax2 = plt.subplot(grid[1], sharex=ax1)
            for i in range(len(df)):
                if i < len(df) - 1:
                    date = mdates.date2num(df.index[i])
                    open_price = (df["Open"].iloc[i] if not pd.isna(df["Open"].iloc[i]) else df["Close"].iloc[i])
                    high_price = (df["High"].iloc[i] if not pd.isna(df["High"].iloc[i]) else df["Close"].iloc[i])
                    low_price = (df["Low"].iloc[i] if not pd.isna(df["Low"].iloc[i]) else df["Close"].iloc[i])
                    close_price = (df["Close"].iloc[i] if not pd.isna(df["Close"].iloc[i]) else open_price)
                    if close_price >= open_price:
                        color = "green"
                        bottom = open_price
                        height = close_price - open_price
                    else:
                        color = "red"
                        bottom = close_price
                        height = open_price - close_price
                    if height > 0:
                        plt.bar(date, height, bottom=bottom, color=color, width=0.6, alpha=0.8)
                    plt.plot([date, date], [low_price, high_price], color="black", linewidth=0.5)

            plt.plot(df.index, df["ichimoku_tenkan_sen"], label=f"Tenkan-sen {df['ichimoku_tenkan_sen'].iloc[-1]:,.2f}",
                     color="red", linewidth=1)
            plt.plot(df.index, df["ichimoku_kijun_sen"], label=f"Kijun-sen {df['ichimoku_kijun_sen'].iloc[-1]:,.2f}",
                     color="blue", linewidth=1)
            plt.plot(df.index, df["ichimoku_senkou_span_a"],
                     label=f"Senkou Span A {df['ichimoku_senkou_span_a'].iloc[-1]:,.2f}", color="green", linewidth=1,
                     alpha=0.7)
            plt.plot(df.index, df["ichimoku_senkou_span_b"],
                     label=f"Senkou Span B {df['ichimoku_senkou_span_b'].iloc[-1]:,.2f}", color="purple", linewidth=1,
                     alpha=0.7)
            plt.plot(df.index, df["ichimoku_chikou_span"], label=f"Chikou Span {df['ichimoku_chikou_span'].iloc[-1]:,.2f}",
                     color="orange", linewidth=1)

            valid_cloud = (df["ichimoku_senkou_span_a"].notna() & df["ichimoku_senkou_span_b"].notna())
            if valid_cloud.any():
                plt.fill_between(df.index[valid_cloud], df["ichimoku_senkou_span_a"][valid_cloud],
                                 df["ichimoku_senkou_span_b"][valid_cloud],
                                 where=(df["ichimoku_senkou_span_a"][valid_cloud] >= df["ichimoku_senkou_span_b"][valid_cloud]),
                                 color="green", alpha=0.2, interpolate=True, label="Bullish Cloud")
                plt.fill_between(df.index[valid_cloud], df["ichimoku_senkou_span_a"][valid_cloud],
                                 df["ichimoku_senkou_span_b"][valid_cloud],
                                 where=(df["ichimoku_senkou_span_a"][valid_cloud] < df["ichimoku_senkou_span_b"][valid_cloud]),
                                 color="red", alpha=0.2, interpolate=True, label="Bearish Cloud")

            plt.plot(df.index, df["Close"], label="Giá đóng cửa", color="black", linewidth=1.5, alpha=0.7)
            plt.title("Ichimoku Cloud", fontsize=12)
            plt.ylabel("Giá", fontsize=10)
            plt.legend(fontsize=7, loc="upper left", ncol=2)
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 3: RSI ===
            ax3 = plt.subplot(grid[2], sharex=ax1)
            plt.plot(df.index, df["RSI"], label=f"RSI {df['RSI'].iloc[-1]:.2f}", color="purple")
            plt.axhline(70, linestyle="--", color="red", alpha=0.7)
            plt.axhline(30, linestyle="--", color="green", alpha=0.7)
            plt.fill_between(df.index, 30, 70, color="gray", alpha=0.1)
            plt.title("RSI", fontsize=12)
            plt.ylabel("RSI", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc="upper left")

            # === Biểu đồ 4: MACD ===
            ax4 = plt.subplot(grid[3], sharex=ax1)
            plt.plot(df.index, df["MACD"], label=f"MACD {df['MACD'].iloc[-1]:.2f}", color="blue")
            plt.plot(df.index, df["MACD_signal"], label=f"Signal Line {df['MACD_signal'].iloc[-1]:.2f}", color="red")
            plt.bar(df.index, df["MACD_Hist"], color=np.where(df["MACD_Hist"] > 0, "green", "red"), alpha=0.5,
                    label=f"Hist {df['MACD_Hist'].iloc[-1]:.2f}")
            plt.title("MACD", fontsize=12)
            plt.ylabel("MACD", fontsize=10)
            plt.legend(fontsize=7, loc="upper left")
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 5: RS (Relative Strength vs VNINDEX) ===
            ax5 = plt.subplot(grid[4], sharex=ax1)
            plt.plot(df.index, df["RS"], label=f"RS (Price / VNINDEX) {df['RS'].iloc[-1]:.2f}", color="brown",
                     linewidth=1)
            plt.plot(df.index, df["RS_SMA_10"], label=f"RS SMA 10 {df['RS_SMA_10'].iloc[-1]:.2f}", color="blue", alpha=0.7,
                     linewidth=1)
            plt.plot(df.index, df["RS_SMA_20"], label=f"RS SMA 20 {df['RS_SMA_20'].iloc[-1]:.2f}", color="orange", alpha=0.7,
                     linewidth=1)
            plt.plot(df.index, df["RS_SMA_50"], label=f"RS SMA 50 {df['RS_SMA_50'].iloc[-1]:.2f}", color="green", alpha=0.7,
                     linewidth=1)
            plt.plot(df.index, df["RS_SMA_200"], label=f"RS SMA 200 {df['RS_SMA_200'].iloc[-1]:.2f}", color="purple", alpha=0.7,
                     linewidth=1)
            plt.title("RS vs VNINDEX", fontsize=12)
            plt.ylabel("RS", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc="upper left")

            # === Biểu đồ 6: RS_Point ===
            ax6 = plt.subplot(grid[5], sharex=ax1)
            plt.plot(df.index, df["RS_Point"], label=f"RS_Point {df['RS_Point'].iloc[-1]:.2f}", color="darkblue",
                     linewidth=1.5)
            plt.plot(df.index, df["RS_Point_SMA_10"], label=f"RS_Point SMA 10 {df['RS_Point_SMA_10'].iloc[-1]:.2f}",
                     color="blue", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_20"], label=f"RS_Point SMA 20 {df['RS_Point_SMA_20'].iloc[-1]:.2f}",
                     color="orange", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_50"], label=f"RS_Point SMA 50 {df['RS_Point_SMA_50'].iloc[-1]:.2f}",
                     color="green", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_200"], label=f"RS_Point SMA 200 {df['RS_Point_SMA_200'].iloc[-1]:.2f}",
                     color="purple", alpha=0.7, linewidth=1)
            plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
            plt.fill_between(df.index, df["RS_Point"], 0, where=(df["RS_Point"] > 0), color="green", alpha=0.3)
            plt.fill_between(df.index, df["RS_Point"], 0, where=(df["RS_Point"] < 0), color="red", alpha=0.3)
            plt.title("RS_Point", fontsize=12)
            plt.ylabel("RS_Point", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc="upper left")

               # === Biểu đồ 7: RS_Point_252 ===
            ax7 = plt.subplot(grid[6], sharex=ax1)
            plt.plot(df.index, df["RS_Point_252"], label=f"RS_Point_252 {df['RS_Point_252'].iloc[-1]:.2f}", color="darkgreen",
                     linewidth=1.5)
            plt.plot(df.index, df["RS_Point_252_SMA_10"], label=f"RS_Point_252 SMA 10 {df['RS_Point_252_SMA_10'].iloc[-1]:.2f}",
                     color="blue", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_252_SMA_20"], label=f"RS_Point_252 SMA 20 {df['RS_Point_252_SMA_20'].iloc[-1]:.2f}",
                     color="orange", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_252_SMA_50"], label=f"RS_Point_252 SMA 50 {df['RS_Point_252_SMA_50'].iloc[-1]:.2f}",
                     color="green", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_252_SMA_200"], label=f"RS_Point_252 SMA 200 {df['RS_Point_252_SMA_200'].iloc[-1]:.2f}",
                     color="purple", alpha=0.7, linewidth=1)
            plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
            plt.fill_between(df.index, df["RS_Point_252"], 0, where=(df["RS_Point_252"] > 0), color="green", alpha=0.2)
            plt.fill_between(df.index, df["RS_Point_252"], 0, where=(df["RS_Point_252"] < 0), color="red", alpha=0.2)
            plt.title("RS_Point_252", fontsize=12)
            plt.ylabel("RS_Point_252", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc="upper left")

            # === Biểu đồ 8: Khối lượng ===
            ax8 = plt.subplot(grid[7], sharex=ax1)
            if show_volume and "Volume" in df.columns:
                volume_sma_plotted = False
                if ("Volume_MA" in df.columns and not df["Volume_MA"].isna().all()):
                    plt.plot(df.index, df["Volume_MA"], label=f"Vol SMA 10 {df['Volume_MA'].iloc[-1]:,.2f}",
                             color="orange", alpha=0.8, linewidth=1.5)
                    volume_sma_plotted = True
                colors = np.where(df["Close"] > df["Open"], "green", "red")
                plt.bar(df.index, df["Volume"], color=colors, alpha=0.7, label="Volume" if not volume_sma_plotted else None)
                handles, labels = ax8.get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), loc="upper left")
                else:
                    plt.legend(fontsize=7, loc="upper left")
                plt.title("Volume & Vol SMA", fontsize=12)
                plt.ylabel("Khối lượng", fontsize=10)
                plt.grid(True, alpha=0.3)
            else:
                plt.title("Khối lượng giao dịch", fontsize=12)
                plt.ylabel("Khối lượng", fontsize=10)
                plt.grid(True, alpha=0.3)

            plt.tight_layout(pad=3.0, h_pad=1.0)
            plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.4)

            plt.savefig(f"vnstocks_data/{symbol}_technical_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Đã lưu biểu đồ phân tích kỹ thuật vào vnstocks_data/{symbol}_technical_analysis.png")

        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")
            traceback.print_exc()

        # Tạo tín hiệu giao dịch
        try:
            last_row = df.iloc[-1]
            current_price = last_row["Close"]
            rsi_value = last_row["RSI"] if not pd.isna(last_row["RSI"]) else 50
            ma10_value = (last_row["SMA_10"] if not pd.isna(last_row["SMA_10"]) else current_price)
            ma20_value = (last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else current_price)
            ma50_value = (last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else current_price)
            ma200_value = (last_row["SMA_200"] if not pd.isna(last_row["SMA_200"]) else current_price)
            rs_value = last_row["RS"] if not pd.isna(last_row["RS"]) else 1.0
            rs_point_value = last_row["RS_Point"] if not pd.isna(last_row["RS_Point"]) else 0.0
            rs_point_252_value = last_row["RS_Point_252"] if not pd.isna(last_row["RS_Point_252"]) else 0.0
            macd_value = last_row["MACD"] if not pd.isna(last_row["MACD"]) else 0
            macd_signal = last_row["MACD_Signal"] if not pd.isna(last_row["MACD_Signal"]) else 0
            macd_hist = last_row["MACD_Hist"] if not pd.isna(last_row["MACD_Hist"]) else 0
            bb_upper = last_row["BB_Upper"] if not pd.isna(last_row["BB_Upper"]) else current_price
            bb_lower = last_row["BB_Lower"] if not pd.isna(last_row["BB_Lower"]) else current_price
            volume_ma = last_row["Volume_MA"] if not pd.isna(last_row["Volume_MA"]) else 0
            tenkan_sen = last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else np.nan
            kijun_sen = last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else np.nan
            senkou_span_a = (tenkan_sen + kijun_sen) / 2 if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen)) else np.nan
            senkou_span_b = df["Close"].rolling(window=52).mean().iloc[-26] if len(df) >= 78 else np.nan
            chikou_span = df["Close"].shift(26).iloc[-1] if len(df) > 26 else np.nan
            # Tính điểm tổng hợp
            score = 50 # Điểm cơ bản
            # 1. RSI - 15 điểm
            rsi_score = 0
            if rsi_value < 30: rsi_score += 15
            elif rsi_value < 50: rsi_score += 7
            elif rsi_value > 70: rsi_score -= 15
            elif rsi_value > 50: rsi_score -= 7
            score += rsi_score
            # 2. MA - 20 điểm
            ma_score = 0
            if current_price > ma10_value: ma_score += 2
            if current_price > ma20_value: ma_score += 3
            if current_price > ma50_value: ma_score += 5
            if current_price > ma200_value: ma_score += 10
            if ma10_value > ma20_value > ma50_value > ma200_value: ma_score += 5
            elif ma10_value < ma20_value < ma50_value < ma200_value: ma_score -= 5
            score += ma_score
            # 3. MACD - 10 điểm
            macd_score = 0
            if macd_value > macd_signal and macd_hist > 0: macd_score += 10
            elif macd_value < macd_signal and macd_hist < 0: macd_score -= 10
            score += macd_score
            # 4. Bollinger Bands - 10 điểm
            bb_score = 0
            if current_price < bb_lower: bb_score += 10
            elif current_price > bb_upper: bb_score -= 10
            score += bb_score
            # 5. Volume - 5 điểm
            volume_score = 0
            if "Volume" in last_row and not pd.isna(last_row["Volume"]) and last_row["Volume"] > volume_ma: volume_score += 5
            score += volume_score
            # 6. RS - 10 điểm
            if symbol.upper() != "VNINDEX":
                rs_score = 0
                if rs_value > last_row.get("RS_SMA_10", rs_value): rs_score += 2
                if rs_value > last_row.get("RS_SMA_20", rs_value): rs_score += 3
                if rs_value > last_row.get("RS_SMA_50", rs_value): rs_score += 5
                score += rs_score
            # 7. RS_Point - 10 điểm
            if symbol.upper() != "VNINDEX":
                rs_point_score = 0
                if rs_point_value > last_row.get("RS_Point_SMA_10", 0): rs_point_score += 2
                if rs_point_value > last_row.get("RS_Point_SMA_20", 0): rs_point_score += 3
                if rs_point_value > last_row.get("RS_Point_SMA_50", 0): rs_point_score += 5
                score += rs_point_score
            # 8. RS_Point_252 - 10 điểm
            if symbol.upper() != "VNINDEX":
                rs_point_252_score = 0
                if rs_point_252_value > last_row.get("RS_Point_252_SMA_10", 0): rs_point_252_score += 2
                if rs_point_252_value > last_row.get("RS_Point_252_SMA_20", 0): rs_point_252_score += 3
                if rs_point_252_value > last_row.get("RS_Point_252_SMA_50", 0): rs_point_252_score += 5
                score += rs_point_252_score
            # 9. Ichimoku Cloud - 15 điểm
            ichimoku_score = 0
            try:
                if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
                    if current_price > max(senkou_span_a, senkou_span_b) and tenkan_sen > kijun_sen: ichimoku_score += 15
                    elif current_price > max(senkou_span_a, senkou_span_b): ichimoku_score += 10
                    elif current_price < min(senkou_span_a, senkou_span_b) and tenkan_sen < kijun_sen: ichimoku_score -= 15
                    elif current_price < min(senkou_span_a, senkou_span_b): ichimoku_score -= 10
                    if tenkan_sen > kijun_sen: ichimoku_score += 5
                    elif tenkan_sen < kijun_sen: ichimoku_score -= 5
                    if kijun_sen > max(senkou_span_a, senkou_span_b): ichimoku_score += 5
                    elif kijun_sen < min(senkou_span_a, senkou_span_b): ichimoku_score -= 5
            except Exception as e:
                print(f"⚠️ Cảnh báo: Lỗi khi tính điểm Ichimoku: {e}")
            score += ichimoku_score
            score = np.clip(score, 0, 100)
            # Xác định tín hiệu và đề xuất
            signal = "TRUNG LẬP"
            recommendation = "GIỮ"
            if score >= 80:
                signal = "MUA MẠNH"
                recommendation = "MUA MẠNH"
            elif score >= 65:
                signal = "MUA"
                recommendation = "MUA"
            elif score <= 20:
                signal = "BÁN MẠNH"
                recommendation = "BÁN MẠNH"
            elif score <= 35:
                signal = "BÁN"
                recommendation = "BÁN"
            # In ra tín hiệu cuối cùng
            analysis_date = df.index[-1].strftime("%d/%m/%Y")
            print(f"\n📊 TÍN HIỆU GIAO DỊCH CUỐI CÙNG CHO {symbol} ({analysis_date}):")
            print(f"  - Giá hiện tại: {current_price:,.2f} VND")
            print(f"  - Đường trung bình:")
            print(f"    * MA10: {ma10_value:,.2f} | MA20: {ma20_value:,.2f} | MA50: {ma50_value:,.2f} | MA200: {ma200_value:,.2f}")
            print(f"  - Chỉ báo dao động:")
            print(f"    * RSI (14): {rsi_value:.2f}")
            print(f"    * MACD: {macd_value:.2f} | Signal: {macd_signal:.2f} | Histogram: {macd_hist:.2f}")
            print(f"    * Bollinger Bands: Trên: {bb_upper:,.2f} | Dưới: {bb_lower:,.2f}")
            if symbol.upper() != "VNINDEX":
                 print(f"  - Sức mạnh tương đối (RS):")
                 print(f"    * RS: {rs_value:.4f}")
                 print(f"    * RS_Point: {rs_point_value:.2f}")
                 print(f"    * RS_Point_252: {rs_point_252_value:.2f}")
            try:
                print(f"  - Mô hình Ichimoku:")
                print(f"    * Tenkan-sen (Chuyển đổi): {tenkan_sen:.2f}")
                print(f"    * Kijun-sen (Cơ sở): {kijun_sen:.2f}")
                print(f"    * Senkou Span A (Leading Span A): {senkou_span_a:.2f}")
                print(f"    * Senkou Span B (Leading Span B): {senkou_span_b:.2f}")
                print(f"    * Chikou Span (Trễ): {chikou_span:.2f}")
                print(f"    * Điểm Ichimoku: ~{ichimoku_score:.2f}")
            except: print(f"  - Ichimoku: Không có đủ dữ liệu.")
            print(f"  - Khối lượng:")
            print(f"    * Khối lượng hiện tại: {last_row.get('Volume', 'N/A')}")
            print(f"    * MA Khối lượng (20): {volume_ma:,.2f}")
            print(f"  🎯 ĐỀ XUẤT CUỐI CÙNG: {recommendation}")
            print(f"  📊 TỔNG ĐIỂM PHÂN TÍCH: {score:.1f}/100")
            print(f"  📈 TÍN HIỆU: {signal}")

            # Trả về kết quả phân tích kỹ thuật (không có dự báo AI)
            return {
                "signal": signal, "score": float(score), "current_price": float(current_price),
                "rsi_value": float(rsi_value),
                "ma10": float(ma10_value), "ma20": float(ma20_value), "ma50": float(ma50_value),
                "ma200": float(ma200_value),
                "rs": float(rs_value), "rs_point": float(rs_point_value),
                "rs_point_252": float(rs_point_252_value),
                "recommendation": recommendation,
                "open": safe_float(last_row.get("Open")), "high": safe_float(last_row.get("High")),
                "low": safe_float(last_row.get("Low")), "volume": safe_float(last_row.get("Volume")),
                "macd": safe_float(macd_value), "macd_signal": safe_float(macd_signal), "macd_hist": safe_float(macd_hist),
                "bb_upper": safe_float(bb_upper), "bb_lower": safe_float(bb_lower),
                "volume_ma": safe_float(last_row.get("Volume_MA")),
                "ichimoku_tenkan_sen": safe_float(tenkan_sen), "ichimoku_kijun_sen": safe_float(kijun_sen),
                "ichimoku_senkou_span_a": safe_float(senkou_span_a),
                "ichimoku_senkou_span_b": safe_float(senkou_span_b),
                "ichimoku_chikou_span": safe_float(chikou_span),
                "rs_sma_10": safe_float(last_row.get("RS_SMA_10")) if symbol.upper() != "VNINDEX" else None,
                "rs_sma_20": safe_float(last_row.get("RS_SMA_20")) if symbol.upper() != "VNINDEX" else None,
                "rs_sma_50": safe_float(last_row.get("RS_SMA_50")) if symbol.upper() != "VNINDEX" else None,
                "rs_sma_200": safe_float(last_row.get("RS_SMA_200")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_252_sma_10": safe_float(last_row.get("RS_Point_252_SMA_10")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_252_sma_20": safe_float(last_row.get("RS_Point_252_SMA_20")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_252_sma_50": safe_float(last_row.get("RS_Point_252_SMA_50")) if symbol.upper() != "VNINDEX" else None,
                "rs_point_252_sma_200": safe_float(last_row.get("RS_Point_252_SMA_200")) if symbol.upper() != "VNINDEX" else None,
                "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
            }
        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu cho {symbol}: {str(e)}")
            traceback.print_exc()
            return {
                "signal": "LỖI", "score": 50,
                "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rsi_value": 50,
                "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rs": 1.0, "rs_point": 0, "rs_point_252": 0.0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "open": None, "high": None, "low": None, "volume": None,
                "macd": None, "macd_signal": None, "macd_hist": None,
                "bb_upper": None, "bb_lower": None,
                "volume_ma": None,
                "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, "ichimoku_chikou_span": None,
                "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
                "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
                "rs_point_252_sma_10": None, "rs_point_252_sma_20": None,
                "rs_point_252_sma_50": None, "rs_point_252_sma_200": None,
                "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
            }
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi phân tích {symbol}: {str(e)}")
        traceback.print_exc()
        return {
            "signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0,
            "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0,
            "rs_point_252": 0.0,
            "recommendation": "KHÔNG XÁC ĐỊNH",
            "open": None, "high": None, "low": None, "volume": None,
            "macd": None, "macd_signal": None, "macd_hist": None,
            "bb_upper": None, "bb_lower": None,
            "volume_ma": None,
            "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
            "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, "ichimoku_chikou_span": None,
            "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
            "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
            "rs_point_252_sma_10": None, "rs_point_252_sma_20": None,
            "rs_point_252_sma_50": None, "rs_point_252_sma_200": None,
            "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
        }

# --- Phân tích bằng Google Gemini ---
def analyze_with_gemini(symbol, trading_signal, financial_data):
    """Phân tích mã chứng khoán bằng Google Gemini."""
    try:
        # Lấy các giá trị cần thiết từ trading_signal
        rs_val = trading_signal.get("rs", 1.0)
        rs_sma10_val = safe_format(trading_signal.get("rs_sma_10"), ".4f")
        rs_sma20_val = safe_format(trading_signal.get("rs_sma_20"), ".4f")
        rs_sma50_val = safe_format(trading_signal.get("rs_sma_50"), ".4f")
        rs_sma200_val = safe_format(trading_signal.get("rs_sma_200"), ".4f")
        rs_point_val = trading_signal["rs_point"]
        rs_point_sma10_val = safe_format(trading_signal.get("rs_point_sma_10"), ".2f")
        rs_point_sma20_val = safe_format(trading_signal.get("rs_point_sma_20"), ".2f")
        rs_point_sma50_val = safe_format(trading_signal.get("rs_point_sma_50"), ".2f")
        rs_point_sma200_val = safe_format(trading_signal.get("rs_point_sma_200"), ".2f")
        rs_point_252_val = trading_signal.get("rs_point_252", 0.0)
        rs_point_252_sma10_val = safe_format(trading_signal.get("rs_point_252_sma_10"), ".2f")
        rs_point_252_sma20_val = safe_format(trading_signal.get("rs_point_252_sma_20"), ".2f")
        rs_point_252_sma50_val = safe_format(trading_signal.get("rs_point_252_sma_50"), ".2f")
        rs_point_252_sma200_val = safe_format(trading_signal.get("rs_point_252_sma_200"), ".2f")
        tenkan_val = safe_format(trading_signal.get("ichimoku_tenkan_sen"))
        kijun_val = safe_format(trading_signal.get("ichimoku_kijun_sen"))
        senkou_a_val = safe_format(trading_signal.get("ichimoku_senkou_span_a"))
        senkou_b_val = safe_format(trading_signal.get("ichimoku_senkou_span_b"))
        chikou_val = safe_format(trading_signal.get("ichimoku_chikou_span"))
        # Lấy đường dẫn ảnh phân tích kỹ thuật
        technical_plot_path = f"vnstocks_data/{symbol}_analysis.png"

        prompt = f"""Bạn là chuyên gia phân tích chứng khoán Việt Nam. Phân tích {symbol}:
1. Kỹ thuật:
- Giá: {trading_signal['current_price']:,.2f}
- RSI: {trading_signal['rsi_value']:.2f}
- MA10: {trading_signal['ma10']:,.2f}
- MA20: {trading_signal['ma20']:,.2f}
- MA50: {trading_signal['ma50']:,.2f}
- MA200: {trading_signal['ma200']:,.2f}
- BB: {safe_format(trading_signal.get('bb_upper'))} / {safe_format(trading_signal.get('bb_lower'))}"""
        if symbol.upper() != "VNINDEX":
            prompt += f"- RS (so với VNINDEX) công thức ( C / PriceVNINDEX): {rs_val:.4f} (SMA10: {safe_format(rs_sma10_val, '.4f')}) (SMA20: {safe_format(rs_sma20_val, '.4f')}) (SMA50: {safe_format(rs_sma50_val, '.4f')}) (SMA200: {safe_format(rs_sma200_val, '.4f')})"
            prompt += f"- RS_Point công thức ( ( ROC( C, 63 ) * .4 + ROC( C, 126 ) * .2 + ROC( C, 189 ) * .2 + ROC( C, 252 ) * .2 ) * 100): {rs_point_val:.2f} (SMA10: {rs_point_sma10_val}) (SMA20: {rs_point_sma20_val}) (SMA50: {rs_point_sma50_val}) (SMA200: {rs_point_sma200_val})"
            prompt += f"- RS_Point_252 công thức ( ( ( C / Ref( C, -252 ) - 1 ) * 100 )): {rs_point_252_val:.2f} (SMA10: {rs_point_252_sma10_val}) (SMA20: {rs_point_252_sma20_val}) (SMA50: {rs_point_252_sma50_val}) (SMA200: {rs_point_252_sma200_val})"
            prompt += f"- Ichimoku: T:{tenkan_val}| K:{kijun_val}| A:{senkou_a_val}| B:{senkou_b_val}| C:{chikou_val}"
        if financial_data is not None and not financial_data.empty:
            prompt += f"2. Tài chính :{financial_data.to_string(index=False)}"
        else:
            prompt += "2. Tài chính : Không có dữ liệu tài chính."
        prompt += """Yêu cầu:
- Cho tôi biết nó đang mẫu hình nào trong phân tích kỹ thuật.
- Phân tích ngắn gọn, chuyên nghiệp.
- Kết luận rõ ràng: MUA MẠNH/MUA/GIỮ/BÁN/BÁN MẠNH.
- Phân tích dựa trên kỹ thuật và phân tích tài chính."""

        # Gửi prompt cho Gemini
        model = genai.GenerativeModel("gemini-2.5-pro")
        response = model.generate_content(prompt)

        if response and response.text:
            return response.text.strip()
        else:
            return "Không nhận được phản hồi từ Google Gemini."
    except Exception as e:
        import traceback
        print(f"❌ Lỗi khi phân tích bằng Google Gemini cho {symbol}: {str(e)}")
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return "Không thể tạo phân tích bằng Google Gemini tại thời điểm này."

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
    financial_data = get_financial_data(symbol)
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"❌ Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    if len(df_processed) < 100:
        print(f"❌ Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_processed)} điểm)")
        return None
    print(f"📈 Đang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_processed)
    print(f"🤖 Đang phân tích bằng Google Gemini ...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, financial_data)
    print(f"\n{'='*20} KẾT QUẢ PHÂN TÍCH CHO MÃ {symbol} {'='*20}")
    print(f"💰 Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"📈 Tín hiệu: {trading_signal['signal']}")
    print(f"🎯 Đề xuất: {trading_signal['recommendation']}")
    print(f"📊 Điểm phân tích: {trading_signal['score']:.2f}/100")
    if symbol.upper() != "VNINDEX":
        print(f"📊 RS (so với VNINDEX: {trading_signal['rs']:.4f}")
        print(f"📊 RS_Point: {trading_signal['rs_point']:.2f}")
        print(f"📊 RS_Point_252: {trading_signal['rs_point_252']:.2f}")
    print(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ GOOGLE GEMINI ---")
    print(gemini_analysis)
    print(f"{'='*60}\n")

    # --- Sửa lỗi TypeError: Object of type Timestamp is not JSON serializable ---
    # Chuyển đổi forecast_dates từ pd.Timestamp sang string (dù list này sẽ rỗng)
    forecast_dates_str = [date.strftime("%Y-%m-%d") for date in trading_signal.get("forecast_dates", [])]
    # --- Kết thúc phần sửa lỗi JSON ---

    report = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": safe_float(trading_signal.get("current_price")), "signal": trading_signal.get("signal"),
        "recommendation": trading_signal.get("recommendation"), "score": safe_float(trading_signal.get("score")),
        "rsi_value": safe_float(trading_signal.get("rsi_value")), "ma10": safe_float(trading_signal.get("ma10")),
        "ma20": safe_float(trading_signal.get("ma20")), "ma50": safe_float(trading_signal.get("ma50")),
        "ma200": safe_float(trading_signal.get("ma200")),
        "rs": safe_float(trading_signal.get("rs")) if symbol.upper() != "VNINDEX" else None,
        "rs_point": safe_float(trading_signal.get("rs_point")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_252": safe_float(trading_signal.get("rs_point_252")) if symbol.upper() != "VNINDEX" else None,
        "forecast": list(zip(forecast_dates_str, trading_signal.get("forecast_prices", []))),  # Sử dụng forecast_dates_str
        "ai_recommendation": "Không có", "ai_reason": "Không chạy mô hình AI riêng", "gemini_analysis": gemini_analysis,
    }
    # report.update(trading_signal) # Không cập nhật toàn bộ trading_signal vì có thể gây trùng lặp key và lỗi JSON
    with open(f"vnstocks_data/{symbol}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")
    return report

# --- Quét danh sách mã cổ phiếu ---
def screen_stocks_parallel():
    """Quét và phân tích nhiều mã chứng khoán tuần tự (sync)."""
    print(f"\n{'=' * 60}")
    print("QUÉT VÀ PHÂN TÍCH DANH SÁCH MÃ CHỨNG KHOÁN")
    print(f"{'=' * 60}")
    stock_list = get_vnstocks_list()
    symbols_to_analyze = stock_list["symbol"].head(20)
    results = []
    for symbol in symbols_to_analyze:
        try:
            result = analyze_stock(symbol)
            if result and result["signal"] != "LỖI":
                results.append(result)
                print(f"✅ Phân tích mã {symbol} hoàn tất.")
            else:
                print(f"⚠️ Phân tích mã {symbol} thất bại hoặc có lỗi.")
        except Exception as e:
            print(f"❌ Lỗi khi phân tích mã {symbol}: {e}")
            traceback.print_exc()
    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        def get_nested_value(report_dict, key_path, default=None):
            keys = key_path.split(".")
            current_dict = report_dict
            try:
                for key in keys:
                    if isinstance(current_dict, dict) and key in current_dict:
                        current_dict = current_dict[key]
                    else:
                        return default
                if current_dict is None or (isinstance(current_dict, float) and (pd.isna(current_dict) or np.isinf(current_dict))):
                    return default
                return float(current_dict)
            except (ValueError, TypeError):
                return default
        data_for_df = []
        for r in results:
            row_data = {
                "Mã": r["symbol"], "Giá": r["current_price"], "Điểm": r["score"], "Tín hiệu": r["signal"],
                "Đề xuất": r["recommendation"],
                "RSI": r["rsi_value"], "MA10": r["ma10"], "MA20": r["ma20"], "MA50": r["ma50"], "MA200": r["ma200"],
                "RS": r["rs"], "RS_Point": r["rs_point"],
                "RS_Point_252": r["rs_point_252"],
                "Ichimoku_Tenkan": r.get("ichimoku_tenkan_sen"),
                "Ichimoku_Kijun": r.get("ichimoku_kijun_sen"),
                "Ichimoku_Senkou_A": r.get("ichimoku_senkou_span_a"),
                "Ichimoku_Senkou_B": r.get("ichimoku_senkou_span_b"),
                "Ichimoku_Chikou": r.get("ichimoku_chikou_span"),
            }
            data_for_df.append(row_data)
        df_results = pd.DataFrame(data_for_df)
        df_results.to_csv("vnstocks_data/stock_screening_report.csv", index=False, encoding='utf-8-sig')
        print(f"✅ Đã lưu báo cáo tổng hợp vào file 'vnstocks_data/stock_screening_report.csv'")
        return df_results
    else:
        print("❌ Không có kết quả phân tích nào để tạo báo cáo tổng hợp.")
        return None

# --- Lọc cổ phiếu ---
def filter_stocks_low_pe_high_cap(min_market_cap=500):
    """Lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao."""
    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=1700)
        df.to_csv("market.csv", index=False, encoding="utf-8-sig")
        if df is None or df.empty:
            print("❌ Không thể lấy dữ liệu danh sách công ty niêm yết.")
            return None
        filtered_df = df[(df['market_cap'] > min_market_cap) &
                         (df['pe'] > 0) &
                         (df['pb'] > 0) &
                         (df['doe'] < 2)]

        return filtered_df
    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình lọc cổ phiếu: {e}")
        return None

# --- Hàm chính ---
def main():
    """Hàm chính để chạy chương trình."""
    print("=" * 60)
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print("TÍCH HỢP VNSTOCK & GOOGLE GEMINI")
    print("=" * 60)
    min_cap = 500
    print(f"🔍 Đang lọc cổ phiếu có P/E thấp và vốn hóa > {min_cap} tỷ VND...")
    filtered_stocks = filter_stocks_low_pe_high_cap(min_market_cap=min_cap)
    # if filtered_stocks is not None and not filtered_stocks.empty:
    #     print("🚀 Bắt đầu quét và phân tích...")
    #     screen_stocks_parallel()
    # else:
    #     print("🔍 Không tìm được cổ phiếu nào phù hợp với tiêu chí lọc.")
    print("\nNhập mã cổ phiếu để phân tích riêng lẻ (ví dụ: VCB, FPT) hoặc 'exit' để thoát:")
    user_input = input("Nhập mã cổ phiếu để phân tích: ").strip().upper()
    if user_input and user_input.lower() != 'exit':
        tickers = [ticker.strip() for ticker in user_input.split(',')]
        for ticker in tickers:
            if ticker:
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
        print("\n✅ Hoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")
    else:
        print("👋 Thoát chương trình.")

if __name__ == "__main__":
    main()