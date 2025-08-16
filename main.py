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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Cấu hình toàn cục
GLOBAL_EPOCHS = 50
GLOBAL_BATCH_SIZE = 32
GLOBAL_PREDICTION_DAYS = 10

start_date = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

# Tải khóa API Google
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Không tìm thấy khóa API Google. Vui lòng kiểm tra file .env")
    exit()

genai.configure(api_key=GOOGLE_API_KEY)

# Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại
if not os.path.exists("vnstocks_data"):
    os.makedirs("vnstocks_data")


def get_vnstocks_list():
    """
    Lấy danh sách tất cả các mã chứng khoán doanh nghiệp trên thị trường Việt Nam.
    Trả về DataFrame chứa cột 'symbol'.
    """
    try:
        df = listing_companies()
        if df is not None and not df.empty:
            df = df[df["organType"] == "DN"]
            symbols = df[["ticker"]].rename(columns={"ticker": "symbol"})
            symbols.to_csv("vnstocks_data/stock_list.csv", index=False)
            print(f"Đã lưu danh sách {len(symbols)} mã chứng khoán vào file 'vnstocks_data/stock_list.csv'")
            return symbols
        else:
            print("Không lấy được danh sách từ vnstock, sử dụng danh sách mẫu")
            sample_stocks = ["VNM", "VCB", "FPT", "GAS", "BID", "CTG", "MWG", "PNJ", "HPG", "STB"]
            df = pd.DataFrame(sample_stocks, columns=["symbol"])
            df.to_csv("vnstocks_data/stock_list.csv", index=False)
            return df
    except Exception as e:
        print(f"Exception khi lấy danh sách mã: {str(e)}")
        sample_stocks = ["VNM", "VCB", "FPT", "GAS", "BID", "CTG", "MWG", "PNJ", "HPG", "STB"]
        df = pd.DataFrame(sample_stocks, columns=["symbol"])
        df.to_csv("vnstocks_data/stock_list.csv", index=False)
        return df


def get_stock_data(symbol):
    """
    Lấy dữ liệu lịch sử giá của một mã chứng khoán từ VCI.
    Lưu dữ liệu vào file CSV.
    """
    try:
        quote = Quote(symbol)
        df = quote.history(start=start_date, end=end_date, interval="1D")
        if df is not None and not df.empty:
            df.rename(columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(f"vnstocks_data/{symbol}_data.csv")
            print(f"Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'")
            return df
        else:
            print(f"Không thể lấy dữ liệu cho mã {symbol} từ vnstock")
            return None
    except Exception as e:
        print(f"Exception khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None


def get_financial_data(symbol):
    """
    Lấy dữ liệu báo cáo tài chính (12 quý gần nhất) của một mã chứng khoán từ VCI.
    Lưu dữ liệu vào file CSV.
    """
    try:
        financial_obj = Finance(symbol=symbol)
        financial_data = financial_obj.ratio(period="quarter", lang="en", flatten_columns=True, limit=12)
        if financial_data is not None and not financial_data.empty:
            financial_data.to_csv(f"vnstocks_data/{symbol}_financial.csv", index=False)
            return financial_data
        else:
            print(f"Không lấy được BCTC cho mã {symbol}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None


def get_market_data():
    """
    Lấy dữ liệu lịch sử của các chỉ số thị trường (VNINDEX, VN30) từ VCI.
    Lưu dữ liệu vào file CSV.
    """
    try:
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(start=start_date, end=end_date, interval="1D")
        if vnindex is not None and not vnindex.empty:
            vnindex.rename(columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            vnindex["Date"] = pd.to_datetime(vnindex["Date"])
            vnindex.set_index("Date", inplace=True)
            vnindex.sort_index(inplace=True)
            vnindex.to_csv("vnstocks_data/vnindex_data.csv")

        quoteVN30 = Quote(symbol="VN30")
        vn30 = quoteVN30.history(start=start_date, end=end_date, interval="1D")
        if vn30 is not None and not vn30.empty:
            vn30.rename(columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            vn30["Date"] = pd.to_datetime(vn30["Date"])
            vn30.set_index("Date", inplace=True)
            vn30.sort_index(inplace=True)
            vn30.to_csv("vnstocks_data/vn30_data.csv")

        print("Đã lưu dữ liệu chỉ số thị trường vào thư mục 'vnstocks_data/'")
        return {"vnindex": vnindex, "vn30": vn30}
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu thị trường: {str(e)}")
        return None


def preprocess_stock_data(df):
    """
    Tiền xử lý dữ liệu giá cổ phiếu cơ bản.
    """
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df["returns"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["volatility"] = df["returns"].rolling(window=10).std()
    df.dropna(inplace=True)
    return df


def create_features(df):
    """
    Tạo các chỉ báo kỹ thuật sử dụng thư viện 'ta'.
    """
    df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()

    bollinger = ta.volatility.BollingerBands(df["Close"])
    df["BB_middle"] = bollinger.bollinger_hband()
    df["BB_std"] = bollinger.bollinger_lband()
    df["BB_upper"] = bollinger.bollinger_hband()
    df["BB_lower"] = bollinger.bollinger_lband()

    df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
    df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
    df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
    df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)

    df["Volume_MA"] = ta.trend.sma_indicator(df["Volume"], window=10)
    df["Volume_Change"] = df["Volume"].pct_change()

    # Ichimoku
    tenkan_window = 9
    kijun_window = 26
    senkou_span_b_window = 52
    tenkan_sen_high = df["High"].rolling(window=tenkan_window).max()
    tenkan_sen_low = df["Low"].rolling(window=tenkan_window).min()
    df["ichimoku_tenkan_sen"] = (tenkan_sen_high + tenkan_sen_low) / 2

    kijun_sen_high = df["High"].rolling(window=kijun_window).max()
    kijun_sen_low = df["Low"].rolling(window=kijun_window).min()
    df["ichimoku_kijun_sen"] = (kijun_sen_high + kijun_sen_low) / 2

    df["ichimoku_senkou_span_a"] = ((df["ichimoku_tenkan_sen"] + df["ichimoku_kijun_sen"]) / 2).shift(kijun_window)
    senkou_span_b_high = df["High"].rolling(window=senkou_span_b_window).max()
    senkou_span_b_low = df["Low"].rolling(window=senkou_span_b_window).min()
    df["ichimoku_senkou_span_b"] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(kijun_window)
    df["ichimoku_chikou_span"] = df["Close"].shift(-kijun_window)

    df.dropna(inplace=True)
    return df


def calculate_relative_strength(df_stock, df_index):
    """
    Tính Relative Strength (RS) và RS Point theo công thức của Amibroker.
    RS = (Giá CP hiện tại / Giá CP 1 kỳ trước) / (Giá Index hiện tại / Giá Index 1 kỳ trước)
    RS Point = (ROC(63)*0.4 + ROC(126)*0.2 + ROC(189)*0.2 + ROC(252)*0.2) * 100
    """
    # Gộp dữ liệu cổ phiếu và chỉ số
    df_merged = df_stock[["Close"]].join(df_index[["Close"]].rename(columns={"Close": "Index_Close"}), how="inner")

    if df_merged.empty or df_merged["Index_Close"].isna().all():
        print("Cảnh báo: Không có dữ liệu chỉ số thị trường để tính RS.")
        df_stock["RS"] = 1.0
        df_stock["RS_Point"] = 0.0
        return df_stock

    df_merged["Index_Close"] = df_merged["Index_Close"].ffill().bfill()

    # Tính RS theo công thức Amibroker
    price_ratio = df_merged["Close"] / df_merged["Close"].shift(1)
    index_ratio = df_merged["Index_Close"] / df_merged["Index_Close"].shift(1)
    df_merged["RS"] = price_ratio / index_ratio

    # Tính ROC cho các kỳ hạn
    roc_63 = df_merged["Close"].pct_change(periods=63) * 100
    roc_126 = df_merged["Close"].pct_change(periods=126) * 100
    roc_189 = df_merged["Close"].pct_change(periods=189) * 100
    roc_252 = df_merged["Close"].pct_change(periods=252) * 100

    # Tính RS Point theo công thức Amibroker
    df_merged["RS_Point"] = (
        roc_63.fillna(0) * 0.4 +
        roc_126.fillna(0) * 0.2 +
        roc_189.fillna(0) * 0.2 +
        roc_252.fillna(0) * 0.2
    )

    # Gán RS và RS_Point trở lại dataframe gốc (df_stock)
    df_stock = df_stock.join(df_merged[["RS", "RS_Point"]], how="left")
    df_stock["RS"].fillna(1.0, inplace=True)
    df_stock["RS_Point"].fillna(0.0, inplace=True)

    # Tính các đường trung bình cho RS và RS_Point
    df_stock["RS_SMA_10"] = ta.trend.sma_indicator(df_stock["RS"], window=10)
    df_stock["RS_SMA_20"] = ta.trend.sma_indicator(df_stock["RS"], window=20)
    df_stock["RS_SMA_50"] = ta.trend.sma_indicator(df_stock["RS"], window=50)
    df_stock["RS_SMA_200"] = ta.trend.sma_indicator(df_stock["RS"], window=200)

    df_stock["RS_Point_SMA_10"] = ta.trend.sma_indicator(df_stock["RS_Point"], window=10)
    df_stock["RS_Point_SMA_20"] = ta.trend.sma_indicator(df_stock["RS_Point"], window=20)
    df_stock["RS_Point_SMA_50"] = ta.trend.sma_indicator(df_stock["RS_Point"], window=50)
    df_stock["RS_Point_SMA_200"] = ta.trend.sma_indicator(df_stock["RS_Point"], window=200)

    return df_stock


def plot_stock_analysis(symbol, df, show_volume=True):
    """
    Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán.
    """
    try:
        if df is None or len(df) == 0:
            print("Dữ liệu phân tích rỗng")
            return {
                "signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0,
                "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0,
                "recommendation": "KHÔNG XÁC ĐỊNH"
            }

        df = df.sort_index()

        # --- BƯỚC 1: Tính các chỉ báo kỹ thuật ---
        df = create_features(df)

        # --- BƯỚC 2: Tính RS (Relative Strength so với VNINDEX) ---
        try:
            quoteVNI = Quote(symbol="VNINDEX")
            vnindex_df = quoteVNI.history(start=start_date, end=end_date, interval="1D")
            if len(vnindex_df) == 0:
                raise ValueError("Không lấy được dữ liệu VNINDEX")

            vnindex_df.rename(columns={
                "time": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume"
            }, inplace=True)
            vnindex_df["Date"] = pd.to_datetime(vnindex_df["Date"])
            vnindex_df.set_index("Date", inplace=True)
            vnindex_df.sort_index(inplace=True)

            df = calculate_relative_strength(df, vnindex_df)

        except Exception as e:
            print(f"Không thể tính RS do lỗi VNINDEX: {e}")
            df["RS"] = 1.0
            df["RS_Point"] = 0.0
            df["RS_SMA_10"] = 1.0
            df["RS_SMA_20"] = 1.0
            df["RS_SMA_50"] = 1.0
            df["RS_SMA_200"] = 1.0
            df["RS_Point_SMA_10"] = 0.0
            df["RS_Point_SMA_20"] = 0.0
            df["RS_Point_SMA_50"] = 0.0
            df["RS_Point_SMA_200"] = 0.0

        # --- BƯỚC 3: Kiểm tra dữ liệu hợp lệ ---
        df = df.dropna(subset=["Close", "SMA_10", "SMA_20", "SMA_50"], how="all")
        if len(df) < 20:
            print("Không đủ dữ liệu hợp lệ để phân tích")
            return {
                "signal": "LỖI", "score": 50,
                "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rsi_value": 50,
                "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rs": 1.0, "rs_point": 0,
                "recommendation": "KHÔNG XÁC ĐỊNH"
            }

        # --- BƯỚC 4: Vẽ biểu đồ ---
        try:
            plot_configs = ["price_sma", "ichimoku", "rsi", "macd", "rs", "rs_point", "volume"]
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
                     linewidth=1.5)
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

            # === Biểu đồ 7: Khối lượng ===
            ax7 = plt.subplot(grid[6], sharex=ax1)
            if show_volume and "Volume" in df.columns:
                volume_sma_plotted = False
                if ("Volume_MA" in df.columns and not df["Volume_MA"].isna().all()):
                    plt.plot(df.index, df["Volume_MA"], label=f"Vol SMA 10 {df['Volume_MA'].iloc[-1]:,.0f}",
                             color="orange", alpha=0.8, linewidth=1.5)
                    volume_sma_plotted = True
                colors = np.where(df["Close"] > df["Open"], "green", "red")
                plt.bar(df.index, df["Volume"], color=colors, alpha=0.7, label="Volume" if not volume_sma_plotted else None)
                handles, labels = ax7.get_legend_handles_labels()
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

        # --- BƯỚC 5: Tạo tín hiệu giao dịch ---
        try:
            last_row = df.iloc[-1]
            current_price = last_row["Close"]
            rsi_value = last_row["RSI"] if not pd.isna(last_row["RSI"]) else 50
            ma10_value = (last_row["SMA_10"] if not pd.isna(last_row["SMA_10"]) else current_price)
            ma20_value = (last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else current_price)
            ma50_value = (last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else current_price)
            ma200_value = (last_row["SMA_200"] if not pd.isna(last_row["SMA_200"]) else current_price)
            rs_value = last_row["RS"] if not pd.isna(last_row["RS"]) else 1.0
            rs_point_value = (last_row["RS_Point"] if not pd.isna(last_row["RS_Point"]) else 0)
            tenkan_sen = last_row.get("ichimoku_tenkan_sen", np.nan)
            kijun_sen = last_row.get("ichimoku_kijun_sen", np.nan)
            senkou_span_a = last_row.get("ichimoku_senkou_span_a", np.nan)
            senkou_span_b = last_row.get("ichimoku_senkou_span_b", np.nan)
            chikou_span = last_row.get("ichimoku_chikou_span", np.nan)

            score = 50

            # 1. RSI - 15 điểm
            if rsi_value < 30:
                score += 15
            elif rsi_value > 70:
                score -= 15
            else:
                score += (50 - abs(rsi_value - 50)) * 0.3

            # 2. Đường trung bình - 25 điểm
            if ma10_value > ma20_value > ma50_value > ma200_value:
                score += 25
            elif ma10_value > ma20_value > ma50_value:
                score += 15
            elif ma10_value > ma20_value:
                score += 8
            elif ma10_value < ma20_value < ma50_value < ma200_value:
                score -= 25
            elif ma10_value < ma20_value < ma50_value:
                score -= 15
            elif ma10_value < ma20_value:
                score -= 8

            # 3. Giá so với các đường trung bình - 10 điểm
            if current_price > ma10_value:
                score += 3
            if current_price > ma20_value:
                score += 3
            if current_price > ma50_value:
                score += 2
            if current_price > ma200_value:
                score += 2

            # 4. MACD - 15 điểm
            macd_value = last_row["MACD"] if not pd.isna(last_row["MACD"]) else 0
            macd_signal = (last_row["MACD_signal"] if not pd.isna(last_row["MACD_signal"]) else 0)
            macd_hist = (last_row["MACD_Hist"] if not pd.isna(last_row["MACD_Hist"]) else 0)
            macd_hist_series = df["MACD_Hist"]
            if len(macd_hist_series) > 1:
                macd_hist_prev = macd_hist_series.iloc[-2] if not pd.isna(macd_hist_series.iloc[-2]) else 0
            else:
                macd_hist_prev = 0

            if macd_value > macd_signal and macd_hist > 0 and macd_hist > macd_hist_prev:
                score += 15
            elif macd_value > macd_signal and macd_hist > 0:
                score += 10
            elif macd_value < macd_signal and macd_hist < 0 and macd_hist < macd_hist_prev:
                score -= 15
            elif macd_value < macd_signal and macd_hist < 0:
                score -= 10
            else:
                score += np.clip(macd_hist * 40, -5, 5)

            # 5. Bollinger Bands - 10 điểm
            bb_upper = (last_row["BB_upper"] if not pd.isna(last_row["BB_upper"]) else current_price)
            bb_lower = (last_row["BB_lower"] if not pd.isna(last_row["BB_lower"]) else current_price)
            if current_price > bb_upper:
                score -= 5
            elif current_price < bb_lower:
                score += 5
            else:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                score += (bb_position - 0.5) * 10

            # 6. RS (Relative Strength) - 10 điểm
            rs_score = 0
            if rs_value > last_row.get("RS_SMA_10", rs_value):
                rs_score += 2
            if rs_value > last_row.get("RS_SMA_20", rs_value):
                rs_score += 3
            if rs_value > last_row.get("RS_SMA_50", rs_value):
                rs_score += 5
            score += rs_score

            # 7. RS_Point - 10 điểm
            rs_point_score = 0
            if rs_point_value > last_row.get("RS_Point_SMA_10", rs_point_value):
                rs_point_score += 2
            if rs_point_value > last_row.get("RS_Point_SMA_20", rs_point_value):
                rs_point_score += 3
            if rs_point_value > last_row.get("RS_Point_SMA_50", rs_point_value):
                rs_point_score += 5
            score += rs_point_score

            # 8. Ichimoku Cloud - 15 điểm
            ichimoku_score = 0
            try:
                if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
                    cloud_top = max(senkou_span_a, senkou_span_b)
                    cloud_bottom = min(senkou_span_a, senkou_span_b)
                    if current_price > cloud_top and tenkan_sen > kijun_sen:
                        ichimoku_score += 15
                    elif current_price > cloud_top:
                        ichimoku_score += 10
                    elif current_price < cloud_bottom and tenkan_sen < kijun_sen:
                        ichimoku_score -= 15
                    elif current_price < cloud_bottom:
                        ichimoku_score -= 10
                    if tenkan_sen > kijun_sen:
                        ichimoku_score += 5
                    elif tenkan_sen < kijun_sen:
                        ichimoku_score -= 5
                    if kijun_sen > cloud_top:
                        ichimoku_score += 5
                    elif kijun_sen < cloud_bottom:
                        ichimoku_score -= 5
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi tính điểm Ichimoku: {e}")
            score += ichimoku_score

            score = np.clip(score, 0, 100)

            # --- XÁC ĐỊNH TÍN HIỆU ---
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

            analysis_date = df.index[-1].strftime("%d/%m/%Y")
            print(f"📊 TÍN HIỆU GIAO DỊCH CUỐI ({analysis_date}):")
            print(
                f"  - Giá & Đường trung bình: Giá={current_price:,.2f} | SMA10={ma10_value:,.2f} | SMA20={ma20_value:,.2f} | SMA50={ma50_value:,.2f} | SMA200={ma200_value:,.2f}")
            try:
                print(f"  - Ichimoku:")
                print(f"    * Tenkan-sen: {tenkan_sen:.2f} | Kijun-sen: {kijun_sen:.2f}")
                print(f"    * Cloud (A/B): {senkou_span_a:.2f} / {senkou_span_b:.2f}")
                print(f"    * Chikou Span: {chikou_span:.2f}")
                print(f"    * Điểm Ichimoku: ~{ichimoku_score:.1f}")
            except:
                print(f"  - Ichimoku: Không có dữ liệu")
            print(f"  - Đề xuất: {recommendation} (Điểm: {score:.1f})")

            def safe_float(val):
                try:
                    if pd.isna(val):
                        return None
                    return float(val)
                except (TypeError, ValueError):
                    return None

            return {
                "signal": signal, "score": float(score), "current_price": float(current_price),
                "rsi_value": float(rsi_value),
                "ma10": float(ma10_value), "ma20": float(ma20_value), "ma50": float(ma50_value),
                "ma200": float(ma200_value),
                "rs": float(rs_value), "rs_point": float(rs_point_value), "recommendation": recommendation,
                "open": safe_float(last_row.get("Open")), "high": safe_float(last_row.get("High")),
                "low": safe_float(last_row.get("Low")), "volume": safe_float(last_row.get("Volume")),
                "macd": safe_float(macd_value), "macd_signal": safe_float(macd_signal),
                "macd_hist": safe_float(macd_hist),
                "bb_upper": safe_float(bb_upper), "bb_lower": safe_float(bb_lower),
                "volume_ma": safe_float(last_row.get("Volume_MA")),
                "ichimoku_tenkan_sen": safe_float(tenkan_sen), "ichimoku_kijun_sen": safe_float(kijun_sen),
                "ichimoku_senkou_span_a": safe_float(senkou_span_a),
                "ichimoku_senkou_span_b": safe_float(senkou_span_b),
                "ichimoku_chikou_span": safe_float(chikou_span),
                "rs_sma_10": safe_float(last_row.get("RS_SMA_10")),
                "rs_sma_20": safe_float(last_row.get("RS_SMA_20")),
                "rs_sma_50": safe_float(last_row.get("RS_SMA_50")),
                "rs_sma_200": safe_float(last_row.get("RS_SMA_200")),
                "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10")),
                "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20")),
                "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50")),
                "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200")),
            }

        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu: {str(e)}")
            traceback.print_exc()
            return {
                "signal": "LỖI", "score": 50,
                "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rsi_value": 50,
                "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rs": 1.0, "rs_point": 0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "open": None, "high": None, "low": None, "volume": None,
                "macd": None, "macd_signal": None, "macd_hist": None,
                "bb_upper": None, "bb_lower": None,
                "volume_ma": None,
                "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
                "ichimoku_chikou_span": None,
                "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
                "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None,
                "rs_point_sma_200": None
            }

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {str(e)}")
        traceback.print_exc()
        return {
            "signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0,
            "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0,
            "recommendation": "KHÔNG XÁC ĐỊNH",
            "open": None, "high": None, "low": None, "volume": None,
            "macd": None, "macd_signal": None, "macd_hist": None,
            "bb_upper": None, "bb_lower": None,
            "volume_ma": None,
            "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
            "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None,
            "ichimoku_chikou_span": None,
            "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
            "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None,
            "rs_point_sma_200": None
        }


def analyze_with_gemini(symbol, trading_signal, forecast, financial_data=None):
    """
    Phân tích cổ phiếu bằng Google Qwen dựa trên dữ liệu kỹ thuật và BCTC.
    """
    try:
        def safe_format(val, fmt=".2f"):
            try:
                if val is None:
                    return "N/A"
                if isinstance(val, float):
                    if pd.isna(val) or np.isinf(val):
                        return "N/A"
                return f"{{:{fmt}}}".format(float(val))
            except (ValueError, TypeError):
                return "N/A"

        rs_val = trading_signal["rs"]
        rs_sma10_val = trading_signal.get("rs_sma_10")
        rs_sma20_val = trading_signal.get("rs_sma_20")
        rs_sma50_val = trading_signal.get("rs_sma_50")
        rs_sma200_val = trading_signal.get("rs_sma_200")
        rs_point_val = trading_signal["rs_point"]
        rs_point_sma10_val = safe_format(trading_signal.get("rs_point_sma_10"), ".2f")
        rs_point_sma20_val = safe_format(trading_signal.get("rs_point_sma_20"), ".2f")
        rs_point_sma50_val = safe_format(trading_signal.get("rs_point_sma_50"), ".2f")
        rs_point_sma200_val = safe_format(trading_signal.get("rs_point_sma_200"), ".2f")

        tenkan_val = safe_format(trading_signal.get("ichimoku_tenkan_sen"))
        kijun_val = safe_format(trading_signal.get("ichimoku_kijun_sen"))
        senkou_a_val = safe_format(trading_signal.get("ichimoku_senkou_span_a"))
        senkou_b_val = safe_format(trading_signal.get("ichimoku_senkou_span_b"))
        chikou_val = safe_format(trading_signal.get("ichimoku_chikou_span"))

        prompt = f"""
Bạn là chuyên gia phân tích chứng khoán Việt Nam. Phân tích {symbol}:
1. Kỹ thuật:
- Giá: {trading_signal['current_price']:,.2f} VND
- RSI: {trading_signal['rsi_value']:.2f}
- MA10: {trading_signal['ma10']:,.2f} VND
- MA20: {trading_signal['ma20']:,.2f} VND
- MA50: {trading_signal['ma50']:,.2f} VND
- MA200: {trading_signal['ma200']:,.2f} VND
- BB: {safe_format(trading_signal.get('bb_upper'))} / {safe_format(trading_signal.get('bb_lower'))}
- RS (Amibroker): {rs_val:.2f} (SMA10: {rs_sma10_val})
- RS_Point: {rs_point_val:.2f} (SMA10: {rs_point_sma10_val})
- Ichimoku: T:{tenkan_val} | K:{kijun_val} | A:{senkou_a_val} | B:{senkou_b_val} | C:{chikou_val}
2. Tín hiệu: {trading_signal['signal']} ({trading_signal['score']:.1f}/100)
"""
        if financial_data is not None and not financial_data.empty:
            prompt += f"\n3. Tài chính (12 quý gần nhất):\n{financial_data.to_string(index=False)}"
        else:
            prompt += "\n3. Không có dữ liệu tài chính."

        prompt += """
Yêu cầu:
- Phân tích ngắn gọn, chuyên nghiệp (dưới 300 từ).
- Kết luận rõ ràng: MUA MẠNH/MUA/GIỮ/BÁN/BÁN MẠNH.
- Lý do dựa trên điểm số và chỉ báo kỹ thuật chính.
"""
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "Không nhận được phản hồi từ Google."

    except Exception as e:
        import traceback
        print(f"Lỗi khi phân tích bằng Qwen: {str(e)}")
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return "Không thể tạo phân tích bằng Google tại thời điểm này."


def analyze_stock(symbol):
    """
    Phân tích toàn diện một mã chứng khoán mà không dùng mô hình AI.
    """
    print(f"\n{'=' * 50}")
    print(f"PHÂN TÍCH MÃ {symbol} (KHÔNG DÙNG AI)")
    print(f"{'=' * 50}")

    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None

    financial_data = get_financial_data(symbol)

    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None

    if len(df_processed) < 100:
        print(f"Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_processed)} điểm)")
        return None

    print(f"\nĐang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_processed)

    print(f"\nĐang phân tích bằng Google ...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, ([], []), financial_data)

    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.2f}/100")
    print(f"RS (Amibroker): {trading_signal['rs']:.2f}")
    print(f"RS_Point (Amibroker): {trading_signal['rs_point']:.2f}")
    print(f"\nPHÂN TÍCH TỔNG HỢP TỪ Google:")
    print(gemini_analysis)

    def safe_float(val):
        try:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    report = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": safe_float(trading_signal.get("current_price")), "signal": trading_signal.get("signal"),
        "recommendation": trading_signal.get("recommendation"), "score": safe_float(trading_signal.get("score")),
        "rsi_value": safe_float(trading_signal.get("rsi_value")), "ma10": safe_float(trading_signal.get("ma10")),
        "ma20": safe_float(trading_signal.get("ma20")), "ma50": safe_float(trading_signal.get("ma50")),
        "ma200": safe_float(trading_signal.get("ma200")), "rs": safe_float(trading_signal.get("rs")),
        "rs_point": safe_float(trading_signal.get("rs_point")),
        "forecast": [],
        "ai_recommendation": "Không có", "ai_reason": "Không chạy mô hình AI", "gemini_analysis": gemini_analysis,
    }
    report.update(trading_signal)

    with open(f"vnstocks_data/{symbol}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")
    return report


def screen_stocks_parallel():
    """
    Quét và phân tích nhiều mã chứng khoán tuần tự (sync).
    """
    print(f"\n{'=' * 50}")
    print("QUÉT VÀ PHÂN TÍCH DANH SÁCH MÃ CHỨNG KHOÁN (TUẦN TỰ - SYNC)")
    print(f"{'=' * 50}")

    stock_list = get_vnstocks_list()
    symbols_to_analyze = stock_list["symbol"].head(20)
    results = []

    for symbol in symbols_to_analyze:
        try:
            result = analyze_stock(symbol)
            if result and result["signal"] != "LỖI":
                results.append(result)
                print(f"✅ Phân tích mã {symbol} hoàn tất (tuần tự - sync).")
            else:
                print(f"⚠️ Phân tích mã {symbol} thất bại hoặc có lỗi (tuần tự - sync).")
        except Exception as e:
            print(f"Lỗi khi phân tích mã {symbol} (tuần tự - sync): {e}")
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
                if current_dict is None or (
                        isinstance(current_dict, float) and (pd.isna(current_dict) or np.isinf(current_dict))):
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
                "Ichimoku_Tenkan": r.get("ichimoku_tenkan_sen"),
                "Ichimoku_Kijun": r.get("ichimoku_kijun_sen"),
                "Ichimoku_Senkou_A": r.get("ichimoku_senkou_span_a"),
                "Ichimoku_Senkou_B": r.get("ichimoku_senkou_span_b"),
                "Ichimoku_Chikou": r.get("ichimoku_chikou_span"),
            }
            data_for_df.append(row_data)

        df_results = pd.DataFrame(data_for_df)
        df_results.to_csv("vnstocks_data/stock_screening_report.csv", index=False)
        print(f"\n{'=' * 50}")
        print("KẾT QUẢ QUÉT MÃ")
        print(f"{'=' * 50}")
        print_cols = ["Mã", "Giá", "Điểm", "Tín hiệu", "Đề xuất"]
        print(df_results[print_cols])

        try:
            plt.figure(figsize=(14, 6))
            sns.barplot(x="Mã", y="Điểm", data=df_results.head(20), palette="viridis")
            plt.title("Top Điểm phân tích các mã chứng khoán")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("vnstocks_data/stock_screening_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ so sánh: {str(e)}")

        print(f"\nĐã lưu báo cáo tổng hợp vào file 'vnstocks_data/stock_screening_report.csv'")
        print("Đã tạo biểu đồ so sánh các mã")
        return df_results
    else:
        print("\nKhông có kết quả phân tích nào để tạo báo cáo tổng hợp.")
    return None


def main():
    """
    Hàm chính để chạy chương trình.
    """
    print("==============================================")
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM")
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE - PHIÊN BẢN KHÔNG AI")
    print("==============================================")

    market_data = get_market_data()
    analyze_stock("DRI")
    # screen_stocks_parallel()

    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")


if __name__ == "__main__":
    main()