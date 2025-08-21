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
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
import traceback
from vnstock.explorer.vci import Quote, Finance
from vnstock import Screener
import matplotlib.dates as mdates
import mplfinance as mpf

warnings.filterwarnings("ignore")

# --- Cấu hình toàn cục cho phân tích dữ liệu ---
# Thời gian lấy dữ liệu (ĐÃ THAY ĐỔI THÀNH 10 NĂM)
GLOBAL_START_DATE = (datetime.today() - timedelta(days=365 * 10)).strftime(
    "%Y-%m-%d"
)  # Lấy dữ liệu 10 năm gần nhất
GLOBAL_END_DATE = datetime.today().strftime("%Y-%m-%d")

# --- Cấu hình API và thư mục lưu trữ ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Khóa API cho AI
OPEN_ROUTER_API_KEY = os.getenv("OPEN_ROUTER_API_KEY")  # Khóa API cho AI
if not GOOGLE_API_KEY or not OPEN_ROUTER_API_KEY:
    raise ValueError("Vui lòng đặt KEY trong file .env")
# CHỈ CẤU HÌNH API KEY, KHÔNG GÁN KẾT QUẢ CHO BIẾN
genai.configure(api_key=GOOGLE_API_KEY)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPEN_ROUTER_API_KEY,
)
os.makedirs(
    "vnstocks_data", exist_ok=True
)  # Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại


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
    """Lấy dữ liệu lịch sử giá cổ phiếu từ VCI và lưu vào file csv."""
    try:
        stock = Quote(symbol=symbol)
        df = stock.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
        if df is not None and not df.empty:
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
            df.to_csv(f"vnstocks_data/{symbol}_data.csv", index=False, encoding='utf-8')
            print(
                f"✅ Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'"
            )
            return df
        else:
            print(f"⚠️ Không lấy được dữ liệu cho mã {symbol}")
            return None
    except Exception as e:
        print(f"❌ Exception khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None


def safe_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # Lọc chỉ giữ lại những key có tồn tại trong df
    valid_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=valid_mapping)


def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính từ VCI và lưu vào file csv."""

    def flatten_columns(df):
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [
                "_".join(col).strip() if col[1] else col[0] for col in df.columns.values
            ]
        return df

    def standardize_columns(df):
        column_mapping = {
            "Meta_ticker": "ticker",
            "Meta_yearReport": "yearReport",
            "Meta_lengthReport": "lengthReport",
        }
        return df.rename(columns=column_mapping)

    try:
        # Khởi tạo đối tượng finance
        stock = Finance(symbol=symbol, period="quarter")

        # Lấy 4 loại báo cáo tài chính
        df_ratio = stock.ratio(period="quarter")
        df_bs = stock.balance_sheet(period="quarter")
        df_is = stock.income_statement(period="quarter")
        df_cf = stock.cash_flow(period="quarter")
        df_ratio = standardize_columns(flatten_columns(df_ratio))

        financial_data = (
            df_bs.merge(df_is, on=["yearReport", "lengthReport", "ticker"], how="outer")
            .merge(df_cf, on=["yearReport", "lengthReport", "ticker"], how="outer")
            .merge(df_ratio, on=["yearReport", "lengthReport", "ticker"], how="outer")
        )

        # Lưu financial_data vào csv
        financial_data.to_csv(
            f"vnstocks_data/{symbol}_financial_statements.csv", index=False, encoding='utf-8'
        )

        print(f"Đã lưu dữ liệu tài chính của mã {symbol} vào file csv")
        return financial_data

    except Exception as e:
        print(f"❌ Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None


def get_market_data():
    """Lấy dữ liệu lịch sử của VNINDEX từ VCI và lưu vào file csv."""
    try:
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(
            start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D"
        )
        if vnindex is not None and not vnindex.empty:
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
            vnindex.to_csv("vnstocks_data/VNINDEX_data.csv", index=False, encoding='utf-8')
            print(
                f"✅ Đã lưu dữ liệu VNINDEX vào file 'vnstocks_data/VNINDEX_data.csv'"
            )
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
    df["volatility"] = df["returns"].rolling(window=10).std()
    return df


def create_features(df):
    """Tạo các chỉ báo kỹ thuật sử dụng thư viện 'ta'."""
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
    df_merged = df_stock[["Close"]].join(
        df_index[["Close"]].rename(columns={"Close": "Index_Close"}), how="inner"
    )
    if df_merged.empty or df_merged["Index_Close"].isna().all():
        print(
            "⚠️ Cảnh báo: Không có dữ liệu chỉ số thị trường để tính RS. Gán giá trị mặc định."
        )
        # Gán giá trị mặc định cho tất cả các chỉ báo RS
        df_stock["RS"] = 1.0
        df_stock["RS_Point"] = 0.0
        df_stock["RS_SMA_10"] = 1.0
        df_stock["RS_SMA_20"] = 1.0
        df_stock["RS_SMA_50"] = 1.0
        df_stock["RS_SMA_200"] = 1.0
        df_stock["RS_Point_SMA_10"] = 0.0
        df_stock["RS_Point_SMA_20"] = 0.0
        df_stock["RS_Point_SMA_50"] = 0.0
        df_stock["RS_Point_SMA_200"] = 0.0
        return df_stock
    df_merged["Index_Close"] = df_merged["Index_Close"]
    # Tính RS
    df_merged["RS"] = df_merged["Close"] / df_merged["Index_Close"]
    # Tính các thành phần ROC cho RS_Point
    roc_63 = ta.momentum.roc(df_merged["Close"], window=63)
    roc_126 = ta.momentum.roc(df_merged["Close"], window=126)
    roc_189 = ta.momentum.roc(df_merged["Close"], window=189)
    roc_252 = ta.momentum.roc(df_merged["Close"], window=252)
    # Tính RS_Point theo công thức: (ROC(63)*0.4 + ROC(126)*0.2 + ROC(189)*0.2 + ROC(252)*0.2)
    # Vì ROC đã được nhân 100, kết quả không cần nhân thêm.
    df_merged["RS_Point"] = (
        roc_63 * 0.4 + roc_126 * 0.2 + roc_189 * 0.2 + roc_252 * 0.2
    ) * 100

    # Tính các đường trung bình cho RS, RS_Point
    df_merged["RS_SMA_10"] = ta.trend.sma_indicator(df_merged["RS"], window=10)
    df_merged["RS_SMA_20"] = ta.trend.sma_indicator(df_merged["RS"], window=20)
    df_merged["RS_SMA_50"] = ta.trend.sma_indicator(df_merged["RS"], window=50)
    df_merged["RS_SMA_200"] = ta.trend.sma_indicator(df_merged["RS"], window=200)
    df_merged["RS_Point_SMA_10"] = ta.trend.sma_indicator(
        df_merged["RS_Point"], window=10
    )
    df_merged["RS_Point_SMA_20"] = ta.trend.sma_indicator(
        df_merged["RS_Point"], window=20
    )
    df_merged["RS_Point_SMA_50"] = ta.trend.sma_indicator(
        df_merged["RS_Point"], window=50
    )
    df_merged["RS_Point_SMA_200"] = ta.trend.sma_indicator(
        df_merged["RS_Point"], window=200
    )
    # Gán các chỉ báo trở lại dataframe gốc
    cols_to_join = [
        "RS",
        "RS_Point",
        "RS_SMA_10",
        "RS_SMA_20",
        "RS_SMA_50",
        "RS_SMA_200",
        "RS_Point_SMA_10",
        "RS_Point_SMA_20",
        "RS_Point_SMA_50",
        "RS_Point_SMA_200",
    ]
    df_stock = df_stock.join(df_merged[cols_to_join], how="left")
    # Xử lý giá trị NaN
    df_stock["RS"].fillna(1.0, inplace=True)
    df_stock["RS_Point"].fillna(0.0, inplace=True)
    df_stock["RS_SMA_10"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_20"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_50"].fillna(1.0, inplace=True)
    df_stock["RS_SMA_200"].fillna(1.0, inplace=True)
    df_stock["RS_Point_SMA_10"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_20"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_50"].fillna(0.0, inplace=True)
    df_stock["RS_Point_SMA_200"].fillna(0.0, inplace=True)
    return df_stock


# --- Phân tích kỹ thuật và vẽ biểu đồ ---
def plot_stock_analysis(symbol, df, show_volume=True):
    """Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán."""
    try:
        if df is None or len(df) == 0:
            print("❌ Dữ liệu phân tích rỗng")
            return {
                "signal": "LỖI",
                "score": 50,
                "current_price": 0,
                "rsi_value": 0,
                "ma10": 0,
                "ma20": 0,
                "ma50": 0,
                "ma200": 0,
                "rs": 1.0,
                "rs_point": 0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None,
                "bb_upper": None,
                "bb_lower": None,
                "volume_ma_20": None,
                "volume_ma_50": None,
                "ichimoku_tenkan_sen": None,
                "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None,
                "ichimoku_senkou_span_b": None,
                "ichimoku_chikou_span": None,
                "rs_sma_10": None,
                "rs_sma_20": None,
                "rs_sma_50": None,
                "rs_sma_200": None,
                "rs_point_sma_10": None,
                "rs_point_sma_20": None,
                "rs_point_sma_50": None,
                "rs_point_sma_200": None,
                "relative_strength_3d": None,
                "relative_strength_1m": None,
                "relative_strength_3m": None,
                "relative_strength_1y": None,
                "forecast_dates": [],
                "forecast_prices": [],
                "forecast_plot_path": "",
            }

        df = df.sort_index()
        df = create_features(df)

        # Tính RS (Relative Strength so với VNINDEX)
        if symbol.upper() != "VNINDEX":
            try:
                quoteVNI = Quote(symbol="VNINDEX")
                vnindex = quoteVNI.history(
                    start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D"
                )
                if vnindex is not None and not vnindex.empty:
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
                    vnindex.to_csv("vnstocks_data/VNINDEX_data.csv", index=False, encoding='utf-8')
                    print(
                        f"✅ Đã lưu dữ liệu VNINDEX vào file 'vnstocks_data/VNINDEX_data.csv'"
                    )

                    df = calculate_relative_strength(df, vnindex)
                else:
                    print("⚠️ Không lấy được dữ liệu VNINDEX")
            except Exception as e:
                print(f"❌ Lỗi khi lấy dữ liệu thị trường (VNINDEX): {str(e)}")

        # Tạo tín hiệu giao dịch
        try:
            last_row = df.iloc[-1]
            current_price = last_row["Close"]
            rsi_value = last_row["RSI"] if not pd.isna(last_row["RSI"]) else 50
            ma10_value = (
                last_row["SMA_10"] if not pd.isna(last_row["SMA_10"]) else current_price
            )
            ma20_value = (
                last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else current_price
            )
            ma50_value = (
                last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else current_price
            )
            ma200_value = (
                last_row["SMA_200"]
                if not pd.isna(last_row["SMA_200"])
                else current_price
            )

            # Lấy giá trị Ichimoku
            ichimoku_indicator = ta.trend.IchimokuIndicator(
                high=df["High"], low=df["Low"], window1=9, window2=26, window3=52
            )
            tenkan_sen_series = ichimoku_indicator.ichimoku_conversion_line()
            kijun_sen_series = ichimoku_indicator.ichimoku_base_line()
            senkou_span_a_series = ichimoku_indicator.ichimoku_a()
            senkou_span_b_series = ichimoku_indicator.ichimoku_b()

            chikou_span_series = df["Close"].shift(26)

            tenkan_sen = (
                tenkan_sen_series.iloc[-1]
                if len(tenkan_sen_series) > 0
                and not pd.isna(tenkan_sen_series.iloc[-1])
                else np.nan
            )
            kijun_sen = (
                kijun_sen_series.iloc[-1]
                if len(kijun_sen_series) > 0 and not pd.isna(kijun_sen_series.iloc[-1])
                else np.nan
            )
            senkou_span_a = (
                senkou_span_a_series.iloc[-1]
                if len(senkou_span_a_series) > 0
                and not pd.isna(senkou_span_a_series.iloc[-1])
                else np.nan
            )
            senkou_span_b = (
                senkou_span_b_series.iloc[-1]
                if len(senkou_span_b_series) > 0
                and not pd.isna(senkou_span_b_series.iloc[-1])
                else np.nan
            )
            chikou_span = (
                chikou_span_series.iloc[-1]
                if len(chikou_span_series) > 26
                and not pd.isna(chikou_span_series.iloc[-1])
                else np.nan
            )

            # Lấy giá trị RS
            rs_value = last_row["RS"] if symbol.upper() != "VNINDEX" else 1.0
            rs_point_value = (
                last_row["RS_Point"] if symbol.upper() != "VNINDEX" else 0.0
            )

            # Lấy giá trị Volume MA
            volume_ma_20 = (
                last_row["Volume_MA_20"]
                if "Volume_MA_20" in last_row
                else last_row["Volume"].rolling(20).mean().iloc[-1]
            )
            volume_ma_50 = (
                last_row["Volume_MA_50"]
                if "Volume_MA_50" in last_row
                else last_row["Volume"].rolling(50).mean().iloc[-1]
            )

            # Đọc dữ liệu từ file market_filtered.csv nếu có
            try:
                file_path = "market_filtered.csv"
                # 1. Đọc file Excel vào DataFrame
                market_df = pd.read_csv(file_path)

                # Kiểm tra xem cột 'ticker' có tồn tại không
                if "ticker" not in market_df.columns:
                    print(f"Lỗi: Không tìm thấy cột 'ticker' trong file {file_path}")
                    print(f"Các cột có trong file: {list(market_df.columns)}")
                else:
                    # 2. Lọc DataFrame theo symbol (không phân biệt chữ hoa/thường)
                    filtered_df = market_df[
                        market_df["ticker"].str.upper() == symbol.upper()
                    ]
                    output_csv_file = f"vnstocks_data/{symbol}_infor.csv"
                    filtered_df.to_csv(output_csv_file, index=False, encoding='utf-8')
                    # 3. Kiểm tra kết quả lọc
                    if not filtered_df.empty:
                        rs_value_3d = (
                            filtered_df["relative_strength_3d"].iloc[0]
                            if symbol.upper() != "VNINDEX"
                            and "relative_strength_3d" in filtered_df.columns
                            else 1.0
                        )
                        rs_value_1m = (
                            filtered_df["rel_strength_1m"].iloc[0]
                            if symbol.upper() != "VNINDEX"
                            and "rel_strength_1m" in filtered_df.columns
                            else 1.0
                        )
                        rs_value_3m = (
                            filtered_df["rel_strength_3m"].iloc[0]
                            if symbol.upper() != "VNINDEX"
                            and "rel_strength_3m" in filtered_df.columns
                            else 1.0
                        )
                        rs_value_1y = (
                            filtered_df["rel_strength_1y"].iloc[0]
                            if symbol.upper() != "VNINDEX"
                            and "rel_strength_1y" in filtered_df.columns
                            else 1.0
                        )
                        print(
                            f"Đã tìm thấy dữ liệu cho mã '{symbol}' trong file market_filtered.csv"
                        )
                    else:
                        print(
                            f"Không tìm thấy dữ liệu cho mã cổ phiếu '{symbol}' trong file."
                        )
                        rs_value_3d = 1.0
                        rs_value_1m = 1.0
                        rs_value_3m = 1.0
                        rs_value_1y = 1.0
            except FileNotFoundError:
                print(f"Lỗi: Không tìm thấy file '{file_path}'")
                rs_value_3d = 1.0
                rs_value_1m = 1.0
                rs_value_3m = 1.0
                rs_value_1y = 1.0
            except Exception as e:
                print(f"Lỗi khi đọc hoặc lọc file: {e}")
                rs_value_3d = 1.0
                rs_value_1m = 1.0
                rs_value_3m = 1.0
                rs_value_1y = 1.0

            # Tính điểm tổng hợp (phiên bản CÂN BẰNG HOÀN TOÀN)
            score = 50  # Điểm cơ bản

            # 1. Đường trung bình (MA) - 14 điểm (cân bằng với các chỉ báo khác)
            ma_score = 0
            # Đánh giá vị trí giá so với các MA
            if current_price > ma10_value:
                ma_score += 3.5
            if current_price > ma20_value:
                ma_score += 3.5
            if current_price > ma50_value:
                ma_score += 3.5
            if current_price > ma200_value:
                ma_score += 3.5

            # Đánh giá cấu trúc xu hướng
            if ma10_value > ma20_value > ma50_value > ma200_value:
                ma_score += 3.5  # Golden cross
            elif ma10_value < ma20_value < ma50_value < ma200_value:
                ma_score -= 3.5  # Death cross
            elif ma10_value > ma20_value and ma50_value > ma200_value:
                ma_score += 1.75  # Xu hướng tăng trung hạn
            elif ma10_value < ma20_value and ma50_value < ma200_value:
                ma_score -= 1.75  # Xu hướng giảm trung hạn

            score += ma_score

            # 2. RSI - 14 điểm (cân bằng với các chỉ báo khác)
            rsi_score = 0
            # Phân chia đều cho 7 mức RSI
            if rsi_value < 30:
                rsi_score += 14  # Quá bán mạnh
            elif 30 <= rsi_value < 40:
                rsi_score += 10  # Xu hướng tăng hình thành
            elif 40 <= rsi_value < 50:
                rsi_score += 7  # Xu hướng tăng nhẹ
            elif 50 <= rsi_value < 60:
                rsi_score += 3.5  # Trung tính
            elif 60 <= rsi_value < 70:
                rsi_score -= 3.5  # Xu hướng giảm nhẹ
            elif 70 <= rsi_value < 80:
                rsi_score -= 7  # Xu hướng giảm hình thành
            else:  # rsi_value >= 80
                rsi_score -= 14  # Quá mua mạnh

            score += rsi_score

            # 3. MACD - 14 điểm (cân bằng với các chỉ báo khác)
            macd_score = 0
            macd_value = last_row["MACD"]
            macd_signal = last_row["MACD_Signal"]
            macd_hist = last_row["MACD_Hist"]

            # Đánh giá trạng thái MACD
            if macd_value > macd_signal and macd_hist > 0:
                macd_score += 7  # Xu hướng tăng
            elif macd_value < macd_signal and macd_hist < 0:
                macd_score -= 7  # Xu hướng giảm

            # Đánh giá động lượng
            if len(df) > 1:
                macd_hist_prev = df["MACD_Hist"].iloc[-2]
                if macd_hist > macd_hist_prev:
                    macd_score += 3.5  # Động lượng tăng
                elif macd_hist < macd_hist_prev:
                    macd_score -= 3.5  # Động lượng giảm

            # Đánh giá cắt chéo
            if len(df) > 1:
                macd_prev = df["MACD"].iloc[-2]
                signal_prev = df["MACD_Signal"].iloc[-2]
                if macd_value > macd_signal and macd_prev <= signal_prev:
                    macd_score += 3.5  # Cắt vàng
                elif macd_value < macd_signal and macd_prev >= signal_prev:
                    macd_score -= 3.5  # Cắt chết

            score += macd_score

            # 4. Ichimoku Cloud - 14 điểm (CHỈ TẬP TRUNG VÀO 3 TRẠNG THÁI CHÍNH)
            ichimoku_score = 0
            if not (
                pd.isna(tenkan_sen)
                or pd.isna(kijun_sen)
                or pd.isna(senkou_span_a)
                or pd.isna(senkou_span_b)
            ):
                kumo_top = max(senkou_span_a, senkou_span_b)
                kumo_bottom = min(senkou_span_a, senkou_span_b)

                # GIÁ TRÊN MÂY - TÍN HIỆU TĂNG
                if current_price > kumo_top:
                    ichimoku_score += 14

                # GIÁ TRONG MÂY - TRUNG TÍNH
                elif current_price >= kumo_bottom and current_price <= kumo_top:
                    ichimoku_score += 0

                # GIÁ DƯỚI MÂY - TÍN HIỆU GIẢM
                elif current_price < kumo_bottom:
                    ichimoku_score -= 14

            score += ichimoku_score

            # 5. Volume - 14 điểm
            volume_score = 0
            if "Volume" in last_row and not pd.isna(last_row["Volume"]):
                current_volume = last_row["Volume"]

                # 1. So sánh với MA20 (4 điểm)
                vol_ratio_to_ma20 = (
                    current_volume / volume_ma_20
                    if volume_ma_20 and volume_ma_20 > 0
                    else 0
                )
                if vol_ratio_to_ma20 > 2.0:
                    volume_score += 4
                elif vol_ratio_to_ma20 > 1.5:
                    volume_score += 3
                elif vol_ratio_to_ma20 > 1.0:
                    volume_score += 1
                elif vol_ratio_to_ma20 < 0.5:
                    volume_score -= 2

                # 2. So sánh với MA50 (3 điểm)
                vol_ratio_to_ma50 = (
                    current_volume / volume_ma_50
                    if volume_ma_50 and volume_ma_50 > 0
                    else 0
                )
                if vol_ratio_to_ma50 > 2.0:
                    volume_score += 3
                elif vol_ratio_to_ma50 > 1.5:
                    volume_score += 2
                elif vol_ratio_to_ma50 > 1.0:
                    volume_score += 1
                elif vol_ratio_to_ma50 < 0.5:
                    volume_score -= 1

                # 3. Xu hướng volume 3 ngày (4 điểm)
                if len(df) > 2:
                    vol_prev = df["Volume"].iloc[-2]
                    vol_prev2 = df["Volume"].iloc[-3]
                    if current_volume > vol_prev > vol_prev2:
                        # Tăng mạnh
                        if current_volume / vol_prev2 > 1.5:
                            volume_score += 4
                        else:
                            volume_score += 2
                    elif current_volume < vol_prev < vol_prev2:
                        # Giảm mạnh
                        if current_volume / vol_prev2 < 0.7:
                            volume_score -= 4
                        else:
                            volume_score -= 2

                # 4. Volume bùng nổ (3 điểm) - So sánh MA20 hiện tại với MA20 của 20 ngày trước
                if len(df) > 40:
                    vol_ma20_prev = df["Volume"].iloc[-21:-1].mean()
                    if vol_ma20_prev > 0 and volume_ma_20 > 0:
                        vol_acc_ratio = volume_ma_20 / vol_ma20_prev
                        if vol_acc_ratio > 2.0:
                            volume_score += 3
                        elif vol_acc_ratio > 1.5:
                            volume_score += 1.5
                        elif vol_acc_ratio < 0.5:
                            volume_score -= 2

                # Giới hạn điểm volume trong khoảng hợp lý nếu cần
                volume_score = np.clip(volume_score, -14, 14)

            score += volume_score

            # 6. RS (Relative Strength) & RS_Point - 14 điểm (cân bằng với các chỉ báo khác)
            # Đảm bảo cả RS và RS_Point đều có ảnh hưởng như nhau đến tổng điểm (7 điểm mỗi cái)
            if symbol.upper() != "VNINDEX":
                rs_score = 0

                # --- Tính điểm cho RS (7 điểm) ---
                # So sánh với SMA ngắn hạn
                if rs_value > last_row.get("RS_SMA_10", rs_value):
                    rs_score += 3.5
                elif rs_value < last_row.get("RS_SMA_10", rs_value):
                    rs_score -= 3.5  # Thêm điều kiện ngược lại

                # So sánh với SMA trung hạn
                if rs_value > last_row.get("RS_SMA_50", rs_value):
                    rs_score += 3.5
                elif rs_value < last_row.get("RS_SMA_50", rs_value):
                    rs_score -= 3.5  # Thêm điều kiện ngược lại

                # --- Tính điểm cho RS_Point (7 điểm) ---
                # Đánh giá xu hướng RS_Point so với SMA20
                rs_point_sma20 = last_row.get("RS_Point_SMA_20", 0)
                if rs_point_value > rs_point_sma20:
                    rs_score += 3.5
                elif rs_point_value < rs_point_sma20:
                    rs_score -= 3.5  # Thêm điều kiện ngược lại

                # Đánh giá mức độ mạnh/yếu của RS_Point (so với 1.0)
                if rs_point_value > 1.0:  # Mạnh hơn thị trường
                    rs_score += 3.5
                elif (
                    rs_point_value < -1.0
                ):  # Yếu hơn thị trường đáng kể (giả sử ngưỡng -1.0)
                    rs_score -= 3.5
                # Ghi chú: Bạn có thể điều chỉnh ngưỡng -1.0 cho phù hợp hoặc bỏ điều kiện này nếu thấy chưa cần thiết.
                # Mục tiêu là đảm bảo tổng điểm cho RS_Point là 7.

                score += rs_score  # Cộng điểm RS & RS_Point vào tổng điểm

            # 7. Bollinger Bands - 14 điểm (cân bằng với các chỉ báo khác)
            bb_score = 0
            bb_upper = last_row["BB_Upper"]
            bb_lower = last_row["BB_Lower"]
            # Tính khoảng cách từ giá đến các dải
            if not pd.isna(bb_upper) and not pd.isna(bb_lower) and bb_upper > bb_lower:
                bb_width = bb_upper - bb_lower
                price_to_upper = (bb_upper - current_price) / bb_width
                price_to_lower = (current_price - bb_lower) / bb_width

                # Đánh giá vị trí giá trong kênh
                if price_to_lower < 0.15:  # Giá gần dải dưới
                    bb_score += 7
                elif price_to_lower < 0.3:  # Giá dưới trung bình
                    bb_score += 3.5

                if price_to_upper < 0.15:  # Giá gần dải trên
                    bb_score -= 7
                elif price_to_upper < 0.3:  # Giá trên trung bình
                    bb_score -= 3.5

                # Đánh giá độ rộng kênh
                if (
                    len(df) > 1
                    and not pd.isna(df["BB_Upper"].iloc[-2])
                    and not pd.isna(df["BB_Lower"].iloc[-2])
                ):
                    bb_width_prev = df["BB_Upper"].iloc[-2] - df["BB_Lower"].iloc[-2]
                    if bb_width > bb_width_prev * 1.1:  # Kênh đang mở rộng
                        bb_score -= 1.75
                    elif bb_width < bb_width_prev * 0.9:  # Kênh đang thu hẹp
                        bb_score += 1.75

            score += bb_score

            # Chuẩn hóa điểm số về thang 0-100
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
            elif score >= 55:
                signal = "TĂNG MẠNH"
                recommendation = "GIỮ - TĂNG"
            elif score >= 45:
                signal = "TRUNG LẬP"
                recommendation = "GIỮ"
            elif score >= 35:
                signal = "GIẢM MẠNH"
                recommendation = "GIỮ - GIẢM"
            elif score >= 20:
                signal = "BÁN"
                recommendation = "BÁN"
            else:
                signal = "BÁN MẠNH"
                recommendation = "BÁN MẠNH"

            # In ra tín hiệu cuối cùng
            analysis_date = df.index[-1].strftime("%d/%m/%Y")
            print(f"📊 TÍN HIỆU GIAO DỊCH CUỐI CÙNG CHO {symbol} ({analysis_date}):")
            print(f" - Giá hiện tại: {current_price:,.2f} VND")
            print(f" - Đường trung bình:")
            print(
                f" * MA10: {ma10_value:,.2f}| MA20: {ma20_value:,.2f}| MA50: {ma50_value:,.2f}| MA200: {ma200_value:,.2f}"
            )
            print(f" - Chỉ báo dao động:")
            print(f" * RSI (14): {rsi_value:.2f}")
            print(
                f" * MACD: {macd_value:.2f}| Signal: {macd_signal:.2f}| Histogram: {macd_hist:.2f}"
            )
            print(f" * Bollinger Bands: Trên: {bb_upper:,.2f}| Dưới: {bb_lower:,.2f}")
            if symbol.upper() != "VNINDEX":
                print(f" - Sức mạnh tương đối (RS):")
                print(f" * RS: {rs_value}")
                print(f" * RS_Point: {rs_point_value:.2f}")
                print(f" * RS3D: {rs_value_3d}")
                print(f" * RS1M: {rs_value_1m}")
                print(f" * RS3M: {rs_value_3m}")
                print(f" * RS1y: {rs_value_1y}")
            try:
                print(f" - Mô hình Ichimoku:")
                print(f" * Tenkan-sen (Chuyển đổi): {tenkan_sen:.2f}")
                print(f" * Kijun-sen (Cơ sở): {kijun_sen:.2f}")
                print(f" * Senkou Span A (Leading Span A): {senkou_span_a:.2f}")
                print(f" * Senkou Span B (Leading Span B): {senkou_span_b:.2f}")
                print(f" * Chikou Span (Trễ): {chikou_span:.2f}")
                print(f" * Điểm Ichimoku: ~{ichimoku_score:.2f}")
            except:
                print(f" - Ichimoku: Không có đủ dữ liệu.")
            print(f" - Khối lượng:")
            print(f" * Khối lượng hiện tại: {last_row.get('Volume', 'N/A')}")
            print(f" * MA Khối lượng (20): {volume_ma_20:,.2f}")
            print(f" * MA Khối lượng (50): {volume_ma_50:,.2f}")
            print(f" 🎯 ĐỀ XUẤT CUỐI CÙNG: {recommendation}")
            print(f" 📊 TỔNG ĐIỂM PHÂN TÍCH: {score:.1f}/100")
            print(f" 📈 TÍN HIỆU: {signal}")

            # Trả về kết quả phân tích kỹ thuật (không có dự báo AI)
            return {
                "signal": signal,
                "score": float(score),
                "current_price": float(current_price),
                "rsi_value": float(rsi_value),
                "ma10": float(ma10_value),
                "ma20": float(ma20_value),
                "ma50": float(ma50_value),
                "ma200": float(ma200_value),
                "rs": float(rs_value),
                "rs_point": float(rs_point_value),
                "recommendation": recommendation,
                "open": safe_float(last_row.get("Open")),
                "high": safe_float(last_row.get("High")),
                "low": safe_float(last_row.get("Low")),
                "volume": safe_float(last_row.get("Volume")),
                "volume_ma_20": safe_float(volume_ma_20),
                "volume_ma_50": safe_float(volume_ma_50),
                "macd": safe_float(macd_value),
                "macd_signal": safe_float(macd_signal),
                "macd_hist": safe_float(macd_hist),
                "bb_upper": safe_float(bb_upper),
                "bb_lower": safe_float(bb_lower),
                "ichimoku_tenkan_sen": safe_float(tenkan_sen),
                "ichimoku_kijun_sen": safe_float(kijun_sen),
                "ichimoku_senkou_span_a": safe_float(senkou_span_a),
                "ichimoku_senkou_span_b": safe_float(senkou_span_b),
                "ichimoku_chikou_span": safe_float(chikou_span),
                "rs_sma_10": safe_float(last_row.get("RS_SMA_10")),
                "relative_strength_3d": safe_float(rs_value_3d),
                "relative_strength_1m": safe_float(rs_value_1m),
                "relative_strength_3m": safe_float(rs_value_3m),
                "relative_strength_1y": safe_float(rs_value_1y)
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_sma_20": safe_float(last_row.get("RS_SMA_20"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_sma_50": safe_float(last_row.get("RS_SMA_50"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_sma_200": safe_float(last_row.get("RS_SMA_200"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50"))
                if symbol.upper() != "VNINDEX"
                else None,
                "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200"))
                if symbol.upper() != "VNINDEX"
                else None,
                "forecast_dates": [],
                "forecast_prices": [],
                "forecast_plot_path": "",
            }
        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu cho {symbol}: {str(e)}")
            traceback.print_exc()
            return {
                "signal": "LỖI",
                "score": 50,
                "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rsi_value": 50,
                "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0,
                "rs": 1.0,
                "rs_point": 0,
                "recommendation": "KHÔNG XÁC ĐỊNH",
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None,
                "bb_upper": None,
                "bb_lower": None,
                "volume_ma_20": None,
                "volume_ma_50": None,
                "ichimoku_tenkan_sen": None,
                "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None,
                "ichimoku_senkou_span_b": None,
                "ichimoku_chikou_span": None,
                "rs_sma_10": None,
                "rs_sma_20": None,
                "rs_sma_50": None,
                "rs_sma_200": None,
                "rs_point_sma_10": None,
                "rs_point_sma_20": None,
                "rs_point_sma_50": None,
                "rs_point_sma_200": None,
                "relative_strength_3d": None,
                "relative_strength_1m": None,
                "relative_strength_3m": None,
                "relative_strength_1y": None,
                "forecast_dates": [],
                "forecast_prices": [],
                "forecast_plot_path": "",
            }
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng khi phân tích {symbol}: {str(e)}")
        traceback.print_exc()
        return {
            "signal": "LỖI",
            "score": 50,
            "current_price": 0,
            "rsi_value": 0,
            "ma10": 0,
            "ma20": 0,
            "ma50": 0,
            "ma200": 0,
            "rs": 1.0,
            "rs_point": 0,
            "recommendation": "KHÔNG XÁC ĐỊNH",
            "open": None,
            "high": None,
            "low": None,
            "volume": None,
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "bb_upper": None,
            "bb_lower": None,
            "volume_ma_20": None,
            "volume_ma_50": None,
            "ichimoku_tenkan_sen": None,
            "ichimoku_kijun_sen": None,
            "ichimoku_senkou_span_a": None,
            "ichimoku_senkou_span_b": None,
            "ichimoku_chikou_span": None,
            "rs_sma_10": None,
            "rs_sma_20": None,
            "rs_sma_50": None,
            "rs_sma_200": None,
            "rs_point_sma_10": None,
            "rs_point_sma_20": None,
            "rs_point_sma_50": None,
            "rs_point_sma_200": None,
            "relative_strength_3d": None,
            "relative_strength_1m": None,
            "relative_strength_3m": None,
            "relative_strength_1y": None,
            "forecast_dates": [],
            "forecast_prices": [],
            "forecast_plot_path": "",
        }


# --- Phân tích bằng AI ---
def analyze_with_gemini(
    symbol: str, trading_signal: dict, financial_data_statement: pd.DataFrame
) -> str:
    """Phân tích tổng hợp với AI, xử lý giá trị None an toàn và kèm theo dữ liệu giá"""
    try:
        # --- MỚI: Đọc dữ liệu từ file csv ---
        csv_file_path = f"vnstocks_data/{symbol}_data.csv"
        infor_csv_file_path = f"vnstocks_data/{symbol}_infor.csv"
        historical_data_str = "Không có dữ liệu lịch sử."
        infor_data_str = "Không có dữ liệu lịch sử."
        if os.path.exists(csv_file_path):
            try:
                # Đọc file csv
                df_history = pd.read_csv(csv_file_path)
                df_infor_history = pd.read_csv(infor_csv_file_path)
                # Chuyển DataFrame thành chuỗi (string) định dạng bảng dễ đọc
                # Có thể điều chỉnh `float_format` nếu cần
                historical_data_str = df_history.to_string(
                    index=False, float_format="{:.2f}".format
                )
                infor_data_str = df_infor_history.to_string(
                    index=False, float_format="{:.2f}".format
                )
                # print(historical_data_str)
                print(
                    f"✅ Đã đọc dữ liệu lịch sử từ '{csv_file_path}' để gửi tới Gemini."
                )
                print(
                    f"✅ Đã đọc dữ liệu lịch sử từ '{infor_csv_file_path}' để gửi tới Gemini."
                )
            except Exception as e:
                print(
                    f"⚠️ Cảnh báo: Không thể đọc file '{csv_file_path}' để gửi tới Gemini: {e}"
                )
                print(
                    f"⚠️ Cảnh báo: Không thể đọc file '{infor_csv_file_path}' để gửi tới Gemini: {e}"
                )
                historical_data_str = "Không thể đọc dữ liệu lịch sử."

        else:
            print(
                f"⚠️ Cảnh báo: File '{csv_file_path}' không tồn tại để gửi tới Gemini."
            )

        # Hàm để chuyển giá trị thành chuỗi, nếu None thì trả về "N/A"
        def to_str(value):
            return str(value) if value is not None else "N/A"

        # Lấy các giá trị trực tiếp từ trading_signal
        current_price = trading_signal.get("current_price")
        rsi_value = trading_signal.get("rsi_value")
        ma10 = trading_signal.get("ma10")
        ma20 = trading_signal.get("ma20")
        ma50 = trading_signal.get("ma50")
        ma200 = trading_signal.get("ma200")
        bb_upper = trading_signal.get("bb_upper")
        bb_lower = trading_signal.get("bb_lower")
        macd = trading_signal.get("macd")
        macd_signal = trading_signal.get("macd_signal")
        hist = trading_signal.get("macd_hist")
        tenkan_val = trading_signal.get("ichimoku_tenkan_sen")
        kijun_val = trading_signal.get("ichimoku_kijun_sen")
        senkou_a_val = trading_signal.get("ichimoku_senkou_span_a")
        senkou_b_val = trading_signal.get("ichimoku_senkou_span_b")
        chikou_val = trading_signal.get("ichimoku_chikou_span")
        volume = trading_signal.get("volume")
        volume_ma_20 = trading_signal.get("volume_ma_20")
        volume_ma_50 = trading_signal.get("volume_ma_50")

        # Tạo prompt với các giá trị trực tiếp
        prompt = f"""
        Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy đánh giá mã {symbol}:
        1. Phân tích kỹ thuật:
        - Giá: {to_str(current_price)}
        - RSI: {to_str(rsi_value)}
        - MA: {to_str(ma10)} (10), {to_str(ma20)} (20), {to_str(ma50)} (50), {to_str(ma200)} (200)
        - Bollinger bands Up: {to_str(bb_upper)}, Bollinger bands Lower: {to_str(bb_lower)}
        - MACD: {to_str(macd)}, Signal: {to_str(macd_signal)}, Histogram: {to_str(hist)}
        - Ichimoku: Tenkan: {to_str(tenkan_val)} | Kijun: {to_str(kijun_val)} | Senkou_A: {to_str(senkou_a_val)} | Senkou_B: {to_str(senkou_b_val)} | Chikou: {to_str(chikou_val)}
        - Khối lượng: {to_str(volume)}
        - Khối lượng trung bình 20 ngày: {to_str(volume_ma_20)}
        - Khối lượng trung bình 50 ngày: {to_str(volume_ma_50)}
        """

        if symbol.upper() != "VNINDEX":
            rs = trading_signal.get("rs")
            rs_point = trading_signal.get("rs_point")

            prompt += f"""
        - RS (Sức mạnh tương đối so với thị trường): C / VNINDEX → {to_str(rs)}
            * RS_SMA_10: {trading_signal.get("rs_sma_10", "N/A")}
            * RS_SMA_20: {trading_signal.get("rs_sma_20", "N/A")}
            * RS_SMA_50: {trading_signal.get("rs_sma_50", "N/A")}
            * RS_SMA_200: {trading_signal.get("rs_sma_200", "N/A")}

        - RS_Point (điểm sức mạnh IBD): 0.4*ROC(63) + 0.2*ROC(126) + 0.2*ROC(189) + 0.2*ROC(252) → {to_str(rs_point)}
            * SMA_10: {to_str(trading_signal.get("rs_point_sma_10"))}
            * SMA_20: {to_str(trading_signal.get("rs_point_sma_20"))}
            * SMA_50: {to_str(trading_signal.get("rs_point_sma_50"))}
            * SMA_200: {to_str(trading_signal.get("rs_point_sma_200"))}
        
        - Sức mạnh RS từ TCBS:
            * RS 3D: {to_str(trading_signal.get("relative_strength_3d"))}
            * RS 1M: {to_str(trading_signal.get("relative_strength_1m"))}
            * RS 3M: {to_str(trading_signal.get("relative_strength_3m"))}
            * RS 1Y: {to_str(trading_signal.get("relative_strength_1y"))}
"""

        if financial_data_statement is not None and not financial_data_statement.empty:
            prompt += "2. Tình hình tài chính (csv).\n"
            if (
                financial_data_statement is not None
                and not financial_data_statement.empty
            ):
                prompt += f"Báo cáo tài chính:\n{financial_data_statement.to_string(index=False)}\n"
        else:
            prompt += "2. Không có dữ liệu tài chính.\n"

        prompt += f"""
        3. Dữ liệu lịch sử giá (csv).\n
        {historical_data_str}
        4. Dữ liệu chung từ TCBS.\n
        {infor_data_str}
"""

        prompt += """
        Nhiệm vụ của bạn:
        - Có thể sử dụng thông tin cung cấp được phân điểm mua đẹp và nhận định báo cáo tài chính.
        - Phân tích kỹ thuật theo Wyckoff, VSA/VPA, Minervini, Alexander Elder: hành động giá, khối lượng, cấu trúc xu hướng, điểm mua/bán.
        - Phân tích cơ bản theo Warren Buffett, Charlie Munger, Peter Lynch, Seth Klarman: tăng trưởng, lợi nhuận, biên lợi nhuận, ROE, nợ, dòng tiền, hàng tồn kho, tài sản cố định, người mua trả trước...
        - Đánh giá mô hình kỹ thuật (nếu có). 
        - Từ dữ liệu lịch sử giá có thể thêm nhận định từ các chỉ báo từ AI tự phân tích.
        - Nhận định xu hướng 1 tuần 1 tháng 3 tháng sắp tới.
        - Kết luận cuối cùng phải rõ ràng, súc tích: **MUA MẠNH / MUA / GIỮ / BÁN / BÁN MẠNH**
        - Chấm điểm từ 1 đến 10 cổ phiếu mua vị thế giá hiện tại.
        - Trình bày phân tích ngắn gọn, chuyên nghiệp, dễ hành động.
"""

        with open("prompt.txt", "w", encoding="utf-8") as file:
            file.write(prompt)

        print(f"✅ Đã lưu nội dung vào file.")

        print(f"📤 Đang upload file dữ liệu giá...")
        fileData = genai.upload_file(path=f"vnstocks_data/{symbol}_data.csv")
        print(f"✅ Upload file dữ liệu giá thành công: {fileData.uri}")

        print(f"📤 Đang upload file báo cáo tài chính...")
        fileStatement = genai.upload_file(
            path=f"vnstocks_data/{symbol}_financial_statements.csv"
        )
        print(f"✅ Upload file báo cáo tài chính thành công: {fileStatement.uri}")

        print(f"📤 Đang upload file tổng quan từ TCBS...")
        fileInfor = genai.upload_file(path=f"vnstocks_data/{symbol}_infor.csv")
        print(f"✅ Upload file dữ liệu TCBS thành công: {fileInfor.uri}")

        # Gọi AI sử dụng
        print(f"🤖 Đang yêu cầu phân tích từ AI...")

        # completion = client.chat.completions.create(
        #     extra_body={},
        #     model="z-ai/glm-4.5-air:free",
        #     messages=[{"role": "user", "content": prompt}],
        # )

        # # In ra câu trả lời
        # if completion.choices and completion.choices[0].message.content:
        #     print("Trả lời từ AI:")
        #     print(completion.choices[0].message.content)
        # else:
        #     print("Không có nội dung trả lời từ mô hình.")
        #     print(completion)

        model = genai.GenerativeModel(model_name="gemini-2.5-flash")
        response = model.generate_content(
            contents=[
                prompt,
                fileData,
                fileStatement,
                fileInfor 
            ],
        )

        if response and response.text:
            return response.text.strip()
        else:
            return "Không nhận được phản hồi từ AI."

    except Exception as e:
        print(f"❌ Lỗi khi phân tích bằng AI cho {symbol}: {str(e)}")
        print("Chi tiết lỗi:")
        traceback.print_exc()
        return "Không thể tạo phân tích bằng AI tại thời điểm này."

    except FileNotFoundError as e:
        print(f"❌ Không tìm thấy file cho {symbol}: {str(e)}")
        return "Không tìm thấy dữ liệu cần thiết để phân tích."


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
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"❌ Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    if len(df_processed) < 100:
        print(
            f"❌ Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_processed)} điểm)"
        )
        return None
    print(f"📈 Đang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_processed)
    print(f"🤖 Đang phân tích bằng AI ...")
    gemini_analysis = analyze_with_gemini(
        symbol, trading_signal, financial_data_statement
    )
    print(f"\n{'=' * 20} KẾT QUẢ PHÂN TÍCH CHO MÃ {symbol} {'=' * 20}")
    print(f"💰 Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"📈 Tín hiệu: {trading_signal['signal']}")
    print(f"🎯 Đề xuất: {trading_signal['recommendation']}")
    print(f"📊 Điểm phân tích: {trading_signal['score']:.2f}/100")
    if symbol.upper() != "VNINDEX":
        print(f"📊 RS (so với VNINDEX: {trading_signal['rs']:.4f}")
        print(f"📊 RS_Point: {trading_signal['rs_point']:.2f}")
    print(f"\n--- PHÂN TÍCH TỔNG HỢP TỪ AI ---")
    print(gemini_analysis)
    print(f"{'=' * 60}\n")

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
        "rs": safe_float(trading_signal.get("rs"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_point": safe_float(trading_signal.get("rs_point"))
        if symbol.upper() != "VNINDEX"
        else None,
        # Thêm các chỉ báo còn thiếu
        "open": safe_float(trading_signal.get("open")),
        "high": safe_float(trading_signal.get("high")),
        "low": safe_float(trading_signal.get("low")),
        "volume": safe_float(trading_signal.get("volume")),
        "macd": safe_float(trading_signal.get("macd")),
        "macd_signal": safe_float(trading_signal.get("macd_signal")),
        "macd_hist": safe_float(trading_signal.get("macd_hist")),
        "bb_upper": safe_float(trading_signal.get("bb_upper")),
        "bb_lower": safe_float(trading_signal.get("bb_lower")),
        "volume_ma": safe_float(trading_signal.get("volume_ma")),
        "ichimoku_tenkan_sen": safe_float(trading_signal.get("ichimoku_tenkan_sen")),
        "ichimoku_kijun_sen": safe_float(trading_signal.get("ichimoku_kijun_sen")),
        "ichimoku_senkou_span_a": safe_float(
            trading_signal.get("ichimoku_senkou_span_a")
        ),
        "ichimoku_senkou_span_b": safe_float(
            trading_signal.get("ichimoku_senkou_span_b")
        ),
        "ichimoku_chikou_span": safe_float(trading_signal.get("ichimoku_chikou_span")),
        "rs_sma_10": safe_float(trading_signal.get("rs_sma_10"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_sma_20": safe_float(trading_signal.get("rs_sma_20"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_sma_50": safe_float(trading_signal.get("rs_sma_50"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_sma_200": safe_float(trading_signal.get("rs_sma_200"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_point_sma_10": safe_float(trading_signal.get("rs_point_sma_10"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_point_sma_20": safe_float(trading_signal.get("rs_point_sma_20"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_point_sma_50": safe_float(trading_signal.get("rs_point_sma_50"))
        if symbol.upper() != "VNINDEX"
        else None,
        "rs_point_sma_200": safe_float(trading_signal.get("rs_point_sma_200"))
        if symbol.upper() != "VNINDEX"
        else None,
        "gemini_analysis": gemini_analysis,
    }
    # report.update(trading_signal) # Không cập nhật toàn bộ trading_signal vì có thể gây trùng lặp key và lỗi JSON
    with open(f"vnstocks_data/{symbol}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")
    return report


# --- Lọc cổ phiếu ---
def filter_stocks_low_pe_high_cap(min_market_cap=500):
    """Lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao."""
    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if df is None or df.empty:
            print("❌ Không thể lấy dữ liệu danh sách công ty niêm yết.")
            return None

        # --- Áp dụng các điều kiện lọc ---
        # 1. Vốn hóa thị trường >= ngưỡng tối thiểu
        condition1 = df["market_cap"] >= min_market_cap

        # 2. P/E dương và thấp hơn 20
        condition2_pe = (df["pe"] > 0) & (df["pe"] < 20)

        # 3. P/B dương
        condition3_pb = df["pb"] > 0

        # 4. Tăng trưởng doanh thu quý gần nhất > 0
        condition4_rev_growth_last = df["last_quarter_revenue_growth"] > 0

        # 5. Tăng trưởng doanh thu quý trước đó > 0
        condition5_rev_growth_second = df["second_quarter_revenue_growth"] > 0

        # 6. Tăng trưởng lợi nhuận quý gần nhất > 0
        condition6_profit_growth_last = df["last_quarter_profit_growth"] > 0

        # 7. Tăng trưởng lợi nhuận quý trước đó > 0
        condition7_profit_growth_second = df["second_quarter_profit_growth"] > 0

        # 8. PEG (Forward) < 1 hoặc NaN (sử dụng pd.isna())
        # Giả sử PEG âm không hợp lệ hoặc không có sẵn
        condition8_peg_forward = (
            (df["peg_forward"] < 1) & (df["peg_forward"] >= 0)
        ) | pd.isna(df["peg_forward"])  # Sử dụng pd.isna() thay cho pd.isnull()

        # 9. PEG (Trailing) < 1 hoặc NaN (sử dụng pd.isna())
        condition9_peg_trailing = (
            (df["peg_trailing"] < 1) & (df["peg_trailing"] >= 0)
        ) | pd.isna(df["peg_trailing"])  # Sử dụng pd.isna() thay cho pd.isnull()

        # --- Kết hợp tất cả các điều kiện ---
        filtered_conditions = (
            condition1
            & condition2_pe
            & condition3_pb
            & condition4_rev_growth_last
            & condition5_rev_growth_second
            & condition6_profit_growth_last
            & condition7_profit_growth_second
            & condition8_peg_forward
            & condition9_peg_trailing
        )

        # Lọc DataFrame dựa trên các điều kiện kết hợp
        filtered_df = df[filtered_conditions]

        # --- Kiểm tra kết quả sau khi lọc ---
        if filtered_df.empty:
            print("⚠️ Không tìm thấy cổ phiếu nào đáp ứng tất cả các tiêu chí lọc.")
            # Có thể trả về DataFrame rỗng thay vì None nếu muốn nhất quán kiểu trả về
            # return filtered_df
            return None  # Trả về None như yêu cầu ban đầu nếu không có kết quả

        # --- Lưu kết quả vào file csv ---
        # Đổi tên file để phân biệt rõ hơn
        output_csv_file = "market_filtered.csv"
        output_csv_file_pe = "market_filtered_pe.csv"
        filtered_df.to_csv(output_csv_file_pe, index=False, encoding='utf-8')
        df.to_csv(output_csv_file, index=False)
        filtered_df.to_csv(output_csv_file_pe, index=False, encoding='utf-8')
        print(
            f"✅ Đã lưu danh sách cổ phiếu được lọc ({len(filtered_df)} mã) vào '{output_csv_file_pe}'"
        )

    except Exception as e:
        print(f"❌ Đã xảy ra lỗi trong quá trình lọc cổ phiếu: {e}")
        # traceback.print_exc() # Bỏ comment nếu muốn xem chi tiết lỗi
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
    print(
        "\nNhập mã cổ phiếu để phân tích riêng lẻ (ví dụ: VCB, FPT) hoặc 'exit' để thoát"
    )
    user_input = input("Nhập mã cổ phiếu để phân tích: ").strip().upper()
    if user_input and user_input.lower() != "exit":
        tickers = [ticker.strip() for ticker in user_input.split(",")]
        for ticker in tickers:
            if ticker:
                print(f"\nPhân tích mã: {ticker}")
                analyze_stock(ticker)
        print(
            "\n✅ Hoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'."
        )
    else:
        print("👋 Thoát chương trình.")


if __name__ == "__main__":
    main()
