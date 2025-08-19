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
from vnstock import Vnstock
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
        stock = Vnstock().stock(symbol= symbol, source='VCI')
        df = stock.quote.history(start=GLOBAL_START_DATE, end=GLOBAL_END_DATE, interval="1D")
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

def safe_rename(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    # Lọc chỉ giữ lại những key có tồn tại trong df
    valid_mapping = {k: v for k, v in mapping.items() if k in df.columns}
    return df.rename(columns=valid_mapping)

def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính từ VCI và lưu vào file CSV."""
    try:

            # Khởi tạo đối tượng finance 
            stock = Vnstock().stock(symbol=symbol)

            # Lấy 4 loại báo cáo tài chính
            df_ratio = stock.finance.ratio(period='quarter',flatten_columns=True)
            df_bs = stock.finance.balance_sheet(period='quarter')
            df_is = stock.finance.income_statement(period='quarter')
            df_cf = stock.finance.cash_flow(period='quarter')
    
            financial_data = df_bs.merge(df_is, on=["yearReport", "lengthReport"], how="outer") \
                    .merge(df_cf, on=["yearReport", "lengthReport"], how="outer")

            return df_ratio, financial_data
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
    df["BB_Upper"] = ta.volatility.bollinger_hband(df["Close"])
    df["BB_Middle"] = ta.volatility.bollinger_mavg(df["Close"])
    df["BB_Lower"] = ta.volatility.bollinger_lband(df["Close"])
    df["Volume_MA_20"] = df["Volume"].rolling(window=20).mean()
    df["Volume_MA_50"] = df["Volume"].rolling(window=50).mean()
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
                "volume_ma_20": None,
                "volume_ma_50": None,
                "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, 
                "ichimoku_chikou_span": None,
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
            ma10_value = (last_row["SMA_10"] if not pd.isna(last_row["SMA_10"]) else current_price)
            ma20_value = (last_row["SMA_20"] if not pd.isna(last_row["SMA_20"]) else current_price)
            ma50_value = (last_row["SMA_50"] if not pd.isna(last_row["SMA_50"]) else current_price)
            ma200_value = (last_row["SMA_200"] if not pd.isna(last_row["SMA_200"]) else current_price)
            
            # Lấy giá trị Ichimoku
            tenkan_sen = df["Close"].rolling(9).mean().iloc[-1]
            kijun_sen = df["Close"].rolling(26).mean().iloc[-1]
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2) if not pd.isna(tenkan_sen) and not pd.isna(kijun_sen) else np.nan
            senkou_span_b = df["Close"].rolling(52).mean().shift(26).iloc[-1] if len(df) >= 78 else np.nan
            chikou_span = df["Close"].shift(26).iloc[-1] if len(df) > 26 else np.nan
            
            # Lấy giá trị RS
            rs_value = last_row["RS"] if symbol.upper() != "VNINDEX" else 1.0
            rs_point_value = last_row["RS_Point"] if symbol.upper() != "VNINDEX" else 0.0
            rs_point_252_value = last_row["RS_Point_252"] if symbol.upper() != "VNINDEX" else 0.0
            
            # Lấy giá trị Volume MA
            volume_ma_20 = last_row["Volume_MA_20"] if "Volume_MA_20" in last_row else last_row["Volume"].rolling(20).mean().iloc[-1]
            volume_ma_50 = last_row["Volume_MA_50"] if "Volume_MA_50" in last_row else last_row["Volume"].rolling(50).mean().iloc[-1]
            
            # Tính điểm tổng hợp (phiên bản CÂN BẰNG HOÀN TOÀN)
            score = 50  # Điểm cơ bản
            
            # 1. Đường trung bình (MA) - 14 điểm (cân bằng với các chỉ báo khác)
            ma_score = 0
            # Đánh giá vị trí giá so với các MA
            if current_price > ma10_value: ma_score += 3.5
            if current_price > ma20_value: ma_score += 3.5
            if current_price > ma50_value: ma_score += 3.5
            if current_price > ma200_value: ma_score += 3.5
            
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
                rsi_score += 7   # Xu hướng tăng nhẹ
            elif 50 <= rsi_value < 60:
                rsi_score += 3.5 # Trung tính
            elif 60 <= rsi_value < 70:
                rsi_score -= 3.5 # Xu hướng giảm nhẹ
            elif 70 <= rsi_value < 80:
                rsi_score -= 7   # Xu hướng giảm hình thành
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
            if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
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
            
            # 5. Volume - 14 điểm (cân bằng với các chỉ báo khác)
            volume_score = 0
            if "Volume" in last_row and not pd.isna(last_row["Volume"]):
                # So sánh với MA20 (ngắn hạn)
                if last_row["Volume"] > volume_ma_20:
                    volume_score += 7
                elif last_row["Volume"] < volume_ma_20 * 0.7:
                    volume_score -= 3.5
                
                # So sánh với MA50 (dài hạn)
                if last_row["Volume"] > volume_ma_50:
                    volume_score += 3.5
                elif last_row["Volume"] < volume_ma_50 * 0.7:
                    volume_score -= 3.5
                
                # Đánh giá xu hướng volume
                if len(df) > 2:
                    vol_prev = df["Volume"].iloc[-2]
                    vol_prev2 = df["Volume"].iloc[-3]
                    if last_row["Volume"] > vol_prev > vol_prev2:
                        volume_score += 3.5  # Volume tăng dần
                    elif last_row["Volume"] < vol_prev < vol_prev2:
                        volume_score -= 3.5  # Volume giảm dần
            
            score += volume_score
            
            # 6. RS (Relative Strength) - 14 điểm (cân bằng với các chỉ báo khác)
            if symbol.upper() != "VNINDEX":
                rs_score = 0
                # So sánh với SMA ngắn hạn
                if rs_value > last_row.get("RS_SMA_10", rs_value):
                    rs_score += 3.5
                
                # So sánh với SMA trung hạn
                if rs_value > last_row.get("RS_SMA_50", rs_value):
                    rs_score += 3.5
                
                # Đánh giá xu hướng RS_Point
                rs_point_sma20 = last_row.get("RS_Point_SMA_20", 0)
                if rs_point_value > rs_point_sma20:
                    rs_score += 3.5
                
                # Đánh giá xu hướng RS_Point_252
                rs_point_252_sma50 = last_row.get("RS_Point_252_SMA_50", 0)
                if rs_point_252_value > rs_point_252_sma50:
                    rs_score += 3.5
                
                score += rs_score
            
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
                if len(df) > 1 and not pd.isna(df["BB_Upper"].iloc[-2]) and not pd.isna(df["BB_Lower"].iloc[-2]):
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
            print(f" * MA10: {ma10_value:,.2f}| MA20: {ma20_value:,.2f}| MA50: {ma50_value:,.2f}| MA200: {ma200_value:,.2f}")
            print(f" - Chỉ báo dao động:")
            print(f" * RSI (14): {rsi_value:.2f}")
            print(f" * MACD: {macd_value:.2f}| Signal: {macd_signal:.2f}| Histogram: {macd_hist:.2f}")
            print(f" * Bollinger Bands: Trên: {bb_upper:,.2f}| Dưới: {bb_lower:,.2f}")
            if symbol.upper() != "VNINDEX":
                print(f" - Sức mạnh tương đối (RS):")
                print(f" * RS: {rs_value:.4f}")
                print(f" * RS_Point: {rs_point_value:.2f}")
                print(f" * RS_Point_252: {rs_point_252_value:.2f}")
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
                "rs_point_252": float(rs_point_252_value),
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
                "forecast_dates": [], 
                "forecast_prices": [], 
                "forecast_plot_path": ""
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
                "volume_ma_20": None,
                "volume_ma_50": None,
                "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
                "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, 
                "ichimoku_chikou_span": None,
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
            "volume_ma_20": None,
            "volume_ma_50": None,
            "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None,
            "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, 
            "ichimoku_chikou_span": None,
            "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None,
            "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None,
            "rs_point_252_sma_10": None, "rs_point_252_sma_20": None,
            "rs_point_252_sma_50": None, "rs_point_252_sma_200": None,
            "forecast_dates": [], "forecast_prices": [], "forecast_plot_path": ""
        }

# --- Phân tích bằng Google Gemini ---
def analyze_with_gemini(symbol: str, trading_signal: dict, financial_data_ratio: pd.DataFrame, financial_data_statement: pd.DataFrame) -> str:
    """Phân tích tổng hợp với Google Gemini, xử lý giá trị None an toàn và kèm theo dữ liệu giá"""
    try:
        # Hàm hỗ trợ định dạng an toàn
        def safe_format(value, fmt=".2f", default="N/A"):
            if value is None or pd.isna(value):
                return default
            try:
                return f"{float(value):{fmt}}"
            except (TypeError, ValueError):
                return default

        # --- MỚI: Đọc dữ liệu từ file CSV ---
        csv_file_path = f"vnstocks_data/{symbol}_data.csv"
        historical_data_str = "Không có dữ liệu lịch sử."
        if os.path.exists(csv_file_path):
            try:
                # Đọc file CSV
                df_history = pd.read_csv(csv_file_path)
                # Giới hạn số dòng dữ liệu gửi đi để tránh vượt quá giới hạn token của API
                # Ví dụ: chỉ lấy 100 dòng cuối cùng
                df_history_limited = df_history
                # Chuyển DataFrame thành chuỗi (string) định dạng bảng dễ đọc
                # Có thể điều chỉnh `float_format` nếu cần
                historical_data_str = df_history_limited.to_string(index=False, float_format="{:.2f}".format)
                #print(historical_data_str)
                print(f"✅ Đã đọc dữ liệu lịch sử từ '{csv_file_path}' để gửi tới Gemini.")
            except Exception as e:
                print(f"⚠️ Cảnh báo: Không thể đọc file '{csv_file_path}' để gửi tới Gemini: {e}")
                historical_data_str = "Không thể đọc dữ liệu lịch sử."
        else:
             print(f"⚠️ Cảnh báo: File '{csv_file_path}' không tồn tại để gửi tới Gemini.")
        
        # Lấy các giá trị cần thiết với xử lý an toàn
        current_price = safe_float(trading_signal.get('current_price'))
        rsi_value = safe_float(trading_signal.get('rsi_value'))
        ma10 = safe_float(trading_signal.get('ma10'))
        ma20 = safe_float(trading_signal.get('ma20'))
        ma50 = safe_float(trading_signal.get('ma50'))
        ma200 = safe_float(trading_signal.get('ma200'))
        bb_upper = safe_float(trading_signal.get('bb_upper'))
        bb_lower = safe_float(trading_signal.get('bb_lower'))
        macd = safe_float(trading_signal.get('macd'))
        macd_signal = safe_float(trading_signal.get('macd_signal'))
        hist = safe_float(trading_signal.get('macd_hist'))
        tenkan_val = safe_format(trading_signal.get("ichimoku_tenkan_sen"))
        kijun_val = safe_format(trading_signal.get("ichimoku_kijun_sen"))
        senkou_a_val = safe_format(trading_signal.get("ichimoku_senkou_span_a"))
        senkou_b_val = safe_format(trading_signal.get("ichimoku_senkou_span_b"))
        chikou_val = safe_format(trading_signal.get("ichimoku_chikou_span"))
        volume = safe_float(trading_signal.get('volume'))
        volume_ma_20 = safe_float(trading_signal.get('volume_ma_20'))
        volume_ma_50 = safe_float(trading_signal.get('volume_ma_50'))
        
        # Tạo prompt với các giá trị đã được xử lý an toàn
        prompt = f"""
Bạn là chuyên gia phân tích chứng khoán Việt Nam. Hãy đánh giá mã {symbol}:
1. Phân tích kỹ thuật:
   - Giá: {safe_format(current_price)}
   - RSI: {safe_format(rsi_value)}
   - MA: {safe_format(ma10)} (10), {safe_format(ma20)} (20), {safe_format(ma50)} (50), {safe_format(ma200)} (200)
   - Bollinger bands Up: {safe_format(bb_upper)}, Bollinger bands Lower: / {safe_format(bb_lower)}
   - MACD: {safe_format(macd)}, Signal: {safe_format(macd_signal)}, Histogram: {safe_format(hist)}
   - Ichimoku: Tenkan: {tenkan_val}| Kijun: {kijun_val}| Senkou_A: {senkou_a_val}| Senkou_B: {senkou_b_val}| Chikou: {chikou_val}"
   - Khối lượng: {safe_format(volume)}
   - Khối lượng trung bình 20 ngày: {safe_format(volume_ma_20)}
   - Khối lượng trung bình 50 ngày: {safe_format(volume_ma_50)}
   """
        
        if symbol.upper() != "VNINDEX":
            rs = safe_float(trading_signal.get('rs'))
            rs_point = safe_float(trading_signal.get('rs_point'))
            rs_point_252 = safe_float(trading_signal.get('rs_point_252'))
            
            prompt += f"""
   - RS (Sức mạnh tương đối so với thị trường): C / VNINDEX → {safe_format(rs, '.4f')}
     * RS_SMA_10: {safe_format(trading_signal.get('rs_sma_10'), '.4f')}
     * RS_SMA_20: {safe_format(trading_signal.get('rs_sma_20'), '.4f')}
     * RS_SMA_50: {safe_format(trading_signal.get('rs_sma_50'), '.4f')}
     * RS_SMA_200: {safe_format(trading_signal.get('rs_sma_200'), '.4f')}

   - RS_Point (điểm sức mạnh IBD): 0.4*ROC(63) + 0.2*ROC(126) + 0.2*ROC(189) + 0.2*ROC(252) → {safe_format(rs_point)}
     * SMA_10: {safe_format(trading_signal.get('rs_point_sma_10'))}*
     * SMA_20: {safe_format(trading_signal.get('rs_point_sma_20'))}
     * SMA_50: {safe_format(trading_signal.get('rs_point_sma_50'))}
     * SMA_200: {safe_format(trading_signal.get('rs_point_sma_200'))}

   - RS_Point_252: ((C / Ref(C, -252)) - 1) * 100 → {safe_format(rs_point_252)}
     * SMA_10: {safe_format(trading_signal.get('rs_point_252_sma_10'))}
     * SMA_20: {safe_format(trading_signal.get('rs_point_252_sma_20'))}
     * SMA_50: {safe_format(trading_signal.get('rs_point_252_sma_50'))}
     * SMA_200: {safe_format(trading_signal.get('rs_point_252_sma_200'))}
            """
        
        if (financial_data_ratio is not None and not financial_data_ratio.empty) or \
           (financial_data_statement is not None and not financial_data_statement.empty):
            prompt += f"2. Tình hình tài chính.\n"
            if financial_data_ratio is not None and not financial_data_ratio.empty:
                prompt += f"Tình hình tỷ lệ tài chính :\n{financial_data_ratio.to_string(index=False)}\n"
            if financial_data_statement is not None and not financial_data_statement.empty:
                prompt += f"Báo cáo tài chính :\n{financial_data_statement.to_string(index=False)}\n"
        else:
            prompt += "2. Không có dữ liệu tài chính.\n"
        prompt += f"""
3. Dữ liệu lịch sử giá:
{historical_data_str}
"""
        prompt += """
Nhiệm vụ của bạn:
- Phân tích kỹ thuật theo Wyckoff, VSA/VPA, Minervini, Alexander Elder: hành động giá, khối lượng, cấu trúc xu hướng, điểm mua/bán.
- Phân tích cơ bản theo Warren Buffett, Charlie Munger, Peter Lynch, Seth Klarman: tăng trưởng, lợi nhuận, biên lợi nhuận, ROE, nợ, dòng tiền.
- Đánh giá mô hình kỹ thuật (nếu có). 
- Từ dữ liệu lịch sử giá có thểm thêm nhận định từ các chỉ báo từ AI tự phân tích.
- Nhận định xu hướng ngắn hạn (1–4 tuần) và trung hạn (1–6 tháng).
- Kết luận cuối cùng phải rõ ràng, súc tích: **MUA MẠNH / MUA / GIỮ / BÁN / BÁN MẠNH**.
- Trình bày phân tích ngắn gọn, chuyên nghiệp, dễ hành động.
"""

        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        if response and response.text:
            return response.text.strip()
        else:
            return "Không nhận được phản hồi từ Google Gemini."
    
    except Exception as e:
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
    financial_data_ratio, financial_data_statement  = get_financial_data(symbol)
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
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, financial_data_ratio, financial_data_statement)
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
        "rs": safe_float(trading_signal.get("rs")) if symbol.upper() != "VNINDEX" else None,
        "rs_point": safe_float(trading_signal.get("rs_point")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_252": safe_float(trading_signal.get("rs_point_252")) if symbol.upper() != "VNINDEX" else None,
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
        "rs_point_252_sma_10": safe_float(trading_signal.get("rs_point_252_sma_10")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_252_sma_20": safe_float(trading_signal.get("rs_point_252_sma_20")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_252_sma_50": safe_float(trading_signal.get("rs_point_252_sma_50")) if symbol.upper() != "VNINDEX" else None,
        "rs_point_252_sma_200": safe_float(trading_signal.get("rs_point_252_sma_200")) if symbol.upper() != "VNINDEX" else None,
        "gemini_analysis": gemini_analysis,
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
    symbols_to_analyze = stock_list["symbol"]
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
def filter_stocks_low_pe_high_cap(min_market_cap= 500):
    """Lọc cổ phiếu theo tiêu chí P/E thấp và vốn hóa cao."""
    try:
        df = Screener().stock(params={"exchangeName": "HOSE,HNX,UPCOM"}, limit=5000)
        if df is None or df.empty:
            print("❌ Không thể lấy dữ liệu danh sách công ty niêm yết.")
            return None
        
        filtered_df = df[
            (df['market_cap'] >= min_market_cap) &
            (df['pe'] > 0 & df['pe'] < 20) &
            (df['pb'] > 0) &
            (df['last_quarter_revenue_growth'] > 0) &
            (df['second_quarter_revenue_growth'] > 0) &
            (df['last_quarter_profit_growth'] > 0) &
            (df['second_quarter_profit_growth'] > 0) &
            ((df['peg_forward'] < 1 & df['peg_forward'] >= 0) | pd.isna(df['peg_forward']) &
            ((df['peg_trailing'] < 1 & df['peg_trailing'] >= 0) | pd.isna(df['peg_trailing'])))
        ]

        filtered_df.to_csv("market.csv", index=False, encoding="utf-8-sig")
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
    # print(f"🔍 Đang lọc cổ phiếu có P/E thấp và vốn hóa > {min_cap} tỷ VND...")
    filtered_stocks = filter_stocks_low_pe_high_cap()
    # if filtered_stocks is not None and not filtered_stocks.empty:
    #     print("🚀 Bắt đầu quét và phân tích...")
    #     screen_stocks_parallel()
    # else:
    #     print("🔍 Không tìm được cổ phiếu nào phù hợp với tiêu chí lọc.")
    print("\nNhập mã cổ phiếu để phân tích riêng lẻ (ví dụ: VCB, FPT) hoặc 'exit' để thoát")
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