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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

# ======================
# CẤU HÌNH VÀ THƯ VIỆN
# ======================
# Tải biến môi trường cho Google
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Không tìm thấy khóa API Google. Vui lòng kiểm tra file .env")
    exit()

# Cấu hình API client cho Google
genai.configure(api_key=GOOGLE_API_KEY)

# Tạo thư mục lưu trữ dữ liệu
if not os.path.exists("vnstocks_data"):
    os.makedirs("vnstocks_data")


# ======================
# PHẦN 1: THU THẬP DỮ LIỆU
# ======================
def get_vnstocks_list():
    """Lấy danh sách tất cả các mã chứng khoán trên thị trường Việt Nam sử dụng vnstock v2"""
    try:
        df = listing_companies()
        if df is not None and not df.empty:
            # Lọc chỉ lấy mã cổ phiếu (loại bỏ chứng chỉ quỹ, trái phiếu...)
            df = df[df["organType"] == "DN"]
            symbols = df[["ticker"]].rename(columns={"ticker": "symbol"})
            symbols.to_csv("vnstocks_data/stock_list.csv", index=False)
            print(
                f"Đã lưu danh sách {len(symbols)} mã chứng khoán vào file 'vnstocks_data/stock_list.csv'"
            )
            return symbols
        else:
            print("Không lấy được danh sách từ vnstock, sử dụng danh sách mẫu")
            sample_stocks = [
                "VNM",
                "VCB",
                "FPT",
                "GAS",
                "BID",
                "CTG",
                "MWG",
                "PNJ",
                "HPG",
                "STB",
            ]
            df = pd.DataFrame(sample_stocks, columns=["symbol"])
            df.to_csv("vnstocks_data/stock_list.csv", index=False)
            return df
    except Exception as e:
        print(f"Exception khi lấy danh sách mã: {str(e)}")
        sample_stocks = [
            "VNM",
            "VCB",
            "FPT",
            "GAS",
            "BID",
            "CTG",
            "MWG",
            "PNJ",
            "HPG",
            "STB",
        ]
        df = pd.DataFrame(sample_stocks, columns=["symbol"])
        df.to_csv("vnstocks_data/stock_list.csv", index=False)
        return df


def get_stock_data(symbol):
    """Lấy dữ liệu lịch sử của một mã chứng khoán sử dụng vnstock v2 mới theo tài liệu"""
    try:
        # Sử dụng cú pháp mới theo tài liệu: stock(symbol, period).price()
        quote = Quote(symbol)
        df = quote.history(start="2012-01-01", end="2030-1-1", interval="1D")
        if df is not None and not df.empty:
            # Chuẩn hóa tên cột theo chuẩn mới
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
            # Xử lý cột Date
            df["Date"] = pd.to_datetime(df["Date"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            # Lưu dữ liệu
            df.to_csv(f"vnstocks_data/{symbol}_data.csv")
            print(
                f"Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'"
            )
            return df
        else:
            print(f"Không thể lấy dữ liệu cho mã {symbol} từ vnstock")
            return None
    except Exception as e:
        print(f"Exception khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        return None


def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính sử dụng vnstock v2"""
    try:
        # Lấy dữ liệu báo cáo tài chính theo quý - CÚ PHÁP MỚI
        financial_obj = Finance(symbol=symbol)
        financial_data = financial_obj.ratio(
            period="quarter", lang="en", flatten_columns=True
        )
        if financial_data is not None and not financial_data.empty:
            # Lưu dữ liệu
            financial_data.to_csv(f"vnstocks_data/{symbol}_financial.csv", index=False)
            return financial_data
        else:
            print(f"Không lấy được BCTC cho mã {symbol}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None


def get_market_data():
    """Lấy dữ liệu thị trường tổng thể sử dụng vnstock v2"""
    try:
        # Lấy dữ liệu VN-Index
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(start="2012-01-01", end="2030-1-1", interval="1D")
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
            vnindex.to_csv("vnstocks_data/vnindex_data.csv")
        # Lấy dữ liệu VN30-Index
        quoteVN30 = Quote(symbol="VN30")
        vn30 = quoteVN30.history(start="2012-01-01", end="2030-1-1", interval="1D")
        if vn30 is not None and not vn30.empty:
            vn30.rename(
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
            vn30["Date"] = pd.to_datetime(vn30["Date"])
            vn30.set_index("Date", inplace=True)
            vn30.sort_index(inplace=True)
            vn30.to_csv("vnstocks_data/vn30_data.csv")
        print("Đã lưu dữ liệu chỉ số thị trường vào thư mục 'vnstocks_data/'")
        return {"vnindex": vnindex, "vn30": vn30}
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu thị trường: {str(e)}")
        return None


# ======================
# PHẦN 2: TIỀN XỬ LÝ VÀ TẠO ĐẶC TRƯNG
# ======================
def preprocess_stock_data(df):
    """
    Preprocesses raw stock data from vnstock:
    - Converts date to datetime index
    - Sorts chronologically
    - Handles missing values
    - Adds technical features
    - Filters relevant columns
    """
    # Convert to datetime and sort
    df.index = pd.to_datetime(df.index)
    df.sort_index(ascending=True, inplace=True)
    # Handle missing values (forward fill then backfill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    # Add technical features (example)
    df["returns"] = df["Close"].pct_change()
    df["MA_10"] = df["Close"].rolling(window=10).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()
    df["volatility"] = df["returns"].rolling(window=10).std()
    # Drop initial rows with NaN values from technical features
    df.dropna(inplace=True)
    return df


def create_features(df):
    """
    Generates technical indicators using pure pandas/numpy
    """
    # Calculate RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    # Calculate MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(window=20).mean()
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
    # ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    # Moving averages
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    # Momentum and volume features
    df["Momentum"] = df["Close"] / df["Close"].shift(4) - 1
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_Change"] = df["Volume"].pct_change()
    # Drop NA values
    df.dropna(inplace=True)
    return df


# ======================
# PHẦN 3: MÔ HÌNH AI - LSTM TĂNG CƯỜNG
# ======================

# --- HÀM LSTM (LSTM TĂNG CƯỜNG HOẶC CƠ BẢN) ---
# ======================
# PHẦN 3: MÔ HÌNH AI - LSTM TĂNG CƯỜNG
# ======================


# - HÀM LSTM (LSTM TĂNG CƯỜNG HOẶC CƠ BẢN) -
def train_stock_model(
    df, target="Close", time_steps=60, test_size=0.2, epochs=0, batch_size=32
):
    """Huấn luyện mô hình LSTM để dự báo giá cổ phiếu."""
    try:

        # Kiểm tra dữ liệu đầu vào
        if df is None or len(df) < time_steps:
            print("Dữ liệu không đủ để huấn luyện mô hình")
            return None, None, None, None, None
        if target not in df.columns:
            print(f"Cột {target} không tồn tại trong dữ liệu")
            return None, None, None, None, None

        data = df[[target]].values
        if len(data) == 0:
            print("Dữ liệu rỗng")
            return None, None, None, None, None

        # Loại bỏ NaN/inf
        data = data[np.isfinite(data)].reshape(-1, 1)
        if len(data) == 0:
            print("Không có dữ liệu hợp lệ sau khi loại bỏ NaN/inf")
            return None, None, None, None, None

        # Chuẩn hóa dữ liệu
        from sklearn.preprocessing import (
            MinMaxScaler,
        )  # Đảm bảo import ở đây nếu chưa có ở đầu file

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        # Tạo dataset cho chuỗi thời gian
        if len(scaled_data) <= time_steps:
            print("Không đủ dữ liệu để tạo chuỗi thời gian")
            return None, None, None, None, None

        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i - time_steps : i, 0])
            y.append(scaled_data[i, 0])

        if len(X) == 0 or len(y) == 0:
            print("Không tạo được dữ liệu huấn luyện")
            return None, None, None, None, None

        X, y = np.array(X), np.array(y)

        # Reshape input để phù hợp với LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        if X.shape[0] < 10:
            print("Dữ liệu quá ít để huấn luyện")
            return None, None, None, None, None

        # Chia train/test
        split_index = max(1, int(len(X) * (1 - test_size)))
        if split_index >= len(X):
            # Nếu test_size quá lớn, đặt split_index ở cuối dữ liệu huấn luyện
            split_index = len(X) - 1

        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]

        if len(X_train) == 0 or len(y_train) == 0:
            print("Dữ liệu train rỗng")
            return None, None, None, None, None

        model = Sequential()

        # Thêm nhiều lớp LSTM với dropout và tăng units
        # Lớp 1: 150 units
        model.add(
            LSTM(units=150, return_sequences=True, input_shape=(X_train.shape[1], 1))
        )
        model.add(Dropout(0.2))
        # Lớp 2: 150 units
        model.add(LSTM(units=150, return_sequences=True))
        model.add(Dropout(0.2))
        # Lớp 3: 100 units
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.2))
        # Lớp 4: 100 units (return_sequences=False cho lớp cuối cùng trước Dense)
        model.add(LSTM(units=100, return_sequences=False))
        model.add(Dropout(0.2))

        # Thêm lớp Dense ẩn với nhiều units hơn và dropout
        model.add(Dense(units=75))
        model.add(Dropout(0.2))

        # Lớp đầu ra
        model.add(Dense(units=1))

        model.compile(optimizer="adam", loss="mean_squared_error")

        # Callbacks để cải thiện huấn luyện
        early_stopping = EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        )  # Tăng patience
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1
        )  # Tăng patience

        print(f"Bắt đầu huấn luyện mô hình LSTM với {epochs} epochs...")
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )
        print("✅ Huấn luyện mô hình LSTM hoàn tất")

        # - KẾT THÚC LSTM TĂNG CƯỜNG (KÍCH THƯỚC LỚN HƠN) -

        # Dự báo trên tập test
        y_pred = model.predict(X_test)
        # Chuyển đổi kết quả về giá trị thực
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred)

        # --- ĐÁNH GIÁ MÔ HÌNH ---
        try:
            from sklearn.metrics import (
                mean_squared_error,
                mean_absolute_error,
                r2_score,
            )

            mse = mean_squared_error(y_test, y_pred)
            rmse_val = np.sqrt(mse)
            mae_val = mean_absolute_error(y_test, y_pred)
            # Tính R2 cho dự báo (so với giá trị thực tế)
            r2 = r2_score(y_test, y_pred)
            print("\n--- ĐÁNH GIÁ MÔ HÌNH DỰ BÁO ---")
            print(f"RMSE: {rmse_val:.2f}")
            print(f"MAE: {mae_val:.2f}")
            print(f"R2: {r2:.2f}")
            print("--- HẾT ĐÁNH GIÁ ---\n")
        except Exception as e:
            print(f"Lỗi khi tính toán đánh giá LSTM: {str(e)}")
            mse, rmse_val, mae_val, r2 = 0, 0, 0, 0
        # --- HẾT ĐÁNH GIÁ ---

        return model, scaler, X_test, y_test, y_pred

    except Exception as e:
        print(f"Lỗi nghiêm trọng khi huấn luyện mô hình LSTM: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, None, None, None, None


def predict_next_days(model, scaler, df, target="Close", time_steps=60, n_days=5):
    """
    Dự báo giá trong n ngày tiếp theo (cho LSTM)
    """
    try:
        # Kiểm tra đầu vào
        if model is None or scaler is None or df is None:
            print("Dữ liệu đầu vào không hợp lệ")
            return np.array([]), np.array([])
        if target not in df.columns:
            print(f"Cột {target} không tồn tại")
            return np.array([]), np.array([])
        if len(df) < time_steps:
            print("Không đủ dữ liệu để dự báo")
            return np.array([]), np.array([])
        # Lấy dữ liệu cuối cùng
        last_data = df[target].values[-time_steps:]
        # Kiểm tra dữ liệu có hợp lệ không
        if len(last_data) == 0:
            print("Dữ liệu dự báo rỗng")
            return np.array([]), np.array([])
        # Loại bỏ NaN/inf
        last_data = last_data[np.isfinite(last_data)]
        if len(last_data) < time_steps:
            print("Dữ liệu không đủ sau khi loại bỏ NaN")
            return np.array([]), np.array([])
        # Chuẩn hóa dữ liệu
        try:
            last_data_scaled = scaler.transform(last_data.reshape(-1, 1))
        except Exception as e:
            print(f"Lỗi khi chuẩn hóa dữ liệu dự báo: {str(e)}")
            return np.array([]), np.array([])
        # Tạo dữ liệu đầu vào
        X = last_data_scaled.reshape(1, time_steps, 1)
        # Dự báo với kiểm tra lỗi
        forecast_scaled = []
        try:
            for _ in range(n_days):
                # Dự báo giá tiếp theo
                pred = model.predict(X, verbose=0)
                forecast_scaled.append(pred[0, 0])
                # Cập nhật dữ liệu đầu vào
                X = np.append(X[:, 1:, :], pred.reshape(1, 1, 1), axis=1)
        except Exception as e:
            print(f"Lỗi khi dự báo từng ngày: {str(e)}")
            return np.array([]), np.array([])
        # Chuyển đổi về giá gốc
        try:
            forecast = scaler.inverse_transform(
                np.array(forecast_scaled).reshape(-1, 1)
            )
        except Exception as e:
            print(f"Lỗi khi chuyển đổi giá gốc: {str(e)}")
            return np.array([]), np.array([])
        # Tạo ngày dự báo
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
        return np.array(forecast_dates), forecast.flatten()
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi dự báo: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])


# --- CẬP NHẬT HÀM ĐÁNH GIÁ DỮ LIỆU ---
def evaluate_data_for_ai(df_features, symbol):
    """
    Đánh giá dữ liệu để đề xuất mô hình AI phù hợp (chỉ LSTM).
    """
    if df_features is None or len(df_features) == 0:
        print(f"❌ Không có dữ liệu để đánh giá cho mã {symbol}.")
        return "Không xác định", "Không có dữ liệu đầu vào."

    num_points = len(df_features)
    num_features = len(df_features.columns)

    print(f"\n--- ĐÁNH GIÁ DỮ LIỆU CHO MÃ {symbol} ---")
    print(f"Số điểm dữ liệu: {num_points}")
    print(f"Số lượng đặc trưng: {num_features}")

    # Cập nhật logic đề xuất (chỉ đề xuất LSTM)
    if num_points > 2000:
        recommendation = "LSTM TĂNG CƯỜNG"
        reason = f"Dữ liệu có {num_points} điểm > 2000, LSTM TĂNG CƯỜNG phù hợp cho chuỗi dài."
    elif num_points > 1000:
        recommendation = "LSTM TĂNG CƯỜNG"
        reason = f"Dữ liệu có {num_points} điểm > 1000, LSTM TĂNG CƯỜNG hiệu quả cho chuỗi dài."
    elif num_features > 50:
        recommendation = "LSTM TĂNG CƯỜNG"
        reason = f"Dữ liệu có {num_features} đặc trưng > 50, LSTM TĂNG CƯỜNG có thể xử lý tốt."
    else:  # Dữ liệu trung bình/trung bình dưới
        recommendation = "LSTM TĂNG CƯỜNG"
        reason = f"Dữ liệu có {num_points} điểm và {num_features} đặc trưng, LSTM TĂNG CƯỜNG là lựa chọn tốt."

    print(f"💡 Đề xuất mô hình AI: {recommendation}")
    print(f"❓ Lý do: {reason}")
    print("--- HẾT ĐÁNH GIÁ ---\n")

    return recommendation, reason


# ======================
# PHẦN 4: PHÂN TÍCH KỸ THUẬT CẢI TIẾN
# ======================
def plot_stock_analysis(symbol, df, show_volume=True):
    """
    Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán
    Có thêm MA10, RS (so với VNINDEX), RS_Point và so sánh với các đường trung bình
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if df is None or len(df) == 0:
            print("Dữ liệu phân tích rỗng")
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
            }

        # Sắp xếp theo ngày tăng dần để tính toán chính xác
        df = df.sort_index()
        # --- BƯỚC 1: Tính các chỉ báo kỹ thuật ---
        # 1. Đường trung bình
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)

        # 2. RSI (Relative Strength Index)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

        # 3. MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()

        # 4. Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bollinger.bollinger_hband()
        df["BB_Lower"] = bollinger.bollinger_lband()

        # 5. Volume Moving Averages
        df["Volume_SMA_20"] = ta.trend.sma_indicator(df["Volume"], window=20)
        df["Volume_SMA_50"] = ta.trend.sma_indicator(df["Volume"], window=50)

        # 6. Ichimoku Cloud
        # Các tham số mặc định cho Ichimoku
        tenkan_window = 9
        kijun_window = 26
        senkou_span_b_window = 52

        # Tenkan-sen (Conversion Line)
        tenkan_sen_high = df["High"].rolling(window=tenkan_window).max()
        tenkan_sen_low = df["Low"].rolling(window=tenkan_window).min()
        df["ichimoku_tenkan_sen"] = (tenkan_sen_high + tenkan_sen_low) / 2

        # Kijun-sen (Base Line)
        kijun_sen_high = df["High"].rolling(window=kijun_window).max()
        kijun_sen_low = df["Low"].rolling(window=kijun_window).min()
        df["ichimoku_kijun_sen"] = (kijun_sen_high + kijun_sen_low) / 2

        # Senkou Span A (Leading Span A)
        df["ichimoku_senkou_span_a"] = (
            (df["ichimoku_tenkan_sen"] + df["ichimoku_kijun_sen"]) / 2
        ).shift(kijun_window)

        # Senkou Span B (Leading Span B)
        senkou_span_b_high = df["High"].rolling(window=senkou_span_b_window).max()
        senkou_span_b_low = df["Low"].rolling(window=senkou_span_b_window).min()
        df["ichimoku_senkou_span_b"] = (
            (senkou_span_b_high + senkou_span_b_low) / 2
        ).shift(kijun_window)

        # Chikou Span (Lagging Span)
        df["ichimoku_chikou_span"] = df["Close"].shift(-kijun_window)

        # --- BƯỚC 2: Tính RS (Relative Strength so với VNINDEX) ---
        try:
            # Lấy dữ liệu VNINDEX trong cùng khoảng thời gian
            quoteVNI = Quote(symbol="VNINDEX")
            vnindex_df = quoteVNI.history(
                start="2012-01-01", end="2030-1-1", interval="1D"
            )

            if len(vnindex_df) == 0:
                raise ValueError("Không lấy được dữ liệu VNINDEX")

            # Đổi tên cột trong vnindex_df để phù hợp với chuẩn
            vnindex_df.rename(
                columns={
                    "time": "Date",
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",  # <-- Đổi 'close' thành 'Close'
                    "volume": "Volume",
                },
                inplace=True,
            )

            # Xử lý cột Date trong vnindex_df
            vnindex_df["Date"] = pd.to_datetime(vnindex_df["Date"])
            vnindex_df.set_index("Date", inplace=True)
            vnindex_df.sort_index(inplace=True)

            # --- SỬA Ở ĐÂY ---
            # Gộp dữ liệu cổ phiếu đang phân tích (df) và VNINDEX (vnindex_df) theo ngày (index)
            # Giả sử df đã được chuẩn hóa tên cột và có index là Date
            # Chúng ta sẽ join df[['Close']] với vnindex_df[['Close']]
            df_merged = df[["Close"]].join(
                vnindex_df[["Close"]].rename(columns={"Close": "VNINDEX_Close"}),
                how="left",
            )

            # Nếu thiếu dữ liệu VNINDEX, không tính RS
            if df_merged["VNINDEX_Close"].isna().all():
                df["RS"] = 1.0  # Mặc định
                df["RS_Point"] = 0.0
                print("Cảnh báo: Không có dữ liệu VNINDEX, bỏ qua RS")
            else:
                # Điền dữ liệu VNINDEX nếu thiếu (forward fill)
                df_merged["VNINDEX_Close"] = df_merged["VNINDEX_Close"].ffill()

                # Tính RS = price / VNINDEX
                df["RS"] = df_merged["Close"] / df_merged["VNINDEX_Close"]

                # Tính RS_Point theo công thức
                roc_63 = ta.momentum.roc(df["Close"], window=63)
                roc_126 = ta.momentum.roc(df["Close"], window=126)
                roc_189 = ta.momentum.roc(df["Close"], window=189)
                roc_252 = ta.momentum.roc(df["Close"], window=252)

                df["RS_Point"] = (
                    roc_63.fillna(0) * 0.4
                    + roc_126.fillna(0) * 0.2
                    + roc_189.fillna(0) * 0.2
                    + roc_252.fillna(0) * 0.2
                ) * 100

                # Tính các đường trung bình cho RS_Point
                df["RS_Point_SMA_10"] = ta.trend.sma_indicator(
                    df["RS_Point"], window=10
                )
                df["RS_Point_SMA_20"] = ta.trend.sma_indicator(
                    df["RS_Point"], window=20
                )
                df["RS_Point_SMA_50"] = ta.trend.sma_indicator(
                    df["RS_Point"], window=50
                )
                df["RS_Point_SMA_200"] = ta.trend.sma_indicator(
                    df["RS_Point"], window=200
                )

                # Tính các đường trung bình cho RS
                df["RS_SMA_10"] = ta.trend.sma_indicator(df["RS"], window=10)
                df["RS_SMA_20"] = ta.trend.sma_indicator(df["RS"], window=20)
                df["RS_SMA_50"] = ta.trend.sma_indicator(df["RS"], window=50)
                df["RS_SMA_200"] = ta.trend.sma_indicator(df["RS"], window=200)

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
            }

        # --- BƯỚC 4: Vẽ biểu đồ (CẬP NHẬT) ---
        try:
            plot_configs = [
                "price_sma",
                "ichimoku",
                "rsi",
                "macd",
                "rs",
                "rs_point",
                "volume",
            ]
            num_subplots = len(plot_configs)  # hoặc đếm số biểu đồ
            height_per_subplot = 2.5
            width = 18
            height = num_subplots * height_per_subplot

            plt.figure(figsize=(width, height), constrained_layout=True)

            # Điều chỉnh GridSpec - GIỮ NGUYÊN 8 HÀNG, TĂNG HSPACE
            grid = plt.GridSpec(
                8, 1, hspace=0.4, height_ratios=[3, 3, 1, 1, 1, 1, 1, 1]
            )

            # === Biểu đồ 1: Giá và các đường trung bình ===
            ax1 = plt.subplot(grid[0])
            plt.plot(
                df.index,
                df["Close"],
                label="Giá đóng cửa",
                color="#1f77b4",
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["SMA_10"],
                label="SMA 10",
                color="blue",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["SMA_20"],
                label="SMA 20",
                color="orange",
                alpha=0.8,
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["SMA_50"],
                label="SMA 50",
                color="green",
                alpha=0.8,
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["SMA_200"],
                label="SMA 200",
                color="purple",
                alpha=0.8,
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["BB_Upper"],
                label="BB Upper",
                color="red",
                alpha=0.5,
                linestyle="--",
            )
            plt.plot(
                df.index,
                df["BB_Lower"],
                label="BB Lower",
                color="green",
                alpha=0.5,
                linestyle="--",
            )
            plt.fill_between(
                df.index, df["BB_Lower"], df["BB_Upper"], color="gray", alpha=0.1
            )

            # Đánh dấu điểm giao nhau SMA10 và SMA20
            cross_10_20_above = (df["SMA_10"] > df["SMA_20"]) & (
                df["SMA_10"].shift(1) <= df["SMA_20"].shift(1)
            )
            cross_10_20_below = (df["SMA_10"] < df["SMA_20"]) & (
                df["SMA_10"].shift(1) >= df["SMA_20"].shift(1)
            )

            if cross_10_20_above.any():
                plt.scatter(
                    df.index[cross_10_20_above],
                    df.loc[cross_10_20_above, "SMA_10"],
                    marker="^",
                    color="lime",
                    s=60,
                    label="SMA10 > SMA20",
                    zorder=5,
                )
            if cross_10_20_below.any():
                plt.scatter(
                    df.index[cross_10_20_below],
                    df.loc[cross_10_20_below, "SMA_10"],
                    marker="v",
                    color="magenta",
                    s=60,
                    label="SMA10 < SMA20",
                    zorder=5,
                )

            # TIÊU ĐỀ CHUNG TRÊN CÙNG
            plt.suptitle(
                f"Phân tích kỹ thuật {symbol} - Giá và Chỉ báo",
                fontsize=16,
                fontweight="bold",
                y=0.98,
            )
            plt.ylabel("Giá (VND)", fontsize=12)
            plt.legend(loc="upper left", fontsize=10)
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 2: Ichimoku Cloud (ĐƯA LÊN THỨ 2) ===
            ax2 = plt.subplot(grid[1], sharex=ax1)

            # Vẽ nến Nhật thủ công
            for i in range(len(df)):
                if i < len(df) - 1:  # Tránh lỗi index
                    date = mdates.date2num(df.index[i])
                    open_price = (
                        df["Open"].iloc[i]
                        if not pd.isna(df["Open"].iloc[i])
                        else df["Close"].iloc[i]
                    )
                    high_price = (
                        df["High"].iloc[i]
                        if not pd.isna(df["High"].iloc[i])
                        else df["Close"].iloc[i]
                    )
                    low_price = (
                        df["Low"].iloc[i]
                        if not pd.isna(df["Low"].iloc[i])
                        else df["Close"].iloc[i]
                    )
                    close_price = (
                        df["Close"].iloc[i]
                        if not pd.isna(df["Close"].iloc[i])
                        else open_price
                    )

                    # Tạo nến
                    if close_price >= open_price:
                        color = "green"
                        bottom = open_price
                        height = close_price - open_price
                    else:
                        color = "red"
                        bottom = close_price
                        height = open_price - close_price

                    # Vẽ thân nến
                    if height > 0:
                        plt.bar(
                            date,
                            height,
                            bottom=bottom,
                            color=color,
                            width=0.6,
                            alpha=0.8,
                        )
                    # Vẽ bấc nến
                    plt.plot(
                        [date, date],
                        [low_price, high_price],
                        color="black",
                        linewidth=0.5,
                    )

            # Vẽ các đường Ichimoku
            plt.plot(
                df.index,
                df["ichimoku_tenkan_sen"],
                label="Tenkan-sen",
                color="red",
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["ichimoku_kijun_sen"],
                label="Kijun-sen",
                color="blue",
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["ichimoku_senkou_span_a"],
                label="Senkou Span A",
                color="green",
                linewidth=1,
                alpha=0.7,
            )
            plt.plot(
                df.index,
                df["ichimoku_senkou_span_b"],
                label="Senkou Span B",
                color="purple",
                linewidth=1,
                alpha=0.7,
            )
            plt.plot(
                df.index,
                df["ichimoku_chikou_span"],
                label="Chikou Span",
                color="orange",
                linewidth=1,
            )

            # Vẽ Cloud (vùng mây)
            valid_cloud = (
                df["ichimoku_senkou_span_a"].notna()
                & df["ichimoku_senkou_span_b"].notna()
            )
            if valid_cloud.any():
                plt.fill_between(
                    df.index[valid_cloud],
                    df["ichimoku_senkou_span_a"][valid_cloud],
                    df["ichimoku_senkou_span_b"][valid_cloud],
                    where=(
                        df["ichimoku_senkou_span_a"][valid_cloud]
                        >= df["ichimoku_senkou_span_b"][valid_cloud]
                    ),
                    color="green",
                    alpha=0.2,
                    interpolate=True,
                    label="Bullish Cloud",
                )
                plt.fill_between(
                    df.index[valid_cloud],
                    df["ichimoku_senkou_span_a"][valid_cloud],
                    df["ichimoku_senkou_span_b"][valid_cloud],
                    where=(
                        df["ichimoku_senkou_span_a"][valid_cloud]
                        < df["ichimoku_senkou_span_b"][valid_cloud]
                    ),
                    color="red",
                    alpha=0.2,
                    interpolate=True,
                    label="Bearish Cloud",
                )

            # Vẽ thêm đường giá đóng cửa để so sánh
            plt.plot(
                df.index,
                df["Close"],
                label="Giá đóng cửa",
                color="black",
                linewidth=1.5,
                alpha=0.7,
            )

            plt.title("Ichimoku Cloud", fontsize=12)
            plt.ylabel("Giá", fontsize=10)
            plt.legend(fontsize=7, loc="upper left", ncol=2)
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 3: RSI ===
            ax3 = plt.subplot(grid[2], sharex=ax1)
            plt.plot(df.index, df["RSI"], label="RSI", color="purple")
            plt.axhline(70, linestyle="--", color="red", alpha=0.7)
            plt.axhline(30, linestyle="--", color="green", alpha=0.7)
            plt.fill_between(df.index, 30, 70, color="gray", alpha=0.1)
            plt.title("RSI", fontsize=12)
            plt.ylim(0, 100)
            plt.ylabel("RSI", fontsize=10)
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 4: MACD ===
            ax4 = plt.subplot(grid[3], sharex=ax1)
            plt.plot(df.index, df["MACD"], label="MACD", color="blue")
            plt.plot(df.index, df["MACD_Signal"], label="Signal Line", color="red")
            plt.bar(
                df.index,
                df["MACD_Hist"],
                color=np.where(df["MACD_Hist"] > 0, "green", "red"),
                alpha=0.5,
            )
            plt.title("MACD", fontsize=12)
            plt.ylabel("MACD", fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # === Biểu đồ 5: RS (Relative Strength vs VNINDEX) ===
            ax5 = plt.subplot(grid[4], sharex=ax1)
            plt.plot(
                df.index,
                df["RS"],
                label="RS (Price / VNINDEX)",
                color="brown",
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["RS_SMA_10"],
                label="RS SMA 10",
                color="blue",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_SMA_20"],
                label="RS SMA 20",
                color="orange",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_SMA_50"],
                label="RS SMA 50",
                color="green",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_SMA_200"],
                label="RS SMA 200",
                color="purple",
                alpha=0.7,
                linewidth=1,
            )
            plt.title("RS vs VNINDEX", fontsize=12)
            plt.ylabel("RS", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            # === Biểu đồ 6: RS_Point ===
            ax6 = plt.subplot(grid[5], sharex=ax1)
            plt.plot(
                df.index,
                df["RS_Point"],
                label="RS_Point",
                color="darkblue",
                linewidth=1.5,
            )
            plt.plot(
                df.index,
                df["RS_Point_SMA_10"],
                label="RS_Point SMA 10",
                color="blue",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_Point_SMA_20"],
                label="RS_Point SMA 20",
                color="orange",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_Point_SMA_50"],
                label="RS_Point SMA 50",
                color="green",
                alpha=0.7,
                linewidth=1,
            )
            plt.plot(
                df.index,
                df["RS_Point_SMA_200"],
                label="RS_Point SMA 200",
                color="purple",
                alpha=0.7,
                linewidth=1,
            )
            plt.axhline(0, color="black", linestyle="-", linewidth=0.8)
            plt.fill_between(
                df.index,
                df["RS_Point"],
                0,
                where=(df["RS_Point"] > 0),
                color="green",
                alpha=0.3,
            )
            plt.fill_between(
                df.index,
                df["RS_Point"],
                0,
                where=(df["RS_Point"] < 0),
                color="red",
                alpha=0.3,
            )
            plt.title("RS_Point", fontsize=12)
            plt.ylabel("RS_Point", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            # === Biểu đồ 7: Khối lượng ===
            ax7 = plt.subplot(grid[6], sharex=ax1)
            if show_volume and "Volume" in df.columns:
                volume_sma_plotted = False
                if (
                    "Volume_SMA_20" in df.columns
                    and not df["Volume_SMA_20"].isna().all()
                ):
                    plt.plot(
                        df.index,
                        df["Volume_SMA_20"],
                        label="Vol SMA 20",
                        color="orange",
                        alpha=0.8,
                        linewidth=1.5,
                    )
                    volume_sma_plotted = True
                if (
                    "Volume_SMA_50" in df.columns
                    and not df["Volume_SMA_50"].isna().all()
                ):
                    plt.plot(
                        df.index,
                        df["Volume_SMA_50"],
                        label="Vol SMA 50",
                        color="purple",
                        alpha=0.8,
                        linewidth=1.5,
                    )
                    volume_sma_plotted = True
                colors = np.where(df["Close"] > df["Open"], "green", "red")
                plt.bar(
                    df.index,
                    df["Volume"],
                    color=colors,
                    alpha=0.7,
                    label="Volume" if not volume_sma_plotted else None,
                )
                handles, labels = ax7.get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), loc="upper left")
                else:
                    plt.legend(loc="upper left")
                plt.title("Volume & Vol SMA", fontsize=12)
                plt.ylabel("Khối lượng", fontsize=10)
                plt.grid(True, alpha=0.3)
            else:
                plt.title("Khối lượng giao dịch", fontsize=12)
                plt.ylabel("Khối lượng", fontsize=10)
                plt.grid(True, alpha=0.3)

            # Điều chỉnh layout để tránh chồng chữ
            plt.tight_layout(pad=3.0, h_pad=1.0)
            plt.subplots_adjust(
                top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.4
            )

            # Lưu biểu đồ
            plt.savefig(
                f"vnstocks_data/{symbol}_technical_analysis.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()
            print(
                f"✅ Đã lưu biểu đồ phân tích kỹ thuật vào vnstocks_data/{symbol}_technical_analysis.png"
            )

        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")

        # --- BƯỚC 5: Tạo tín hiệu giao dịch ---
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
            rs_value = last_row["RS"] if not pd.isna(last_row["RS"]) else 1.0
            rs_point_value = (
                last_row["RS_Point"] if not pd.isna(last_row["RS_Point"]) else 0
            )

            # Lấy giá trị Ichimoku từ hàng cuối
            tenkan_sen = last_row.get("ichimoku_tenkan_sen", np.nan)
            kijun_sen = last_row.get("ichimoku_kijun_sen", np.nan)
            senkou_span_a = last_row.get("ichimoku_senkou_span_a", np.nan)
            senkou_span_b = last_row.get("ichimoku_senkou_span_b", np.nan)
            chikou_span = last_row.get("ichimoku_chikou_span", np.nan)

            # --- HỆ THỐNG TÍNH ĐIỂM TOÀN DIỆN ---
            score = 0

            # 1. RSI - 10 điểm
            if rsi_value < 30:
                score += 10  # Quá bán
            elif rsi_value > 70:
                score -= 10  # Quá mua
            else:
                score += (50 - abs(rsi_value - 50)) * 0.2  # 0-10 điểm

            # 2. Đường trung bình - 20 điểm
            # SMA10 vs SMA20
            if ma10_value > ma20_value:
                score += 6
            # SMA20 vs SMA50
            if ma20_value > ma50_value:
                score += 6
            # SMA50 vs SMA200
            if ma50_value > ma200_value:
                score += 8

            # 3. Giá so với các đường trung bình - 10 điểm
            if current_price > ma10_value:
                score += 3
            if current_price > ma20_value:
                score += 3
            if current_price > ma50_value:
                score += 2
            if current_price > ma200_value:
                score += 2

            # 4. MACD - 10 điểm
            macd_value = last_row["MACD"] if not pd.isna(last_row["MACD"]) else 0
            macd_signal = (
                last_row["MACD_Signal"] if not pd.isna(last_row["MACD_Signal"]) else 0
            )
            macd_hist = (
                last_row["MACD_Hist"] if not pd.isna(last_row["MACD_Hist"]) else 0
            )
            if macd_value > macd_signal and macd_hist > 0:
                score += 7  # Tín hiệu mua
            elif macd_value < macd_signal and macd_hist < 0:
                score -= 7  # Tín hiệu bán
            else:
                score += np.clip(macd_hist * 30, -3, 3)  # Dựa trên histogram

            # 5. Bollinger Bands - 5 điểm
            bb_upper = (
                last_row["BB_Upper"]
                if not pd.isna(last_row["BB_Upper"])
                else current_price
            )
            bb_lower = (
                last_row["BB_Lower"]
                if not pd.isna(last_row["BB_Lower"])
                else current_price
            )
            if current_price > bb_upper:
                score -= 3  # Quá mua
            elif current_price < bb_lower:
                score += 3  # Quá bán
            else:
                # Ở giữa, tính theo vị trí tương đối
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                score += (bb_position - 0.5) * 6

            # 6. RS (Relative Strength) - 10 điểm
            # So sánh RS với các đường trung bình của chính nó
            rs_score = 0
            if rs_value > last_row.get("RS_SMA_10", rs_value):
                rs_score += 2
            if rs_value > last_row.get("RS_SMA_20", rs_value):
                rs_score += 3
            if rs_value > last_row.get("RS_SMA_50", rs_value):
                rs_score += 5
            score += rs_score

            # 7. RS_Point - 10 điểm
            # So sánh RS_Point với các đường trung bình của chính nó
            rs_point_score = 0
            if rs_point_value > last_row.get("RS_Point_SMA_10", rs_point_value):
                rs_point_score += 2
            if rs_point_value > last_row.get("RS_Point_SMA_20", rs_point_value):
                rs_point_score += 3
            if rs_point_value > last_row.get("RS_Point_SMA_50", rs_point_value):
                rs_point_score += 5
            score += rs_point_score
            # 8. Ichimoku Cloud - 10 điểm
            ichimoku_score = 0
            try:
                # Kiểm tra nếu các giá trị không phải NaN
                if not (
                    pd.isna(tenkan_sen)
                    or pd.isna(kijun_sen)
                    or pd.isna(senkou_span_a)
                    or pd.isna(senkou_span_b)
                ):
                    # 8.1 Vị trí giá so với Cloud (5 điểm)
                    cloud_top = max(senkou_span_a, senkou_span_b)
                    cloud_bottom = min(senkou_span_a, senkou_span_b)
                    if current_price > cloud_top:
                        ichimoku_score += 5  # Giá trên Cloud - Tín hiệu Mua
                    elif current_price < cloud_bottom:
                        ichimoku_score -= 5  # Giá dưới Cloud - Tín hiệu Bán
                    # (Nếu giá trong cloud, điểm = 0)
                    # 8.2 Tenkan-sen vs Kijun-sen (3 điểm)
                    if tenkan_sen > kijun_sen:
                        ichimoku_score += 3  # Tenkan > Kijun - Tín hiệu Mua
                    elif tenkan_sen < kijun_sen:
                        ichimoku_score -= 3  # Tenkan < Kijun - Tín hiệu Bán
                    # 8.3 Kijun-sen vs Cloud (2 điểm)
                    if kijun_sen > cloud_top:
                        ichimoku_score += 2  # Kijun trên Cloud - Xu hướng tăng mạnh
                    elif kijun_sen < cloud_bottom:
                        ichimoku_score -= 2  # Kijun dưới Cloud - Xu hướng giảm mạnh
            except Exception as e:
                print(f"Cảnh báo: Lỗi khi tính điểm Ichimoku: {e}")
            score += ichimoku_score

            # Chuẩn hóa điểm về thang 0-100
            score = np.clip(score, 0, 100)

            # --- XÁC ĐỊNH TÍN HIỆU ---
            signal = "TRUNG LẬP"
            recommendation = "GIỮ"

            # Điều kiện mua mạnh (giản lược để phù hợp với logic điểm số)
            if score > 75:
                signal = "MUA"
                recommendation = "MUA MẠNH" if score > 85 else "MUA"

            # Điều kiện bán mạnh (giản lược để phù hợp với logic điểm số)
            elif score < 25:
                signal = "BÁN"
                recommendation = "BÁN MẠNH" if score < 15 else "BÁN"

            # Hiển thị kết quả phân tích
            analysis_date = df.index[-1].strftime("%d/%m/%Y")
            print(f"\n📊 TÍN HIỆU GIAO DỊCH CUỐI ({analysis_date}):")
            print(
                f"  - Giá & Đường trung bình: Giá={current_price:,.2f} | SMA10={ma10_value:,.2f} | SMA20={ma20_value:,.2f} | SMA50={ma50_value:,.2f} | SMA200={ma200_value:,.2f}"
            )

            # In thông tin Ichimoku
            try:
                print(f"  - Ichimoku:")
                print(
                    f"    * Tenkan-sen: {tenkan_sen:.2f} | Kijun-sen: {kijun_sen:.2f}"
                )
                print(f"    * Cloud (A/B): {senkou_span_a:.2f} / {senkou_span_b:.2f}")
                print(f"    * Chikou Span: {chikou_span:.2f}")
                print(f"    * Điểm Ichimoku: ~{ichimoku_score:.1f}")
            except:
                print(f"  - Ichimoku: Không có dữ liệu")

            print(f"  - Đề xuất: {recommendation} (Điểm: {score:.1f})")

            # --- BƯỚC 6: Trả về kết quả phân tích với nhiều thông tin hơn ---
            # Hàm trợ giúp để chuyển đổi giá trị an toàn sang float hoặc None
            def safe_float(val):
                """Chuyển đổi giá trị sang float, xử lý NaN/None."""
                try:
                    if pd.isna(val):
                        return None
                    return float(val)
                except (TypeError, ValueError):
                    return None

            # Trả về dictionary phong phú hơn
            return {
                # Thông tin cơ bản (giữ nguyên)
                "signal": signal,
                "score": float(score),  # Đảm bảo là kiểu float
                "current_price": float(current_price),
                "rsi_value": float(rsi_value),
                "ma10": float(ma10_value),
                "ma20": float(ma20_value),
                "ma50": float(ma50_value),
                "ma200": float(ma200_value),
                "rs": float(rs_value),
                "rs_point": float(rs_point_value),
                "recommendation": recommendation,
                # --- BẮT ĐẦU: Thêm nhiều thông tin kỹ thuật chi tiết ---
                # Giá và Volume
                "open": safe_float(last_row.get("Open")),
                "high": safe_float(last_row.get("High")),
                "low": safe_float(last_row.get("Low")),
                "volume": safe_float(last_row.get("Volume")),
                # MACD
                "macd": safe_float(macd_value),
                "macd_signal": safe_float(macd_signal),
                "macd_hist": safe_float(macd_hist),
                # Bollinger Bands
                "bb_upper": safe_float(bb_upper),
                "bb_lower": safe_float(bb_lower),
                # Volume Moving Averages
                "volume_sma_20": safe_float(last_row.get("Volume_SMA_20")),
                "volume_sma_50": safe_float(last_row.get("Volume_SMA_50")),
                # Ichimoku Cloud
                "ichimoku_tenkan_sen": safe_float(tenkan_sen),
                "ichimoku_kijun_sen": safe_float(kijun_sen),
                "ichimoku_senkou_span_a": safe_float(senkou_span_a),
                "ichimoku_senkou_span_b": safe_float(senkou_span_b),
                "ichimoku_chikou_span": safe_float(chikou_span),
                # RS & RS_Point (đã có, nhưng có thể thêm SMA nếu cần truy cập trực tiếp)
                "rs_sma_10": safe_float(last_row.get("RS_SMA_10")),
                "rs_sma_20": safe_float(last_row.get("RS_SMA_20")),
                "rs_sma_50": safe_float(last_row.get("RS_SMA_50")),
                "rs_sma_200": safe_float(last_row.get("RS_SMA_200")),
                "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10")),
                "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20")),
                "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50")),
                "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200")),
                # --- KẾT THÚC: Thêm nhiều thông tin kỹ thuật chi tiết ---
            }
            # --- HẾT PHẦN SỬA ĐỔI RETURN ---

        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu: {str(e)}")
            # Trả về dictionary lỗi với các khóa mới cũng được thêm vào để đảm bảo cấu trúc nhất quán
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
                # --- Thêm None cho các khóa mới khi có lỗi ---
                "open": None,
                "high": None,
                "low": None,
                "volume": None,
                "macd": None,
                "macd_signal": None,
                "macd_hist": None,
                "bb_upper": None,
                "bb_lower": None,
                "volume_sma_20": None,
                "volume_sma_50": None,
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
                # --- Hết phần thêm None ---
            }

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {str(e)}")
        traceback.print_exc()
        # Trả về dictionary lỗi với cấu trúc đầy đủ
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
            # --- Thêm None cho các khóa mới khi có lỗi nghiêm trọng ---
            "open": None,
            "high": None,
            "low": None,
            "volume": None,
            "macd": None,
            "macd_signal": None,
            "macd_hist": None,
            "bb_upper": None,
            "bb_lower": None,
            "volume_sma_20": None,
            "volume_sma_50": None,
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
            # --- Hết phần thêm None ---
        }


# ======================
# PHẦN 5: TÍCH HỢP PHÂN TÍCH BẰNG Google
# ======================
def analyze_with_gemini(symbol, trading_signal, forecast, financial_data=None):
    """Phân tích cổ phiếu bằng Google Qwen dựa trên dữ liệu kỹ thuật và BCTC"""

    try:
        # --- BẮT ĐẦU: Tính toán các giá trị trước để tránh lỗi định dạng ---
        # Hàm lambda nội tuyến để định dạng an toàn
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

        # Lấy và định dạng các giá trị RS và RS_Point trước
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

        # Lấy và định dạng các giá trị Ichimoku trước
        tenkan_val = safe_format(trading_signal.get("ichimoku_tenkan_sen"))
        kijun_val = safe_format(trading_signal.get("ichimoku_kijun_sen"))
        senkou_a_val = safe_format(trading_signal.get("ichimoku_senkou_span_a"))
        senkou_b_val = safe_format(trading_signal.get("ichimoku_senkou_span_b"))
        chikou_val = safe_format(trading_signal.get("ichimoku_chikou_span"))
        # --- KẾT THÚC: Tính toán các giá trị trước ---

        # Tạo prompt cho Qwen
        prompt = f"""
Hãy đóng vai một chuyên gia phân tích chứng khoán tại Việt Nam. Phân tích cổ phiếu {symbol} dựa trên các thông tin sau:

1. Phân tích kỹ thuật:
    - Giá hiện tại: {trading_signal['current_price']:,.0f} VND
    - RSI: {trading_signal['rsi_value']:.2f}
    - MA10: {trading_signal['ma10']:,.0f} VND
    - MA20: {trading_signal['ma20']:,.0f} VND
    - MA50: {trading_signal['ma50']:,.0f} VND
    - MA200: {trading_signal['ma200']:,.0f} VND
    --- Thông tin Bollinger Bands ---
    - BB Upper (Band trên): {safe_format(trading_signal.get('bb_upper'))}
    - BB Lower (Band dưới): {safe_format(trading_signal.get('bb_lower'))}
    --- Thông tin RS & RS_Point ---
    - RS (so với VNINDEX): {rs_val:.3f}
        * So sánh sức mạnh tương đối của cổ phiếu với chỉ số VNINDEX.
        * RS_SMA_10: {rs_sma10_val}
        * RS_SMA_20: {rs_sma20_val}
        * RS_SMA_50: {rs_sma50_val}
        * RS_SMA_200: {rs_sma200_val}
    - RS_Point: {rs_point_val:.2f}
        * Công thức: RS_Point = (ROC_63_ngày * 0.4) + (ROC_126_ngày * 0.2) + (ROC_189_ngày * 0.2) + (ROC_252_ngày * 0.2) * 100
        * Đo lường động lượng giá trong dài hạn.
        * RS_Point_SMA_10: {rs_point_sma10_val}
        * RS_Point_SMA_20: {rs_point_sma20_val}
        * RS_Point_SMA_50: {rs_point_sma50_val}
        * RS_Point_SMA_200: {rs_point_sma200_val}
    --- Thông tin Ichimoku Cloud ---
    - Tenkan-sen (Đường chuyển đổi): {tenkan_val}
    - Kijun-sen (Đường chuẩn): {kijun_val}
    - Senkou Span A (Leading Span A): {senkou_a_val}
    - Senkou Span B (Leading Span B): {senkou_b_val}
    - Chikou Span (Lagging Span): {chikou_val}
    -----------------------------

"""

        if financial_data is not None and not financial_data.empty:
            prompt += "\n2. Dữ liệu tài chính (BCTC) gần nhất:\n"
            try:
                # Lấy quý gần nhất
                financial_data_sorted = financial_data.copy()
                # Giới hạn số cột để tránh prompt quá dài
                prompt += f"{financial_data_sorted.head(20).to_string(index=False)}\n"
            except Exception as e:
                print(f"Lỗi khi xử lý dữ liệu tài chính: {str(e)}")
                prompt += "   - Không có dữ liệu tài chính chi tiết\n"
        else:
            prompt += "\n2. Không có dữ liệu tài chính\n"

        prompt += """
Yêu cầu phân tích:
- Tổng hợp phân tích kỹ thuật và cơ bản dựa trên tất cả các thông tin được cung cấp.
- Đánh giá sức mạnh tài chính của công ty dựa trên báo cáo tài chính (nếu có).
- Phân tích xu hướng giá hiện tại và các tín hiệu kỹ thuật, đặc biệt chú ý đến:
  * RS và các đường trung bình của nó (RS_SMA_10, RS_SMA_20, v.v.) để đánh giá sức mạnh tương đối.
  * RS_Point và công thức tính của nó để đánh giá động lượng dài hạn.
  * Các thành phần của Ichimoku Cloud để xác định xu hướng và vùng hỗ trợ/kháng cự.
- Nhận định rủi ro tiềm ẩn dựa trên phân tích kỹ thuật và bối cảnh thị trường.
- Dự báo triển vọng ngắn hạn (1-2 tuần) và trung hạn (1-3 tháng).
- Đưa ra khuyến nghị đầu tư cụ thể (Mua/Mua mạnh/Bán/Bán mạnh/Nắm giữ) kèm theo lý do phân tích.
Kết quả phân tích cần:
- Ngắn gọn, súc tích (không quá 500 từ).
- Chuyên nghiệp như một nhà phân tích chứng khoán.
- Bao gồm cả yếu tố thị trường tổng thể (VNINDEX).
- Có số liệu minh họa cụ thể từ các chỉ báo được cung cấp.
"""
        # Sử dụng Google để phân tích
        # print(prompt) # Bỏ comment nếu muốn xem prompt được tạo ra
        model = genai.GenerativeModel("gemini-2.5-flash")  # Hoặc model bạn đang dùng
        response = model.generate_content(prompt)

        if response and response.text:
            return response.text.strip()
        else:
            return "Không nhận được phản hồi từ Google."

    except Exception as e:
        import traceback

        print(f"Lỗi khi phân tích bằng Qwen: {str(e)}")
        print("Chi tiết lỗi:")
        traceback.print_exc()  # In traceback đầy đủ để dễ gỡ lỗi
        return "Không thể tạo phân tích bằng Google tại thời điểm này."


# ======================
# PHẦN 6: CHỨC NĂNG CHÍNH - CẢI TIẾN
# ======================
def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán với tích hợp Google và lựa chọn mô hình AI phù hợp (chỉ LSTM)"""
    print(f"\n{'='*50}")
    print(f"PHÂN TÍCH MÃ {symbol} VỚI AI")
    print(f"{'='*50}")
    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    financial_data = get_financial_data(symbol)
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    df_features = create_features(df_processed)
    if len(df_features) < 100:
        print(f"Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_features)} điểm)")
        return None

    # --- ĐÁNH GIÁ DỮ LIỆU VÀ ĐỀ XUẤT MÔ HÌNH AI ---
    ai_recommendation, ai_reason = evaluate_data_for_ai(df_features, symbol)

    model, scaler = None, None
    X_test_or_actual, y_test_or_pred, forecast_source = None, None, None
    forecast_dates, forecast_values = np.array([]), np.array([])

    # --- HUẤN LUYỆN MÔ HÌNH DỰA TRÊN ĐỀ XUẤT ---
    if len(df_features) < 100:
        print(
            f"Cảnh báo: Dữ liệu cho mã {symbol} quá ít ({len(df_features)} điểm) để huấn luyện mô hình AI hiệu quả."
        )
    else:
        print(f"\n🔔 ĐỀ XUẤT MỞ RỘNG: {ai_recommendation}")
        print(f"   Lý do: {ai_reason}")

        # Dù đề xuất gì, chỉ dùng LSTM
        print(f"\nĐang huấn luyện mô hình AI (LSTM) cho mã {symbol}...")
        model, scaler, X_test, y_test, y_pred = train_stock_model(
            df_features
        )  # Dùng phiên bản cải tiến
        if model is not None:
            X_test_or_actual = y_test
            y_test_or_pred = y_pred
            print(f"\nĐang dự báo giá cho 5 ngày tới bằng LSTM...")
            forecast_dates, forecast_values = predict_next_days(
                model, scaler, df_features
            )
        else:
            print("\n⚠️ Không thể huấn luyện mô hình LSTM.")

    # --- KẾT THÚC PHẦN ĐÁNH GIÁ VÀ AI ---
    print(f"\nĐang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_features)
    print(f"\nĐang phân tích bằng Google ...")
    gemini_analysis = analyze_with_gemini(
        symbol, trading_signal, (forecast_dates, forecast_values), financial_data
    )

    # In kết quả
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.2f}/100")

    # Kiểm tra điều kiện cho forecast
    if len(forecast_dates) > 0 and len(forecast_values) > 0:
        print(f"\nDỰ BÁO GIÁ CHO {len(forecast_dates)} NGÀY TIẾP THEO:")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            change = (
                (price - trading_signal["current_price"])
                / trading_signal["current_price"]
            ) * 100
            print(f"Ngày {i+1} ({date.date()}): {price:,.2f} VND ({change:+.2f}%)")
    else:
        print("\nKhông có dự báo giá do lỗi trong quá trình huấn luyện mô hình")
    print(f"\nPHÂN TÍCH TỔNG HỢP TỪ Google:")
    print(gemini_analysis)

    # - LƯU BÁO CÁO -
    # --- BẮT ĐẦU: Tạo báo cáo đầy đủ với nhiều dữ liệu hơn ---
    # Hàm trợ giúp để chuyển đổi giá trị an toàn sang float hoặc None khi cần thiết cho các phần riêng lẻ
    def safe_float(val):
        """Chuyển đổi giá trị sang float, xử lý NaN/None."""
        try:
            if val is None or (
                isinstance(val, float) and (np.isnan(val) or np.isinf(val))
            ):
                return None
            return float(val)
        except (TypeError, ValueError):
            return None

    # Bắt đầu với các thông tin cơ bản và các phần khác (forecast, AI)
    # Sau đó cập nhật/bổ sung với tất cả dữ liệu từ trading_signal
    report = {
        "symbol": symbol,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Các khóa cơ bản từ trading_signal (đã có)
        # Chúng ta sẽ cập nhật chúng từ trading_signal để đảm bảo nhất quán
        "current_price": safe_float(trading_signal.get("current_price")),
        "signal": trading_signal.get("signal"),
        "recommendation": trading_signal.get("recommendation"),
        "score": safe_float(trading_signal.get("score")),
        "rsi_value": safe_float(trading_signal.get("rsi_value")),
        "ma10": safe_float(trading_signal.get("ma10")),
        "ma20": safe_float(trading_signal.get("ma20")),
        "ma50": safe_float(trading_signal.get("ma50")),
        "ma200": safe_float(trading_signal.get("ma200")),
        "rs": safe_float(trading_signal.get("rs")),
        "rs_point": safe_float(trading_signal.get("rs_point")),
        # Forecast
        "forecast": (
            [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "price": safe_float(price),
                    "change_percent": safe_float(change),
                }
                for date, price, change in zip(
                    forecast_dates,
                    forecast_values,
                    [
                        (
                            (price - (trading_signal.get("current_price") or 0))
                            / (trading_signal.get("current_price") or 1)
                        )
                        * 100
                        for price in forecast_values
                    ],
                )
            ]
            if len(forecast_dates) > 0
            and len(forecast_values) > 0
            and trading_signal.get("current_price") is not None
            and trading_signal.get("current_price") != 0
            else []
        ),
        # AI Analysis
        "ai_recommendation": ai_recommendation,
        "ai_reason": ai_reason,
        "gemini_analysis": gemini_analysis,
    }

    # --- BẮT ĐẦU: Thêm tất cả các khóa từ trading_signal vào report ---
    # Điều này đảm bảo rằng bất kỳ khóa nào được thêm vào kết quả trả về của plot_stock_analysis
    # (bao gồm các chỉ báo kỹ thuật chi tiết) đều được lưu vào báo cáo.
    # Nó ghi đè lên các khóa đã được định nghĩa ở trên nếu có, đảm bảo giá trị từ trading_signal được sử dụng.
    # Ví dụ: nếu trading_signal có khóa 'open', nó sẽ được thêm/cập nhật vào report.
    report.update(trading_signal)
    # --- KẾT THÚC: Thêm tất cả các khóa từ trading_signal ---

    # Lưu báo cáo
    with open(f"vnstocks_data/{symbol}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")

    return report
    # --- KẾT THÚC: Tạo báo cáo đầy đủ ---


def screen_stocks():
    """Quét và phân tích nhiều mã chứng khoán"""
    print(f"\n{'='*50}")
    print("QUÉT VÀ PHÂN TÍCH DANH SÁCH MÃ CHỨNG KHOÁN")
    print(f"{'='*50}")
    # Lấy danh sách mã
    stock_list = get_vnstocks_list()
    # Danh sách để lưu kết quả
    results = []
    # Phân tích từng mã
    # --- SỬA Ở ĐÂY: Tăng số lượng mã để quét, ví dụ lên 20 ---
    # for symbol in stock_list['symbol'].head(10):  # Phân tích 10 mã đầu tiên để demo
    for symbol in stock_list["symbol"].head(20):  # Phân tích 20 mã đầu tiên
        # --- HẾT PHẦN SỬA ---
        try:
            print(f"\nPhân tích mã {symbol}...")
            report = analyze_stock(symbol)  # Gọi trực tiếp hàm analyze_stock
            if report and report["signal"] != "LỖI":
                results.append(report)
            time.sleep(1)  # Dừng 1 giây giữa các request
        except Exception as e:
            print(f"Lỗi khi phân tích mã {symbol}: {str(e)}")
            # traceback.print_exc() # Có thể bỏ comment nếu muốn xem chi tiết lỗi
            continue
    # Tạo báo cáo tổng hợp
    if results:
        # Sắp xếp theo điểm phân tích
        results.sort(key=lambda x: x["score"], reverse=True)

        # --- BẮT ĐẦU: Cập nhật phần tạo df_results để thêm thông tin Ichimoku ---
        # Hàm trợ giúp để lấy giá trị an toàn từ report, đặc biệt là các khóa lồng nhau
        def get_nested_value(report_dict, key_path, default=None):
            """
            Lấy giá trị từ dictionary lồng nhau.
            key_path là một chuỗi với các khóa ngăn cách bởi dấu chấm, ví dụ: 'technical_data.ichimoku_tenkan_sen'
            """
            keys = key_path.split(".")
            current_dict = report_dict
            try:
                for key in keys:
                    # Kiểm tra nếu current_dict là dict và key tồn tại
                    if isinstance(current_dict, dict) and key in current_dict:
                        current_dict = current_dict[key]
                    else:
                        # Nếu key không tồn tại hoặc current_dict không phải dict, trả về default
                        return default
                # Nếu vòng lặp hoàn tất, current_dict là giá trị cuối cùng
                # Chuyển sang float nếu có thể, nếu không trả về nguyên gốc hoặc default nếu NaN/None
                if current_dict is None or (
                    isinstance(current_dict, float)
                    and (pd.isna(current_dict) or np.isinf(current_dict))
                ):
                    return default
                return float(current_dict)
            except (ValueError, TypeError):
                # Nếu không chuyển được sang float, trả về default
                return default

        # Tạo danh sách dictionary cho DataFrame, bao gồm các chỉ báo mới
        data_for_df = []
        for r in results:
            row_data = {
                "Mã": r["symbol"],
                "Giá": r["current_price"],
                "Điểm": r["score"],
                "Tín hiệu": r["signal"],
                "Đề xuất": r["recommendation"],
                "RSI": r["rsi_value"],
                "MA10": r["ma10"],
                "MA20": r["ma20"],
                "MA50": r["ma50"],
                "MA200": r["ma200"],
                "RS": r["rs"],
                "RS_Point": r["rs_point"],
                # --- Thêm các thành phần Ichimoku ---
                # Giả sử các giá trị Ichimoku được lưu trực tiếp trong report như các khóa riêng lẻ
                # (đây là cách trong phiên bản code bạn cung cấp gần đây)
                "Ichimoku_Tenkan": r.get(
                    "ichimoku_tenkan_sen"
                ),  # Truy cập trực tiếp khóa
                "Ichimoku_Kijun": r.get("ichimoku_kijun_sen"),
                "Ichimoku_Senkou_A": r.get("ichimoku_senkou_span_a"),
                "Ichimoku_Senkou_B": r.get("ichimoku_senkou_span_b"),
                "Ichimoku_Chikou": r.get("ichimoku_chikou_span"),
                # Nếu bạn lưu chúng trong một khóa lồng như 'technical_data', dùng get_nested_value:
                # 'Ichimoku_Tenkan': get_nested_value(r, 'technical_data.ichimoku_tenkan_sen'),
                # 'Ichimoku_Kijun': get_nested_value(r, 'technical_data.ichimoku_kijun_sen'),
                # 'Ichimoku_Senkou_A': get_nested_value(r, 'technical_data.ichimoku_senkou_span_a'),
                # 'Ichimoku_Senkou_B': get_nested_value(r, 'technical_data.ichimoku_senkou_span_b'),
                # 'Ichimoku_Chikou': get_nested_value(r, 'technical_data.ichimoku_chikou_span'),
                # --- Thêm các chỉ báo khác nếu cần ---
                # 'MACD': r.get('macd'),
                # 'BB_Upper': r.get('bb_upper'),
                # 'Volume': r.get('volume'),
            }
            # Áp dụng safe_float cho các cột mới nếu cần đảm bảo kiểu dữ liệu
            # row_data['Ichimoku_Tenkan'] = get_nested_value(r, 'ichimoku_tenkan_sen') # Nếu dùng hàm trợ giúp
            data_for_df.append(row_data)

        # Tạo DataFrame từ danh sách đã cập nhật
        df_results = pd.DataFrame(data_for_df)
        # --- KẾT THÚC: Cập nhật phần tạo df_results ---

        # Lưu báo cáo tổng hợp
        df_results.to_csv("vnstocks_data/stock_screening_report.csv", index=False)
        print(f"\n{'='*50}")
        print("KẾT QUẢ QUÉT MÃ")
        print(f"{'='*50}")
        # Cập nhật danh sách cột được in ra để bao gồm Ichimoku nếu muốn, hoặc giữ nguyên
        print_cols = [
            "Mã",
            "Giá",
            "Điểm",
            "Tín hiệu",
            "Đề xuất",
        ]  # Có thể thêm 'Ichimoku_Tenkan', v.v.
        print(df_results[print_cols])  # In các cột đã chọn

        # Vẽ biểu đồ so sánh (giữ nguyên)
        try:
            plt.figure(figsize=(14, 6))  # Tăng chiều rộng một chút cho 20 mã
            # --- SỬA Ở ĐÂY: Vẽ top 20 thay vì top 10 nếu muốn ---
            # sns.barplot(x='Mã', y='Điểm', data=df_results.head(10), palette='viridis')
            sns.barplot(
                x="Mã", y="Điểm", data=df_results.head(20), palette="viridis"
            )  # Vẽ top 20
            # --- HẾT PHẦN SỬA ---
            plt.title("Top Điểm phân tích các mã chứng khoán")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig("vnstocks_data/stock_screening_comparison.png")
            plt.close()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ so sánh: {str(e)}")

        print(
            f"\nĐã lưu báo cáo tổng hợp vào file 'vnstocks_data/stock_screening_report.csv'"
        )
        print("Đã tạo biểu đồ so sánh các mã")
        return df_results
    else:
        print("\nKhông có kết quả phân tích nào để tạo báo cáo tổng hợp.")
    return None


# ======================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# ======================
if __name__ == "__main__":
    print("==============================================")
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM VỚI AI")
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE")
    print("==============================================")
    # Lấy dữ liệu thị trường
    market_data = get_market_data()
    analyze_stock("DRI")
    # screen_stocks() # Bỏ comment nếu muốn quét nhiều mã
    print(
        "\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'."
    )
