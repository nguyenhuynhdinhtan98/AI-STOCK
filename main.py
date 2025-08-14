import os
import time
import json
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import ta  # Technical Analysis library
import warnings
warnings.filterwarnings('ignore')

# ======================
# CẤU HÌNH VÀ THƯ VIỆN
# ======================

# Tạo thư mục lưu trữ dữ liệu
if not os.path.exists('vnstocks_data'):
    os.makedirs('vnstocks_data')

# Cấu hình API
API_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'application/json',
    'Referer': 'https://vnstocks.com/'
}

# ======================
# PHẦN 1: THU THẬP DỮ LIỆU
# ======================

def get_vnstocks_list():
    """Lấy danh sách tất cả các mã chứng khoán trên thị trường Việt Nam"""
    try:
        # API của vnstocks.com (giả định)
        url = "https://api.vnstocks.com/v1/stocks/list"
        response = requests.get(url, headers=API_HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            df.to_csv('vnstocks_data/stock_list.csv', index=False)
            print(f"Đã lưu danh sách {len(df)} mã chứng khoán vào file 'vnstocks_data/stock_list.csv'")
            return df
        else:
            print(f"Lỗi khi lấy danh sách mã chứng khoán: {response.status_code}")
            # Nếu API không hoạt động, sử dụng danh sách mẫu
            print("Đang sử dụng danh sách mã chứng khoán mẫu...")
            sample_stocks = ['VNM', 'VCB', 'FPT', 'GAS', 'BID', 'CTG', 'MWG', 'PNJ', 'HPG', 'STB']
            df = pd.DataFrame(sample_stocks, columns=['symbol'])
            df.to_csv('vnstocks_data/stock_list.csv', index=False)
            return df
            
    except Exception as e:
        print(f"Exception khi lấy danh sách mã: {str(e)}")
        # Sử dụng danh sách mẫu
        sample_stocks = ['VNM', 'VCB', 'FPT', 'GAS', 'BID', 'CTG', 'MWG', 'PNJ', 'HPG', 'STB']
        df = pd.DataFrame(sample_stocks, columns=['symbol'])
        df.to_csv('vnstocks_data/stock_list.csv', index=False)
        return df

def get_stock_data(symbol, period="1y"):
    """Lấy dữ liệu lịch sử của một mã chứng khoán"""
    try:
        # Thử lấy dữ liệu từ API của vnstocks.com
        url = f"https://api.vnstocks.com/v1/stocks/{symbol}/history"
        params = {
            'period': period,
            'interval': '1d'
        }
        response = requests.get(url, headers=API_HEADERS, params=params)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data['data'])
            # Chuyển đổi định dạng thời gian
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            df.sort_index(inplace=True)
            
            # Lưu dữ liệu
            df.to_csv(f'vnstocks_data/{symbol}_data.csv')
            print(f"Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'")
            return df
        else:
            print(f"Lỗi API cho mã {symbol}: {response.status_code}")
            # Thử dùng Yahoo Finance làm phương án dự phòng
            print(f"Đang sử dụng Yahoo Finance để lấy dữ liệu cho mã {symbol}...")
            df = yf.download(f"{symbol}.VN", period=period, interval='1d')
            if not df.empty:
                df.to_csv(f'vnstocks_data/{symbol}_data.csv')
                return df
            else:
                print(f"Không thể lấy dữ liệu cho mã {symbol}")
                return None
                
    except Exception as e:
        print(f"Exception khi lấy dữ liệu cho mã {symbol}: {str(e)}")
        try:
            # Thử dùng Yahoo Finance
            print(f"Đang sử dụng Yahoo Finance để lấy dữ liệu cho mã {symbol}...")
            df = yf.download(f"{symbol}.VN", period=period, interval='1d')
            if not df.empty:
                df.to_csv(f'vnstocks_data/{symbol}_data.csv')
                return df
        except:
            print(f"Không thể lấy dữ liệu cho mã {symbol} từ bất kỳ nguồn nào")
            return None

def get_market_data():
    """Lấy dữ liệu thị trường tổng thể"""
    try:
        # Lấy dữ liệu VN-Index
        vnindex = yf.download("^VNI", period="5y", interval="1d")
        vnindex.to_csv('vnstocks_data/vnindex_data.csv')
        
        # Lấy dữ liệu VN30-Index
        vn30 = yf.download("^VN30", period="5y", interval="1d")
        vn30.to_csv('vnstocks_data/vn30_data.csv')
        
        print("Đã lưu dữ liệu chỉ số thị trường vào thư mục 'vnstocks_data/'")
        return {
            'vnindex': vnindex,
            'vn30': vn30
        }
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu thị trường: {str(e)}")
        return None

# ======================
# PHẦN 2: TIỀN XỬ LÝ DỮ LIỆU
# ======================

def preprocess_stock_data(df):
    """Tiền xử lý dữ liệu chứng khoán"""
    if df is None or df.empty:
        return None
    
    # Sao chép DataFrame để tránh thay đổi dữ liệu gốc
    df_processed = df.copy()
    
    # Xử lý dữ liệu thiếu
    df_processed = df_processed.interpolate(method='linear', limit_direction='both')
    
    # Tính toán các chỉ báo kỹ thuật
    # Đường trung bình động
    df_processed['MA20'] = df_processed['Close'].rolling(window=20).mean()
    df_processed['MA50'] = df_processed['Close'].rolling(window=50).mean()
    df_processed['MA200'] = df_processed['Close'].rolling(window=200).mean()
    
    # RSI
    delta = df_processed['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_processed['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df_processed['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_processed['Close'].ewm(span=26, adjust=False).mean()
    df_processed['MACD'] = exp1 - exp2
    df_processed['Signal_Line'] = df_processed['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df_processed['20_std'] = df_processed['Close'].rolling(window=20).std()
    df_processed['upper_band'] = df_processed['MA20'] + (df_processed['20_std'] * 2)
    df_processed['lower_band'] = df_processed['MA20'] - (df_processed['20_std'] * 2)
    
    # Volume Moving Average
    df_processed['Vol_MA20'] = df_processed['Volume'].rolling(window=20).mean()
    
    # Tỷ lệ thanh khoản
    df_processed['liquidity_ratio'] = df_processed['Volume'] / df_processed['Vol_MA20']
    
    # Xử lý giá trị NaN
    df_processed = df_processed.dropna()
    
    return df_processed

def create_features(df):
    """Tạo các đặc trưng cho mô hình AI"""
    df_features = df.copy()
    
    # Thêm các đặc trưng kỹ thuật
    df_features['return'] = df_features['Close'].pct_change()
    df_features['volatility'] = df_features['return'].rolling(window=20).std() * np.sqrt(252)
    
    # Đặc trưng theo ngày trong tuần
    df_features['day_of_week'] = df_features.index.dayofweek
    
    # Đặc trưng theo tháng
    df_features['month'] = df_features.index.month
    
    # Đặc trưng theo quý
    df_features['quarter'] = df_features.index.quarter
    
    # Tín hiệu cắt nhau của MA
    df_features['ma_crossover'] = np.where(df_features['MA20'] > df_features['MA50'], 1, 0)
    
    # Tín hiệu RSI
    df_features['rsi_signal'] = np.where(df_features['RSI'] < 30, 1, 
                                       np.where(df_features['RSI'] > 70, -1, 0))
    
    # Tín hiệu MACD
    df_features['macd_signal'] = np.where(df_features['MACD'] > df_features['Signal_Line'], 1, 0)
    
    # Tỷ lệ giá so với Bollinger Bands
    df_features['bb_position'] = (df_features['Close'] - df_features['lower_band']) / \
                              (df_features['upper_band'] - df_features['lower_band'])
    
    # Xử lý giá trị NaN
    df_features = df_features.dropna()
    
    return df_features

def prepare_time_series_data(df, target_col='Close', sequence_length=60):
    """
    Chuẩn bị dữ liệu chuỗi thời gian cho mô hình LSTM
    sequence_length: Số ngày lịch sử để dự báo
    """
    # Chọn các cột cần thiết
    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA20', 'MA50', 
                   'RSI', 'MACD', 'Signal_Line', 'volatility', 'bb_position']
    
    # Chỉ lấy các cột có trong DataFrame
    available_features = [col for col in feature_cols if col in df.columns]
    df = df[available_features].copy()
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)
    
    # Tạo chuỗi dữ liệu
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        # Dự báo giá đóng cửa tiếp theo
        y.append(scaled_data[i, df.columns.get_loc(target_col)])
    
    X, y = np.array(X), np.array(y)
    
    return X, y, scaler, df.columns.tolist()

# ======================
# PHẦN 3: MÔ HÌNH AI
# ======================

def build_lstm_model(input_shape):
    """Xây dựng mô hình LSTM cho dự báo giá cổ phiếu"""
    model = Sequential()
    
    # Lớp LSTM đầu tiên
    model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Lớp LSTM thứ hai
    model.add(LSTM(units=70, return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
    # Lớp LSTM thứ ba
    model.add(LSTM(units=50))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))
    
    # Lớp đầu ra
    model.add(Dense(units=1))
    
    # Biên dịch mô hình
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    return model

def train_stock_model(symbol, sequence_length=60, epochs=50, batch_size=32):
    """Huấn luyện mô hình dự báo cho một mã cổ phiếu"""
    # Lấy và xử lý dữ liệu
    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"Không có dữ liệu cho mã {symbol}")
        return None, None, None
    
    # Tiền xử lý
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None, None, None
    
    # Tạo đặc trưng
    df_features = create_features(df_processed)
    
    # Chuẩn bị dữ liệu cho mô hình
    X, y, scaler, feature_names = prepare_time_series_data(
        df_features, 
        target_col='Close',
        sequence_length=sequence_length
    )
    
    # Chia tập train/test
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Xây dựng và huấn luyện mô hình
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # Early stopping để tránh overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    print(f"Đang huấn luyện mô hình cho mã {symbol}...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop],
        verbose=1
    )
    
    # Đánh giá mô hình
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Chuẩn hóa lại dự báo
    train_predictions = train_predictions.reshape(-1, 1)
    test_predictions = test_predictions.reshape(-1, 1)
    
    # Tạo mảng đầy đủ để inverse transform
    dummy = np.zeros((len(train_predictions), len(feature_names)))
    dummy[:, feature_names.index('Close')] = train_predictions.flatten()
    train_predictions = scaler.inverse_transform(dummy)[:, feature_names.index('Close')]
    
    dummy = np.zeros((len(test_predictions), len(feature_names)))
    dummy[:, feature_names.index('Close')] = test_predictions.flatten()
    test_predictions = scaler.inverse_transform(dummy)[:, feature_names.index('Close')]
    
    # Lấy giá thực tế
    actual_train = df_features['Close'].values[sequence_length:train_size+sequence_length]
    actual_test = df_features['Close'].values[train_size+sequence_length:]
    
    # Tính toán các chỉ số đánh giá
    train_rmse = np.sqrt(mean_squared_error(actual_train, train_predictions))
    test_rmse = np.sqrt(mean_squared_error(actual_test, test_predictions))
    train_mae = mean_absolute_error(actual_train, train_predictions)
    test_mae = mean_absolute_error(actual_test, test_predictions)
    train_r2 = r2_score(actual_train, train_predictions)
    test_r2 = r2_score(actual_test, test_predictions)
    
    print(f"\nĐánh giá mô hình cho mã {symbol}:")
    print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
    print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
    print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")
    
    # Lưu mô hình
    model.save(f'vnstocks_data/{symbol}_model.h5')
    
    # Lưu scaler
    np.save(f'vnstocks_data/{symbol}_scaler.npy', scaler)
    
    return model, scaler, feature_names

def predict_next_days(model, scaler, feature_names, df, sequence_length=60, days=5):
    """Dự báo giá cho các ngày tiếp theo"""
    # Lấy dữ liệu gần nhất
    last_sequence = df[feature_names].values[-sequence_length:]
    
    # Chuẩn hóa
    last_sequence_scaled = scaler.transform(last_sequence)
    
    # Tạo dự báo cho các ngày tiếp theo
    predictions = []
    current_sequence = last_sequence_scaled.copy()
    
    for _ in range(days):
        # Dự báo ngày tiếp theo
        next_pred = model.predict(current_sequence.reshape(1, sequence_length, len(feature_names)))[0][0]
        
        # Thêm dự báo vào kết quả
        predictions.append(next_pred)
        
        # Cập nhật chuỗi để dự báo ngày tiếp theo
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, feature_names.index('Close')] = next_pred
    
    # Chuyển đổi dự báo về giá trị thực
    dummy = np.zeros((len(predictions), len(feature_names)))
    dummy[:, feature_names.index('Close')] = predictions
    predictions = scaler.inverse_transform(dummy)[:, feature_names.index('Close')]
    
    # Tạo index cho các ngày dự báo
    last_date = df.index[-1]
    forecast_dates = [last_date + timedelta(days=i+1) for i in range(days)]
    
    return forecast_dates, predictions

# ======================
# PHẦN 4: PHÂN TÍCH VÀ TRỰC QUAN HÓA
# ======================

def plot_stock_analysis(symbol, df, model=None, scaler=None, feature_names=None, sequence_length=60):
    """Vẽ biểu đồ phân tích kỹ thuật và dự báo"""
    plt.figure(figsize=(14, 10))
    
    # Biểu đồ giá
    plt.subplot(3, 1, 1)
    plt.plot(df.index, df['Close'], label='Giá đóng cửa', color='blue')
    plt.plot(df.index, df['MA20'], label='MA20', alpha=0.7)
    plt.plot(df.index, df['MA50'], label='MA50', alpha=0.7)
    plt.title(f'Phân tích kỹ thuật - {symbol}')
    plt.legend()
    plt.grid(True)
    
    # Vẽ vùng RSI
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df['RSI'], label='RSI', color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.5)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.5)
    plt.fill_between(df.index, 70, 100, color='red', alpha=0.1)
    plt.fill_between(df.index, 0, 30, color='green', alpha=0.1)
    plt.title('Chỉ báo RSI')
    plt.legend()
    plt.grid(True)
    
    # Vẽ Bollinger Bands
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df['Close'], label='Giá đóng cửa', color='blue', alpha=0.7)
    plt.plot(df.index, df['MA20'], label='MA20', color='orange', alpha=0.7)
    plt.plot(df.index, df['upper_band'], label='Upper Band', color='gray', linestyle='--')
    plt.plot(df.index, df['lower_band'], label='Lower Band', color='gray', linestyle='--')
    plt.title('Bollinger Bands')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'vnstocks_data/{symbol}_analysis.png')
    plt.close()
    
    # Nếu có mô hình, vẽ dự báo
    if model is not None and scaler is not None and feature_names is not None:
        try:
            forecast_dates, forecast_values = predict_next_days(model, scaler, feature_names, df, sequence_length)
            
            plt.figure(figsize=(12, 6))
            plt.plot(df.index, df['Close'], label='Giá thực tế', color='blue')
            
            # Chỉ lấy dữ liệu 60 ngày gần nhất để hiển thị rõ hơn
            last_60_days = max(60, len(df) - len(forecast_dates))
            plt.plot(df.index[-last_60_days:], df['Close'].values[-last_60_days:], 
                    label='Giá thực tế (60 ngày gần nhất)', color='blue')
            
            plt.plot(forecast_dates, forecast_values, label='Dự báo', color='red', linestyle='--', marker='o')
            
            plt.title(f'Dự báo giá {symbol} cho {len(forecast_dates)} ngày tiếp theo')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'vnstocks_data/{symbol}_forecast.png')
            plt.close()
            
            print(f"Đã tạo biểu đồ dự báo cho {symbol}")
            return forecast_dates, forecast_values
        except Exception as e:
            print(f"Lỗi khi tạo dự báo cho {symbol}: {str(e)}")
    
    return None, None

def generate_trading_signal(symbol, df):
    """Tạo tín hiệu giao dịch dựa trên phân tích kỹ thuật"""
    latest = df.iloc[-1]
    
    # Tín hiệu từ MA
    ma_signal = 0
    if latest['MA20'] > latest['MA50']:
        ma_signal = 1  # Tín hiệu mua
    elif latest['MA20'] < latest['MA50']:
        ma_signal = -1  # Tín hiệu bán
    
    # Tín hiệu từ RSI
    rsi_signal = 0
    if latest['RSI'] < 30:
        rsi_signal = 1  # Quá bán, tín hiệu mua
    elif latest['RSI'] > 70:
        rsi_signal = -1  # Quá mua, tín hiệu bán
    
    # Tín hiệu từ MACD
    macd_signal = 0
    if latest['MACD'] > latest['Signal_Line']:
        macd_signal = 1  # Tín hiệu mua
    elif latest['MACD'] < latest['Signal_Line']:
        macd_signal = -1  # Tín hiệu bán
    
    # Tín hiệu từ Bollinger Bands
    bb_signal = 0
    if latest['Close'] < latest['lower_band']:
        bb_signal = 1  # Giá ở dưới lower band, tín hiệu mua
    elif latest['Close'] > latest['upper_band']:
        bb_signal = -1  # Giá ở trên upper band, tín hiệu bán
    
    # Tổng hợp tín hiệu
    total_signal = ma_signal + rsi_signal + macd_signal + bb_signal
    
    # Xác định mức độ tín hiệu
    strength = abs(total_signal)
    if strength >= 3:
        strength_text = "Mạnh"
    elif strength == 2:
        strength_text = "Trung bình"
    else:
        strength_text = "Yếu"
    
    # Xác định hướng tín hiệu
    if total_signal > 0:
        signal = f"Tín hiệu MUA {strength_text}"
        recommendation = "Nên xem xét mua"
    elif total_signal < 0:
        signal = f"Tín hiệu BÁN {strength_text}"
        recommendation = "Nên xem xét bán"
    else:
        signal = "Không có tín hiệu rõ ràng"
        recommendation = "Theo dõi thêm"
    
    # Tính toán điểm số
    score = (total_signal + 4) * 25  # Chuyển đổi thành thang điểm 0-100
    
    return {
        'symbol': symbol,
        'signal': signal,
        'recommendation': recommendation,
        'score': score,
        'ma_signal': ma_signal,
        'rsi_signal': rsi_signal,
        'macd_signal': macd_signal,
        'bb_signal': bb_signal,
        'current_price': latest['Close'],
        'rsi_value': latest['RSI'],
        'ma20': latest['MA20'],
        'ma50': latest['MA50']
    }

# ======================
# PHẦN 5: CHỨC NĂNG CHÍNH
# ======================

def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán"""
    print(f"\n{'='*50}")
    print(f"PHÂN TÍCH MÃ {symbol}")
    print(f"{'='*50}")
    
    # Lấy và xử lý dữ liệu
    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    
    # Tiền xử lý
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    
    # Tạo đặc trưng
    df_features = create_features(df_processed)
    
    # Huấn luyện mô hình AI
    print(f"\nĐang huấn luyện mô hình AI cho mã {symbol}...")
    model, scaler, feature_names = train_stock_model(symbol)
    
    # Tạo phân tích kỹ thuật và dự báo
    print(f"\nĐang tạo biểu đồ phân tích cho mã {symbol}...")
    forecast_dates, forecast_values = plot_stock_analysis(symbol, df_features, model, scaler, feature_names)
    
    # Tạo tín hiệu giao dịch
    print(f"\nĐang tạo tín hiệu giao dịch cho mã {symbol}...")
    trading_signal = generate_trading_signal(symbol, df_features)
    
    # In kết quả
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:.2f}")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.1f}/100")
    
    if forecast_dates and len(forecast_dates) > 0:
        print(f"\nDỰ BÁO GIÁ CHO {len(forecast_dates)} NGÀY TIẾP THEO:")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            change = ((price - trading_signal['current_price']) / trading_signal['current_price']) * 100
            print(f"Ngày {i+1} ({date.date()}): {price:.2f} ({change:+.2f}%)")
    
    # Lưu báo cáo
    report = {
        'symbol': symbol,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'current_price': float(trading_signal['current_price']),
        'signal': trading_signal['signal'],
        'recommendation': trading_signal['recommendation'],
        'score': float(trading_signal['score']),
        'rsi': float(trading_signal['rsi_value']),
        'ma20': float(trading_signal['ma20']),
        'ma50': float(trading_signal['ma50']),
        'forecast': [{
            'date': date.strftime("%Y-%m-%d"),
            'price': float(price),
            'change_percent': float(((price - trading_signal['current_price']) / trading_signal['current_price']) * 100)
        } for date, price in zip(forecast_dates, forecast_values)] if forecast_dates else []
    }
    
    with open(f'vnstocks_data/{symbol}_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    
    print(f"\nĐã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")
    
    return report

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
    for symbol in stock_list['symbol'].head(10):  # Chỉ phân tích 10 mã đầu tiên để thử nghiệm
        try:
            print(f"\nPhân tích mã {symbol}...")
            report = analyze_stock(symbol)
            if report:
                results.append(report)
            time.sleep(1)  # Dừng 1 giây giữa các request để không bị chặn
        except Exception as e:
            print(f"Lỗi khi phân tích mã {symbol}: {str(e)}")
            continue
    
    # Tạo báo cáo tổng hợp
    if results:
        # Sắp xếp theo điểm phân tích
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Tạo DataFrame
        df_results = pd.DataFrame([{
            'Mã': r['symbol'],
            'Giá': r['current_price'],
            'Điểm': r['score'],
            'Tín hiệu': r['signal'],
            'Đề xuất': r['recommendation'],
            'RSI': r['rsi'],
            'MA20': r['ma20'],
            'MA50': r['ma50']
        } for r in results])
        
        # Lưu báo cáo tổng hợp
        df_results.to_csv('vnstocks_data/stock_screening_report.csv', index=False)
        
        print(f"\n{'='*50}")
        print("KẾT QUẢ QUÉT MÃ")
        print(f"{'='*50}")
        print(df_results[['Mã', 'Giá', 'Điểm', 'Tín hiệu', 'Đề xuất']])
        
        # Vẽ biểu đồ so sánh
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Mã', y='Điểm', data=df_results, palette='viridis')
        plt.title('Điểm phân tích các mã chứng khoán')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('vnstocks_data/stock_screening_comparison.png')
        plt.close()
        
        print(f"\nĐã lưu báo cáo tổng hợp vào file 'vnstocks_data/stock_screening_report.csv'")
        print("Đã tạo biểu đồ so sánh các mã")
        
        return df_results
    
    return None

# ======================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# ======================

if __name__ == "__main__":
    print("==============================================")
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM VỚI AI")
    print("==============================================")
    
    # Lấy dữ liệu thị trường
    market_data = get_market_data()
    
    # Lựa chọn chế độ
    print("\nChọn chế độ phân tích:")
    print("1. Phân tích một mã cụ thể")
    print("2. Quét và phân tích danh sách mã")
    
    choice = input("\nNhập lựa chọn của bạn (1/2): ")
    
    if choice == "1":
        symbol = input("Nhập mã chứng khoán cần phân tích (ví dụ: VNM, VCB): ").strip().upper()
        analyze_stock(symbol)
    elif choice == "2":
        screen_stocks()
    else:
        print("Lựa chọn không hợp lệ!")
    
    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")