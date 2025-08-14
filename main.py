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
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import ta
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
from vnstock import *
import traceback
from vnstock.explorer.vci import Quote, Finance
from google import genai
warnings.filterwarnings('ignore')

# ======================
# CẤU HÌNH VÀ THƯ VIỆN
# ======================

# Tải biến môi trường cho Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Không tìm thấy khóa API Gemini. Vui lòng kiểm tra file .env")
    exit()

client = genai.Client(api_key= GOOGLE_API_KEY)

# Tạo thư mục lưu trữ dữ liệu
if not os.path.exists('vnstocks_data'):
    os.makedirs('vnstocks_data')

# ======================
# PHẦN 1: THU THẬP DỮ LIỆU
# ======================

def get_vnstocks_list():
    """Lấy danh sách tất cả các mã chứng khoán trên thị trường Việt Nam sử dụng vnstock v2"""
    try:
        df = listing_companies()
        
        if df is not None and not df.empty:
            # Lọc chỉ lấy mã cổ phiếu (loại bỏ chứng chỉ quỹ, trái phiếu...)
            df = df[df['organType'] == 'DN']
            symbols = df[['ticker']].rename(columns={'ticker': 'symbol'})
            symbols.to_csv('vnstocks_data/stock_list.csv', index=False)
            print(f"Đã lưu danh sách {len(symbols)} mã chứng khoán vào file 'vnstocks_data/stock_list.csv'")
            return symbols
        else:
            print("Không lấy được danh sách từ vnstock, sử dụng danh sách mẫu")
            sample_stocks = ['VNM', 'VCB', 'FPT', 'GAS', 'BID', 'CTG', 'MWG', 'PNJ', 'HPG', 'STB']
            df = pd.DataFrame(sample_stocks, columns=['symbol'])
            df.to_csv('vnstocks_data/stock_list.csv', index=False)
            return df
            
    except Exception as e:
        print(f"Exception khi lấy danh sách mã: {str(e)}")
        sample_stocks = ['VNM', 'VCB', 'FPT', 'GAS', 'BID', 'CTG', 'MWG', 'PNJ', 'HPG', 'STB']
        df = pd.DataFrame(sample_stocks, columns=['symbol'])
        df.to_csv('vnstocks_data/stock_list.csv', index=False)
        return df

def get_stock_data(symbol):
    """Lấy dữ liệu lịch sử của một mã chứng khoán sử dụng vnstock v2 mới theo tài liệu"""
    try:
        # Sử dụng cú pháp mới theo tài liệu: stock(symbol, period).price()
        quote = Quote(symbol)
        df = quote.history(start='2012-01-01', end='2030-1-1', interval='1D')
        if df is not None and not df.empty:
            # Chuẩn hóa tên cột theo chuẩn mới
            df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            
            # Xử lý cột Date
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)
            
            # Lưu dữ liệu
            df.to_csv(f'vnstocks_data/{symbol}_data.csv')
            print(f"Đã lưu dữ liệu cho mã {symbol} vào file 'vnstocks_data/{symbol}_data.csv'")
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
        financial_data = financial_obj.ratio(period='quarter', lang='en', flatten_columns=True)
        if financial_data is not None and not financial_data.empty:
            # Lưu dữ liệu
            financial_data.to_csv(f'vnstocks_data/{symbol}_financial.csv', index=False)
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
        quoteVNI = Quote(symbol='VNINDEX')
        vnindex = quoteVNI.history(start='2012-01-01', end='2030-1-1', interval='1D')
        if vnindex is not None and not vnindex.empty:
            vnindex.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            vnindex['Date'] = pd.to_datetime(vnindex['Date'])
            vnindex.set_index('Date', inplace=True)
            vnindex.sort_index(inplace=True)
            vnindex.to_csv('vnstocks_data/vnindex_data.csv')
        
        # Lấy dữ liệu VN30-Index
        quoteVN30 = Quote(symbol='VN30')
        vn30 = quoteVN30.history(start='2012-01-01', end='2030-1-1', interval='1D')
        if vn30 is not None and not vn30.empty:
            vn30.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            }, inplace=True)
            vn30['Date'] = pd.to_datetime(vn30['Date'])
            vn30.set_index('Date', inplace=True)
            vn30.sort_index(inplace=True)
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
    df['returns'] = df['Close'].pct_change()
    df['MA_10'] = df['Close'].rolling(window=10).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['volatility'] = df['returns'].rolling(window=10).std()
    
    # Drop initial rows with NaN values from technical features
    df.dropna(inplace=True)
    
    return df

def create_features(df):
    """
    Generates technical indicators using pure pandas/numpy
    """
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Momentum and volume features
    df['Momentum'] = df['Close'] / df['Close'].shift(4) - 1
    df['Volume_MA'] = df['Volume'].rolling(window=10).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Drop NA values
    df.dropna(inplace=True)
    
    return df

# ======================
# PHẦN 3: MÔ HÌNH AI
# ======================

def train_stock_model(df, target='Close', time_steps=60, test_size=0.2, epochs=50, batch_size=32):
    """
    Huấn luyện mô hình LSTM để dự báo giá cổ phiếu
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if df is None or len(df) < time_steps:
            print("Dữ liệu không đủ để huấn luyện mô hình")
            return None, None, None, None, None
        
        if target not in df.columns:
            print(f"Cột {target} không tồn tại trong dữ liệu")
            return None, None, None, None, None
            
        data = df[[target]].values
        
        # Kiểm tra dữ liệu có hợp lệ không
        if len(data) == 0:
            print("Dữ liệu rỗng")
            return None, None, None, None, None
        
        # Loại bỏ các giá trị NaN/inf
        data = data[np.isfinite(data)].reshape(-1, 1)
        if len(data) == 0:
            print("Không có dữ liệu hợp lệ sau khi loại bỏ NaN/inf")
            return None, None, None, None, None
        
        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Kiểm tra đủ dữ liệu để tạo chuỗi thời gian
        if len(scaled_data) <= time_steps:
            print("Không đủ dữ liệu để tạo chuỗi thời gian")
            return None, None, None, None, None
        
        # Tạo dataset theo chuỗi thời gian
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i, 0])
            y.append(scaled_data[i, 0])
        
        if len(X) == 0 or len(y) == 0:
            print("Không tạo được dữ liệu huấn luyện")
            return None, None, None, None, None
            
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Kiểm tra kích thước dữ liệu
        if X.shape[0] < 10:  # Cần ít nhất 10 mẫu để chia train/test
            print("Dữ liệu quá ít để huấn luyện")
            return None, None, None, None, None
        
        # Chia dữ liệu train/test
        split_index = max(1, int(len(X) * (1 - test_size)))
        if split_index >= len(X):
            split_index = len(X) - 1
            
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Kiểm tra dữ liệu train có hợp lệ không
        if len(X_train) == 0 or len(y_train) == 0:
            print("Dữ liệu train rỗng")
            return None, None, None, None, None
        
        # Xây dựng mô hình LSTM
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=25))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Huấn luyện mô hình với kiểm tra lỗi
        try:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=[early_stopping],
                verbose=1
            )
        except Exception as e:
            print(f"Lỗi khi huấn luyện mô hình: {str(e)}")
            # Thử với ít epochs hơn
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=batch_size,
                validation_split=0.1,
                verbose=1
            )
        
        # Dự báo với kiểm tra lỗi
        try:
            y_pred = model.predict(X_test)
        except Exception as e:
            print(f"Lỗi khi dự báo: {str(e)}")
            return None, None, None, None, None
        
        # Chuyển đổi về giá gốc
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred)
        
        # Vẽ biểu đồ quá trình huấn luyện
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            if 'val_loss' in history.history:
                plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Lịch sử huấn luyện mô hình')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            plt.savefig('training_history.png')
            plt.close()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ huấn luyện: {str(e)}")
        
        # Vẽ biểu đồ dự báo vs thực tế
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(y_test, label='Giá thực tế', color='blue')
            plt.plot(y_pred, label='Dự báo', color='red', linestyle='--')
            plt.title('So sánh giá thực tế và dự báo')
            plt.xlabel('Thời gian')
            plt.ylabel('Giá cổ phiếu')
            plt.legend()
            plt.grid(True)
            plt.savefig('forecast_vs_actual.png')
            plt.close()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ dự báo: {str(e)}")
        
        # Tính toán các chỉ số đánh giá
        try:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            print("\nĐÁNH GIÁ MÔ HÌNH:")
            print(f"MSE: {mse:.2f}")
            print(f"RMSE: {rmse:.2f}")
            print(f"MAE: {mae:.2f}")
            print(f"R2: {r2:.2f}")
        except Exception as e:
            print(f"Lỗi khi tính toán đánh giá: {str(e)}")
            mse, rmse, mae, r2 = 0, 0, 0, 0
        
        return model, scaler, X_test, y_test, y_pred
        
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi huấn luyện mô hình: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None

def predict_next_days(model, scaler, df, target='Close', time_steps=60, n_days=5):
    """
    Dự báo giá trong n ngày tiếp theo
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
            forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1))
        except Exception as e:
            print(f"Lỗi khi chuyển đổi giá gốc: {str(e)}")
            return np.array([]), np.array([])
        
        # Tạo ngày dự báo
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
        
        return np.array(forecast_dates), forecast.flatten()
    
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi dự báo: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])

# ======================
# PHẦN 4: PHÂN TÍCH KỸ THUẬT
# ======================

def plot_stock_analysis(symbol, df, show_volume=True):
    """
    Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán
    """
    try:
        # Kiểm tra dữ liệu đầu vào
        if df is None or len(df) == 0:
            print("Dữ liệu phân tích rỗng")
            return {
                'signal': 'LỖI',
                'score': 50,
                'current_price': 0,
                'rsi_value': 0,
                'ma20': 0,
                'ma50': 0,
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }
        
        # Tính toán chỉ báo kỹ thuật
        # 1. Đường trung bình
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        
        # 2. RSI (Relative Strength Index)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        
        # 3. MACD (Moving Average Convergence Divergence)
        macd = ta.trend.MACD(df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()
        
        # 4. Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Close'])
        df['BB_Upper'] = bollinger.bollinger_hband()
        df['BB_Lower'] = bollinger.bollinger_lband()
        
        # Kiểm tra dữ liệu sau khi tính toán
        if len(df.dropna()) < 20:  # Cần ít nhất 20 điểm dữ liệu hợp lệ
            print("Không đủ dữ liệu hợp lệ để phân tích kỹ thuật")
            return {
                'signal': 'LỖI',
                'score': 50,
                'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'rsi_value': 50,
                'ma20': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'ma50': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }
        
        # Lọc dữ liệu hợp lệ
        df = df.dropna()
        
        # Tạo layout biểu đồ
        try:
            plt.figure(figsize=(16, 12))
            grid = plt.GridSpec(4, 1, hspace=0.2, height_ratios=[3, 1, 1, 1])
            # Biểu đồ 1: Giá và Bollinger Bands
            ax1 = plt.subplot(grid[0])
            plt.plot(df.index, df['Close'], label='Giá đóng cửa', color='#1f77b4')
            plt.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['SMA_50'], label='SMA 50', color='green', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['SMA_200'], label='SMA 200', color='purple', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', alpha=0.5, linestyle='--')
            plt.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', alpha=0.5, linestyle='--')
            plt.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='gray', alpha=0.1)
            
            # Đánh dấu điểm giao nhau quan trọng
            cross_above = df[df['SMA_20'] > df['SMA_50']].index
            cross_below = df[df['SMA_20'] < df['SMA_50']].index
            
            if len(cross_above) > 0:
                plt.scatter(cross_above, df.loc[cross_above, 'SMA_20'], 
                           marker='^', color='green', s=80, label='SMA20 > SMA50')
            if len(cross_below) > 0:
                plt.scatter(cross_below, df.loc[cross_below, 'SMA_20'], 
                           marker='v', color='red', s=80, label='SMA20 < SMA50')
            
            plt.title(f'Phân tích kỹ thuật {symbol} - Giá và Chỉ báo', fontsize=14)
            plt.ylabel('Giá (VND)', fontsize=12)
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
            
            # Biểu đồ 2: RSI
            ax2 = plt.subplot(grid[1], sharex=ax1)
            plt.plot(df.index, df['RSI'], label='RSI', color='purple')
            plt.axhline(70, linestyle='--', color='red', alpha=0.7)
            plt.axhline(30, linestyle='--', color='green', alpha=0.7)
            plt.fill_between(df.index, 30, 70, color='gray', alpha=0.1)
            plt.title('Chỉ số Sức mạnh Tương đối (RSI)', fontsize=12)
            plt.ylim(0, 100)
            plt.ylabel('RSI', fontsize=10)
            plt.grid(True, alpha=0.3)
            
            # Biểu đồ 3: MACD
            ax3 = plt.subplot(grid[2], sharex=ax1)
            plt.plot(df.index, df['MACD'], label='MACD', color='blue')
            plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
            plt.bar(df.index, df['MACD_Hist'], 
                    color=np.where(df['MACD_Hist'] > 0, 'green', 'red'), 
                    alpha=0.5, label='Histogram')
            plt.title('MACD (Moving Average Convergence Divergence)', fontsize=12)
            plt.ylabel('MACD', fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Biểu đồ 4: Khối lượng
            ax4 = plt.subplot(grid[3], sharex=ax1)
            if show_volume and 'Volume' in df.columns:
                plt.bar(df.index, df['Volume'], 
                       color=np.where(df['Close'] > df['Open'], 'green', 'red'), 
                       alpha=0.7)
                plt.title('Khối lượng giao dịch', fontsize=12)
                plt.ylabel('Khối lượng', fontsize=10)
                plt.grid(True, alpha=0.3)
            
            # Định dạng trục x
            plt.gcf().autofmt_xdate()
            
            # Lưu biểu đồ
            plt.tight_layout()
            plt.savefig(f'vnstocks_data/{symbol}_technical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Đã lưu biểu đồ phân tích kỹ thuật vào vnstocks_data/{symbol}_technical_analysis.png")
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ phân tích: {str(e)}")
        
        # Hiển thị tín hiệu giao dịch cuối
        try:
            last_signal = "TRUNG LẬP"
            current_price = df['Close'].iloc[-1]
            rsi_value = df['RSI'].iloc[-1] if not pd.isna(df['RSI'].iloc[-1]) else 50
            ma20_value = df['SMA_20'].iloc[-1] if not pd.isna(df['SMA_20'].iloc[-1]) else current_price
            ma50_value = df['SMA_50'].iloc[-1] if not pd.isna(df['SMA_50'].iloc[-1]) else current_price
            
            if rsi_value < 30 and current_price > ma50_value:
                last_signal = "MUA (RSI quá bán + Giá trên SMA50)"
            elif rsi_value > 70 and current_price < ma50_value:
                last_signal = "BÁN (RSI quá mua + Giá dưới SMA50)"
                
            print(f"\nTÍN HIỆU GIAO DỊCH CUỐI CÙNG ({df.index[-1].strftime('%d/%m/%Y')}):")
            print(f"- Giá đóng cửa: {current_price:,.2f} VND")
            print(f"- RSI: {rsi_value:.2f}")
            print(f"- MACD: {df['MACD'].iloc[-1]:.2f} | Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
            print(f"- SMA20/SMA50: {ma20_value:,.2f} / {ma50_value:,.2f} VND")
            print(f"- Đề xuất: {last_signal}")
            
            return {
                'signal': 'MUA' if last_signal.startswith("MUA") else 'BÁN' if last_signal.startswith("BÁN") else 'TRUNG LẬP',
                'score': 100 - abs(rsi_value - 50) * 2,  # Điểm từ 0-100
                'current_price': current_price,
                'rsi_value': rsi_value,
                'ma20': ma20_value,
                'ma50': ma50_value,
                'recommendation': 'MUA MẠNH' if last_signal.startswith("MUA") else 'BÁN MẠNH' if last_signal.startswith("BÁN") else 'GIỮ'
            }
        except Exception as e:
            print(f"Lỗi khi tạo tín hiệu giao dịch: {str(e)}")
            return {
                'signal': 'LỖI',
                'score': 50,
                'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'rsi_value': 50,
                'ma20': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'ma50': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }
    
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi phân tích kỹ thuật: {str(e)}")
        traceback.print_exc()
        return {
            'signal': 'LỖI',
            'score': 50,
            'current_price': 0,
            'rsi_value': 0,
            'ma20': 0,
            'ma50': 0,
            'recommendation': 'KHÔNG XÁC ĐỊNH'
        }

# ======================
# PHẦN 5: TÍCH HỢP PHÂN TÍCH BẰNG GEMINI
# ======================

def analyze_with_gemini(symbol, trading_signal, forecast, financial_data=None):
    """Phân tích cổ phiếu bằng Google Gemini dựa trên dữ liệu kỹ thuật và BCTC"""
    try:
        # Tạo prompt cho Gemini
        prompt = f"""
Hãy đóng vai một chuyên gia phân tích chứng khoán tại Việt Nam. Phân tích cổ phiếu {symbol} dựa trên các thông tin sau:

1. Tín hiệu giao dịch:
   - Tín hiệu: {trading_signal['signal']}
   - Điểm phân tích: {trading_signal['score']}/100
   - Giá hiện tại: {trading_signal['current_price']:,.0f} VND
   - RSI: {trading_signal['rsi_value']:.2f}
   - MA20: {trading_signal['ma20']:,.0f} VND
   - MA50: {trading_signal['ma50']:,.0f} VND

2. Dự báo giá trong 5 ngày tới:
"""
        # Sửa lỗi kiểm tra điều kiện cho forecast
        if len(forecast[0]) > 0 and len(forecast[1]) > 0:
            for i, (date, price) in enumerate(zip(forecast[0], forecast[1])):
                change = ((price - trading_signal['current_price']) / trading_signal['current_price']) * 100
                prompt += f"   - Ngày {i+1} ({date.strftime('%d/%m/%Y')}): {price:,.0f} VND ({change:+.2f}%)\n"
        else:
            prompt += "   - Không có dự báo\n"
         
        if financial_data is not None and not financial_data.empty:
            prompt += "\n3. Dữ liệu tài chính (BCTC) gần nhất:\n"
            try:
                # Lấy quý gần nhất
                financial_data_sorted = financial_data.copy()
                prompt += f"{financial_data_sorted}"
            except Exception as e:
                print(f"Lỗi khi xử lý dữ liệu tài chính: {str(e)}")
                prompt += "   - Không có dữ liệu tài chính chi tiết\n"
        else:
            prompt += "\n3. Không có dữ liệu tài chính\n"

        prompt += """
\nYêu cầu phân tích:
- Tổng hợp phân tích kỹ thuật và cơ bản
- Đánh giá sức mạnh tài chính của công ty
- Phân tích xu hướng giá và tín hiệu kỹ thuật
- Nhận định rủi ro tiềm ẩn
- Dự báo triển vọng ngắn hạn và trung hạn
- Đưa ra khuyến nghị đầu tư (Mua/Bán/Nắm giữ) với lý do cụ thể

Kết quả phân tích cần:
- Ngắn gọn, súc tích (không quá 500 từ)
- Chuyên nghiệp như một nhà phân tích chứng khoán
- Bao gồm cả yếu tố thị trường tổng thể
- Có số liệu minh họa cụ thể
"""
        
        # Sử dụng Gemini Pro để phân tích
        response = client.models.generate_content(
        model="gemini-2.5-pro", contents= prompt
)       
        
        return response.text
    
    except Exception as e:
        print(f"Lỗi khi phân tích bằng Gemini: {str(e)}")
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."

# ======================
# PHẦN 6: CHỨC NĂNG CHÍNH
# ======================

def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán với tích hợp Gemini"""
    print(f"\n{'='*50}")
    print(f"PHÂN TÍCH MÃ {symbol} VỚI AI")
    print(f"{'='*50}")
    
    # Lấy dữ liệu lịch sử
    df = get_stock_data(symbol)
    if df is None or df.empty:
        print(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    
    # Lấy dữ liệu BCTC
    financial_data = get_financial_data(symbol)
    
    # Tiền xử lý dữ liệu
    df_processed = preprocess_stock_data(df)
    if df_processed is None or df_processed.empty:
        print(f"Không thể tiền xử lý dữ liệu cho mã {symbol}")
        return None
    
    # Tạo đặc trưng
    df_features = create_features(df_processed)
    
    # Kiểm tra dữ liệu sau khi tạo đặc trưng
    if len(df_features) < 100:  # Cần ít nhất 100 điểm dữ liệu
        print(f"Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_features)} điểm)")
        return None
    
    # Huấn luyện mô hình AI
    print(f"\nĐang huấn luyện mô hình AI cho mã {symbol}...")
    model, scaler, X_test, y_test, y_pred = train_stock_model(df_features)
    
    # Dự báo 5 ngày tiếp theo
    print(f"\nĐang dự báo giá cho 5 ngày tới...")
    forecast_dates, forecast_values = predict_next_days(model, scaler, df_features)
    
    # Phân tích kỹ thuật và tạo tín hiệu giao dịch
    print(f"\nĐang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_features)
    
    # Phân tích bằng Gemini
    print(f"\nĐang phân tích bằng Google Gemini...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, (forecast_dates, forecast_values), financial_data)
    
    # In kết quả
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.2} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.2}/100")
    
    # Sửa lỗi kiểm tra điều kiện cho forecast
    if len(forecast_dates) > 0 and len(forecast_values) > 0:
        print(f"\nDỰ BÁO GIÁ CHO {len(forecast_dates):,.2} NGÀY TIẾP THEO:")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            change = ((price - trading_signal['current_price']) / trading_signal['current_price']) * 100
            print(f"Ngày {i+1} ({date.date()}): {price:,.2f} VND ({change:+.2f}%)")
    else:
        print("\nKhông có dự báo giá do lỗi trong quá trình huấn luyện mô hình")
    
    print(f"\nPHÂN TÍCH TỔNG HỢP TỪ GEMINI:")
    print(gemini_analysis)
    
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
            'change_percent': float(change)
        } for date, price, change in zip(
            forecast_dates, 
            forecast_values, 
            [((price - trading_signal['current_price']) / trading_signal['current_price']) * 100 
             for price in forecast_values]
        )] if len(forecast_dates) > 0 and len(forecast_values) > 0 else [],
        'gemini_analysis': gemini_analysis
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
    for symbol in stock_list['symbol'].head(5):  # Chỉ phân tích 5 mã đầu tiên để demo
        try:
            print(f"\nPhân tích mã {symbol}...")
            report = analyze_stock(symbol)
            if report:
                results.append(report)
            time.sleep(2)  # Dừng 2 giây giữa các request
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
        try:
            plt.figure(figsize=(12, 6))
            sns.barplot(x='Mã', y='Điểm', data=df_results, palette='viridis')
            plt.title('Điểm phân tích các mã chứng khoán')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('vnstocks_data/stock_screening_comparison.png')
            plt.close()
        except Exception as e:
            print(f"Lỗi khi vẽ biểu đồ so sánh: {str(e)}")
        
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
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE GEMINI")
    print("==============================================")
    
    # Lấy dữ liệu thị trường
    market_data = get_market_data()
    
    # Test với mã ACB
    analyze_stock('DBC')
    
    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")