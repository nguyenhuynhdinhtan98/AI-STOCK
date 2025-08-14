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
import ta
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
from vnstock import *  # Import vnstock library
import matplotlib.pyplot as plt
import mplfinance as mpf
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
#from vnstock_ta import Indicator, Plotter, DataSource

warnings.filterwarnings('ignore')

# ======================
# CẤU HÌNH VÀ THƯ VIỆN
# ======================

# Tải biến môi trường cho Gemini
load_dotenv()
GOOGLE_API_KEY = os.getenv('AIzaSyBh4yjR8V6ZNNUFsS-d_m3A9JWIKB__0n4')
genai.configure(api_key=GOOGLE_API_KEY)

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
# PHẦN 1: THU THẬP DỮ LIỆU (SỬA ĐỔI SỬ DỤNG VNSTOCK)
# ======================

def get_vnstocks_list():
    """Lấy danh sách tất cả các mã chứng khoán trên thị trường Việt Nam sử dụng vnstock"""
    try:
        # Sử dụng vnstock để lấy danh sách công ty niêm yết
        listing_companies = listing_companies()
        
        if listing_companies is not None and not listing_companies.empty:
            df = listing_companies[['ticker']].rename(columns={'ticker': 'symbol'})
            df.to_csv('vnstocks_data/stock_list.csv', index=False)
            print(f"Đã lưu danh sách {len(df)} mã chứng khoán vào file 'vnstocks_data/stock_list.csv'")
            return df
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

def get_stock_data(symbol, period="1y"):
    """Lấy dữ liệu lịch sử của một mã chứng khoán sử dụng vnstock"""
    try:
        # Xác định ngày bắt đầu và kết thúc
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        
        # Sử dụng vnstock để lấy dữ liệu
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution="1D",
            type="stock"
        )
        
        if df is not None and not df.empty:
            # Chuẩn hóa tên cột
            df.rename(columns={
                'time': 'Date',
                'open': 'Open',
                'high': 'High',
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume',
                'ticker': 'Ticker'
            }, inplace=True)
            
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
    """Lấy dữ liệu báo cáo tài chính sử dụng vnstock"""
    try:
        # Lấy dữ liệu báo cáo tài chính
        financial_report = financial_report(symbol, 'quarterly', 2023)
        
        if financial_report is not None and not financial_report.empty:
            # Lưu dữ liệu
            financial_report.to_csv(f'vnstocks_data/{symbol}_financial.csv')
            return financial_report
        else:
            print(f"Không lấy được BCTC cho mã {symbol}")
            return None
    except Exception as e:
        print(f"Lỗi khi lấy BCTC cho {symbol}: {str(e)}")
        return None

def get_market_data():
    """Lấy dữ liệu thị trường tổng thể sử dụng vnstock"""
    try:
        # Lấy dữ liệu VN-Index
        vnindex = stock_historical_data(symbol='VNINDEX', 
                                       start_date="2018-01-01", 
                                       end_date=datetime.now().strftime("%Y-%m-%d"),
                                       resolution="1D")
        vnindex.rename(columns={'close': 'Close'}, inplace=True)
        vnindex.to_csv('vnstocks_data/vnindex_data.csv')
        
        # Lấy dữ liệu VN30-Index
        vn30 = stock_historical_data(symbol='VN30', 
                                    start_date="2018-01-01", 
                                    end_date=datetime.now().strftime("%Y-%m-%d"),
                                    resolution="1D")
        vn30.rename(columns={'close': 'Close'}, inplace=True)
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
# PHẦN 2-4: GIỮ NGUYÊN CÁC HÀM TIỀN XỬ LÝ, MÔ HÌNH AI VÀ PHÂN TÍCH
# ======================
# [Các hàm preprocess_stock_data, create_features, prepare_time_series_data,
#  build_lstm_model, train_stock_model, predict_next_days, 
#  plot_stock_analysis, generate_trading_signal giữ nguyên]
# ======================

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
        for i, (date, price) in enumerate(zip(forecast[0], forecast[1])):
            change = ((price - trading_signal['current_price']) / trading_signal['current_price']) * 100
            prompt += f"   - Ngày {i+1} ({date.strftime('%d/%m/%Y')}): {price:,.0f} VND ({change:+.2f}%)\n"

        if financial_data is not None:
            prompt += "\n3. Dữ liệu tài chính (BCTC) gần nhất:\n"
            # Tạo bản tóm tắt các chỉ số tài chính quan trọng
            financial_summary = financial_data[['quarter', 'year', 'revenue', 'grossProfit', 'netProfit', 
                                              'roe', 'debtToEquity', 'eps']].tail(4).to_string()
            prompt += financial_summary

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
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        
        return response.text
    
    except Exception as e:
        print(f"Lỗi khi phân tích bằng Gemini: {str(e)}")
        return "Không thể tạo phân tích bằng Gemini tại thời điểm này."

# ======================
# PHẦN 5: CHỨC NĂNG CHÍNH (SỬA ĐỔI)
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
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    
    # Handle missing values (forward fill then backfill)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    
    # Add technical features (example)
    df['returns'] = df['close'].pct_change()
    df['MA_10'] = df['close'].rolling(window=10).mean()
    df['MA_50'] = df['close'].rolling(window=50).mean()
    df['volatility'] = df['returns'].rolling(window=10).std()
    
    # Drop initial rows with NaN values from technical features
    df.dropna(inplace=True)
    
    # Select relevant columns
    processed_df = df[['open', 'high', 'low', 'close', 'volume', 
                       'returns', 'MA_10', 'MA_50', 'volatility']]
    
    return processed_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def create_features(df):
    """
    Generates technical indicators using pure pandas/numpy
    """
    # Calculate RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bollinger Bands
    df['BB_middle'] = df['close'].rolling(window=20).mean()
    df['BB_std'] = df['close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
    df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
    
    # ATR (Average True Range)
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR'] = true_range.rolling(window=14).mean()
    
    # Moving averages
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Momentum and volume features
    df['Momentum'] = df['close'] / df['close'].shift(4) - 1
    df['Volume_MA'] = df['volume'].rolling(window=10).mean()
    df['Volume_Change'] = df['volume'].pct_change()
    
    # Drop NA values
    df.dropna(inplace=True)
    
    return df

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

def train_stock_model(df, target='close', time_steps=60, test_size=0.2, epochs=50, batch_size=32):
    """
    Huấn luyện mô hình LSTM để dự báo giá cổ phiếu
    
    Parameters:
        df (pd.DataFrame): DataFrame chứa dữ liệu đã tiền xử lý
        target (str): Tên cột mục tiêu (mặc định 'close')
        time_steps (int): Số ngày trong quá khứ để dự báo ngày tiếp theo
        test_size (float): Tỷ lệ dữ liệu kiểm tra (0.0 - 1.0)
        epochs (int): Số epoch huấn luyện
        batch_size (int): Kích thước batch
    
    Returns:
        model: Mô hình đã huấn luyện
        scaler: Đối tượng scaler
        X_test: Dữ liệu kiểm tra
        y_test: Giá trị thực tế
        y_pred: Dự báo
    """
    try:
        # Chuẩn bị dữ liệu
        data = df[[target]].values
        
        # Chuẩn hóa dữ liệu
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        # Tạo dataset theo chuỗi thời gian
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Chia dữ liệu train/test
        split_index = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        
        # Xây dựng mô hình LSTM
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.3))
        model.add(LSTM(units=100, return_sequences=True))
        model.add(Dropout(0.3))
        model.add(LSTM(units=50))
        model.add(Dropout(0.3))
        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        # Huấn luyện mô hình
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Dự báo
        y_pred = model.predict(X_test)
        
        # Chuyển đổi về giá gốc
        y_test = scaler.inverse_transform(y_test.reshape(-1, 1))
        y_pred = scaler.inverse_transform(y_pred)
        
        # Vẽ biểu đồ quá trình huấn luyện
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Lịch sử huấn luyện mô hình')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.show()
        
        # Vẽ biểu đồ dự báo vs thực tế
        plt.figure(figsize=(12, 6))
        plt.plot(y_test, label='Giá thực tế', color='blue')
        plt.plot(y_pred, label='Dự báo', color='red', linestyle='--')
        plt.title('So sánh giá thực tế và dự báo')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá cổ phiếu')
        plt.legend()
        plt.grid(True)
        plt.savefig('forecast_vs_actual.png')
        plt.show()
        
        # Tính toán các chỉ số đánh giá
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print("\nĐÁNH GIÁ MÔ HÌNH:")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"R2: {r2:.2f}")
        
        return model, scaler, X_test, y_test, y_pred
        
    except Exception as e:
        print(f"Lỗi khi huấn luyện mô hình: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None
    

def plot_stock_analysis(symbol, period="1y", show_volume=True):
    """
    Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán sử dụng vnstock
    Bao gồm: Giá, khối lượng, RSI, MACD, Bollinger Bands
    
    Parameters:
    - symbol: Mã chứng khoán (VD: 'VIC', 'VNM')
    - period: Khoảng thời gian ('1d', '1w', '1m', '3m', '6m', '1y', '2y', '5y')
    - show_volume: Hiển thị biểu đồ khối lượng
    """
    try:
        # Lấy dữ liệu từ vnstock
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=365*5)).strftime("%Y-%m-%d")
        
        df = stock_historical_data(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            resolution="1D",
            type="stock"
        )
        
        if df is None or df.empty:
            print(f"Không lấy được dữ liệu cho mã {symbol}")
            return None
        
        # Chuẩn hóa dữ liệu
        df.rename(columns={
            'time': 'Date',
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        }, inplace=True)
        
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df.sort_index(inplace=True)
        
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
        
        # Tạo layout biểu đồ
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
        if show_volume:
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
        plt.savefig(f'{symbol}_technical_analysis.png', dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ phân tích kỹ thuật vào {symbol}_technical_analysis.png")
        
        # Hiển thị tín hiệu giao dịch cuối
        last_signal = "TRUNG LẬP"
        if df['RSI'].iloc[-1] < 30 and df['Close'].iloc[-1] > df['SMA_50'].iloc[-1]:
            last_signal = "MUA (RSI quá bán + Giá trên SMA50)"
        elif df['RSI'].iloc[-1] > 70 and df['Close'].iloc[-1] < df['SMA_50'].iloc[-1]:
            last_signal = "BÁN (RSI quá mua + Giá dưới SMA50)"
            
        print(f"\nTÍN HIỆU GIAO DỊCH CUỐI CÙNG ({df.index[-1].strftime('%d/%m/%Y')}):")
        print(f"- Giá đóng cửa: {df['Close'].iloc[-1]:,.0f} VND")
        print(f"- RSI: {df['RSI'].iloc[-1]:.2f}")
        print(f"- MACD: {df['MACD'].iloc[-1]:.2f} | Signal: {df['MACD_Signal'].iloc[-1]:.2f}")
        print(f"- SMA20/SMA50: {df['SMA_20'].iloc[-1]:,.0f} / {df['SMA_50'].iloc[-1]:,.0f} VND")
        print(f"- Đề xuất: {last_signal}")
        
        return df
    
    except Exception as e:
        print(f"Lỗi khi phân tích kỹ thuật: {str(e)}")
        traceback.print_exc()
        return None
    
def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán với tích hợp Gemini"""
    print(f"\n{'='*50}")
    print(f"PHÂN TÍCH MÃ {symbol} VỚI AI")
    print(f"{'='*50}")
    
    # Lấy và xử lý dữ liệu
    # Get historical data
    quote = Quote(symbol= symbol,source="TCBS") # tạo đối tượng quote như 1 biến để tái sử dụng
    df = quote.history(start='2022-01-01', end='2024-07-10', interval='1D')
    if df is None or df.empty:
        print(f"Không thể phân tích mã {symbol} do thiếu dữ liệu")
        return None
    
    # Lấy dữ liệu BCTC
    financial_data = Finance(symbol=symbol, source='VCI')
    
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
    forecast_dates, forecast_values = plot_stock_analysis(symbol, period="1y")
    
    # Tạo tín hiệu giao dịch
    print(f"\nĐang tạo tín hiệu giao dịch cho mã {symbol}...")
    trading_signal = generate_trading_signal(symbol, df_features)
    
    # Phân tích bằng Gemini
    print(f"\nĐang phân tích bằng Google Gemini...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, (forecast_dates, forecast_values), financial_data)
    
    # In kết quả
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.0f} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.1f}/100")
    
    if forecast_dates and len(forecast_dates) > 0:
        print(f"\nDỰ BÁO GIÁ CHO {len(forecast_dates)} NGÀY TIẾP THEO:")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            change = ((price - trading_signal['current_price']) / trading_signal['current_price']) * 100
            print(f"Ngày {i+1} ({date.date()}): {price:,.0f} VND ({change:+.2f}%)")
    
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
            'change_percent': float(((price - trading_signal['current_price']) / trading_signal['current_price']) * 100)
        } for date, price in zip(forecast_dates, forecast_values)] if forecast_dates else [],
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
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE GEMINI")
    print("==============================================")
    
    # Lấy dữ liệu thị trường
    market_data = get_market_data()
    
    # Lựa chọn chế độ
    print("\nChọn chế độ phân tích:")
    print("1. Phân tích một mã cụ thể")
    print("2. Quét và phân tích danh sách mã")
    
    analyze_stock("ACB")
    # choice = input("\nNhập lựa chọn của bạn (1/2): ")
    
    # if choice == "1":
    #     symbol = input("Nhập mã chứng khoán cần phân tích (ví dụ: VNM, VCB): ").strip().upper()
    #     analyze_stock(symbol)
    # elif choice == "2":
    #     screen_stocks()
    # else:
    #     print("Lựa chọn không hợp lệ!")
    
    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")
