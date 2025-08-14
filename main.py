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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import yfinance as yf
import ta
import warnings
import google.generativeai as genai
from dotenv import load_dotenv
from vnstock import *  # Import vnstock library
import mplfinance as mpf
import traceback
from pandas.plotting import register_matplotlib_converters

# Cài đặt môi trường
warnings.filterwarnings('ignore')
register_matplotlib_converters()
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')
load_dotenv()

# Cấu hình API Gemini
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-pro')
else:
    print("Warning: Gemini API key not found. AI analysis will be limited.")

class StockAnalyzer:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.technical_indicators = None
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def fetch_data(self):
        """Thu thập dữ liệu từ VNStock hoặc Yahoo Finance dự phòng"""
        print(f"Đang thu thập dữ liệu cho {self.symbol} từ {self.start_date} đến {self.end_date}")
        
        try:
            # Thử với VNStock trước
            df = stock_historical_data(self.symbol, self.start_date, self.end_date, "1D")
            if df is not None and not df.empty:
                df.rename(columns={
                    'date': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                self.data = df
                print(f"Thu thập thành công từ VNStock: {len(df)} bản ghi")
                return
        except Exception as e:
            print(f"Lỗi VNStock: {str(e)}")
        
        try:
            # Dự phòng với Yahoo Finance
            ticker = self.symbol + ".VN"
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            if not df.empty:
                df.reset_index(inplace=True)
                df.rename(columns={'Date': 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                self.data = df
                print(f"Thu thập thành công từ Yahoo Finance: {len(df)} bản ghi")
            else:
                raise ValueError("Không thể thu thập dữ liệu từ cả hai nguồn")
        except Exception as e:
            print(f"Lỗi Yahoo Finance: {str(e)}")
            traceback.print_exc()
            raise

    def add_technical_indicators(self):
        """Thêm các chỉ báo kỹ thuật vào dữ liệu"""
        if self.data is None:
            raise ValueError("Dữ liệu chưa được tải")
            
        print("Đang tính toán chỉ báo kỹ thuật...")
        df = self.data.copy()
        
        # Chỉ báo xu hướng
        df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['SMA_200'] = ta.trend.sma_indicator(df['Close'], window=200)
        df['EMA_20'] = ta.trend.ema_indicator(df['Close'], window=20)
        
        # Chỉ báo động lượng
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
        df['MACD'] = ta.trend.macd_diff(df['Close'])
        df['Stoch_%K'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14)
        
        # Chỉ báo biến động
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        df['Bollinger_Upper'] = ta.volatility.bollinger_hband(df['Close'])
        df['Bollinger_Lower'] = ta.volatility.bollinger_lband(df['Close'])
        
        # Chỉ báo khối lượng
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        df['Volume_SMA'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
        
        # Tạo tín hiệu giao dịch
        df['Signal'] = np.where(
            (df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 70) & (df['Close'] > df['Bollinger_Upper']), 
            'MUA', 
            np.where(
                (df['SMA_50'] < df['SMA_200']) & (df['RSI'] > 30) & (df['Close'] < df['Bollinger_Lower']),
                'BÁN',
                'GIỮ'
            )
        )
        
        self.technical_indicators = df.dropna()
        print("Đã thêm 10 chỉ báo kỹ thuật")
        
    def visualize_technical_analysis(self):
        """Trực quan hóa phân tích kỹ thuật"""
        if self.technical_indicators is None:
            self.add_technical_indicators()
            
        df = self.technical_indicators
        plt.figure(figsize=(18, 22))
        
        # Biểu đồ giá với Bollinger Bands
        plt.subplot(4, 1, 1)
        plt.plot(df['Close'], label='Giá đóng cửa', color='blue')
        plt.plot(df['SMA_50'], label='SMA 50', color='orange', linestyle='--')
        plt.plot(df['SMA_200'], label='SMA 200', color='red', linestyle='--')
        plt.plot(df['Bollinger_Upper'], label='Bollinger Upper', color='green', alpha=0.5)
        plt.plot(df['Bollinger_Lower'], label='Bollinger Lower', color='red', alpha=0.5)
        plt.fill_between(df.index, df['Bollinger_Lower'], df['Bollinger_Upper'], color='gray', alpha=0.1)
        plt.title(f'Phân tích kỹ thuật {self.symbol} - Dải Bollinger & Đường MA')
        plt.legend()
        
        # Biểu đồ RSI và MACD
        plt.subplot(4, 1, 2)
        plt.plot(df['RSI'], label='RSI', color='purple')
        plt.axhline(70, linestyle='--', color='red', alpha=0.5)
        plt.axhline(30, linestyle='--', color='green', alpha=0.5)
        plt.title('Chỉ số Sức mạnh Tương đối (RSI)')
        
        ax2 = plt.gca().twinx()
        ax2.plot(df['MACD'], label='MACD', color='blue', alpha=0.7)
        ax2.axhline(0, linestyle='--', color='black', alpha=0.3)
        plt.legend()
        
        # Biểu đồ khối lượng
        plt.subplot(4, 1, 3)
        plt.bar(df.index, df['Volume'], color=np.where(df['Close'] > df['Open'], 'g', 'r'), alpha=0.8)
        plt.plot(df['Volume_SMA'], color='blue', label='Khối lượng trung bình 20 ngày')
        plt.title('Khối lượng giao dịch')
        plt.legend()
        
        # Biểu đồ nến
        plt.subplot(4, 1, 4)
        mpf.plot(df.tail(60), type='candle', style='charles', title=f'Biểu đồ nến 60 ngày - {self.symbol}', 
                volume=True, show_nontrading=False, ax=plt.gca())
        
        plt.tight_layout()
        plt.savefig(f'{self.symbol}_technical_analysis.png')
        plt.show()
        
    def prepare_lstm_data(self, time_steps=60):
        """Chuẩn bị dữ liệu cho mô hình LSTM"""
        if self.technical_indicators is None:
            self.add_technical_indicators()
            
        df = self.technical_indicators
        features = ['Close', 'Volume', 'RSI', 'MACD', 'SMA_50', 'SMA_200']
        df_features = df[features]
        
        # Chuẩn hóa dữ liệu
        scaled_data = self.scaler.fit_transform(df_features)
        
        # Tạo dataset theo chuỗi thời gian
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i-time_steps:i])
            y.append(scaled_data[i, 0])  # Dự báo giá đóng cửa
            
        X, y = np.array(X), np.array(y)
        return X, y
    
    def build_lstm_model(self, input_shape):
        """Xây dựng mô hình LSTM"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            LSTM(64, return_sequences=False),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        print(model.summary())
        return model
    
    def train_lstm_model(self, epochs=100, batch_size=32, test_size=0.2):
        """Huấn luyện mô hình dự báo"""
        X, y = self.prepare_lstm_data()
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=False
        )
        
        # Xây dựng mô hình
        self.model = self.build_lstm_model((X_train.shape[1], X_train.shape[2]))
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)
        ]
        
        # Huấn luyện
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Đánh giá
        self.evaluate_model(X_test, y_test)
        self.plot_training_history(history)
        
        return history
    
    def evaluate_model(self, X_test, y_test):
        """Đánh giá mô hình"""
        predictions = self.model.predict(X_test)
        
        # Chuyển đổi về giá gốc
        dummy_array = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy_array[:, 0] = predictions.flatten()
        predictions_orig = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        dummy_array[:, 0] = y_test.flatten()
        y_test_orig = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        # Tính toán metrics
        mse = mean_squared_error(y_test_orig, predictions_orig)
        mae = mean_absolute_error(y_test_orig, predictions_orig)
        r2 = r2_score(y_test_orig, predictions_orig)
        
        # Tính MAPE
        with np.errstate(divide='ignore', invalid='ignore'):
            absolute_percentage_error = np.abs((y_test_orig - predictions_orig) / y_test_orig)
            mask = np.isfinite(absolute_percentage_error)
            valid_ape = absolute_percentage_error[mask]
            mape = np.mean(valid_ape) * 100 if len(valid_ape) > 0 else 0
            accuracy = 100 - mape
        
        print("\n" + "="*50)
        print("ĐÁNH GIÁ MÔ HÌNH")
        print("="*50)
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {np.sqrt(mse):.2f}")
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Độ chính xác: {accuracy:.2f}%")
        print(f"R²: {r2:.4f}")
        print("="*50)
        
        # Trực quan hóa kết quả
        plt.figure(figsize=(16, 8))
        plt.plot(y_test_orig, label='Giá thực tế', alpha=0.8)
        plt.plot(predictions_orig, label='Dự báo AI', linestyle='--', alpha=0.9)
        plt.title(f'So sánh dự báo và thực tế - {self.symbol}')
        plt.xlabel('Thời gian')
        plt.ylabel('Giá cổ phiếu')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.symbol}_forecast_vs_actual.png')
        plt.show()
        
        return predictions_orig, y_test_orig
    
    def plot_training_history(self, history):
        """Vẽ biểu đồ quá trình huấn luyện"""
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Tiến trình huấn luyện mô hình')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{self.symbol}_training_history.png')
        plt.show()
    
    def generate_ai_analysis(self):
        """Tạo phân tích AI bằng Gemini"""
        if not GEMINI_API_KEY:
            print("Không có API key Gemini, bỏ qua phân tích AI")
            return None
            
        if self.technical_indicators is None:
            self.add_technical_indicators()
            
        df = self.technical_indicators
        last_record = df.iloc[-1]
        
        # Chuẩn bị dữ liệu cho AI
        prompt = f"""
        Bạn là một chuyên gia phân tích chứng khoán. Hãy phân tích cổ phiếu {self.symbol} dựa trên các chỉ số sau:
        
        - Giá hiện tại: {last_record['Close']}
        - SMA 50 ngày: {last_record['SMA_50']}
        - SMA 200 ngày: {last_record['SMA_200']}
        - RSI: {last_record['RSI']}
        - MACD: {last_record['MACD']}
        - Tín hiệu hiện tại: {last_record['Signal']}
        
        Dựa trên các chỉ số kỹ thuật này:
        1. Đánh giá xu hướng ngắn hạn và dài hạn
        2. Phân tích sức mạnh thị trường
        3. Đưa ra khuyến nghị giao dịch (Mua/Bán/Giữ)
        4. Dự báo ngắn hạn (1-2 tuần)
        5. Cảnh báo rủi ro tiềm ẩn
        
        Trả lời bằng tiếng Việt, trình bày rõ ràng, mạch lạc.
        """
        
        try:
            response = gemini_model.generate_content(prompt)
            analysis = response.text
            print("\n" + "="*50)
            print("PHÂN TÍCH AI TỪ GEMINI")
            print("="*50)
            print(analysis)
            print("="*50)
            
            # Lưu phân tích vào file
            with open(f'{self.symbol}_ai_analysis.txt', 'w', encoding='utf-8') as f:
                f.write(analysis)
                
            return analysis
        except Exception as e:
            print(f"Lỗi khi tạo phân tích AI: {str(e)}")
            return None
    
    def full_analysis(self):
        """Thực hiện toàn bộ quy trình phân tích"""
        start_time = time.time()
        
        try:
            self.fetch_data()
            self.add_technical_indicators()
            self.visualize_technical_analysis()
            self.train_lstm_model(epochs=50)
            self.generate_ai_analysis()
            
            # Tạo báo cáo tổng hợp
            report = {
                'symbol': self.symbol,
                'last_price': self.technical_indicators['Close'].iloc[-1],
                'last_signal': self.technical_indicators['Signal'].iloc[-1],
                'analysis_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'execution_time': f"{time.time() - start_time:.2f} giây"
            }
            
            with open(f'{self.symbol}_report.json', 'w') as f:
                json.dump(report, f)
                
            print(f"\nPhân tích hoàn tất! Kết quả đã được lưu vào các file {self.symbol}_*")
            return True
        except Exception as e:
            print(f"Lỗi trong quá trình phân tích: {str(e)}")
            traceback.print_exc()
            return False

if __name__ == "__main__":
    # Cấu hình
    symbol = "VIC"  # Mã cổ phiếu VinGroup
    start_date = "2020-01-01"
    end_date = datetime.now().strftime("%Y-%m-%d")
    
    # Thực hiện phân tích
    analyzer = StockAnalyzer(symbol, start_date, end_date)
    analyzer.full_analysis()