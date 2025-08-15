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
# from sklearn.model_selection import train_test_split # Không dùng trong LSTM cơ bản này
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import ta
import warnings
# SDK cho Google Qwen
import google.generativeai as genai
from dotenv import load_dotenv
# Giả sử vnstock và các module con đã được cài đặt và import đúng cách
from vnstock import *
import traceback
from vnstock.explorer.vci import Quote, Finance
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

# ======================
# CẤU HÌNH VÀ THƯ VIỆN
# ======================
# Tải biến môi trường cho Qwen
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("Không tìm thấy khóa API Qwen. Vui lòng kiểm tra file .env")
    exit()

# Cấu hình API client cho Qwen
genai.configure(api_key=GOOGLE_API_KEY)

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
                'signal': 'LỖI',
                'score': 50,
                'current_price': 0,
                'rsi_value': 0,
                'ma10': 0,
                'ma20': 0,
                'ma50': 0,
                'ma200': 0, # Thêm ma200
                'rs': 1.0,  # Thêm rs
                'rs_point': 0, # Thêm rs_point
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }

        # Sắp xếp theo ngày tăng dần để tính toán chính xác
        df = df.sort_index()

        # --- BƯỚC 1: Tính các chỉ báo kỹ thuật ---
        # 1. Đường trung bình
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
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

        # 5. Volume Moving Averages
        df['Volume_SMA_20'] = ta.trend.sma_indicator(df['Volume'], window=20)
        df['Volume_SMA_50'] = ta.trend.sma_indicator(df['Volume'], window=50)

        # 6. Ichimoku Cloud
        # Các tham số mặc định cho Ichimoku
        tenkan_window = 9
        kijun_window = 26
        senkou_span_b_window = 52

        # Tenkan-sen (Conversion Line)
        tenkan_sen_high = df['High'].rolling(window=tenkan_window).max()
        tenkan_sen_low = df['Low'].rolling(window=tenkan_window).min()
        df['ichimoku_tenkan_sen'] = (tenkan_sen_high + tenkan_sen_low) / 2

        # Kijun-sen (Base Line)
        kijun_sen_high = df['High'].rolling(window=kijun_window).max()
        kijun_sen_low = df['Low'].rolling(window=kijun_window).min()
        df['ichimoku_kijun_sen'] = (kijun_sen_high + kijun_sen_low) / 2

        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_span_a'] = ((df['ichimoku_tenkan_sen'] + df['ichimoku_kijun_sen']) / 2).shift(kijun_window)

        # Senkou Span B (Leading Span B)
        senkou_span_b_high = df['High'].rolling(window=senkou_span_b_window).max()
        senkou_span_b_low = df['Low'].rolling(window=senkou_span_b_window).min()
        df['ichimoku_senkou_span_b'] = ((senkou_span_b_high + senkou_span_b_low) / 2).shift(kijun_window)

        # Chikou Span (Lagging Span)
        df['ichimoku_chikou_span'] = df['Close'].shift(-kijun_window)

        # --- BƯỚC 2: Tính RS (Relative Strength so với VNINDEX) ---
        try:
            # Lấy dữ liệu VNINDEX trong cùng khoảng thời gian
            start_date = df.index[0].strftime('%Y-%m-%d')
            end_date = df.index[-1].strftime('%Y-%m-%d')
            vnindex_df = stock_historical_data("VNINDEX", start_date, end_date, "1D")

            if len(vnindex_df) == 0:
                raise ValueError("Không lấy được dữ liệu VNINDEX")

            vnindex_df.set_index('TradingDate', inplace=True)
            vnindex_df.sort_index(inplace=True)

            # Gộp dữ liệu cổ phiếu và VNINDEX theo ngày
            df_merged = df[['Close']].join(vnindex_df[['Close']].rename(columns={'Close': 'VNINDEX_Close'}), how='left')

            # Nếu thiếu dữ liệu VNINDEX, không tính RS
            if df_merged['VNINDEX_Close'].isna().all():
                df['RS'] = 1.0  # Mặc định
                df['RS_Point'] = 0.0
                print("Cảnh báo: Không có dữ liệu VNINDEX, bỏ qua RS")
            else:
                # Điền dữ liệu VNINDEX nếu thiếu (forward fill)
                df_merged['VNINDEX_Close'] = df_merged['VNINDEX_Close'].ffill()

                # Tính RS = price / VNINDEX
                df['RS'] = df_merged['Close'] / df_merged['VNINDEX_Close']

                # Tính RS_Point theo công thức
                roc_63 = ta.momentum.roc(df['Close'], window=63)
                roc_126 = ta.momentum.roc(df['Close'], window=126)
                roc_189 = ta.momentum.roc(df['Close'], window=189)
                roc_252 = ta.momentum.roc(df['Close'], window=252)

                df['RS_Point'] = (
                    roc_63.fillna(0) * 0.4 +
                    roc_126.fillna(0) * 0.2 +
                    roc_189.fillna(0) * 0.2 +
                    roc_252.fillna(0) * 0.2
                ) * 100

                # Tính các đường trung bình cho RS_Point
                df['RS_Point_SMA_10'] = ta.trend.sma_indicator(df['RS_Point'], window=10)
                df['RS_Point_SMA_20'] = ta.trend.sma_indicator(df['RS_Point'], window=20)
                df['RS_Point_SMA_50'] = ta.trend.sma_indicator(df['RS_Point'], window=50)
                df['RS_Point_SMA_200'] = ta.trend.sma_indicator(df['RS_Point'], window=200)

                # Tính các đường trung bình cho RS
                df['RS_SMA_10'] = ta.trend.sma_indicator(df['RS'], window=10)
                df['RS_SMA_20'] = ta.trend.sma_indicator(df['RS'], window=20)
                df['RS_SMA_50'] = ta.trend.sma_indicator(df['RS'], window=50)
                df['RS_SMA_200'] = ta.trend.sma_indicator(df['RS'], window=200)

        except Exception as e:
            print(f"Không thể tính RS do lỗi VNINDEX: {e}")
            df['RS'] = 1.0
            df['RS_Point'] = 0.0
            df['RS_SMA_10'] = 1.0
            df['RS_SMA_20'] = 1.0
            df['RS_SMA_50'] = 1.0
            df['RS_SMA_200'] = 1.0
            df['RS_Point_SMA_10'] = 0.0
            df['RS_Point_SMA_20'] = 0.0
            df['RS_Point_SMA_50'] = 0.0
            df['RS_Point_SMA_200'] = 0.0

        # --- BƯỚC 3: Kiểm tra dữ liệu hợp lệ ---
        df = df.dropna(subset=['Close', 'SMA_10', 'SMA_20', 'SMA_50'], how='all')
        if len(df) < 20:
            print("Không đủ dữ liệu hợp lệ để phân tích")
            return {
                'signal': 'LỖI',
                'score': 50,
                'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'rsi_value': 50, # Mặc định
                'ma10': df['Close'].iloc[-1] if len(df) > 0 else 0, # Nếu có dữ liệu
                'ma20': df['Close'].iloc[-1] if len(df) > 0 else 0, # Nếu có dữ liệu
                'ma50': df['Close'].iloc[-1] if len(df) > 0 else 0, # Nếu có dữ liệu
                'ma200': df['Close'].iloc[-1] if len(df) > 0 else 0, # Nếu có dữ liệu
                'rs': 1.0, # Mặc định
                'rs_point': 0, # Mặc định
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }

        # --- BƯỚC 4: Vẽ biểu đồ ---
        try:
            plt.figure(figsize=(16, 16))
            # Điều chỉnh GridSpec để thêm biểu đồ RS và RS_Point
            grid = plt.GridSpec(6, 1, hspace=0.2, height_ratios=[3, 1, 1, 1, 1, 1])

            # Biểu đồ 1: Giá và các đường trung bình
            ax1 = plt.subplot(grid[0])
            plt.plot(df.index, df['Close'], label='Giá đóng cửa', color='#1f77b4', linewidth=1.5)
            plt.plot(df.index, df['SMA_10'], label='SMA 10', color='blue', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['SMA_20'], label='SMA 20', color='orange', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['SMA_50'], label='SMA 50', color='green', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['SMA_200'], label='SMA 200', color='purple', alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df['BB_Upper'], label='BB Upper', color='red', alpha=0.5, linestyle='--')
            plt.plot(df.index, df['BB_Lower'], label='BB Lower', color='green', alpha=0.5, linestyle='--')
            plt.fill_between(df.index, df['BB_Lower'], df['BB_Upper'], color='gray', alpha=0.1)

            # Đánh dấu điểm giao nhau SMA10 và SMA20
            cross_10_20_above = (df['SMA_10'] > df['SMA_20']) & (df['SMA_10'].shift(1) <= df['SMA_20'].shift(1))
            cross_10_20_below = (df['SMA_10'] < df['SMA_20']) & (df['SMA_10'].shift(1) >= df['SMA_20'].shift(1))

            if cross_10_20_above.any():
                plt.scatter(df.index[cross_10_20_above], df.loc[cross_10_20_above, 'SMA_10'],
                            marker='^', color='lime', s=60, label='SMA10 > SMA20', zorder=5)
            if cross_10_20_below.any():
                plt.scatter(df.index[cross_10_20_below], df.loc[cross_10_20_below, 'SMA_10'],
                            marker='v', color='magenta', s=60, label='SMA10 < SMA20', zorder=5)

            plt.title(f'Phân tích kỹ thuật {symbol} - Giá và Chỉ báo', fontsize=14, fontweight='bold')
            plt.ylabel('Giá (VND)', fontsize=12)
            plt.legend(loc='upper left', fontsize=10)
            plt.grid(True, alpha=0.3)

            # Biểu đồ 2: RSI
            ax2 = plt.subplot(grid[1], sharex=ax1)
            plt.plot(df.index, df['RSI'], label='RSI', color='purple')
            plt.axhline(70, linestyle='--', color='red', alpha=0.7)
            plt.axhline(30, linestyle='--', color='green', alpha=0.7)
            plt.fill_between(df.index, 30, 70, color='gray', alpha=0.1)
            plt.title('RSI (Relative Strength Index)', fontsize=12)
            plt.ylim(0, 100)
            plt.ylabel('RSI', fontsize=10)
            plt.grid(True, alpha=0.3)

            # Biểu đồ 3: MACD
            ax3 = plt.subplot(grid[2], sharex=ax1)
            plt.plot(df.index, df['MACD'], label='MACD', color='blue')
            plt.plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
            plt.bar(df.index, df['MACD_Hist'], color=np.where(df['MACD_Hist'] > 0, 'green', 'red'), alpha=0.5)
            plt.title('MACD', fontsize=12)
            plt.ylabel('MACD', fontsize=10)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Biểu đồ 4: RS (Relative Strength vs VNINDEX) với các đường trung bình
            ax4 = plt.subplot(grid[3], sharex=ax1)
            plt.plot(df.index, df['RS'], label='RS (Price / VNINDEX)', color='brown', linewidth=1.5)
            plt.plot(df.index, df['RS_SMA_10'], label='RS SMA 10', color='blue', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_SMA_20'], label='RS SMA 20', color='orange', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_SMA_50'], label='RS SMA 50', color='green', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_SMA_200'], label='RS SMA 200', color='purple', alpha=0.7, linewidth=1)
            plt.title('Relative Strength (RS) so với VNINDEX và các đường trung bình', fontsize=12)
            plt.ylabel('RS', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            # Biểu đồ 5: RS_Point với các đường trung bình
            ax5 = plt.subplot(grid[4], sharex=ax1)
            plt.plot(df.index, df['RS_Point'], label='RS_Point', color='darkblue', linewidth=1.5)
            plt.plot(df.index, df['RS_Point_SMA_10'], label='RS_Point SMA 10', color='blue', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_Point_SMA_20'], label='RS_Point SMA 20', color='orange', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_Point_SMA_50'], label='RS_Point SMA 50', color='green', alpha=0.7, linewidth=1)
            plt.plot(df.index, df['RS_Point_SMA_200'], label='RS_Point SMA 200', color='purple', alpha=0.7, linewidth=1)
            plt.axhline(0, color='black', linestyle='-', linewidth=0.8)
            plt.fill_between(df.index, df['RS_Point'], 0, where=(df['RS_Point'] > 0), color='green', alpha=0.3)
            plt.fill_between(df.index, df['RS_Point'], 0, where=(df['RS_Point'] < 0), color='red', alpha=0.3)
            plt.title('RS_Point và các đường trung bình', fontsize=12)
            plt.ylabel('RS_Point', fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)

            # Biểu đồ 6: Khối lượng với Volume SMA
            ax6 = plt.subplot(grid[5], sharex=ax1)
            if show_volume and 'Volume' in df.columns:
                # Vẽ Volume SMA nếu có dữ liệu
                volume_sma_plotted = False
                if 'Volume_SMA_20' in df.columns and not df['Volume_SMA_20'].isna().all():
                    plt.plot(df.index, df['Volume_SMA_20'], label='Vol SMA 20', color='orange', alpha=0.8, linewidth=1.5)
                    volume_sma_plotted = True
                if 'Volume_SMA_50' in df.columns and not df['Volume_SMA_50'].isna().all():
                    plt.plot(df.index, df['Volume_SMA_50'], label='Vol SMA 50', color='purple', alpha=0.8, linewidth=1.5)
                    volume_sma_plotted = True

                # Vẽ biểu đồ cột khối lượng
                colors = np.where(df['Close'] > df['Open'], 'green', 'red')
                plt.bar(df.index, df['Volume'], color=colors, alpha=0.7, label='Volume' if not volume_sma_plotted else None)

                # Cập nhật legend
                handles, labels = ax6.get_legend_handles_labels()
                if handles:
                    by_label = dict(zip(labels, handles))
                    plt.legend(by_label.values(), by_label.keys(), loc='upper left')
                else:
                    plt.legend(loc='upper left')

                plt.title('Khối lượng giao dịch & Volume SMA', fontsize=12)
                plt.ylabel('Khối lượng', fontsize=10)
                plt.grid(True, alpha=0.3)
            else:
                plt.title('Khối lượng giao dịch', fontsize=12)
                plt.ylabel('Khối lượng', fontsize=10)
                plt.grid(True, alpha=0.3)

            # Định dạng trục x
            plt.gcf().autofmt_xdate()
            ax6.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax6.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

            # Lưu biểu đồ
            plt.tight_layout()
            plt.savefig(f'vnstocks_data/{symbol}_technical_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Đã lưu biểu đồ phân tích kỹ thuật vào vnstocks_data/{symbol}_technical_analysis.png")

        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")

        # --- BƯỚC 5: Tạo tín hiệu giao dịch ---
        try:
            last_row = df.iloc[-1]
            current_price = last_row['Close']
            rsi_value = last_row['RSI'] if not pd.isna(last_row['RSI']) else 50
            ma10_value = last_row['SMA_10'] if not pd.isna(last_row['SMA_10']) else current_price
            ma20_value = last_row['SMA_20'] if not pd.isna(last_row['SMA_20']) else current_price
            ma50_value = last_row['SMA_50'] if not pd.isna(last_row['SMA_50']) else current_price
            ma200_value = last_row['SMA_200'] if not pd.isna(last_row['SMA_200']) else current_price
            rs_value = last_row['RS'] if not pd.isna(last_row['RS']) else 1.0
            rs_point_value = last_row['RS_Point'] if not pd.isna(last_row['RS_Point']) else 0

            # Lấy giá trị Ichimoku từ hàng cuối
            tenkan_sen = last_row.get('ichimoku_tenkan_sen', np.nan)
            kijun_sen = last_row.get('ichimoku_kijun_sen', np.nan)
            senkou_span_a = last_row.get('ichimoku_senkou_span_a', np.nan)
            senkou_span_b = last_row.get('ichimoku_senkou_span_b', np.nan)
            chikou_span = last_row.get('ichimoku_chikou_span', np.nan)

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
            macd_value = last_row['MACD'] if not pd.isna(last_row['MACD']) else 0
            macd_signal = last_row['MACD_Signal'] if not pd.isna(last_row['MACD_Signal']) else 0
            macd_hist = last_row['MACD_Hist'] if not pd.isna(last_row['MACD_Hist']) else 0
            if macd_value > macd_signal and macd_hist > 0:
                score += 7  # Tín hiệu mua
            elif macd_value < macd_signal and macd_hist < 0:
                score -= 7  # Tín hiệu bán
            else:
                score += np.clip(macd_hist * 30, -3, 3) # Dựa trên histogram

            # 5. Bollinger Bands - 5 điểm
            bb_upper = last_row['BB_Upper'] if not pd.isna(last_row['BB_Upper']) else current_price
            bb_lower = last_row['BB_Lower'] if not pd.isna(last_row['BB_Lower']) else current_price
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
            if rs_value > last_row.get('RS_SMA_10', rs_value):
                rs_score += 2
            if rs_value > last_row.get('RS_SMA_20', rs_value):
                rs_score += 3
            if rs_value > last_row.get('RS_SMA_50', rs_value):
                rs_score += 5
            score += rs_score

            # 7. RS_Point - 10 điểm
            # So sánh RS_Point với các đường trung bình của chính nó
            rs_point_score = 0
            if rs_point_value > last_row.get('RS_Point_SMA_10', rs_point_value):
                rs_point_score += 2
            if rs_point_value > last_row.get('RS_Point_SMA_20', rs_point_value):
                rs_point_score += 3
            if rs_point_value > last_row.get('RS_Point_SMA_50', rs_point_value):
                rs_point_score += 5
            score += rs_point_score

            # 8. Ichimoku Cloud - 10 điểm
            ichimoku_score = 0
            try:
                # Kiểm tra nếu các giá trị không phải NaN
                if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
                    # 8.1 Vị trí giá so với Cloud (5 điểm)
                    cloud_top = max(senkou_span_a, senkou_span_b)
                    cloud_bottom = min(senkou_span_a, senkou_span_b)
                    if current_price > cloud_top:
                        ichimoku_score += 5 # Giá trên Cloud - Tín hiệu Mua
                    elif current_price < cloud_bottom:
                        ichimoku_score -= 5 # Giá dưới Cloud - Tín hiệu Bán
                    # (Nếu giá trong cloud, điểm = 0)
                    # 8.2 Tenkan-sen vs Kijun-sen (3 điểm)
                    if tenkan_sen > kijun_sen:
                        ichimoku_score += 3 # Tenkan > Kijun - Tín hiệu Mua
                    elif tenkan_sen < kijun_sen:
                        ichimoku_score -= 3 # Tenkan < Kijun - Tín hiệu Bán
                    # 8.3 Kijun-sen vs Cloud (2 điểm)
                    if kijun_sen > cloud_top:
                        ichimoku_score += 2 # Kijun trên Cloud - Xu hướng tăng mạnh
                    elif kijun_sen < cloud_bottom:
                        ichimoku_score -= 2 # Kijun dưới Cloud - Xu hướng giảm mạnh
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
            analysis_date = df.index[-1].strftime('%d/%m/%Y')
            print(f"\n📊 TÍN HIỆU GIAO DỊCH CUỐI ({analysis_date}):")
            print(f"  - Giá & Đường trung bình: Giá={current_price:,.2f} | SMA10={ma10_value:,.2f} | SMA20={ma20_value:,.2f} | SMA50={ma50_value:,.2f} | SMA200={ma200_value:,.2f}")
            print(f"  - RS: {rs_value:.3f} (SMA10={last_row.get('RS_SMA_10', np.nan):.3f} | SMA20={last_row.get('RS_SMA_20', np.nan):.3f} | SMA50={last_row.get('RS_SMA_50', np.nan):.3f})")
            print(f"  - RS_Point: {rs_point_value:.2f} (SMA10={last_row.get('RS_Point_SMA_10', np.nan):.2f} | SMA20={last_row.get('RS_Point_SMA_20', np.nan):.2f} | SMA50={last_row.get('RS_Point_SMA_50', np.nan):.2f})")

            # In thông tin Ichimoku
            try:
                print(f"  - Ichimoku:")
                print(f"    * Tenkan-sen: {tenkan_sen:.2f} | Kijun-sen: {kijun_sen:.2f}")
                print(f"    * Cloud (A/B): {senkou_span_a:.2f} / {senkou_span_b:.2f}")
                print(f"    * Chikou Span: {chikou_span:.2f}")
                print(f"    * Điểm Ichimoku: ~{ichimoku_score:.1f}")
            except:
                print(f"  - Ichimoku: Không có dữ liệu")

            print(f"  - Đề xuất: {recommendation} (Điểm: {score:.1f})")

            # --- Đảm bảo trả về đầy đủ các khóa ---
            return {
                'signal': signal,
                'score': score,
                'current_price': current_price,
                'rsi_value': rsi_value, # Đảm bảo có khóa rsi_value
                'ma10': ma10_value,      # Đảm bảo có khóa ma10
                'ma20': ma20_value,      # Đảm bảo có khóa ma20
                'ma50': ma50_value,      # Đảm bảo có khóa ma50
                'ma200': ma200_value,    # Đảm bảo có khóa ma200
                'rs': rs_value,          # Đảm bảo có khóa rs
                'rs_point': rs_point_value, # Đảm bảo có khóa rs_point
                'recommendation': recommendation
            }

        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu: {str(e)}")
            traceback.print_exc() # In traceback đầy đủ để debug
            return {
                'signal': 'LỖI',
                'score': 50,
                'current_price': df['Close'].iloc[-1] if len(df) > 0 else 0,
                'rsi_value': 50, # Mặc định nếu lỗi tính RSI
                'ma10': df['Close'].iloc[-1] if len(df) > 0 and 'SMA_10' in df.columns else 0, # An toàn hơn
                'ma20': df['Close'].iloc[-1] if len(df) > 0 and 'SMA_20' in df.columns else 0, # An toàn hơn
                'ma50': df['Close'].iloc[-1] if len(df) > 0 and 'SMA_50' in df.columns else 0, # An toàn hơn
                'ma200': df['Close'].iloc[-1] if len(df) > 0 and 'SMA_200' in df.columns else 0, # An toàn hơn, đảm bảo có khóa ma200
                'rs': 1.0, # Mặc định nếu lỗi tính RS, đảm bảo có khóa rs
                'rs_point': 0, # Mặc định nếu lỗi tính RS_Point, đảm bảo có khóa rs_point
                'recommendation': 'KHÔNG XÁC ĐỊNH'
            }

    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {str(e)}")
        traceback.print_exc() # In traceback đầy đủ để debug
        return {
            'signal': 'LỖI',
            'score': 50,
            'current_price': 0,
            'rsi_value': 0,
            'ma10': 0,
            'ma20': 0,
            'ma50': 0,
            'ma200': 0, # Đảm bảo có khóa ma200
            'rs': 1.0,  # Đảm bảo có khóa rs
            'rs_point': 0, # Đảm bảo có khóa rs_point
            'recommendation': 'KHÔNG XÁC ĐỊNH'
        }

# ======================
# PHẦN 5: TÍCH HỢP PHÂN TÍCH BẰNG QWEN
# ======================
def analyze_with_gemini(symbol, trading_signal, financial_data=None):
    """Phân tích cổ phiếu bằng Google Qwen dựa trên dữ liệu kỹ thuật và BCTC"""
    try:
        # --- Cập nhật Prompt với đầy đủ thông tin ---
        prompt = f"""
Hãy đóng vai một chuyên gia phân tích chứng khoán tại Việt Nam. Phân tích cổ phiếu {symbol} dựa trên các thông tin sau:
1. Tín hiệu giao dịch:
   - Tín hiệu: {trading_signal['signal']}
   - Điểm phân tích: {trading_signal['score']}/100
   - Giá hiện tại: {trading_signal['current_price']:,.0f} VND
   - RSI: {trading_signal['rsi_value']:.2f} # Sử dụng đúng khóa
   - MA10: {trading_signal['ma10']:,.0f} VND # Sử dụng đúng khóa
   - MA20: {trading_signal['ma20']:,.0f} VND # Sử dụng đúng khóa
   - MA50: {trading_signal['ma50']:,.0f} VND # Sử dụng đúng khóa
   - MA200: {trading_signal['ma200']:,.0f} VND # Sử dụng đúng khóa
   - RS (so với VNINDEX): {trading_signal['rs']:.3f} # Sử dụng đúng khóa
   - RS_Point: {trading_signal['rs_point']:.2f} # Sử dụng đúng khóa
2. Không có dự báo giá từ mô hình AI (đã loại bỏ LSTM/N-BEATS theo yêu cầu)
3. Dữ liệu tài chính (BCTC) gần nhất:
"""
        if financial_data is not None and not financial_data.empty:
            try:
                # Lấy quý gần nhất
                financial_data_sorted = financial_data.copy()
                # Giới hạn số cột để tránh prompt quá dài
                prompt += f"{financial_data_sorted.head(5).to_string(index=False)}\n"
            except Exception as e:
                print(f"Lỗi khi xử lý dữ liệu tài chính: {str(e)}")
                prompt += "   - Không có dữ liệu tài chính chi tiết\n"
        else:
            prompt += "   - Không có dữ liệu tài chính\n"
        prompt += """
Yêu cầu phân tích:
- Tổng hợp phân tích kỹ thuật và cơ bản
- Đánh giá sức mạnh tài chính của công ty
- Phân tích xu hướng giá và tín hiệu kỹ thuật (bao gồm RS, RS_Point, Ichimoku, RSI, Bollinger Bands, các đường trung bình nếu có)
- Nhận định rủi ro tiềm ẩn
- Dự báo triển vọng ngắn hạn và trung hạn
- Đưa ra khuyến nghị đầu tư (Mua/Bán/Nắm giữ) với lý do cụ thể
Kết quả phân tích cần:
- Ngắn gọn, súc tích (không quá 500 từ)
- Chuyên nghiệp như một nhà phân tích chứng khoán
- Bao gồm cả yếu tố thị trường tổng thể
- Có số liệu minh họa cụ thể
"""
        # Sử dụng Qwen Pro để phân tích
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Lỗi khi phân tích bằng Qwen: {str(e)}")
        return "Không thể tạo phân tích bằng Qwen tại thời điểm này."

# ======================
# PHẦN 6: CHỨC NĂNG CHÍNH
# ======================
def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán với tích hợp Qwen (đã loại bỏ LSTM)"""
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
    if len(df_features) < 100:  # Cần ít nhất 100 điểm dữ liệu
        print(f"Dữ liệu cho mã {symbol} quá ít để phân tích ({len(df_features)} điểm)")
        return None

    print(f"\nĐang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_features)
    print(f"\nĐang phân tích bằng Google Qwen...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, financial_data)

    # In kết quả
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.2f}/100")
    print(f"\nPHÂN TÍCH TỔNG HỢP TỪ QWEN:")
    print(gemini_analysis)

    # --- Đảm bảo report có đầy đủ các khóa ---
    report = {
        'symbol': symbol,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'current_price': float(trading_signal['current_price']),
        'signal': trading_signal['signal'],
        'recommendation': trading_signal['recommendation'],
        'score': float(trading_signal['score']),
        'rsi_value': float(trading_signal['rsi_value']), # Sử dụng đúng khóa
        'ma10': float(trading_signal['ma10']),          # Sử dụng đúng khóa
        'ma20': float(trading_signal['ma20']),          # Sử dụng đúng khóa
        'ma50': float(trading_signal['ma50']),          # Sử dụng đúng khóa
        'ma200': float(trading_signal['ma200']),        # Sử dụng đúng khóa
        'rs': float(trading_signal['rs']),              # Sử dụng đúng khóa
        'rs_point': float(trading_signal['rs_point']),  # Sử dụng đúng khóa
        'forecast': [], # Không có dự báo từ mô hình
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
    for symbol in stock_list['symbol'].head(10):  # Phân tích 10 mã đầu tiên để demo
        try:
            print(f"\nPhân tích mã {symbol}...")
            report = analyze_stock(symbol) # Gọi trực tiếp hàm analyze_stock
            if report and report['signal'] != 'LỖI':
                results.append(report)
            time.sleep(1)  # Dừng 1 giây giữa các request
        except Exception as e:
            print(f"Lỗi khi phân tích mã {symbol}: {str(e)}")
            traceback.print_exc()
            continue
    # Tạo báo cáo tổng hợp
    if results:
        # Sắp xếp theo điểm phân tích
        results.sort(key=lambda x: x['score'], reverse=True)
        # Tạo DataFrame - Đảm bảo tên cột khớp với khóa trong report
        df_results = pd.DataFrame([{
            'Mã': r['symbol'],
            'Giá': r['current_price'],
            'Điểm': r['score'],
            'Tín hiệu': r['signal'],
            'Đề xuất': r['recommendation'],
            'RSI': r['rsi_value'],     # Sử dụng đúng khóa
            'MA10': r['ma10'],         # Sử dụng đúng khóa
            'MA20': r['ma20'],         # Sử dụng đúng khóa
            'MA50': r['ma50'],         # Sử dụng đúng khóa
            'MA200': r['ma200'],       # Sử dụng đúng khóa
            'RS': r['rs'],             # Sử dụng đúng khóa
            'RS_Point': r['rs_point']  # Sử dụng đúng khóa
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
            sns.barplot(x='Mã', y='Điểm', data=df_results.head(10), palette='viridis') # Chỉ vẽ top 10
            plt.title('Top 10 Điểm phân tích các mã chứng khoán')
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
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE QWEN")
    print("==============================================")
    # Lấy dữ liệu thị trường
    market_data = get_market_data()
    analyze_stock('DRI')
    # screen_stocks() # Bỏ comment nếu muốn quét nhiều mã
    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")
