import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from vnstock import *  # Thư viện chính thức của VNStocks

# 1. Thu thập dữ liệu từ VNStocks
def get_stock_data(symbol, start_date, end_date):
    """
    Lấy dữ liệu chứng khoán từ VNStocks
    """
    df = stock_historical_data(symbol, start_date, end_date, "1D")
    return df

# 2. Tiền xử lý dữ liệu
def preprocess_data(df):
    """
    Chuẩn bị dữ liệu cho mô hình AI
    """
    # Chọn giá đóng cửa
    data = df[['close']].values
    
    # Chuẩn hóa dữ liệu
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Tạo dataset theo chuỗi thời gian
    time_step = 60
    X, y = [], []
    for i in range(time_step, len(scaled_data)):
        X.append(scaled_data[i-time_step:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler

# 3. Xây dựng mô hình AI (LSTM)
def build_lstm_model(input_shape):
    """
    Tạo mô hình LSTM để dự báo giá cổ phiếu
    """
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 4. Dự báo và trực quan hóa
def predict_and_visualize(model, X_test, y_test, scaler):
    """
    Dự báo và hiển thị kết quả
    """
    # Dự báo
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    
    # Trực quan hóa
    plt.figure(figsize=(16,8))
    plt.plot(actual, color='blue', label='Giá thực tế')
    plt.plot(predictions, color='red', label='Dự báo AI')
    plt.title(f'Dự báo giá cổ phiếu {symbol}')
    plt.xlabel('Thời gian')
    plt.ylabel('Giá cổ phiếu')
    plt.legend()
    plt.savefig('stock_forecast.png')
    plt.show()
    
    return predictions

# 5. Phân tích kỹ thuật tự động
def technical_analysis(df):
    """
    Tạo tín hiệu giao dịch dựa trên phân tích kỹ thuật
    """
    # Tính SMA
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()
    
    # Tính RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Tạo tín hiệu
    df['Signal'] = np.where(
        (df['SMA_50'] > df['SMA_200']) & (df['RSI'] < 70), 
        'MUA', 
        np.where(
            (df['SMA_50'] < df['SMA_200']) & (df['RSI'] > 30),
            'BÁN',
            'GIỮ'
        )
    )
    
    return df

# Main execution
if __name__ == "__main__":
    # Cấu hình
    symbol = "VIC"  # Mã cổ phiếu VinGroup
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    # Thu thập dữ liệu
    stock_data = get_stock_data(symbol, start_date, end_date)
    print(f"Thu thập {len(stock_data)} bản ghi dữ liệu")
    
    # Phân tích kỹ thuật
    analyzed_data = technical_analysis(stock_data)
    print(analyzed_data[['date', 'close', 'SMA_50', 'SMA_200', 'RSI', 'Signal']].tail(10))
    
    # Chuẩn bị dữ liệu cho AI
    X, y, scaler = preprocess_data(analyzed_data.dropna())
    
    # Chia dữ liệu train/test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Xây dựng và huấn luyện mô hình
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)
    
    # Dự báo và hiển thị kết quả
    predictions = predict_and_visualize(model, X_test, y_test, scaler)
    
    # Lưu kết quả
    results = pd.DataFrame({
        'Actual': scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
        'Predicted': predictions.flatten()
    })
    results.to_csv(f'{symbol}_predictions.csv', index=False)
    print("Kết quả đã được lưu vào file CSV")