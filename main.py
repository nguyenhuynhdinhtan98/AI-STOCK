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
# --- CẢI TIẾN 1: Import PyTorch và kiểm tra thiết bị nâng cao ---
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.multiprocessing as mp # Có thể dùng cho song song hóa dữ liệu nếu cần mở rộng
# --- HẾT CẢI TIẾN 1 ---
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
# --- BIẾN GLOBAL ---
GLOBAL_EPOCHS = 50 # Đã sửa thành 50 để mô hình có thể chạy
GLOBAL_BATCH_SIZE = 128 # Tăng batch size để tận dụng tốt hơn GPU
GLOBAL_PREDICTION_DAYS = 10 # Số ngày dự báo
# --- HẾT PHẦN BIẾN GLOBAL ---
# --- BIẾN GLOBAL CHO KHOẢNG THỜI GIAN ---
start_date = (datetime.now() - timedelta(days=365 * 10)).strftime("%Y-%m-%d")  # 10 năm trước
end_date = datetime.now().strftime("%Y-%m-%d")
# --- HẾT PHẦN BIẾN GLOBAL ---
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
# PHẦN 1: THU THẬP DỮ LIỆU (CẬP NHẬT)
# ======================
def get_vnstocks_list():
    """Lấy danh sách tất cả các mã chứng khoán trên thị trường Việt Nam sử dụng vnstock v2"""
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
    """Lấy dữ liệu lịch sử của một mã chứng khoán sử dụng vnstock v2 mới theo tài liệu"""
    try:
        quote = Quote(symbol)
        df = quote.history(start=start_date, end=end_date, interval="1D")
        if df is not None and not df.empty:
            df.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
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
# ======================
# PHẦN 1B: THU THẬP DỮ LIỆU BCTC (CẬP NHẬT)
# ======================
def get_financial_data(symbol):
    """Lấy dữ liệu báo cáo tài chính sử dụng vnstock v2 - 12 quý gần nhất"""
    try:
        financial_obj = Finance(symbol=symbol)
        # --- CẬP NHẬT: Thêm tham số limit=12 ---
        financial_data = financial_obj.ratio(period="quarter", lang="en", flatten_columns=True, limit=12)
        # --- HẾT CẬP NHẬT ---
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
    """Lấy dữ liệu thị trường tổng thể sử dụng vnstock v2"""
    try:
        quoteVNI = Quote(symbol="VNINDEX")
        vnindex = quoteVNI.history(start=start_date, end=end_date, interval="1D")
        if vnindex is not None and not vnindex.empty:
            vnindex.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            vnindex["Date"] = pd.to_datetime(vnindex["Date"])
            vnindex.set_index("Date", inplace=True)
            vnindex.sort_index(inplace=True)
            vnindex.to_csv("vnstocks_data/vnindex_data.csv")
        quoteVN30 = Quote(symbol="VN30")
        vn30 = quoteVN30.history(start=start_date, end=end_date, interval="1D")
        if vn30 is not None and not vn30.empty:
            vn30.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
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
    """Preprocesses raw stock data from vnstock"""
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
    """Generates technical indicators using pure pandas/numpy"""
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["BB_middle"] = df["Close"].rolling(window=20).mean()
    df["BB_std"] = df["Close"].rolling(window=20).std()
    df["BB_upper"] = df["BB_middle"] + (df["BB_std"] * 2)
    df["BB_lower"] = df["BB_middle"] - (df["BB_std"] * 2)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["ATR"] = true_range.rolling(window=14).mean()
    df["SMA_20"] = df["Close"].rolling(window=20).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["Momentum"] = df["Close"] / df["Close"].shift(4) - 1
    df["Volume_MA"] = df["Volume"].rolling(window=10).mean()
    df["Volume_Change"] = df["Volume"].pct_change()
    df.dropna(inplace=True)
    return df
# ======================
# PHẦN 3A: MÔ HÌNH AI - LSTM PYTORCH TỐI ƯU
# ======================
# --- CẢI TIẾN 2: Hàm kiểm tra thiết bị nâng cao ---
def check_device_and_configure():
    """Kiểm tra và cấu hình thiết bị tốt nhất (MPS, CUDA, CPU) cho PyTorch."""
    print("Kiểm tra thiết bị:")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ CUDA GPU được phát hiện: {torch.cuda.get_device_name(0)}")
        print("Các thiết bị CUDA khả dụng:")
        for i in range(torch.cuda.device_count()):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
        print("✅ MPS (Apple Silicon) được phát hiện và sẽ được sử dụng.")
    else:
        device = torch.device("cpu")
        print("⚠️ Không tìm thấy GPU (CUDA/MPS), sẽ sử dụng CPU.")
    return device
# --- HẾT CẢI TIẾN 2 ---
# --- CẢI TIẾN 3: Định nghĩa mô hình PyTorch cải tiến ---
class OptimizedLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=150, num_layers=4, output_size=1, dropout=0.2):
        super(OptimizedLSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_layer_size, 75)
        self.fc2 = nn.Linear(75, output_size)
    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        lstm_out = self.dropout(lstm_out[:, -1, :]) # Lấy output cuối cùng
        out = torch.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        predictions = self.fc2(out)
        return predictions
# --- HẾT CẢI TIẾN 3 ---
# --- CẢI TIẾN 4: Hàm huấn luyện PyTorch tối ưu ---
def train_stock_model_pytorch_optimized(df, target="Close", time_steps=60, test_size=0.2, epochs=GLOBAL_EPOCHS, batch_size=GLOBAL_BATCH_SIZE):
    """Huấn luyện mô hình LSTM PyTorch được tối ưu."""
    try:
        if df is None or len(df) < time_steps: return None, None, None, None, None
        if target not in df.columns: return None, None, None, None, None
        data = df[[target]].values.astype(np.float32)
        data = data[np.isfinite(data)].reshape(-1, 1)
        if len(data) == 0: return None, None, None, None, None
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        if len(scaled_data) <= time_steps: return None, None, None, None, None
        X, y = [], []
        for i in range(time_steps, len(scaled_data)):
            X.append(scaled_data[i - time_steps : i, 0])
            y.append(scaled_data[i, 0])
        if len(X) == 0 or len(y) == 0: return None, None, None, None, None
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        split_index = max(1, int(len(X) * (1 - test_size)))
        if split_index >= len(X): split_index = len(X) - 1
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        if len(X_train) == 0 or len(y_train) == 0: return None, None, None, None, None
        device = check_device_and_configure()
        # Sử dụng DataLoader để quản lý batch hiệu quả hơn
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type != 'cpu'))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type != 'cpu'))
        model = OptimizedLSTMModel(input_size=1, hidden_layer_size=150, num_layers=4, output_size=1, dropout=0.2)
        model.to(device)
        # Cố gắng compile model để tăng tốc (nếu hỗ trợ)
        try:
            model = torch.compile(model)
            print("✅ Model đã được compile để tăng hiệu suất.")
        except Exception as e:
            print(f"⚠️ Không thể compile model: {e}")
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        early_stopping_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        print(f"Bắt đầu huấn luyện mô hình LSTM PyTorch tối ưu với {epochs} epochs và batch_size={batch_size}...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')): # Mixed precision cho CUDA
                    y_pred = model(batch_x).squeeze()
                    loss = loss_function(y_pred, batch_y)
                scaler_loss = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
                if scaler_loss:
                    scaler_loss.scale(loss).backward()
                    scaler_loss.step(optimizer)
                    scaler_loss.update()
                else:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        y_pred = model(batch_x).squeeze()
                        loss = loss_function(y_pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0) # Nhân với batch size để tính trung bình đúng
            avg_val_loss = val_loss / len(test_loader.dataset)
            scheduler.step(avg_val_loss) # Cập nhật learning rate dựa trên val loss
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Lưu model tốt nhất
                torch.save(model.state_dict(), f'vnstocks_data/best_model_{df.index.name if df.index.name else "unknown"}.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break
        print("✅ Huấn luyện mô hình LSTM PyTorch tối ưu hoàn tất")
        model.eval()
        y_pred_list, y_test_list = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    y_pred = model(batch_x).cpu().numpy()
                y_pred_list.append(y_pred)
                y_test_list.append(batch_y.numpy())
        y_pred_all = np.concatenate(y_pred_list)
        y_test_all = np.concatenate(y_test_list)
        y_pred_inv = scaler.inverse_transform(y_pred_all)
        y_test_inv = scaler.inverse_transform(y_test_all.reshape(-1, 1))
        try:
            mse = mean_squared_error(y_test_inv, y_pred_inv)
            rmse_val = np.sqrt(mse)
            mae_val = mean_absolute_error(y_test_inv, y_pred_inv)
            r2 = r2_score(y_test_inv, y_pred_inv)
            print("\n--- ĐÁNH GIÁ MÔ HÌNH DỰ BÁO ---")
            print(f"RMSE: {rmse_val:.2f}")
            print(f"MAE: {mae_val:.2f}")
            print(f"R2: {r2:.2f}")
            print("--- HẾT ĐÁNH GIÁ ---\n")
        except Exception as e:
            print(f"Lỗi khi tính toán đánh giá LSTM PyTorch: {str(e)}")
            mse, rmse_val, mae_val, r2 = 0, 0, 0, 0
        return model, scaler, None, y_test_inv, y_pred_inv # Trả về None cho X_test_tensor vì không còn dùng
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi huấn luyện mô hình LSTM PyTorch tối ưu: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None
# --- HẾT CẢI TIẾN 4 ---
# --- CẢI TIẾN 5: Hàm dự báo PyTorch tối ưu (ĐÃ SỬA) ---
def predict_next_days_pytorch_optimized(model, scaler, df, target="Close", time_steps=60, n_days=GLOBAL_PREDICTION_DAYS):
    """Dự báo giá trong n ngày tiếp theo (cho LSTM PyTorch tối ưu)"""
    try:
        if model is None or scaler is None or df is None:
            print("Dữ liệu đầu vào không hợp lệ cho PyTorch predict")
            return np.array([]), np.array([])
        if target not in df.columns:
            print(f"Cột {target} không tồn tại")
            return np.array([]), np.array([])
        if len(df) < time_steps:
            print("Không đủ dữ liệu để dự báo")
            return np.array([]), np.array([])
        
        # --- BƯỚC 1: LẤY DỮ LIỆU CUỐI CÙNG ĐÚNG CÁCH ---
        # Lấy dữ liệu cuối cùng (time_steps điểm gần nhất)
        last_data = df[target].values[-time_steps:]
        # Kiểm tra xem có NaN hay không
        if pd.isna(last_data).any():
            # Nếu có NaN, cần xử lý (fill forward/backward hoặc bỏ qua)
            # Dưới đây là fill forward
            last_data_series = pd.Series(last_data)
            last_data_series = last_data_series.fillna(method='ffill').fillna(method='bfill')
            last_data = last_data_series.values
        
        # Đảm bảo dữ liệu có đủ chiều dài
        if len(last_data) != time_steps:
            print(f"Lỗi: Không đủ dữ liệu để dự báo ({len(last_data)} điểm, cần {time_steps} điểm)")
            return np.array([]), np.array([])
        
        # --- BƯỚC 2: CHUẨN HÓA DỮ LIỆU ---
        # Chuẩn hóa bằng scaler đã dùng khi huấn luyện
        try:
            # Chuyển thành array 2D để phù hợp với scaler.transform
            last_data_reshaped = last_data.reshape(-1, 1)
            last_data_scaled = scaler.transform(last_data_reshaped)
            # Trở lại dạng 1D để đưa vào mô hình
            last_data_scaled_flat = last_data_scaled.flatten()
        except Exception as e:
            print(f"Lỗi khi chuẩn hóa dữ liệu dự báo: {str(e)}")
            return np.array([]), np.array([])
        
        # --- BƯỚC 3: DỰ BÁO ---
        forecast_scaled = []
        model.eval()
        device = next(model.parameters()).device # Lấy thiết bị từ model
        with torch.no_grad():
            # Chuẩn bị input tensor (1 batch, time_steps, 1 feature)
            x_input = torch.tensor(last_data_scaled_flat.reshape(1, time_steps, 1), dtype=torch.float32).to(device)
            
            for _ in range(n_days):
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    pred = model(x_input)
                forecast_scaled.append(pred.item())
                # Cập nhật input cho bước dự báo tiếp theo
                # Lấy phần còn lại của chuỗi cũ và nối với dự báo mới
                x_input = torch.cat((x_input[:, 1:, :], pred.reshape(1, 1, 1)), dim=1)
        
        # --- BƯỚC 4: CHUYỂN ĐỔI GIÁ TRỞ LẠI ---
        try:
            # Chuyển đổi sang giá trị thực
            forecast_values = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"Lỗi khi chuyển đổi giá gốc: {str(e)}")
            return np.array([]), np.array([])
        
        # --- BƯỚC 5: TẠO NGÀY DỰ BÁO ---
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
        
        return np.array(forecast_dates), forecast_values
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi dự báo PyTorch tối ưu: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])
# --- HẾT CẢI TIẾN 5 ---
# --- CẬP NHẬT HÀM ĐÁNH GIÁ DỮ LIỆU (CHO LSTM) ---
def evaluate_data_for_ai(df_features, symbol):
    """Đánh giá dữ liệu để đề xuất mô hình AI phù hợp (chỉ LSTM)."""
    if df_features is None or len(df_features) == 0:
        print(f"❌ Không có dữ liệu để đánh giá cho mã {symbol}.")
        return "Không xác định", "Không có dữ liệu đầu vào."
    num_points = len(df_features)
    num_features = len(df_features.columns)
    print(f"--- ĐÁNH GIÁ DỮ LIỆU CHO MÃ {symbol} ---")
    print(f"Số điểm dữ liệu: {num_points}")
    print(f"Số lượng đặc trưng: {num_features}")
    if num_points > 2500: # Ngưỡng cao hơn một chút
        recommendation = "LSTM PYTORCH TỐI ƯU"
        reason = f"Dữ liệu phong phú ({num_points} điểm), LSTM PYTORCH TỐI ƯU sẽ phát huy hiệu quả cao."
    elif num_points > 1500:
        recommendation = "LSTM PYTORCH TỐI ƯU"
        reason = f"Dữ liệu ổn định ({num_points} điểm), LSTM PYTORCH TỐI ƯU là lựa chọn phù hợp."
    elif num_features > 60:
        recommendation = "LSTM PYTORCH TỐI ƯU"
        reason = f"Dữ liệu đa chiều ({num_features} đặc trưng), LSTM PYTORCH TỐI ƯU có khả năng xử lý tốt."
    else:
        recommendation = "LSTM PYTORCH TỐI ƯU"
        reason = f"Dữ liệu tiêu chuẩn ({num_points} điểm, {num_features} đặc trưng), LSTM PYTORCH TỐI ƯU đảm bảo hiệu suất & độ chính xác."
    print(f"💡 Đề xuất mô hình AI: {recommendation}")
    print(f"❓ Lý do: {reason}")
    print("--- HẾT ĐÁNH GIÁ ---")
    return recommendation, reason

# ======================
# PHẦN 3B: MÔ HÌNH AI - N-BEATS (BỔ SUNG)
# ======================

# --- THÊM: Định nghĩa mô hình N-BEATS ---
class NBeatsBlock(nn.Module):
    def __init__(self, input_size: int, theta_size: int, basis_function, layers: int, layer_size: int):
        super().__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.basis_function = basis_function
        self.layers = nn.ModuleList([nn.Linear(in_features=input_size, out_features=layer_size)] +
                                    [nn.Linear(in_features=layer_size, out_features=layer_size)
                                     for _ in range(layers - 1)])
        self.basis_parameters = nn.Linear(in_features=layer_size, out_features=theta_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        block_input = x
        for layer in self.layers:
            block_input = self.relu(layer(block_input))
        basis_parameters = self.basis_parameters(block_input)
        return self.basis_function(basis_parameters)


def seasonality_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "Nhiều tham số hơn các bước thời gian"
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    s1 = torch.tensor([torch.cos(2 * torch.pi * i * t) for i in range(p1)]).float().to(device)
    s2 = torch.tensor([torch.sin(2 * torch.pi * i * t) for i in range(p2)]).float().to(device)
    S = torch.cat([s1, s2], dim=0) # Đảm bảo kích thước đúng
    # Tính toán theo batch
    return torch.sum(thetas.unsqueeze(-1) * S.unsqueeze(0), dim=1)


def trend_model(thetas, t, device):
    p = thetas.size()[-1]
    assert p <= thetas.shape[1], "Nhiều tham số hơn các bước thời gian"
    T = torch.tensor([t ** i for i in range(p)]).float().to(device)
    # Tính toán theo batch
    return torch.sum(thetas.unsqueeze(-1) * T.unsqueeze(0), dim=1)


class NBeats(nn.Module):
    def __init__(self, device, input_size: int = 60, output_size: int = 10,
                 stacks: int = 3, blocks_per_stack: int = 1,
                 forecast_length: int = GLOBAL_PREDICTION_DAYS, backcast_length: int = None, # Cho phép None
                 thetas_dims: list = [4, 8], share_weights_in_stack: bool = False,
                 hidden_layer_units: int = 256):
        super(NBeats, self).__init__()
        self.device = device
        self.forecast_length = forecast_length
        # Xử lý backcast_length: nếu None thì dùng input_size, đảm bảo input_size là số nguyên hợp lệ
        if backcast_length is None:
            if not isinstance(input_size, int) or input_size <= 0:
                 raise ValueError(f"input_size phải là số nguyên dương nếu backcast_length không được cung cấp. Nhận được input_size={input_size}")
            self.backcast_length = input_size
        else:
            if not isinstance(backcast_length, int) or backcast_length <= 0:
                raise ValueError(f"backcast_length phải là số nguyên dương. Nhận được backcast_length={backcast_length}")
            self.backcast_length = backcast_length
            
        # Kiểm tra input_size có khớp với backcast_length không?
        # Trong hầu hết các trường hợp, input_size nên bằng backcast_length
        # Có thể thêm cảnh báo nếu chúng khác nhau, nhưng ở đây ta ưu tiên backcast_length
        if input_size != self.backcast_length:
             print(f"Cảnh báo: input_size ({input_size}) khác với backcast_length ({self.backcast_length}). Dùng backcast_length cho tính toán.")

        self.hidden_layer_units = hidden_layer_units
        self.stacks = stacks
        self.blocks_per_stack = blocks_per_stack
        self.share_weights_in_stack = share_weights_in_stack

        # Tạo tensor thời gian cho trend và seasonality
        # Đảm bảo rằng các giá trị này là số nguyên trước khi truyền cho linspace
        forecast_steps = int(self.forecast_length)
        backcast_steps = int(self.backcast_length)
        
        if forecast_steps <= 0:
            raise ValueError(f"forecast_length phải là số nguyên dương sau khi chuyển đổi. Nhận được forecast_length={forecast_length}")
        if backcast_steps <= 0:
            raise ValueError(f"backcast_length (sau khi xử lý) phải là số nguyên dương. Nhận được backcast_length={backcast_length}, input_size={input_size}")

        self.t_forecast = torch.linspace(0, 1, forecast_steps).to(self.device) # [forecast_length]
        self.t_backcast = torch.linspace(0, 1, backcast_steps).to(self.device) # [backcast_length]

        self.stack_list = nn.ModuleList()

        # Stack 1: Seasonality
        for _ in range(blocks_per_stack):
            block = NBeatsBlock(input_size=self.backcast_length, theta_size=thetas_dims[0],
                                basis_function=lambda thetas: seasonality_model(thetas, self.t_forecast, self.device),
                                layers=4, layer_size=hidden_layer_units)
            self.stack_list.append(block)

        # Stack 2: Trend
        for _ in range(blocks_per_stack):
            block = NBeatsBlock(input_size=self.backcast_length, theta_size=thetas_dims[1],
                                basis_function=lambda thetas: trend_model(thetas, self.t_forecast, self.device),
                                layers=4, layer_size=hidden_layer_units)
            self.stack_list.append(block)

        # Stack 3: Generic
        for _ in range(blocks_per_stack):
            block = NBeatsBlock(input_size=self.backcast_length,
                                theta_size=self.forecast_length + self.backcast_length,
                                basis_function=lambda thetas: thetas[:, :self.forecast_length], # Forecast part
                                layers=4, layer_size=hidden_layer_units)
            self.stack_list.append(block)

    def forward(self, x):
        # Kiểm tra kích thước đầu vào
        if x.dim() != 2:
            raise ValueError(f"Đầu vào x phải có 2 chiều [batch, seq_len], nhưng nhận được {x.dim()} chiều")
        if x.size(1) != self.backcast_length:
            raise ValueError(f"Chiều dài chuỗi đầu vào ({x.size(1)}) phải bằng backcast_length ({self.backcast_length})")
            
        # Chuẩn hóa đầu vào
        x_mean = torch.mean(x, dim=1, keepdim=True) # [batch, 1]
        x_centered = x - x_mean # [batch, backcast_length]
        # Khởi tạo backcast và forecast
        backcast = x_centered # [batch, backcast_length]
        forecast = torch.zeros(size=(x.size(0), self.forecast_length), device=self.device) # [batch, forecast_length]
        # Forward qua các stack
        for i, block in enumerate(self.stack_list):
            # block_input = backcast # [batch, backcast_length]
            block_forecast = block(backcast) # [batch, forecast_length] hoặc [batch, theta_size]
            if i < self.blocks_per_stack: # Seasonality
                 block_backcast = seasonality_model(block.basis_parameters(block.layers[-1](block.relu(block.layers[0](backcast)))), self.t_backcast, self.device) # Tính lại backcast từ theta
            elif i < 2 * self.blocks_per_stack: # Trend
                 block_backcast = trend_model(block.basis_parameters(block.layers[-1](block.relu(block.layers[0](backcast)))), self.t_backcast, self.device) # Tính lại backcast từ theta
            else: # Generic
                 theta_full = block.basis_parameters(block.layers[-1](block.relu(block.layers[0](backcast)))) # [batch, theta_size]
                 block_backcast = theta_full[:, self.forecast_length:] # [batch, backcast_length]
                 # block_forecast đã được tính trong block.forward
            backcast = backcast - block_backcast # [batch, backcast_length]
            forecast = forecast + block_forecast # [batch, forecast_length]
        # Thêm lại giá trị trung bình
        forecast = forecast + x_mean # [batch, forecast_length]
        return forecast

# --- HẾT THÊM: Định nghĩa mô hình N-BEATS ---

# --- THÊM: Hàm huấn luyện N-BEATS ---
def train_stock_model_nbeats(df, target="Close", time_steps=60, test_size=0.2, epochs=GLOBAL_EPOCHS, batch_size=GLOBAL_BATCH_SIZE):
    """Huấn luyện mô hình N-BEATS."""
    try:
        if df is None or len(df) < time_steps: return None, None, None, None, None
        if target not in df.columns: return None, None, None, None, None
        data = df[[target]].values.astype(np.float32)
        data = data[np.isfinite(data)].reshape(-1, 1)
        if len(data) == 0: return None, None, None, None, None
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        if len(scaled_data) <= time_steps: return None, None, None, None, None
        X, y = [], []
        for i in range(time_steps, len(scaled_data) - GLOBAL_PREDICTION_DAYS + 1): # Điều chỉnh để lấy chuỗi dự báo
            X.append(scaled_data[i - time_steps : i, 0])
            y.append(scaled_data[i : i + GLOBAL_PREDICTION_DAYS, 0]) # Dự báo nhiều ngày
        if len(X) == 0 or len(y) == 0: return None, None, None, None, None
        X, y = np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        # Không cần reshape cho N-BEATS, nó làm việc với (batch, seq_len)
        split_index = max(1, int(len(X) * (1 - test_size)))
        if split_index >= len(X): split_index = len(X) - 1
        X_train, X_test = X[:split_index], X[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        if len(X_train) == 0 or len(y_train) == 0: return None, None, None, None, None
        device = check_device_and_configure() # Sử dụng hàm kiểm tra thiết bị chung
        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type != 'cpu'))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type != 'cpu'))
        model = NBeats(device=device, input_size=time_steps, output_size=GLOBAL_PREDICTION_DAYS)
        model.to(device)
        try:
            model = torch.compile(model)
            print("✅ N-BEATS Model đã được compile để tăng hiệu suất.")
        except Exception as e:
            print(f"⚠️ Không thể compile N-BEATS model: {e}")
        loss_function = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)
        early_stopping_patience = 10
        best_val_loss = float('inf')
        patience_counter = 0
        print(f"Bắt đầu huấn luyện mô hình N-BEATS với {epochs} epochs và batch_size={batch_size}...")
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    y_pred = model(batch_x)
                    loss = loss_function(y_pred, batch_y)
                scaler_loss = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
                if scaler_loss:
                    scaler_loss.scale(loss).backward()
                    scaler_loss.step(optimizer)
                    scaler_loss.update()
                else:
                    loss.backward()
                    optimizer.step()
                epoch_loss += loss.item()
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        y_pred = model(batch_x)
                        loss = loss_function(y_pred, batch_y)
                    val_loss += loss.item() * batch_x.size(0)
            avg_val_loss = val_loss / len(test_loader.dataset)
            scheduler.step(avg_val_loss)
            avg_epoch_loss = epoch_loss / len(train_loader)
            print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(model.state_dict(), f'vnstocks_data/best_model_nbeats_{df.index.name if df.index.name else "unknown"}.pth')
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered cho N-BEATS.")
                    break
        print("✅ Huấn luyện mô hình N-BEATS hoàn tất")
        model.eval()
        y_pred_list, y_test_list = [], []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    y_pred = model(batch_x).cpu().numpy()
                y_pred_list.append(y_pred)
                y_test_list.append(batch_y.numpy())
        y_pred_all = np.concatenate(y_pred_list, axis=0)
        y_test_all = np.concatenate(y_test_list, axis=0)
        # Chỉ lấy giá trị cuối cùng để so sánh RMSE/MAE/R2 (hoặc có thể tính trung bình lỗi cho toàn chuỗi)
        y_pred_last = y_pred_all[:, -1].reshape(-1, 1) # Dự báo ngày cuối cùng
        y_test_last = y_test_all[:, -1].reshape(-1, 1) # Giá trị thực tế ngày cuối cùng
        y_pred_inv_last = scaler.inverse_transform(y_pred_last)
        y_test_inv_last = scaler.inverse_transform(y_test_last)
        try:
            mse = mean_squared_error(y_test_inv_last, y_pred_inv_last)
            rmse_val = np.sqrt(mse)
            mae_val = mean_absolute_error(y_test_inv_last, y_pred_inv_last)
            r2 = r2_score(y_test_inv_last, y_pred_inv_last)
            print("\n--- ĐÁNH GIÁ MÔ HÌNH DỰ BÁO N-BEATS (Ngày cuối) ---")
            print(f"RMSE (Ngày cuối): {rmse_val:.2f}")
            print(f"MAE (Ngày cuối): {mae_val:.2f}")
            print(f"R2 (Ngày cuối): {r2:.2f}")
            print("--- HẾT ĐÁNH GIÁ ---\n")
        except Exception as e:
            print(f"Lỗi khi tính toán đánh giá N-BEATS: {str(e)}")
            mse, rmse_val, mae_val, r2 = 0, 0, 0, 0
        # Trả về toàn bộ dự báo để vẽ biểu đồ
        # Chuyển đổi ngược toàn bộ chuỗi dự báo
        y_pred_full_inv = scaler.inverse_transform(y_pred_all.reshape(-1, GLOBAL_PREDICTION_DAYS)).reshape(y_pred_all.shape)
        y_test_full_inv = scaler.inverse_transform(y_test_all.reshape(-1, GLOBAL_PREDICTION_DAYS)).reshape(y_test_all.shape)

        return model, scaler, None, y_test_full_inv, y_pred_full_inv
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi huấn luyện mô hình N-BEATS: {str(e)}")
        traceback.print_exc()
        return None, None, None, None, None
# --- HẾT THÊM: Hàm huấn luyện N-BEATS ---

# --- THÊM: Hàm dự báo N-BEATS ---
def predict_next_days_nbeats(model, scaler, df, target="Close", time_steps=60, n_days=GLOBAL_PREDICTION_DAYS):
    """Dự báo giá trong n ngày tiếp theo (cho N-BEATS)"""
    try:
        if model is None or scaler is None or df is None:
            print("Dữ liệu đầu vào không hợp lệ cho N-BEATS predict")
            return np.array([]), np.array([])
        if target not in df.columns:
            print(f"Cột {target} không tồn tại")
            return np.array([]), np.array([])
        if len(df) < time_steps:
            print("Không đủ dữ liệu để dự báo")
            return np.array([]), np.array([])
        last_data = df[target].values[-time_steps:]
        last_data = last_data[np.isfinite(last_data)]
        if len(last_data) < time_steps:
            print("Dữ liệu không đủ sau khi loại bỏ NaN")
            return np.array([]), np.array([])
        try:
            last_data_scaled = scaler.transform(last_data.reshape(-1, 1)).flatten() # Đảm bảo là 1D array
        except Exception as e:
            print(f"Lỗi khi chuẩn hóa dữ liệu dự báo N-BEATS: {str(e)}")
            return np.array([]), np.array([])
        model.eval()
        device = next(model.parameters()).device # Lấy thiết bị từ model
        with torch.no_grad():
            x_input = torch.tensor(last_data_scaled.reshape(1, time_steps), dtype=torch.float32).to(device)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                forecast_scaled = model(x_input).cpu().numpy().flatten() # Dự báo cho n_days
        try:
            # N-BEATS trả về trực tiếp chuỗi dự báo, không cần nối thêm
            forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)).flatten()
        except Exception as e:
            print(f"Lỗi khi chuyển đổi giá gốc N-BEATS: {str(e)}")
            return np.array([]), np.array([])
        last_date = df.index[-1]
        forecast_dates = [last_date + timedelta(days=i + 1) for i in range(n_days)]
        return np.array(forecast_dates), forecast
    except Exception as e:
        print(f"Lỗi nghiêm trọng khi dự báo N-BEATS: {str(e)}")
        traceback.print_exc()
        return np.array([]), np.array([])
# --- HẾT THÊM: Hàm dự báo N-BEATS ---

# --- CẬP NHẬT HÀM ĐÁNH GIÁ DỮ LIỆU (CHO N-BEATS) ---
def evaluate_data_for_ai_nbeats(df_features, symbol):
    """Đánh giá dữ liệu để đề xuất mô hình AI phù hợp (N-BEATS)."""
    if df_features is None or len(df_features) == 0:
        print(f"❌ Không có dữ liệu để đánh giá cho mã {symbol}.")
        return "Không xác định", "Không có dữ liệu đầu vào."
    num_points = len(df_features)
    num_features = len(df_features.columns)
    print(f"--- ĐÁNH GIÁ DỮ LIỆU CHO MÃ {symbol} (N-BEATS) ---")
    print(f"Số điểm dữ liệu: {num_points}")
    print(f"Số lượng đặc trưng: {num_features}")
    # N-BEATS thường hoạt động tốt với chuỗi thời gian dài
    if num_points > 3000:
        recommendation = "N-BEATS"
        reason = f"Dữ liệu rất phong phú ({num_points} điểm), N-BEATS có thể tận dụng tốt."
    elif num_points > 2000:
        recommendation = "N-BEATS"
        reason = f"Dữ liệu phong phú ({num_points} điểm), N-BEATS là lựa chọn phù hợp."
    else:
        recommendation = "LSTM PYTORCH TỐI ƯU" # Mặc định nếu dữ liệu không đủ cho N-BEATS
        reason = f"Dữ liệu ({num_points} điểm) có thể phù hợp hơn với LSTM PYTORCH TỐI ƯU."
    print(f"💡 Đề xuất mô hình AI: {recommendation}")
    print(f"❓ Lý do: {reason}")
    print("--- HẾT ĐÁNH GIÁ ---")
    return recommendation, reason
# --- HẾT CẬP NHẬT HÀM ĐÁNH GIÁ DỮ LIỆU ---

# ======================
# PHẦN 4: PHÂN TÍCH KỸ THUẬT CẢI TIẾN
# ======================
def plot_stock_analysis(symbol, df, show_volume=True):
    """Phân tích kỹ thuật và vẽ biểu đồ cho mã chứng khoán"""
    try:
        if df is None or len(df) == 0:
            print("Dữ liệu phân tích rỗng")
            return {"signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0, "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0, "recommendation": "KHÔNG XÁC ĐỊNH"}
        df = df.sort_index()
        # --- BƯỚC 1: Tính các chỉ báo kỹ thuật ---
        df["SMA_10"] = ta.trend.sma_indicator(df["Close"], window=10)
        df["SMA_20"] = ta.trend.sma_indicator(df["Close"], window=20)
        df["SMA_50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["SMA_200"] = ta.trend.sma_indicator(df["Close"], window=200)
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)
        macd = ta.trend.MACD(df["Close"])
        df["MACD"] = macd.macd()
        df["MACD_Signal"] = macd.macd_signal()
        df["MACD_Hist"] = macd.macd_diff()
        bollinger = ta.volatility.BollingerBands(df["Close"])
        df["BB_Upper"] = bollinger.bollinger_hband()
        df["BB_Lower"] = bollinger.bollinger_lband()
        df["Volume_SMA_20"] = ta.trend.sma_indicator(df["Volume"], window=20)
        df["Volume_SMA_50"] = ta.trend.sma_indicator(df["Volume"], window=50)
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
        # --- BƯỚC 2: Tính RS (Relative Strength so với VNINDEX) theo công thức Amibroker ---
        try:
            quoteVNI = Quote(symbol="VNINDEX")
            vnindex_df = quoteVNI.history(start=start_date, end=end_date, interval="1D")
            if len(vnindex_df) == 0: raise ValueError("Không lấy được dữ liệu VNINDEX")
            vnindex_df.rename(columns={"time": "Date", "open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            vnindex_df["Date"] = pd.to_datetime(vnindex_df["Date"])
            vnindex_df.set_index("Date", inplace=True)
            vnindex_df.sort_index(inplace=True)
            df_merged = df[["Close"]].join(vnindex_df[["Close"]].rename(columns={"Close": "VNINDEX_Close"}), how="left")
            if df_merged["VNINDEX_Close"].isna().all():
                df["RS"] = 1.0
                df["RS_Point"] = 0.0
                print("Cảnh báo: Không có dữ liệu VNINDEX, bỏ qua RS")
            else:
                df_merged["VNINDEX_Close"] = df_merged["VNINDEX_Close"].ffill()
                price_return = df_merged["Close"] / df_merged["Close"].shift(1)
                index_return = df_merged["VNINDEX_Close"] / df_merged["VNINDEX_Close"].shift(1)
                df["RS"] = price_return / index_return
                roc_63 = ta.momentum.roc(df["Close"], window=63)
                roc_126 = ta.momentum.roc(df["Close"], window=126)
                roc_189 = ta.momentum.roc(df["Close"], window=189)
                roc_252 = ta.momentum.roc(df["Close"], window=252)
                df["RS_Point"] = (roc_63.fillna(0) * 0.4 + roc_126.fillna(0) * 0.2 + roc_189.fillna(0) * 0.2 + roc_252.fillna(0) * 0.2) * 100
                df["RS_Point_SMA_10"] = ta.trend.sma_indicator(df["RS_Point"], window=10)
                df["RS_Point_SMA_20"] = ta.trend.sma_indicator(df["RS_Point"], window=20)
                df["RS_Point_SMA_50"] = ta.trend.sma_indicator(df["RS_Point"], window=50)
                df["RS_Point_SMA_200"] = ta.trend.sma_indicator(df["RS_Point"], window=200)
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
            return {"signal": "LỖI", "score": 50, "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0, "rsi_value": 50, "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0, "rs": 1.0, "rs_point": 0, "recommendation": "KHÔNG XÁC ĐỊNH"}
      # --- BƯỚC 4: Vẽ biểu đồ (CẬP NHẬT) ---
        # --- BƯỚC 4: Vẽ biểu đồ (CẬP NHẬT) ---
        try:
            plot_configs = ["price_sma", "ichimoku", "rsi", "macd", "rs", "rs_point", "volume"]
            num_subplots = len(plot_configs)
            height_per_subplot = 3
            width = 18
            height = num_subplots * height_per_subplot
            plt.figure(figsize=(width, height), constrained_layout=True)
            # Điều chỉnh GridSpec - TĂNG KÍCH THƯỚC CHO RSI & MACD
            # height_ratios: [Price, Ichimoku, RSI, MACD, RS, RS_Point, Volume, (placeholder)]
            grid = plt.GridSpec(
                8, 1, hspace=0.3, height_ratios=[3, 3, 2, 2, 2, 2, 2, 2]
            )
            # === Biểu đồ 1: Giá và các đường trung bình ===
            ax1 = plt.subplot(grid[0])
            plt.plot(df.index, df["Close"], label=f"Giá đóng cửa {df['Close'].iloc[-1]:,.2f}", color="#1f77b4", linewidth=1.5)
            plt.plot(df.index, df["SMA_10"], label=f"SMA 10 {df['SMA_10'].iloc[-1]:,.2f}", color="blue", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["SMA_20"], label=f"SMA 20 {df['SMA_20'].iloc[-1]:,.2f}", color="orange", alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df["SMA_50"], label=f"SMA 50 {df['SMA_50'].iloc[-1]:,.2f}", color="green", alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df["SMA_200"], label=f"SMA 200 {df['SMA_200'].iloc[-1]:,.2f}", color="purple", alpha=0.8, linewidth=1.5)
            plt.plot(df.index, df["BB_Upper"], label=f"BB Upper {df['BB_Upper'].iloc[-1]:,.2f}", color="red", alpha=0.5, linestyle="--")
            plt.plot(df.index, df["BB_Lower"], label=f"BB Lower {df['BB_Lower'].iloc[-1]:,.2f}", color="green", alpha=0.5, linestyle="--")
            plt.fill_between(df.index, df["BB_Lower"], df["BB_Upper"], color="gray", alpha=0.1)
            cross_10_20_above = (df["SMA_10"] > df["SMA_20"]) & (df["SMA_10"].shift(1) <= df["SMA_20"].shift(1))
            cross_10_20_below = (df["SMA_10"] < df["SMA_20"]) & (df["SMA_10"].shift(1) >= df["SMA_20"].shift(1))
            if cross_10_20_above.any():
                plt.scatter(df.index[cross_10_20_above], df.loc[cross_10_20_above, "SMA_10"], marker="^", color="lime", s=60, label="SMA10 > SMA20", zorder=5)
            if cross_10_20_below.any():
                plt.scatter(df.index[cross_10_20_below], df.loc[cross_10_20_below, "SMA_10"], marker="v", color="magenta", s=60, label="SMA10 < SMA20", zorder=5)
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
            plt.plot(df.index, df["ichimoku_tenkan_sen"], label=f"Tenkan-sen {df['ichimoku_tenkan_sen'].iloc[-1]:,.2f}", color="red", linewidth=1)
            plt.plot(df.index, df["ichimoku_kijun_sen"], label=f"Kijun-sen {df['ichimoku_kijun_sen'].iloc[-1]:,.2f}", color="blue", linewidth=1)
            plt.plot(df.index, df["ichimoku_senkou_span_a"], label=f"Senkou Span A {df['ichimoku_senkou_span_a'].iloc[-1]:,.2f}", color="green", linewidth=1, alpha=0.7)
            plt.plot(df.index, df["ichimoku_senkou_span_b"], label=f"Senkou Span B {df['ichimoku_senkou_span_b'].iloc[-1]:,.2f}", color="purple", linewidth=1, alpha=0.7)
            plt.plot(df.index, df["ichimoku_chikou_span"], label=f"Chikou Span {df['ichimoku_chikou_span'].iloc[-1]:,.2f}", color="orange", linewidth=1)
            valid_cloud = (df["ichimoku_senkou_span_a"].notna() & df["ichimoku_senkou_span_b"].notna())
            if valid_cloud.any():
                plt.fill_between(df.index[valid_cloud], df["ichimoku_senkou_span_a"][valid_cloud], df["ichimoku_senkou_span_b"][valid_cloud], where=(df["ichimoku_senkou_span_a"][valid_cloud] >= df["ichimoku_senkou_span_b"][valid_cloud]), color="green", alpha=0.2, interpolate=True, label="Bullish Cloud")
                plt.fill_between(df.index[valid_cloud], df["ichimoku_senkou_span_a"][valid_cloud], df["ichimoku_senkou_span_b"][valid_cloud], where=(df["ichimoku_senkou_span_a"][valid_cloud] < df["ichimoku_senkou_span_b"][valid_cloud]), color="red", alpha=0.2, interpolate=True, label="Bearish Cloud")
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
            plt.plot(df.index, df["MACD_Signal"], label=f"Signal Line {df['MACD_Signal'].iloc[-1]:.2f}", color="red")
            # Chỉ vẽ histogram bằng bar, không cần plot thêm line cho hist
            plt.bar(df.index, df["MACD_Hist"], color=np.where(df["MACD_Hist"] > 0, "green", "red"), alpha=0.5, label=f"Hist {df['MACD_Hist'].iloc[-1]:.2f}")
            plt.title("MACD", fontsize=12)
            plt.ylabel("MACD", fontsize=10)
            plt.legend(fontsize=7, loc="upper left")
            plt.grid(True, alpha=0.3)
            # === Biểu đồ 5: RS (Relative Strength vs VNINDEX) ===
            ax5 = plt.subplot(grid[4], sharex=ax1)
            plt.plot(df.index, df["RS"], label=f"RS (Price / VNINDEX) {df['RS'].iloc[-1]:.2f}", color="brown", linewidth=1.5)
            plt.plot(df.index, df["RS_SMA_10"], label=f"RS SMA 10 {df['RS_SMA_10'].iloc[-1]:.2f}", color="blue", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_SMA_20"], label=f"RS SMA 20 {df['RS_SMA_20'].iloc[-1]:.2f}", color="orange", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_SMA_50"], label=f"RS SMA 50 {df['RS_SMA_50'].iloc[-1]:.2f}", color="green", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_SMA_200"], label=f"RS SMA 200 {df['RS_SMA_200'].iloc[-1]:.2f}", color="purple", alpha=0.7, linewidth=1)
            plt.title("RS vs VNINDEX", fontsize=12)
            plt.ylabel("RS", fontsize=10)
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=7, loc="upper left")
            # === Biểu đồ 6: RS_Point ===
            ax6 = plt.subplot(grid[5], sharex=ax1)
            plt.plot(df.index, df["RS_Point"], label=f"RS_Point {df['RS_Point'].iloc[-1]:.2f}", color="darkblue", linewidth=1.5)
            plt.plot(df.index, df["RS_Point_SMA_10"], label=f"RS_Point SMA 10 {df['RS_Point_SMA_10'].iloc[-1]:.2f}", color="blue", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_20"], label=f"RS_Point SMA 20 {df['RS_Point_SMA_20'].iloc[-1]:.2f}", color="orange", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_50"], label=f"RS_Point SMA 50 {df['RS_Point_SMA_50'].iloc[-1]:.2f}", color="green", alpha=0.7, linewidth=1)
            plt.plot(df.index, df["RS_Point_SMA_200"], label=f"RS_Point SMA 200 {df['RS_Point_SMA_200'].iloc[-1]:.2f}", color="purple", alpha=0.7, linewidth=1)
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
                if ("Volume_SMA_20" in df.columns and not df["Volume_SMA_20"].isna().all()):
                    plt.plot(df.index, df["Volume_SMA_20"], label=f"Vol SMA 20 {df['Volume_SMA_20'].iloc[-1]:,.0f}", color="orange", alpha=0.8, linewidth=1.5)
                    volume_sma_plotted = True
                if ("Volume_SMA_50" in df.columns and not df["Volume_SMA_50"].isna().all()):
                    plt.plot(df.index, df["Volume_SMA_50"], label=f"Vol SMA 50 {df['Volume_SMA_50'].iloc[-1]:,.0f}", color="purple", alpha=0.8, linewidth=1.5)
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
            # Điều chỉnh layout để tránh chồng chữ
            plt.tight_layout(pad=3.0, h_pad=1.0)
            plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.4)
            # Lưu biểu đồ
            plt.savefig(f"vnstocks_data/{symbol}_technical_analysis.png", dpi=300, bbox_inches="tight")
            plt.close()
            print(f"✅ Đã lưu biểu đồ phân tích kỹ thuật vào vnstocks_data/{symbol}_technical_analysis.png")
        except Exception as e:
            print(f"❌ Lỗi khi vẽ biểu đồ: {str(e)}")
            import traceback
            traceback.print_exc() # In traceback đầy đủ để dễ gỡ lỗi
        # --- BƯỚC 5: Tạo tín hiệu giao dịch (CẢI TIẾN LOGIC) ---
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
            # --- HỆ THỐNG TÍNH ĐIỂM TOÀN DIỆN CẢI TIẾN ---
            score = 50 # Điểm cơ bản
            # 1. RSI - 15 điểm
            if rsi_value < 30: score += 15
            elif rsi_value > 70: score -= 15
            else: score += (50 - abs(rsi_value - 50)) * 0.3
            # 2. Đường trung bình - 25 điểm (Tăng trọng số)
            if ma10_value > ma20_value > ma50_value > ma200_value: score += 25 # Xu hướng tăng mạnh
            elif ma10_value > ma20_value > ma50_value: score += 15 # Xu hướng tăng trung bình
            elif ma10_value > ma20_value: score += 8 # Xu hướng tăng yếu
            elif ma10_value < ma20_value < ma50_value < ma200_value: score -= 25 # Xu hướng giảm mạnh
            elif ma10_value < ma20_value < ma50_value: score -= 15 # Xu hướng giảm trung bình
            elif ma10_value < ma20_value: score -= 8 # Xu hướng giảm yếu
            # 3. Giá so với các đường trung bình - 10 điểm
            if current_price > ma10_value: score += 3
            if current_price > ma20_value: score += 3
            if current_price > ma50_value: score += 2
            if current_price > ma200_value: score += 2
            # 4. MACD - 15 điểm (Tăng trọng số)
            macd_value = last_row["MACD"] if not pd.isna(last_row["MACD"]) else 0
            macd_signal = (last_row["MACD_Signal"] if not pd.isna(last_row["MACD_Signal"]) else 0)
            macd_hist = (last_row["MACD_Hist"] if not pd.isna(last_row["MACD_Hist"]) else 0)
            # Sửa lỗi: Chuyển đổi sang Series để có thể dùng .shift()
            macd_hist_series = df["MACD_Hist"]
            if len(macd_hist_series) > 1:
                macd_hist_prev = macd_hist_series.iloc[-2] if not pd.isna(macd_hist_series.iloc[-2]) else 0
            else:
                macd_hist_prev = 0

            if macd_value > macd_signal and macd_hist > 0 and macd_hist > macd_hist_prev: score += 15 # Tín hiệu mua mạnh
            elif macd_value > macd_signal and macd_hist > 0: score += 10 # Tín hiệu mua
            elif macd_value < macd_signal and macd_hist < 0 and macd_hist < macd_hist_prev: score -= 15 # Tín hiệu bán mạnh
            elif macd_value < macd_signal and macd_hist < 0: score -= 10 # Tín hiệu bán
            else: score += np.clip(macd_hist * 40, -5, 5) # Dựa trên histogram
            # 5. Bollinger Bands - 10 điểm
            bb_upper = (last_row["BB_Upper"] if not pd.isna(last_row["BB_Upper"]) else current_price)
            bb_lower = (last_row["BB_Lower"] if not pd.isna(last_row["BB_Lower"]) else current_price)
            if current_price > bb_upper: score -= 5 # Quá mua
            elif current_price < bb_lower: score += 5 # Quá bán
            else:
                bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
                score += (bb_position - 0.5) * 10
            # 6. RS (Relative Strength) - 10 điểm
            rs_score = 0
            if rs_value > last_row.get("RS_SMA_10", rs_value): rs_score += 2
            if rs_value > last_row.get("RS_SMA_20", rs_value): rs_score += 3
            if rs_value > last_row.get("RS_SMA_50", rs_value): rs_score += 5
            score += rs_score
            # 7. RS_Point - 10 điểm
            rs_point_score = 0
            if rs_point_value > last_row.get("RS_Point_SMA_10", rs_point_value): rs_point_score += 2
            if rs_point_value > last_row.get("RS_Point_SMA_20", rs_point_value): rs_point_score += 3
            if rs_point_value > last_row.get("RS_Point_SMA_50", rs_point_value): rs_point_score += 5
            score += rs_point_score
            # 8. Ichimoku Cloud - 15 điểm (Tăng trọng số)
            ichimoku_score = 0
            try:
                if not (pd.isna(tenkan_sen) or pd.isna(kijun_sen) or pd.isna(senkou_span_a) or pd.isna(senkou_span_b)):
                    cloud_top = max(senkou_span_a, senkou_span_b)
                    cloud_bottom = min(senkou_span_a, senkou_span_b)
                    if current_price > cloud_top and tenkan_sen > kijun_sen: ichimoku_score += 15 # Mua mạnh
                    elif current_price > cloud_top: ichimoku_score += 10 # Mua
                    elif current_price < cloud_bottom and tenkan_sen < kijun_sen: ichimoku_score -= 15 # Bán mạnh
                    elif current_price < cloud_bottom: ichimoku_score -= 10 # Bán
                    if tenkan_sen > kijun_sen: ichimoku_score += 5
                    elif tenkan_sen < kijun_sen: ichimoku_score -= 5
                    if kijun_sen > cloud_top: ichimoku_score += 5
                    elif kijun_sen < cloud_bottom: ichimoku_score -= 5
            except Exception as e: print(f"Cảnh báo: Lỗi khi tính điểm Ichimoku: {e}")
            score += ichimoku_score
            score = np.clip(score, 0, 100)
            # --- XÁC ĐỊNH TÍN HIỆU ---
            signal = "TRUNG LẬP"
            recommendation = "GIỮ"
            if score >= 80: signal = "MUA MẠNH"; recommendation = "MUA MẠNH"
            elif score >= 65: signal = "MUA"; recommendation = "MUA"
            elif score <= 20: signal = "BÁN MẠNH"; recommendation = "BÁN MẠNH"
            elif score <= 35: signal = "BÁN"; recommendation = "BÁN"
            analysis_date = df.index[-1].strftime("%d/%m/%Y")
            print(f"📊 TÍN HIỆU GIAO DỊCH CUỐI ({analysis_date}):")
            print(f"  - Giá & Đường trung bình: Giá={current_price:,.2f} | SMA10={ma10_value:,.2f} | SMA20={ma20_value:,.2f} | SMA50={ma50_value:,.2f} | SMA200={ma200_value:,.2f}")
            try:
                print(f"  - Ichimoku:")
                print(f"    * Tenkan-sen: {tenkan_sen:.2f} | Kijun-sen: {kijun_sen:.2f}")
                print(f"    * Cloud (A/B): {senkou_span_a:.2f} / {senkou_span_b:.2f}")
                print(f"    * Chikou Span: {chikou_span:.2f}")
                print(f"    * Điểm Ichimoku: ~{ichimoku_score:.1f}")
            except: print(f"  - Ichimoku: Không có dữ liệu")
            print(f"  - Đề xuất: {recommendation} (Điểm: {score:.1f})")
            # --- BƯỚC 6: Trả về kết quả phân tích với nhiều thông tin hơn ---
            def safe_float(val):
                """Chuyển đổi giá trị sang float, xử lý NaN/None."""
                try:
                    if pd.isna(val): return None
                    return float(val)
                except (TypeError, ValueError): return None
            return {
                "signal": signal, "score": float(score), "current_price": float(current_price), "rsi_value": float(rsi_value),
                "ma10": float(ma10_value), "ma20": float(ma20_value), "ma50": float(ma50_value), "ma200": float(ma200_value),
                "rs": float(rs_value), "rs_point": float(rs_point_value), "recommendation": recommendation,
                "open": safe_float(last_row.get("Open")), "high": safe_float(last_row.get("High")), "low": safe_float(last_row.get("Low")), "volume": safe_float(last_row.get("Volume")),
                "macd": safe_float(macd_value), "macd_signal": safe_float(macd_signal), "macd_hist": safe_float(macd_hist),
                "bb_upper": safe_float(bb_upper), "bb_lower": safe_float(bb_lower),
                "volume_sma_20": safe_float(last_row.get("Volume_SMA_20")), "volume_sma_50": safe_float(last_row.get("Volume_SMA_50")),
                "ichimoku_tenkan_sen": safe_float(tenkan_sen), "ichimoku_kijun_sen": safe_float(kijun_sen),
                "ichimoku_senkou_span_a": safe_float(senkou_span_a), "ichimoku_senkou_span_b": safe_float(senkou_span_b),
                "ichimoku_chikou_span": safe_float(chikou_span),
                "rs_sma_10": safe_float(last_row.get("RS_SMA_10")), "rs_sma_20": safe_float(last_row.get("RS_SMA_20")),
                "rs_sma_50": safe_float(last_row.get("RS_SMA_50")), "rs_sma_200": safe_float(last_row.get("RS_SMA_200")),
                "rs_point_sma_10": safe_float(last_row.get("RS_Point_SMA_10")), "rs_point_sma_20": safe_float(last_row.get("RS_Point_SMA_20")),
                "rs_point_sma_50": safe_float(last_row.get("RS_Point_SMA_50")), "rs_point_sma_200": safe_float(last_row.get("RS_Point_SMA_200")),
            }
        except Exception as e:
            print(f"❌ Lỗi khi tạo tín hiệu: {str(e)}")
            traceback.print_exc() # In traceback để dễ debug
            return {"signal": "LỖI", "score": 50, "current_price": df["Close"].iloc[-1] if len(df) > 0 else 0, "rsi_value": 50, "ma10": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma20": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma50": df["Close"].iloc[-1] if len(df) > 0 else 0, "ma200": df["Close"].iloc[-1] if len(df) > 0 else 0, "rs": 1.0, "rs_point": 0, "recommendation": "KHÔNG XÁC ĐỊNH", "open": None, "high": None, "low": None, "volume": None, "macd": None, "macd_signal": None, "macd_hist": None, "bb_upper": None, "bb_lower": None, "volume_sma_20": None, "volume_sma_50": None, "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None, "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, "ichimoku_chikou_span": None, "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None, "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None}
    except Exception as e:
        print(f"❌ Lỗi nghiêm trọng: {str(e)}")
        traceback.print_exc()
        return {"signal": "LỖI", "score": 50, "current_price": 0, "rsi_value": 0, "ma10": 0, "ma20": 0, "ma50": 0, "ma200": 0, "rs": 1.0, "rs_point": 0, "recommendation": "KHÔNG XÁC ĐỊNH", "open": None, "high": None, "low": None, "volume": None, "macd": None, "macd_signal": None, "macd_hist": None, "bb_upper": None, "bb_lower": None, "volume_sma_20": None, "volume_sma_50": None, "ichimoku_tenkan_sen": None, "ichimoku_kijun_sen": None, "ichimoku_senkou_span_a": None, "ichimoku_senkou_span_b": None, "ichimoku_chikou_span": None, "rs_sma_10": None, "rs_sma_20": None, "rs_sma_50": None, "rs_sma_200": None, "rs_point_sma_10": None, "rs_point_sma_20": None, "rs_point_sma_50": None, "rs_point_sma_200": None}
# ======================
# PHẦN 5: TÍCH HỢP PHÂN TÍCH BẰNG Google
# ======================
def analyze_with_gemini(symbol, trading_signal, forecast, financial_data=None):
    """Phân tích cổ phiếu bằng Google Qwen dựa trên dữ liệu kỹ thuật và BCTC"""
    try:
        def safe_format(val, fmt=".2f"):
            try:
                if val is None: return "N/A"
                if isinstance(val, float): 
                    if pd.isna(val) or np.isinf(val): return "N/A"
                return f"{{:{fmt}}}".format(float(val))
            except (ValueError, TypeError): return "N/A"
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
        # --- CẢI TIẾN LOGIC TẠO PROMPT ---
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
- RS (Amibroker): {rs_val} (SMA10: {rs_sma10_val})
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
        # --- HẾT CẢI TIẾN LOGIC TẠO PROMPT ---
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
# ======================
# PHẦN 6: CHỨC NĂNG CHÍNH - CẢI TIẾN
# ======================
# --- Hàm vẽ biểu đồ Actual vs Forecast ---
def plot_actual_vs_forecast(symbol, df, forecast_dates, forecast_values):
    """Vẽ biểu đồ so sánh giá thực tế và giá dự báo."""
    try:
        if len(forecast_dates) == 0 or len(forecast_values) == 0:
            print(f"Không có dữ liệu dự báo để vẽ cho {symbol}")
            return
        lookback_days = min(120, len(df))
        actual_data = df['Close'].tail(lookback_days)
        actual_dates = actual_data.index
        plt.figure(figsize=(12, 6))
        plt.plot(actual_dates, actual_data, label='Giá thực tế (Close)', color='blue', marker='o', markersize=3)
        all_dates = list(actual_dates) + list(forecast_dates)
        connection_x = [actual_dates[-1], forecast_dates[0]]
        connection_y = [actual_data.iloc[-1], forecast_values[0]]
        plt.plot(connection_x, connection_y, color='orange', linestyle='--', alpha=0.7)
        plt.plot(forecast_dates, forecast_values, label=f'Giá dự báo (AI - {GLOBAL_PREDICTION_DAYS} ngày)', color='red', marker='x', markersize=5)
        plt.title(f'Giá thực tế và dự báo cho {symbol}')
        plt.xlabel('Ngày')
        plt.ylabel('Giá (VND)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        filename = f"vnstocks_data/{symbol}_actual_vs_forecast.png"
        plt.savefig(filename)
        plt.close()
        print(f"✅ Đã lưu biểu đồ Actual vs Forecast vào {filename}")
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ Actual vs Forecast cho {symbol}: {e}")

# --- CẬP NHẬT: Logic lựa chọn mô hình trong analyze_stock ---
def analyze_stock(symbol):
    """Phân tích toàn diện một mã chứng khoán với tích hợp Google và lựa chọn mô hình AI phù hợp (LSTM hoặc N-BEATS)"""
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
    
    # --- CẬP NHẬT: Logic lựa chọn mô hình ---
    ai_recommendation, ai_reason = evaluate_data_for_ai(df_features, symbol) # Giữ lại để so sánh chung
    ai_recommendation_nbeats, ai_reason_nbeats = evaluate_data_for_ai_nbeats(df_features, symbol) # Thêm đánh giá cho N-BEATS

    # Chọn mô hình dựa trên tiêu chí đơn giản (có thể tinh chỉnh)
    use_nbeats = ai_recommendation_nbeats == "N-BEATS"

    model, scaler = None, None
    X_test_or_actual, y_test_or_pred, forecast_source = None, None, None
    forecast_dates, forecast_values = np.array([]), np.array([])

    if len(df_features) < 100:
        print(f"Cảnh báo: Dữ liệu cho mã {symbol} quá ít ({len(df_features)} điểm) để huấn luyện mô hình AI hiệu quả.")
    else:
        if use_nbeats:
            print(f"\n🔔 ĐỀ XUẤT MỞ RỘNG: {ai_recommendation_nbeats}")
            print(f"   Lý do: {ai_reason_nbeats}")
            print(f"\nĐang huấn luyện mô hình AI (N-BEATS) cho mã {symbol}...")
            model, scaler, X_test, y_test_full, y_pred_full = train_stock_model_nbeats(df_features) # Gọi trực tiếp
            if model is not None:
                 # Lấy giá trị cuối cùng để vẽ biểu đồ so sánh (hoặc có thể lấy toàn bộ)
                 X_test_or_actual = y_test_full[:, -1] if y_test_full.ndim > 1 else y_test_full # Giá trị thực tế ngày cuối
                 y_test_or_pred = y_pred_full[:, -1] if y_pred_full.ndim > 1 else y_pred_full # Dự báo ngày cuối
                 print(f"\nĐang dự báo giá cho {GLOBAL_PREDICTION_DAYS} ngày tới bằng N-BEATS...")
                 forecast_dates, forecast_values = predict_next_days_nbeats(model, scaler, df_features) # Gọi trực tiếp
                 if len(forecast_dates) > 0 and len(forecast_values) > 0:
                      plot_actual_vs_forecast(symbol, df_features, forecast_dates, forecast_values)
            else:
                 print("\n⚠️ Không thể huấn luyện mô hình N-BEATS.")

        else: # Mặc định sử dụng LSTM
            print(f"\n🔔 ĐỀ XUẤT MỞ RỘNG: {ai_recommendation}")
            print(f"   Lý do: {ai_reason}")
            print(f"\nĐang huấn luyện mô hình AI (LSTM PyTorch tối ưu) cho mã {symbol}...")
            model, scaler, X_test, y_test, y_pred = train_stock_model_pytorch_optimized(df_features)
            if model is not None:
                X_test_or_actual = y_test
                y_test_or_pred = y_pred
                print(f"\nĐang dự báo giá cho {GLOBAL_PREDICTION_DAYS} ngày tới bằng LSTM PyTorch tối ưu...")
                forecast_dates, forecast_values = predict_next_days_pytorch_optimized(model, scaler, df_features)
                if len(forecast_dates) > 0 and len(forecast_values) > 0:
                     plot_actual_vs_forecast(symbol, df_features, forecast_dates, forecast_values)
            else:
                print("\n⚠️ Không thể huấn luyện mô hình LSTM PyTorch tối ưu.")
    # ... (phần code sau đó trong analyze_stock không thay đổi) ...
    print(f"\nĐang phân tích kỹ thuật cho mã {symbol}...")
    trading_signal = plot_stock_analysis(symbol, df_features)
    print(f"\nĐang phân tích bằng Google ...")
    gemini_analysis = analyze_with_gemini(symbol, trading_signal, (forecast_dates, forecast_values), financial_data)
    print(f"\nKẾT QUẢ PHÂN TÍCH CHO MÃ {symbol}:")
    print(f"Giá hiện tại: {trading_signal['current_price']:,.2f} VND")
    print(f"Tín hiệu: {trading_signal['signal']}")
    print(f"Đề xuất: {trading_signal['recommendation']}")
    print(f"Điểm phân tích: {trading_signal['score']:.2f}/100")
    if len(forecast_dates) > 0 and len(forecast_values) > 0:
        print(f"\nDỰ BÁO GIÁ CHO {len(forecast_dates)} NGÀY TIẾP THEO:")
        for i, (date, price) in enumerate(zip(forecast_dates, forecast_values)):
            change = ((price - trading_signal["current_price"])/ trading_signal["current_price"]) * 100
            print(f"Ngày {i+1} ({date.date()}): {price:,.2f} VND ({change:+.2f}%)")
    else:
        print("\nKhông có dự báo giá do lỗi trong quá trình huấn luyện mô hình")
    print(f"\nPHÂN TÍCH TỔNG HỢP TỪ Google:")
    print(gemini_analysis)
    def safe_float(val):
        """Chuyển đổi giá trị sang float, xử lý NaN/None."""
        try:
            if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))): return None
            return float(val)
        except (TypeError, ValueError): return None
    report = {
        "symbol": symbol, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "current_price": safe_float(trading_signal.get("current_price")), "signal": trading_signal.get("signal"),
        "recommendation": trading_signal.get("recommendation"), "score": safe_float(trading_signal.get("score")),
        "rsi_value": safe_float(trading_signal.get("rsi_value")), "ma10": safe_float(trading_signal.get("ma10")),
        "ma20": safe_float(trading_signal.get("ma20")), "ma50": safe_float(trading_signal.get("ma50")),
        "ma200": safe_float(trading_signal.get("ma200")), "rs": safe_float(trading_signal.get("rs")),
        "rs_point": safe_float(trading_signal.get("rs_point")),
        "forecast": (
            [{"date": date.strftime("%Y-%m-%d"), "price": safe_float(price), "change_percent": safe_float(change)}
             for date, price, change in zip(forecast_dates, forecast_values,
                [((price - (trading_signal.get("current_price") or 0)) / (trading_signal.get("current_price") or 1)) * 100 for price in forecast_values])]
            if len(forecast_dates) > 0 and len(forecast_values) > 0 and trading_signal.get("current_price") is not None and trading_signal.get("current_price") != 0 else []
        ),
        "ai_recommendation": ai_recommendation, "ai_reason": ai_reason, "gemini_analysis": gemini_analysis,
    }
    report.update(trading_signal)
    with open(f"vnstocks_data/{symbol}_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)
    print(f"✅ Đã lưu báo cáo phân tích vào file 'vnstocks_data/{symbol}_report.json'")
    return report

def screen_stocks_parallel():
    """Quét và phân tích nhiều mã chứng khoán tuần tự (sync)."""
    print(f"\n{'='*50}")
    print("QUÉT VÀ PHÂN TÍCH DANH SÁCH MÃ CHỨNG KHOÁN (TUẦN TỰ - SYNC)")
    print(f"{'='*50}")
    stock_list = get_vnstocks_list()
    symbols_to_analyze = stock_list["symbol"].head(20)
    results = []
    for symbol in symbols_to_analyze: # Thay vì chạy song song, chạy tuần tự
        try:
            result = analyze_stock(symbol)
            if result and result["signal"] != "LỖI":
                results.append(result)
                print(f"✅ Phân tích mã {symbol} hoàn tất (tuần tự - sync).")
            else:
                print(f"⚠️ Phân tích mã {symbol} thất bại hoặc có lỗi (tuần tự - sync).")
        except Exception as e:
            print(f"Lỗi khi phân tích mã {symbol} (tuần tự - sync): {e}")
            import traceback
            traceback.print_exc()

    if results:
        results.sort(key=lambda x: x["score"], reverse=True)
        def get_nested_value(report_dict, key_path, default=None):
            keys = key_path.split(".")
            current_dict = report_dict
            try:
                for key in keys:
                    if isinstance(current_dict, dict) and key in current_dict: current_dict = current_dict[key]
                    else: return default
                if current_dict is None or (isinstance(current_dict, float) and (pd.isna(current_dict) or np.isinf(current_dict))): return default
                return float(current_dict)
            except (ValueError, TypeError): return default
        data_for_df = []
        for r in results:
            row_data = {
                "Mã": r["symbol"], "Giá": r["current_price"], "Điểm": r["score"], "Tín hiệu": r["signal"], "Đề xuất": r["recommendation"],
                "RSI": r["rsi_value"], "MA10": r["ma10"], "MA20": r["ma20"], "MA50": r["ma50"], "MA200": r["ma200"],
                "RS": r["rs"], "RS_Point": r["rs_point"],
                "Ichimoku_Tenkan": r.get("ichimoku_tenkan_sen"), "Ichimoku_Kijun": r.get("ichimoku_kijun_sen"),
                "Ichimoku_Senkou_A": r.get("ichimoku_senkou_span_a"), "Ichimoku_Senkou_B": r.get("ichimoku_senkou_span_b"),
                "Ichimoku_Chikou": r.get("ichimoku_chikou_span"),
            }
            data_for_df.append(row_data)
        df_results = pd.DataFrame(data_for_df)
        df_results.to_csv("vnstocks_data/stock_screening_report.csv", index=False)
        print(f"\n{'='*50}")
        print("KẾT QUẢ QUÉT MÃ")
        print(f"{'='*50}")
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
# ======================
# CHẠY CHƯƠNG TRÌNH CHÍNH
# ======================
def main():
    print("==============================================")
    print("HỆ THỐNG PHÂN TÍCH CHỨNG KHOÁN VIỆT NAM VỚI AI")
    print("TÍCH HỢP VNSTOCK VÀ GOOGLE - PHIÊN BẢN TỐI ƯU")
    print("==============================================")
    market_data = get_market_data()
    analyze_stock("DRI") # Có thể thay bằng mã khác hoặc bỏ comment dòng dưới để quét danh sách
    # screen_stocks_parallel() # Gọi trực tiếp, không dùng await
    print("\nHoàn thành phân tích. Các báo cáo đã được lưu trong thư mục 'vnstocks_data/'.")
if __name__ == "__main__":
    main() # Gọi trực tiếp, không dùng asyncio.run()
# --- KẾT THÚC: TOÀN BỘ MÃ NGUỒN ĐÃ CẬP NHẬT & TỐI ƯU TOÀN DIỆN ---
