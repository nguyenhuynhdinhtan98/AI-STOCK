# stock_price_prediction_transformer_heavy.py

# --- GLOBAL PARAMETERS ---
# 1. Data Parameters
DATA_FILE_PATH = 'data.csv'
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume'] # Dùng nhiều cột
TARGET_COL = 'Close' # Cột mục tiêu để dự đoán
SEQUENCE_LENGTH = 250 # Tăng độ dài chuỗi đầu vào
PREDICT_STEPS = 1 # Dự đoán 1 bước tới

# 2. Model Parameters (Transformer)
INPUT_SIZE = len(FEATURE_COLS) # Đầu vào là 5 đặc trưng
D_MODEL = 256 # Kích thước embedding và hidden states (tăng lên)
NHEAD = 8 # Số lượng attention heads
NUM_ENCODER_LAYERS = 5 # Số lớp encoder (tăng lên)
NUM_DECODER_LAYERS = 5 # Số lớp decoder (tăng lên)
DIM_FEEDFORWARD = 512 # Kích thước lớp feedforward (tăng lên)
DROPOUT = 0.2

# 3. Training Parameters
EPOCHS = 200 # Tăng số lượng epochs
BATCH_SIZE = 128 # Có thể thử 64, 128, 256
LEARNING_RATE = 0.0001 # Giảm learning rate cho mô hình phức tạp
WEIGHT_DECAY = 1e-4 # Điều chỉnh weight decay
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1 # 10% của train set dùng cho validation
GRADIENT_CLIP_MAX_NORM = 1.0

# 4. Scheduler & Early Stopping Parameters
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 20 # Dừng nếu val loss không cải thiện sau 20 epochs

# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# --- 1. Cấu hình thiết bị (Device) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Đọc và tiền xử lý dữ liệu ---
def load_and_preprocess_data(file_path, feature_cols, target_col):
    """
    Đọc dữ liệu từ file CSV chuẩn và tiền xử lý.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    df = pd.read_csv(file_path)
    
    if 'Date' not in df.columns:
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
        
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(f"Các cột đặc trưng sau không tồn tại trong dữ liệu: {missing_features}")
    
    if target_col not in df.columns:
        raise ValueError(f"Cột mục tiêu '{target_col}' không tồn tại trong dữ liệu.")
    
    # Chỉ giữ các cột cần thiết
    data = df[['Date'] + feature_cols].copy()
    data.set_index('Date', inplace=True)
    
    print(f"Tải dữ liệu thành công. Phạm vi ngày: {data.index.min().date()} đến {data.index.max().date()}")
    print(f"Số lượng bản ghi: {len(data)}")
    print(data.head())
    print(data.tail())
    
    return data

# --- 3. Tạo Sliding Windows ---
def create_sequences(data, seq_length, predict_steps=1):
    """Tạo chuỗi dữ liệu dạng sliding window cho dự đoán nhiều bước hoặc một bước."""
    xs, ys = [], []
    for i in range(len(data) - seq_length - predict_steps + 1):
        x = data[i:i+seq_length] # Dữ liệu đầu vào (seq_length, num_features)
        if predict_steps == 1:
            y = data[i+seq_length] # Dữ liệu đầu ra (num_features,) hoặc (1,) nếu chỉ target
        else:
            y = data[i+seq_length : i+seq_length+predict_steps] # (predict_steps, num_features)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 4. Định nghĩa Dataset ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        # Chuyển đổi sang tensor float32
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx] # Trả về (seq_len, features), (features,) hoặc (steps, features)

# --- 5. Định nghĩa Mô hình PyTorch (Transformer cải tiến) ---
class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, pred_steps=1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.pred_steps = pred_steps
        
        # Linear layer để chiếu dữ liệu đầu vào lên không gian d_model
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional Encoding (có thể thêm nếu cần, nhưng thường không bắt buộc với dữ liệu tài chính)
        # self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer model
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True # Rất quan trọng để input shape là (batch, seq, feature)
        )
        
        # Linear layer để chiếu từ d_model về không gian đầu ra (target size)
        # Giả sử chỉ dự đoán 1 giá trị (Close) -> output_size = 1
        self.output_projection = nn.Linear(d_model, 1) # Chỉ dự đoán cột Close
        
        # Tham số để tạo tgt cho decoder (có thể dùng learnable hoặc zeros)
        # Ở đây ta dùng learnable parameter
        self.tgt_token = nn.Parameter(torch.randn(1, 1, d_model)) # (1, 1, d_model)

    def forward(self, src):
        # src shape: (batch_size, seq_len, input_size)
        
        # 1. Chiếu lên không gian d_model
        src = self.input_projection(src) # (batch_size, seq_len, d_model)
        # src = self.pos_encoder(src) # Nếu dùng positional encoding
        
        # 2. Tạo tgt cho decoder
        # tgt shape cần là (batch_size, tgt_seq_len, d_model)
        # Ở đây, tgt_seq_len = pred_steps
        batch_size = src.size(0)
        tgt = self.tgt_token.repeat(batch_size, self.pred_steps, 1) # (batch_size, pred_steps, d_model)
        
        # 3. Truyền qua Transformer
        # memory là đầu ra của encoder
        memory = self.transformer.encoder(src) # (batch_size, seq_len, d_model)
        
        # tgt_mask để đảm bảo decoder không nhìn thấy tương lai (cho dự đoán nhiều bước)
        tgt_mask = self.transformer.generate_square_subsequent_mask(self.pred_steps).to(src.device)
        
        # Truyền tgt và memory qua decoder
        transformer_out = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask) # (batch_size, pred_steps, d_model)
        
        # 4. Chiếu đầu ra về không gian mục tiêu
        out = self.output_projection(transformer_out) # (batch_size, pred_steps, 1)
        
        # Nếu chỉ dự đoán 1 bước, trả về shape (batch_size, 1)
        if self.pred_steps == 1:
            out = out.squeeze(1) # (batch_size, 1)
            
        return out

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load dữ liệu
    print("Loading and preprocessing data...")
    raw_data = load_and_preprocess_data(DATA_FILE_PATH, FEATURE_COLS, TARGET_COL)
    
    # Chuẩn hóa dữ liệu riêng biệt cho features và target
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(raw_data[FEATURE_COLS])
    scaled_target = target_scaler.fit_transform(raw_data[[TARGET_COL]]) # Giữ dạng 2D
    
    # Tạo chuỗi
    X, y = create_sequences(scaled_features, SEQUENCE_LENGTH, PREDICT_STEPS)
    # y hiện tại có shape (num_samples, num_features) nếu predict_steps=1
    # hoặc (num_samples, predict_steps, num_features) nếu predict_steps > 1
    # Ta cần y chỉ chứa cột target (Close)
    if PREDICT_STEPS == 1:
         # y có shape (num_samples, num_features), ta lấy cột tương ứng với TARGET_COL
         # TARGET_COL là 'Close', nằm ở index 3 trong FEATURE_COLS
         target_col_index = FEATURE_COLS.index(TARGET_COL)
         y = y[:, target_col_index:target_col_index+1] # Shape: (num_samples, 1)
    else:
        # y có shape (num_samples, predict_steps, num_features)
        target_col_index = FEATURE_COLS.index(TARGET_COL)
        y = y[:, :, target_col_index:target_col_index+1] # Shape: (num_samples, predict_steps, 1)
        y = y.squeeze(-1) # Shape: (num_samples, predict_steps) nếu predict_steps > 1
         
    print(f"X shape: {X.shape}, y shape: {y.shape}")

    if len(X) == 0:
        raise ValueError(f"Không đủ dữ liệu để tạo chuỗi với seq_length={SEQUENCE_LENGTH}.")

    # Chia train/val/test
    total_size = len(X)
    train_size = int(total_size * TRAIN_SPLIT_RATIO)
    val_size = int(total_size * VAL_SPLIT_RATIO)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
    X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]

    train_dates = raw_data.index[SEQUENCE_LENGTH : SEQUENCE_LENGTH + train_size]
    val_dates = raw_data.index[SEQUENCE_LENGTH + train_size : SEQUENCE_LENGTH + train_size + val_size]
    test_dates = raw_data.index[SEQUENCE_LENGTH + train_size + val_size : SEQUENCE_LENGTH + len(X)]

    print(f"\n--- Phân chia dữ liệu ---")
    print(f"Train set shape: X_train {X_train.shape}, y_train {y_train.shape}")
    print(f"  - Phạm vi ngày train: {train_dates[0].date()} đến {train_dates[-1].date()}")
    print(f"Validation set shape: X_val {X_val.shape}, y_val {y_val.shape}")
    print(f"  - Phạm vi ngày val: {val_dates[0].date()} đến {val_dates[-1].date()}")
    print(f"Test set shape: X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"  - Phạm vi ngày test: {test_dates[0].date()} đến {test_dates[-1].date()}")
    print(f"-------------------------\n")

    # Tạo dataset và dataloader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo mô hình
    model = TimeSeriesTransformer(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        pred_steps=PREDICT_STEPS
    )
    model.to(device)
    print(model)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    # Loss function và Optimizer
    loss_function = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=SCHEDULER_MODE, 
        factor=SCHEDULER_FACTOR, patience=SCHEDULER_PATIENCE
    )

    # Lưu loss để vẽ đồ thị và early stopping
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        num_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_function(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRADIENT_CLIP_MAX_NORM)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
        train_losses.append(avg_train_loss)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_pred = model(X_batch)
                loss = loss_function(y_pred, y_batch)
                val_loss += loss.item()
                num_val_batches += 1
        
        avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_losses.append(avg_val_loss)
        
        # --- Scheduler and Early Stopping ---
        scheduler.step(avg_val_loss)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Lưu model tốt nhất
            torch.save(model.state_dict(), 'best_model_transformer.pth')
        else:
            epochs_no_improve += 1
            
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Epoch [{epoch+1:3d}/{EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.6f}')
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
            print(f'Early stopping at epoch {epoch+1} vì Val Loss không cải thiện sau {EARLY_STOPPING_PATIENCE} epochs.')
            break
            
    print("Training completed.")

    # --- Load best model for evaluation ---
    model.load_state_dict(torch.load('best_model_transformer.pth'))
    model.eval()

    # --- Đánh giá và Vẽ biểu đồ ---
    predictions = []
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())

    if predictions:
        predictions = np.concatenate(predictions, axis=0)
    else:
        raise ValueError("Không có dự đoán nào được tạo ra từ tập test.")

    # Đảo ngược chuẩn hóa cho target
    # predictions có shape (num_test_samples, 1) hoặc (num_test_samples, pred_steps)
    if PREDICT_STEPS == 1:
        predicted_prices = target_scaler.inverse_transform(predictions) # (num_test_samples, 1)
        true_prices = target_scaler.inverse_transform(y_test) # (num_test_samples, 1)
    else:
        # Nếu dự đoán nhiều bước, ta chỉ lấy bước đầu tiên để so sánh đơn giản
        predicted_prices = target_scaler.inverse_transform(predictions[:, 0:1]) # (num_test_samples, 1)
        true_prices = target_scaler.inverse_transform(y_test[:, 0:1]) # (num_test_samples, 1)

    # Tính RMSE
    rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
    print(f'\nTest RMSE (using best model): {rmse:.2f}')

    # Vẽ biểu đồ so sánh giá thực tế và dự đoán (Tập Test)
    plt.figure(figsize=(14, 6))
    indices = np.arange(len(true_prices))
    plt.plot(indices, true_prices, label='Giá thực tế (Actual)', color='blue')
    plt.plot(indices, predicted_prices, label='Giá dự đoán (Predicted)', color='red', alpha=0.7)
    plt.title('So sánh Giá cổ phiếu Thực tế và Dự đoán (Tập Test - Transformer)')
    plt.xlabel('Ngày (Index)')
    plt.ylabel('Giá đóng cửa')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Vẽ loss
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
