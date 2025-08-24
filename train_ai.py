# --- GLOBAL PARAMETERS ---
# 1. Data Parameters
DATA_FILE_PATH = 'data.csv'
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'
SEQUENCE_LENGTH = 200
PREDICT_STEPS = 5  # Dự đoán 5 bước tới

# 2. Model Parameters (Enhanced Transformer)
INPUT_SIZE = len(FEATURE_COLS)
D_MODEL = 512
NHEAD = 8
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 4
DIM_FEEDFORWARD = 1024
DROPOUT = 0.1

# 3. Training Parameters
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-4
TRAIN_SPLIT_RATIO = 0.8
VAL_SPLIT_RATIO = 0.1
GRADIENT_CLIP_MAX_NORM = 1.0

# 4. Scheduler & Early Stopping Parameters
SCHEDULER_MODE = 'min'
SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 10
EARLY_STOPPING_PATIENCE = 20

# --- IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import math

# --- 1. Cấu hình thiết bị (Device) ---
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- 2. Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- 3. Enhanced Transformer Model (Fixed Version) ---
class EnhancedTimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_encoder_layers, num_decoder_layers, 
                 dim_feedforward, dropout, pred_steps=1):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.pred_steps = pred_steps
        
        # Linear projection
        self.input_projection = nn.Linear(input_size, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Standard Transformer encoder và decoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection - dự đoán 1 giá trị (Close) cho mỗi bước
        self.output_projection = nn.Linear(d_model, 1)
        
        # Learnable target tokens cho decoder
        self.tgt_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, src, return_attention=False):
        # src shape: (batch_size, seq_len, input_size)
        batch_size = src.size(0)
        
        # Project to model dimension
        src = self.input_projection(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, d_model)
        
        # Encode
        memory = self.transformer_encoder(src)
        
        # Tạo target sequence cho decoder
        # tgt shape: (batch_size, pred_steps, d_model)
        tgt = self.tgt_token.repeat(batch_size, self.pred_steps, 1)
        
        # Tạo causal mask cho decoder
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(self.pred_steps).to(src.device)
        
        # Decode
        decoder_output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        
        # Project to output - shape: (batch_size, pred_steps, 1)
        output = self.output_projection(decoder_output)
        
        # Trả về shape: (batch_size, pred_steps) nếu chỉ dự đoán 1 feature
        output = output.squeeze(-1)
        
        return output

# --- 4. Data Processing Functions ---
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
    
    return data

def create_sequences(data, seq_length, predict_steps=1):
    """Tạo chuỗi dữ liệu dạng sliding window."""
    xs, ys = [], []
    for i in range(len(data) - seq_length - predict_steps + 1):
        x = data[i:i+seq_length]
        if predict_steps == 1:
            y = data[i+seq_length, FEATURE_COLS.index(TARGET_COL):FEATURE_COLS.index(TARGET_COL)+1]
        else:
            # Lấy cột target cho predict_steps bước
            target_idx = FEATURE_COLS.index(TARGET_COL)
            y = data[i+seq_length:i+seq_length+predict_steps, target_idx]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- 5. Dataset Class ---
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- 6. Visualization Functions ---
def plot_predictions(true_prices, predicted_prices, title="Stock Price Prediction"):
    """Vẽ biểu đồ so sánh giá thực tế và dự đoán."""
    plt.figure(figsize=(14, 6))
    indices = np.arange(len(true_prices))
    
    if len(true_prices.shape) > 1 and true_prices.shape[1] > 1:
        # Nếu là multi-step, chỉ lấy bước đầu tiên để so sánh
        true_prices = true_prices[:, 0]
        predicted_prices = predicted_prices[:, 0]
    elif len(true_prices.shape) > 1:
        true_prices = true_prices.flatten()
        predicted_prices = predicted_prices.flatten()
    
    plt.plot(indices, true_prices, label='Giá thực tế (Actual)', color='blue')
    plt.plot(indices, predicted_prices, label='Giá dự đoán (Predicted)', color='red', alpha=0.7)
    plt.title(title)
    plt.xlabel('Thời gian')
    plt.ylabel('Giá đóng cửa')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_losses(train_losses, val_losses):
    """Vẽ biểu đồ loss."""
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

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Load dữ liệu
    print("Loading and preprocessing data...")
    try:
        raw_data = load_and_preprocess_data(DATA_FILE_PATH, FEATURE_COLS, TARGET_COL)
    except Exception as e:
        print(f"Lỗi khi tải dữ liệu: {e}")
        print("Tạo dữ liệu mẫu để test...")
        # Tạo dữ liệu mẫu nếu không có file
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        sample_data = {
            'Date': dates,
            'Open': np.random.randn(1000) * 100 + 1000,
            'High': np.random.randn(1000) * 100 + 1050,
            'Low': np.random.randn(1000) * 100 + 950,
            'Close': np.random.randn(1000) * 100 + 1000,
            'Volume': np.random.randint(1000, 10000, 1000)
        }
        raw_data = pd.DataFrame(sample_data)
        raw_data.set_index('Date', inplace=True)
        print("Đã tạo dữ liệu mẫu.")
    
    # Chuẩn hóa dữ liệu
    feature_scaler = MinMaxScaler(feature_range=(0, 1))
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    
    scaled_features = feature_scaler.fit_transform(raw_data[FEATURE_COLS])
    
    # Tạo chuỗi
    X, y = create_sequences(scaled_features, SEQUENCE_LENGTH, PREDICT_STEPS)
    
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

    # Tạo dataset và dataloader
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Khởi tạo mô hình
    model = EnhancedTimeSeriesTransformer(
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

    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        # Training Phase
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

        # Validation Phase
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
        
        # Scheduler and Early Stopping
        scheduler.step(avg_val_loss)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model_enhanced_transformer.pth')
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

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_model_enhanced_transformer.pth'))
    model.eval()

    # Evaluation
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in test_loader:
            X_batch = X_batch.to(device)
            y_pred = model(X_batch)
            predictions.append(y_pred.cpu().numpy())

    predictions = np.concatenate(predictions, axis=0)

    # Đảo ngược chuẩn hóa
    if PREDICT_STEPS == 1:
        predicted_prices = target_scaler.inverse_transform(predictions.reshape(-1, 1))
        true_prices = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    else:
        # Lấy bước đầu tiên để so sánh
        predicted_prices = target_scaler.inverse_transform(predictions[:, 0:1])
        true_prices = target_scaler.inverse_transform(y_test[:, 0:1])

    # Tính metrics
    rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
    mae = mean_absolute_error(true_prices, predicted_prices)
    print(f'\nTest RMSE (using best model): {rmse:.2f}')
    print(f'Test MAE (using best model): {mae:.2f}')

    # Visualization
    try:
        plot_predictions(true_prices, predicted_prices, "So sánh Giá cổ phiếu Thực tế và Dự đoán (Enhanced Transformer)")
        plot_losses(train_losses, val_losses)
    except Exception as e:
        print(f"Lỗi khi vẽ biểu đồ: {e}")

    print("Enhanced Transformer model completed!")