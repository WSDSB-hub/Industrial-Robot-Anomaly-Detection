import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ======================================================
# Temporal Anomaly Detection v3 — 修复阈值选择
# 改进：
# 1. 数据标准化
# 2. 阈值基于正常验证集误差的高分位数(99%)
# 3. 训练/验证/测试分离
# ======================================================

torch.manual_seed(42)
np.random.seed(42)

def generate_trajectory_data(n_timesteps=1000, n_joints=6, anomaly_start=700, anomaly_length=100):
    t = np.linspace(0, 10, n_timesteps)
    normal_trajectory = np.zeros((n_timesteps, n_joints))
    for j in range(n_joints):
        amplitude = 10 + j * 2
        frequency = 0.5 + j * 0.1
        phase = j * 0.5
        normal_trajectory[:, j] = amplitude * np.sin(frequency * t + phase) + 20 * np.cos(0.2 * t)
    trajectory = normal_trajectory.copy()
    if anomaly_start + anomaly_length < n_timesteps:
        trajectory[anomaly_start:anomaly_start+anomaly_length, :] += np.random.normal(0, 5, (anomaly_length, n_joints))
        trajectory[anomaly_start:anomaly_start+anomaly_length, 2] += 15 * np.sin(np.linspace(0, 4*np.pi, anomaly_length))
    return trajectory, anomaly_start, anomaly_length

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.output_layer = nn.Linear(hidden_dim, input_dim)
    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(decoder_input)
        return self.output_layer(decoder_output)

def prepare_sequences(data, seq_len=50):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)

def train_lstm_autoencoder(train_data, val_data, seq_len=50, epochs=100, batch_size=32, lr=0.001):
    input_dim = train_data.shape[1]
    model = LSTMAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    X_train = torch.FloatTensor(prepare_sequences(train_data, seq_len))
    X_val = torch.FloatTensor(prepare_sequences(val_data, seq_len))
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch+1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                val_recon = model(X_val)
                val_loss = criterion(val_recon, X_val).item()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {total_loss/len(dataloader):.6f}, Val Loss: {val_loss:.6f}")
    return model

def compute_reconstruction_error(model, data, seq_len=50):
    sequences = prepare_sequences(data, seq_len)
    X = torch.FloatTensor(sequences)
    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = torch.mean((recon - X) ** 2, dim=(1, 2)).numpy()
    return errors

def main():
    print("=" * 70)
    print("Temporal Anomaly Detection v3 (Fixed Threshold)")
    print("=" * 70)

    trajectory, anomaly_start, anomaly_length = generate_trajectory_data()

    # 标准化
    scaler = StandardScaler()
    trajectory_scaled = scaler.fit_transform(trajectory)

    # 数据划分
    normal_data = trajectory_scaled[:700, :]
    train_data = normal_data[:500, :]
    val_data = normal_data[500:650, :]
    test_data = trajectory_scaled[500:, :]

    print(f"数据形状: {trajectory_scaled.shape}")
    print(f"异常段: 时间步 {anomaly_start} 到 {anomaly_start+anomaly_length}")

    print("\n训练LSTM-Autoencoder...")
    model = train_lstm_autoencoder(train_data, val_data, epochs=100)

    # 验证集重建误差（全正常）
    val_errors = compute_reconstruction_error(model, val_data)

    # 基于正常验证集误差的99分位数设置阈值
    threshold = np.percentile(val_errors, 99)
    print(f"\n正常验证集重建误差均值: {np.mean(val_errors):.6f}")
    print(f"正常验证集重建误差99分位数(阈值): {threshold:.6f}")

    # 测试集评估
    test_errors = compute_reconstruction_error(model, test_data)
    test_labels = np.zeros(len(test_errors))
    seq_len = 50
    anomaly_start_in_test = 700 - 500
    anomaly_end_in_test = 800 - 500
    for i in range(len(test_errors)):
        window_start = i
        window_end = i + seq_len
        if window_end >= anomaly_start_in_test and window_start <= anomaly_end_in_test:
            test_labels[i] = 1

    pred = (test_errors > threshold).astype(int)
    TP = np.sum((test_labels == 1) & (pred == 1))
    FP = np.sum((test_labels == 0) & (pred == 1))
    FN = np.sum((test_labels == 1) & (pred == 0))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n测试集结果: TP={TP}, FP={FP}, FN={FN}")
    print(f"精确率: {precision:.3f}")
    print(f"召回率: {recall:.3f}")
    print(f"F1: {f1:.3f}")

    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    axes[0].plot(test_data[:, 2], color='#3498db', alpha=0.7, label='J3 angle (scaled)')
    axes[0].axvspan(anomaly_start_in_test, anomaly_end_in_test, color='red', alpha=0.2, label='True anomaly')
    axes[0].set_xlabel('Time step (test set)')
    axes[0].set_ylabel('J3 angle (scaled)')
    axes[0].set_title('J3 Joint Angle Trajectory (Test Set)')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(test_errors, color='#2c3e50', alpha=0.8, label='Reconstruction error')
    axes[1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold:.3f})')
    axes[1].axvspan(anomaly_start_in_test, anomaly_end_in_test, color='red', alpha=0.1)
    axes[1].set_xlabel('Time window (test set)')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Reconstruction Error (Test Set)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'temporal_anomaly_detection_v3.png'), dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到 {os.path.join(images_dir, 'temporal_anomaly_detection_v3.png')}")
    plt.show()

if __name__ == "__main__":
    main()