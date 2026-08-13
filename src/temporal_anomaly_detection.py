import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================================================
# Temporal Anomaly Detection using LSTM-Autoencoder
# 用于检测连续运动轨迹中的时间序列异常
# ======================================================

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ======================================================
# 1. 生成模拟的连续关节角度时间序列
# ======================================================

def generate_trajectory_data(n_timesteps=1000, n_joints=6, anomaly_start=700, anomaly_length=100):
    """
    生成6轴关节角度的连续时间序列
    正常段：平滑的正弦运动
    异常段：加入突变和噪声
    """
    t = np.linspace(0, 10, n_timesteps)
    
    # 正常轨迹：每个关节都有平滑的角度变化
    normal_trajectory = np.zeros((n_timesteps, n_joints))
    for j in range(n_joints):
        amplitude = 10 + j * 2
        frequency = 0.5 + j * 0.1
        phase = j * 0.5
        normal_trajectory[:, j] = amplitude * np.sin(frequency * t + phase) + 20 * np.cos(0.2 * t)
    
    # 注入异常段：在 anomaly_start 到 anomaly_start+anomaly_length 之间加入突变
    trajectory = normal_trajectory.copy()
    if anomaly_start + anomaly_length < n_timesteps:
        trajectory[anomaly_start:anomaly_start+anomaly_length, :] += np.random.normal(0, 5, (anomaly_length, n_joints))
        # 特别让J3关节出现大幅偏移
        trajectory[anomaly_start:anomaly_start+anomaly_length, 2] += 15 * np.sin(np.linspace(0, 4*np.pi, anomaly_length))
    
    return trajectory, anomaly_start, anomaly_length


# ======================================================
# 2. 定义LSTM-Autoencoder模型
# ======================================================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, num_layers=2):
        super(LSTMAutoencoder, self).__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True, dropout=0.1)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        _, (hidden, _) = self.encoder(x)
        # 重复最后一个隐藏状态，作为解码器输入
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        decoder_output, _ = self.decoder(decoder_input)
        reconstructed = self.output_layer(decoder_output)
        return reconstructed


# ======================================================
# 3. 准备训练数据
# ======================================================

def prepare_sequences(data, seq_len=50):
    """将时间序列切分为窗口序列"""
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i+seq_len])
    return np.array(sequences)


# ======================================================
# 4. 训练模型
# ======================================================

def train_lstm_autoencoder(train_data, seq_len=50, epochs=50, batch_size=32, lr=0.001):
    """用正常数据训练LSTM-Autoencoder"""
    input_dim = train_data.shape[1]
    model = LSTMAutoencoder(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 准备训练序列（使用正常段数据）
    normal_sequences = prepare_sequences(train_data, seq_len)
    X_train = torch.FloatTensor(normal_sequences)
    dataset = TensorDataset(X_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            x = batch[0]
            optimizer.zero_grad()
            reconstructed = model(x)
            loss = criterion(reconstructed, x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    return model


# ======================================================
# 5. 计算重建误差并检测异常
# ======================================================

def compute_reconstruction_error(model, data, seq_len=50):
    """计算每个窗口的重建误差"""
    sequences = prepare_sequences(data, seq_len)
    X = torch.FloatTensor(sequences)
    model.eval()
    with torch.no_grad():
        reconstructed = model(X)
        # 计算每个窗口的MSE
        errors = torch.mean((reconstructed - X) ** 2, dim=(1, 2)).numpy()
    return errors


# ======================================================
# 6. 主程序
# ======================================================

def main():
    print("=" * 70)
    print("Temporal Anomaly Detection using LSTM-Autoencoder")
    print("=" * 70)

    # 生成模拟数据
    trajectory, anomaly_start, anomaly_length = generate_trajectory_data()
    print(f"数据形状: {trajectory.shape}")
    print(f"异常段: 时间步 {anomaly_start} 到 {anomaly_start+anomaly_length}")

    # 用正常段训练（取前700个时间步作为正常数据）
    normal_data = trajectory[:700, :]
    print("\n训练LSTM-Autoencoder（仅使用正常数据）...")
    model = train_lstm_autoencoder(normal_data, seq_len=50, epochs=50)

    # 计算整个轨迹的重建误差
    print("\n计算重建误差...")
    errors = compute_reconstruction_error(model, trajectory, seq_len=50)

    # 设定异常阈值（基于正常段误差的均值+3标准差）
    normal_errors = compute_reconstruction_error(model, normal_data, seq_len=50)
    threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
    print(f"正常段重建误差均值: {np.mean(normal_errors):.6f}")
    print(f"异常阈值: {threshold:.6f}")

    # 检测异常窗口
    anomaly_flags = errors > threshold
    detected_anomalies = np.where(anomaly_flags)[0]

    # 计算检测准确率
    # 真实异常窗口：anomaly_start 到 anomaly_start+anomaly_length
    true_anomaly_windows = np.arange(anomaly_start - 50, anomaly_start + anomaly_length)
    true_anomaly_windows = true_anomaly_windows[(true_anomaly_windows >= 0) & (true_anomaly_windows < len(errors))]
    
    if len(detected_anomalies) > 0:
        TP = len(set(detected_anomalies) & set(true_anomaly_windows))
        FP = len(set(detected_anomalies) - set(true_anomaly_windows))
        FN = len(set(true_anomaly_windows) - set(detected_anomalies))
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        print(f"\n检测结果: TP={TP}, FP={FP}, FN={FN}")
        print(f"精确率: {precision:.3f}")
        print(f"召回率: {recall:.3f}")
        print(f"F1: {f1:.3f}")

    # ======================================================
    # 7. 可视化
    # ======================================================
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))

    # 上图：J3关节轨迹
    axes[0].plot(trajectory[:, 2], color='#3498db', alpha=0.7, label='J3 angle')
    axes[0].axvspan(anomaly_start, anomaly_start+anomaly_length, color='red', alpha=0.2, label='True anomaly')
    axes[0].set_xlabel('Time step')
    axes[0].set_ylabel('J3 angle (deg)')
    axes[0].set_title('J3 Joint Angle Trajectory')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 下图：重建误差
    axes[1].plot(errors, color='#2c3e50', alpha=0.8, label='Reconstruction error')
    axes[1].axhline(y=threshold, color='red', linestyle='--', label='Threshold')
    axes[1].axvspan(anomaly_start, anomaly_start+anomaly_length, color='red', alpha=0.1)
    axes[1].set_xlabel('Time window')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('Reconstruction Error with Anomaly Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'temporal_anomaly_detection.png'), dpi=150, bbox_inches='tight')
    print(f"\n图表已保存到 {os.path.join(images_dir, 'temporal_anomaly_detection.png')}")
    plt.show()


if __name__ == "__main__":
    main()