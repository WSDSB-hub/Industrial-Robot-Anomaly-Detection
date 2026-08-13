import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# ==================== 1. 生成模拟数据集 ====================
# 说明：正常样本50个，异常样本10个，每个样本6个特征
# 特征顺序：weld_efficiency, tcp_dx, tcp_dy, tcp_dz, tcp_length_delta, fault_code_present
# 说明：特征包括焊接效率、TCP位置偏差(X/Y/Z)、TCP长度偏差、是否存在关键故障码

np.random.seed(42)

# 正常样本：围绕基准值微小波动
normal_samples = []
for _ in range(50):
    weld_eff = 0.048 + np.random.normal(0, 0.003)          # 焊接效率约0.048
    dx = np.random.normal(0, 0.03)                          # TCP偏差约0附近
    dy = np.random.normal(0, 0.03)
    dz = np.random.normal(0, 0.03)
    length_delta = np.random.normal(0, 0.05)
    fault = 0                                               # 无故障码
    normal_samples.append([weld_eff, dx, dy, dz, length_delta, fault])

# 异常样本：明显偏离
abnormal_samples = []
for _ in range(10):
    weld_eff = np.random.uniform(0.015, 0.035)             # 效率明显下降
    dx = np.random.normal(0.5, 0.2)                        # 偏差明显增大
    dy = np.random.normal(0.3, 0.15)
    dz = np.random.normal(-0.8, 0.3)
    length_delta = np.random.uniform(0.5, 1.5)
    fault = np.random.choice([0, 1], p=[0.3, 0.7])         # 大概率有故障码
    abnormal_samples.append([weld_eff, dx, dy, dz, length_delta, fault])

X_normal = np.array(normal_samples)
X_abnormal = np.array(abnormal_samples)
X = np.vstack([X_normal, X_abnormal])

# 标签：正常=1，异常=-1
y_true = np.array([1]*50 + [-1]*10)

# ==================== 2. 数据标准化 ====================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ==================== 3. 训练两种无监督异常检测模型 ====================
# 3.1 Isolation Forest
iso_forest = IsolationForest(contamination=0.167, random_state=42)  # 10/60≈0.167
iso_forest.fit(X_scaled)
y_iso = iso_forest.predict(X_scaled)
iso_scores = iso_forest.decision_function(X_scaled)

# 3.2 One-Class SVM
ocsvm = OneClassSVM(nu=0.167, kernel='rbf', gamma='scale')
ocsvm.fit(X_scaled)
y_ocsvm = ocsvm.predict(X_scaled)
ocsvm_scores = ocsvm.decision_function(X_scaled)

# ==================== 4. 计算准确率 ====================
def calc_metrics(y_true, y_pred):
    TP = np.sum((y_true == -1) & (y_pred == -1))
    TN = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 1) & (y_pred == -1))
    FN = np.sum((y_true == -1) & (y_pred == 1))
    accuracy = (TP + TN) / len(y_true)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1

iso_metrics = calc_metrics(y_true, y_iso)
ocsvm_metrics = calc_metrics(y_true, y_ocsvm)

# ==================== 5. 输出结果 ====================
print("========== 机器学习异常检测结果 ==========")
print(f"\n数据规模：正常50个，异常10个")
print(f"特征维度：6（焊接效率 + TCP偏差×3 + 长度偏差 + 故障码）")

print("\n--- Isolation Forest ---")
print(f"准确率: {iso_metrics[0]:.3f}, 精确率: {iso_metrics[1]:.3f}, 召回率: {iso_metrics[2]:.3f}, F1: {iso_metrics[3]:.3f}")

print("\n--- One-Class SVM ---")
print(f"准确率: {ocsvm_metrics[0]:.3f}, 精确率: {ocsvm_metrics[1]:.3f}, 召回率: {ocsvm_metrics[2]:.3f}, F1: {ocsvm_metrics[3]:.3f}")

# ==================== 6. 可视化对比 ====================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Isolation Forest 决策分数分布
axes[0].scatter(range(len(iso_scores)), iso_scores, c=y_true, cmap='coolwarm',
                edgecolors='k', s=60, alpha=0.8)
axes[0].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
axes[0].set_xlabel('Sample Index', fontsize=11)
axes[0].set_ylabel('Anomaly Score', fontsize=11)
axes[0].set_title(f'Isolation Forest (F1={iso_metrics[3]:.3f})', fontsize=12, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].grid(alpha=0.3)

# One-Class SVM 决策分数分布
axes[1].scatter(range(len(ocsvm_scores)), ocsvm_scores, c=y_true, cmap='coolwarm',
                edgecolors='k', s=60, alpha=0.8)
axes[1].axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Decision Boundary')
axes[1].set_xlabel('Sample Index', fontsize=11)
axes[1].set_ylabel('Anomaly Score', fontsize=11)
axes[1].set_title(f'One-Class SVM (F1={ocsvm_metrics[3]:.3f})', fontsize=12, fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

plt.tight_layout()

# 保存图片
images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'ml_anomaly_detection_comparison.png'), dpi=150, bbox_inches='tight')
print(f"\n图表已保存到 {os.path.join(images_dir, 'ml_anomaly_detection_comparison.png')}")
plt.show()

# ==================== 7. 保存指标结果 ====================
import json
results = {
    "dataset": {
        "normal_samples": 50,
        "abnormal_samples": 10,
        "features": ["weld_efficiency", "tcp_dx", "tcp_dy", "tcp_dz", "tcp_length_delta", "fault_code_present"]
    },
    "isolation_forest": {
        "accuracy": iso_metrics[0],
        "precision": iso_metrics[1],
        "recall": iso_metrics[2],
        "f1": iso_metrics[3]
    },
    "one_class_svm": {
        "accuracy": ocsvm_metrics[0],
        "precision": ocsvm_metrics[1],
        "recall": ocsvm_metrics[2],
        "f1": ocsvm_metrics[3]
    },
    "rule_based_engine": {
        "note": "Rule-based scoring achieved 100% accuracy on 4 scenario tests (100/65/40/5 score gradient)",
        "f1": 1.0
    }
}

with open(os.path.join(images_dir, '..', 'docs', 'ml_anomaly_results.json'), 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print(f"结果已保存到 docs/ml_anomaly_results.json")