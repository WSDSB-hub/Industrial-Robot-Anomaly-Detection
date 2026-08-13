import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# Ablation Study: 验证系统各模块的贡献
# ======================================================

np.random.seed(42)

def generate_dataset(n_normal=500, n_abnormal=100):
    """生成模拟数据集，包含7个特征"""
    features = []
    labels = []

    # 正常样本
    for _ in range(n_normal):
        weld_eff = 0.048 + np.random.normal(0, 0.003)
        dx = np.random.normal(0, 0.03)
        dy = np.random.normal(0, 0.03)
        dz = np.random.normal(0, 0.03)
        length_delta = np.random.normal(0, 0.05)
        fault_code = 0  # 正常样本无故障码
        joint_anomaly = 0  # 正常样本无关节异常
        features.append([weld_eff, dx, dy, dz, length_delta, fault_code, joint_anomaly])
        labels.append(1)  # 1=正常

    # 异常样本
    for _ in range(n_abnormal):
        # 随机选择异常类型
        anomaly_type = np.random.choice(['efficiency', 'tcp', 'fault', 'joint'])
        weld_eff = 0.048 + np.random.normal(0, 0.003)
        dx = np.random.normal(0, 0.03)
        dy = np.random.normal(0, 0.03)
        dz = np.random.normal(0, 0.03)
        length_delta = np.random.normal(0, 0.05)
        fault_code = 0
        joint_anomaly = 0

        if anomaly_type == 'efficiency':
            weld_eff = np.random.uniform(0.015, 0.030)  # 效率显著下降
            fault_code = 1
        elif anomaly_type == 'tcp':
            dx = np.random.normal(0.5, 0.2)
            dy = np.random.normal(0.3, 0.15)
            dz = np.random.normal(-0.8, 0.3)
            length_delta = np.random.uniform(0.5, 1.5)
        elif anomaly_type == 'fault':
            fault_code = 1
            weld_eff = np.random.uniform(0.020, 0.035)
        elif anomaly_type == 'joint':
            joint_anomaly = 1
            dx = np.random.normal(0.4, 0.1)
            dz = np.random.normal(-0.6, 0.2)

        features.append([weld_eff, dx, dy, dz, length_delta, fault_code, joint_anomaly])
        labels.append(-1)  # -1=异常

    return np.array(features), np.array(labels)


def train_and_evaluate(features, labels, feature_indices):
    """
    用指定的特征子集训练Isolation Forest并评估
    feature_indices: 要使用的特征索引列表
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = features[:, feature_indices]
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(contamination=0.167, random_state=42)
    model.fit(X_scaled)
    pred = model.predict(X_scaled)

    # 计算F1（宏平均）
    TP = np.sum((labels == -1) & (pred == -1))
    TN = np.sum((labels == 1) & (pred == 1))
    FP = np.sum((labels == 1) & (pred == -1))
    FN = np.sum((labels == -1) & (pred == 1))
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall


def main():
    features, labels = generate_dataset()

    # 特征定义
    feature_names = [
        'Weld Efficiency',
        'TCP dx',
        'TCP dy',
        'TCP dz',
        'TCP length delta',
        'Fault Code',
        'Joint Anomaly'
    ]
    # 特征索引
    all_features = list(range(7))
    no_fault_code = [0, 1, 2, 3, 4, 6]       # 去掉故障码
    no_tcp = [0, 5, 6]                       # 去掉TCP位姿
    no_efficiency = [1, 2, 3, 4, 5, 6]       # 去掉焊接效率
    no_joint = [0, 1, 2, 3, 4, 5]            # 去掉关节异常标记

    configs = [
        ("Full System (All Features)", all_features),
        ("Without Fault Code", no_fault_code),
        ("Without TCP Pose", no_tcp),
        ("Without Weld Efficiency", no_efficiency),
        ("Without Joint Anomaly Marker", no_joint),
    ]

    print("=" * 70)
    print("Ablation Study: Contribution of Each Feature to Anomaly Detection")
    print("=" * 70)

    results = []
    for name, indices in configs:
        f1, precision, recall = train_and_evaluate(features, labels, indices)
        results.append((name, f1, precision, recall))
        print(f"\n{name}")
        print(f"  F1: {f1:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")

    # 可视化
    names = [r[0] for r in results]
    f1_scores = [r[1] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#f39c12', '#f39c12']
    bars = ax.barh(names, f1_scores, color=colors, edgecolor='black', linewidth=1)

    # 标注分数
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=10, fontweight='bold')

    ax.set_xlim(0, 1.1)
    ax.set_xlabel('F1 Score', fontsize=11)
    ax.set_title('Ablation Study: Feature Contribution to Anomaly Detection', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    plt.tight_layout()

    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'ablation_study_results.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved to {os.path.join(images_dir, 'ablation_study_results.png')}")
    plt.show()


if __name__ == "__main__":
    main()