import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from pathlib import Path
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import StandardScaler

# 路径配置
BASE_DIR = Path(r"D:/VisionBot/robot-anomaly-detection")
DOCS_DIR = BASE_DIR / "docs"
IMAGES_DIR = BASE_DIR / "images"
DOCS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

def load_real_data():
    """加载真实数据（硬编码关键特征）"""
    normal_samples = [
        [3.7, 0.35, 39.1],
        [4.1, 0.38, 39.8],
        [13.2, 0.79, 44.7],
        [13.7, 0.82, 46.4],
        [3.9, 0.36, 46.8],
        [15.8, 0.95, 48.7],
        [13.5, 0.81, 49.5],
        [11.3, 0.68, 50.2],
        [3.8, 0.36, 50.6],
        [18.9, 1.13, 51.9],
        [4.0, 0.37, 62.4],
        [13.4, 0.80, 58.7],
        [13.9, 0.83, 55.3],
        [3.7, 0.35, 54.8],
        [12.9, 0.77, 55.4],
        [3.6, 0.34, 55.7],
        [3.8, 0.36, 45.4],
        [13.3, 0.80, 49.3],
        [15.6, 0.94, 51.4],
        [4.0, 0.37, 51.8],
        [7.5, 0.52, 52.6],
        [14.7, 0.88, 53.5],
        [15.1, 0.91, 54.3],
        [3.9, 0.36, 54.7],
        [17.6, 1.06, 55.6],
        [14.5, 0.87, 56.2],
        [3.7, 0.35, 56.5],
        [13.6, 0.81, 57.3],
        [3.6, 0.34, 57.8],
    ]
    anomaly_samples = [[127.4, 2.31, 71.8]]

    X_normal = np.array(normal_samples)
    X_anomaly = np.array(anomaly_samples)
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([0]*len(normal_samples) + [1]*len(anomaly_samples))
    return X, y

def generate_sim_data(X_real_normal, X_real_anomaly, num_normal=500, num_anomaly=100):
    """基于真实数据统计生成仿真数据，增强数据多样性"""
    mean_normal = np.mean(X_real_normal, axis=0)
    std_normal = np.std(X_real_normal, axis=0) * 1.2
    std_normal = np.where(std_normal < 0.1, 0.1, std_normal)
    X_sim_normal = np.random.normal(loc=mean_normal, scale=std_normal, size=(num_normal, 3))
    X_sim_normal = np.clip(X_sim_normal, a_min=0, a_max=None)

    # 异常样本：两类
    loads_extreme = np.random.uniform(100, 160, num_anomaly // 2)
    currents_extreme = np.random.uniform(2.0, 3.5, num_anomaly // 2)
    temps_extreme = np.random.uniform(65, 85, num_anomaly // 2)

    loads_marginal = np.random.uniform(30, 80, num_anomaly - num_anomaly // 2)
    currents_marginal = np.random.uniform(0.8, 1.8, num_anomaly - num_anomaly // 2)
    temps_marginal = np.random.uniform(50, 65, num_anomaly - num_anomaly // 2)

    loads = np.concatenate([loads_extreme, loads_marginal])
    currents = np.concatenate([currents_extreme, currents_marginal])
    temps = np.concatenate([temps_extreme, temps_marginal])

    X_sim_anomaly = np.column_stack([loads, currents, temps])
    X_sim = np.vstack([X_sim_normal, X_sim_anomaly])
    y_sim = np.array([0]*num_normal + [1]*num_anomaly)
    return X_sim, y_sim

def evaluate_unsupervised(clf, X_test, y_test):
    """无监督评估：阈值优化"""
    scores = clf.decision_function(X_test)
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_f1 = -1
    best_thresh = scores.min()
    best_prec = 0
    best_rec = 0
    for thresh in thresholds:
        y_pred = (scores < thresh).astype(int)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_prec = prec
            best_rec = rec
    auc = roc_auc_score(y_test, -scores)
    return best_prec, best_rec, best_f1, auc, best_thresh

def main():
    print("="*70)
    print("数字孪生仿真：数据增强与异常检测实验")
    print("="*70)

    # 1. 加载真实数据
    X_real, y_real = load_real_data()
    X_real_normal = X_real[y_real==0]
    X_real_anomaly = X_real[y_real==1]

    print(f"\n真实数据：{len(X_real)} 个样本（正常 {sum(y_real==0)}，异常 {sum(y_real==1)}）")

    # 2. 生成仿真数据
    print("\n生成仿真数据（基于真实数据统计特征）...")
    X_sim, y_sim = generate_sim_data(X_real_normal, X_real_anomaly, num_normal=500, num_anomaly=100)
    print(f"仿真数据：{len(X_sim)} 个样本（正常 {sum(y_sim==0)}，异常 {sum(y_sim==1)}）")

    # 3. 标准化
    scaler = StandardScaler()
    scaler.fit(X_real_normal)  # 以真实正常样本为基准
    X_real_normal_scaled = scaler.transform(X_real_normal)
    X_real_scaled = scaler.transform(X_real)
    X_sim_scaled = scaler.transform(X_sim)

    # ===================== 实验一：无监督异常检测 =====================
    print("\n" + "="*50)
    print("实验一：无监督异常检测（Isolation Forest）")
    print("="*50)

    # 仅真实正常样本训练
    clf_real = IsolationForest(contamination=0.01, random_state=42)
    clf_real.fit(X_real_normal_scaled)
    prec_real, rec_real, f1_real, auc_real, _ = evaluate_unsupervised(clf_real, X_real_scaled, y_real)

    # 仅仿真数据训练
    X_sim_normal_scaled = scaler.transform(X_sim[y_sim==0])
    clf_sim = IsolationForest(contamination=0.01, random_state=42)
    clf_sim.fit(X_sim_normal_scaled)
    prec_sim, rec_sim, f1_sim, auc_sim, _ = evaluate_unsupervised(clf_sim, X_real_scaled, y_real)

    # 混合数据训练
    X_train_mix = np.vstack([X_real_normal, X_sim])
    y_train_mix = np.array([0]*len(X_real_normal) + list(y_sim))
    X_mix_normal = X_train_mix[y_train_mix==0]
    X_mix_normal_scaled = scaler.transform(X_mix_normal)
    clf_mix = IsolationForest(contamination=0.01, random_state=42)
    clf_mix.fit(X_mix_normal_scaled)
    prec_mix, rec_mix, f1_mix, auc_mix, _ = evaluate_unsupervised(clf_mix, X_real_scaled, y_real)

    print("\n--- 无监督检测性能（在真实数据上测试，阈值优化） ---")
    print(f"{'实验':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'AUC':<10}")
    print(f"{'仅真实数据':<20} {prec_real:<10.3f} {rec_real:<10.3f} {f1_real:<10.3f} {auc_real:<10.3f}")
    print(f"{'仅仿真数据':<20} {prec_sim:<10.3f} {rec_sim:<10.3f} {f1_sim:<10.3f} {auc_sim:<10.3f}")
    print(f"{'混合数据':<20} {prec_mix:<10.3f} {rec_mix:<10.3f} {f1_mix:<10.3f} {auc_mix:<10.3f}")

    # ===================== 实验二：监督学习（数据增强价值展示） =====================
    print("\n" + "="*50)
    print("实验二：监督学习（随机森林）")
    print("="*50)

    # 训练配置
    # 配置A：仅仿真数据训练
    X_train_sim = X_sim
    y_train_sim = y_sim

    # 配置B：混合数据训练（真实正常 + 仿真全部）
    X_train_mix = np.vstack([X_real_normal, X_sim])
    y_train_mix = np.array([0]*len(X_real_normal) + list(y_sim))

    # 测试集：真实30组数据
    X_test = X_real
    y_test = y_real

    # 标准化（使用仿真数据训练集的统计量）
    scaler_sim = StandardScaler()
    scaler_sim.fit(X_train_sim)
    X_train_sim_scaled = scaler_sim.transform(X_train_sim)
    X_test_sim_scaled = scaler_sim.transform(X_test)

    scaler_mix = StandardScaler()
    scaler_mix.fit(X_train_mix)
    X_train_mix_scaled = scaler_mix.transform(X_train_mix)
    X_test_mix_scaled = scaler_mix.transform(X_test)

    # 训练随机森林
    rf_sim = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sim.fit(X_train_sim_scaled, y_train_sim)
    y_pred_sim = rf_sim.predict(X_test_sim_scaled)

    rf_mix = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_mix.fit(X_train_mix_scaled, y_train_mix)
    y_pred_mix = rf_mix.predict(X_test_mix_scaled)

    print("\n--- 监督学习在真实数据上的表现 ---")
    print("\n配置A：仅仿真数据训练")
    print(classification_report(y_test, y_pred_sim, digits=3, target_names=['Normal', 'Anomaly']))
    print("配置B：混合数据训练")
    print(classification_report(y_test, y_pred_mix, digits=3, target_names=['Normal', 'Anomaly']))

    # 仅真实数据无法训练监督模型（异常样本只有1个），因此不列出
    print("注意：仅真实数据无法训练监督模型（异常样本仅1个），因此监督学习实验只比较仿真与混合训练。")

    # ===================== 保存结果 =====================
    sim_df = pd.DataFrame(X_sim, columns=['J6_load', 'J6_current', 'J6_temp'])
    sim_df['label'] = y_sim
    sim_csv = DOCS_DIR / "simulation_data_600.csv"
    sim_df.to_csv(sim_csv, index=False)
    print(f"\n仿真数据已保存至 {sim_csv}")

    # 保存图表
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 左图：数据分布
    axes[0].scatter(X_real_normal[:,0], X_real_normal[:,1], c='blue', label='Real Normal', alpha=0.6)
    axes[0].scatter(X_real_anomaly[:,0], X_real_anomaly[:,1], c='red', label='Real Anomaly', s=100, marker='x')
    axes[0].scatter(X_sim[y_sim==0][:,0], X_sim[y_sim==0][:,1], c='cyan', label='Sim Normal', alpha=0.3)
    axes[0].scatter(X_sim[y_sim==1][:,0], X_sim[y_sim==1][:,1], c='magenta', label='Sim Anomaly', alpha=0.3)
    axes[0].set_xlabel('J6 Load (%)')
    axes[0].set_ylabel('J6 Current (A)')
    axes[0].set_title('Data Distribution: Real vs Simulation')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 右图：无监督性能对比
    f1_scores = [f1_real, f1_sim, f1_mix]
    auc_scores = [auc_real, auc_sim, auc_mix]
    x = np.arange(3)
    width = 0.35
    axes[1].bar(x - width/2, f1_scores, width, label='F1 Score')
    axes[1].bar(x + width/2, auc_scores, width, label='ROC AUC')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(['Real Only', 'Sim Only', 'Mixed'])
    axes[1].set_ylabel('Score')
    axes[1].set_title('Unsupervised Detection Performance (Threshold Optimized)')
    axes[1].legend()
    axes[1].grid(alpha=0.3, axis='y')

    plt.tight_layout()
    img_path = IMAGES_DIR / "digital_twin_results.png"
    plt.savefig(img_path, dpi=150, bbox_inches='tight')
    print(f"结果图表已保存至 {img_path}")

    # 保存 JSON
    results = {
        "real_data_samples": len(X_real),
        "simulation_data_samples": len(X_sim),
        "unsupervised_performance": {
            "real_only": {"precision": float(prec_real), "recall": float(rec_real),
                          "f1": float(f1_real), "auc": float(auc_real)},
            "sim_only": {"precision": float(prec_sim), "recall": float(rec_sim),
                         "f1": float(f1_sim), "auc": float(auc_sim)},
            "mixed": {"precision": float(prec_mix), "recall": float(rec_mix),
                      "f1": float(f1_mix), "auc": float(auc_mix)}
        },
        "supervised_performance": {
            "sim_only": classification_report(y_test, y_pred_sim, output_dict=True, zero_division=0),
            "mixed": classification_report(y_test, y_pred_mix, output_dict=True, zero_division=0)
        },
        "note": "Simulation data generated from real data statistics; supervised learning demonstrates value of augmentation."
    }
    json_path = DOCS_DIR / "digital_twin_results.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"结构化结果已保存至 {json_path}")

    plt.show()

if __name__ == "__main__":
    main()