import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ======================================================
# 基于真实负载率信号的异常检测
# 数据来源：FANUC M-20iD/25 现场30组记录
# ======================================================

# 30组数据的J1-J6负载率（从你提供的完整记录中提取）
load_rate_data = np.array([
    [7.2, 9.5, 6.8, 4.3, 5.1, 3.7],    # 1 冷机待机
    [8.1, 10.7, 7.4, 4.8, 5.6, 4.1],   # 2 空程后待机
    [27.4, 38.7, 22.9, 18.6, 24.1, 13.2],  # 3 直线焊接
    [28.1, 39.4, 23.5, 19.2, 24.7, 13.7],  # 4 直线焊接
    [7.8, 10.2, 7.1, 4.6, 5.4, 3.9],    # 5 工件更换待机
    [31.2, 42.6, 25.7, 21.4, 28.3, 15.8],  # 6 曲线焊接
    [27.9, 39.1, 23.2, 18.9, 24.5, 13.5],  # 7 直线角焊
    [22.6, 33.4, 19.8, 15.7, 20.2, 11.3],  # 8 收弧
    [7.5, 9.9, 7.0, 4.5, 5.3, 3.8],     # 9 工装切换
    [33.5, 44.8, 27.4, 24.7, 30.6, 18.9],  # 10 大姿态焊接
    [30.7, 41.9, 25.1, 22.3, 27.8, 127.4], # 11 异常报警(J6过载)
    [7.9, 10.4, 7.3, 4.7, 5.5, 4.0],    # 12 复位后待机
    [27.6, 38.9, 23.1, 18.8, 24.3, 13.4],  # 13 恢复焊接
    [28.3, 39.6, 23.7, 19.4, 25.0, 13.9],  # 14 连续焊接
    [7.3, 9.6, 6.9, 4.4, 5.2, 3.7],     # 15 半批次待机
    [26.8, 37.9, 22.4, 18.2, 23.6, 12.9],  # 16 补焊
    [7.1, 9.4, 6.7, 4.2, 5.0, 3.6],     # 17 午间待机
    [7.4, 9.7, 6.9, 4.4, 5.2, 3.8],     # 18 下午待机
    [27.5, 38.8, 23.0, 18.7, 24.2, 13.3],  # 19 下午首批
    [31.0, 42.4, 25.5, 21.2, 28.1, 15.6],  # 20 曲线焊接
    [7.9, 10.3, 7.2, 4.7, 5.5, 4.0],    # 21 工件更换
    [15.3, 21.7, 13.8, 10.2, 13.1, 7.5],   # 22 高速空程
    [29.7, 41.2, 24.8, 20.5, 26.4, 14.7],  # 23 厚板焊接
    [30.2, 41.8, 25.3, 21.0, 26.9, 15.1],  # 24 连续厚板
    [7.7, 10.1, 7.1, 4.6, 5.4, 3.9],    # 25 TCP校准
    [32.6, 43.9, 26.8, 23.5, 29.7, 17.6],  # 26 立角焊
    [29.1, 40.5, 24.3, 20.1, 25.8, 14.5],  # 27 连续焊接
    [7.2, 9.5, 6.8, 4.3, 5.1, 3.7],     # 28 批次待机
    [28.0, 39.3, 23.4, 19.1, 24.8, 13.6],  # 29 收尾批次
    [7.1, 9.4, 6.7, 4.2, 5.0, 3.6],     # 30 生产结束
])

# 标记第11组为异常（索引10）
anomaly_index = 10

# ======================================================
# 分析1：J6轴负载率的正常范围 vs 异常值
# ======================================================
j6_loads = load_rate_data[:, 5]
j6_normal = np.delete(j6_loads, anomaly_index)
j6_anomaly = j6_loads[anomaly_index]

j6_mean = np.mean(j6_normal)
j6_std = np.std(j6_normal)
threshold_3sigma = j6_mean + 3 * j6_std
threshold_95 = np.percentile(j6_normal, 95)

print("=" * 70)
print("Signal-Based Health Analysis: J6 Load Rate")
print("=" * 70)
print(f"\nJ6 正常负载率统计:")
print(f"  均值: {j6_mean:.2f}%")
print(f"  标准差: {j6_std:.2f}%")
print(f"  95分位数阈值: {threshold_95:.2f}%")
print(f"  3σ阈值: {threshold_3sigma:.2f}%")
print(f"\nJ6 异常样本负载率: {j6_anomaly:.1f}%")
print(f"  与均值偏差: {j6_anomaly - j6_mean:.1f}%")
print(f"  超过3σ阈值: {j6_anomaly > threshold_3sigma}")
print(f"  异常倍数: {j6_anomaly / j6_mean:.2f}倍")

# ======================================================
# 分析2：所有轴的异常检测（基于3σ规则）
# ======================================================
print(f"\n{'轴':>4} {'正常均值':>8} {'正常std':>8} {'3σ阈值':>8} {'异常值':>8} {'检测结果':>10}")
print("-" * 60)
for j in range(6):
    loads = load_rate_data[:, j]
    normal = np.delete(loads, anomaly_index)
    anomaly_val = loads[anomaly_index]
    m = np.mean(normal)
    s = np.std(normal)
    thr = m + 3 * s
    detected = anomaly_val > thr
    print(f"J{j+1:>4} {m:>8.2f}% {s:>8.2f}% {thr:>8.2f}% {anomaly_val:>8.1f}% {'异常' if detected else '正常':>10}")

# ======================================================
# 分析3：可视化
# ======================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 上图：J6负载率随时间变化
axes[0].plot(range(1, 31), j6_loads, 'o-', color='#3498db', markersize=4, alpha=0.7, label='J6 Load Rate')
axes[0].axhline(y=threshold_3sigma, color='red', linestyle='--', label=f'3σ Threshold ({threshold_3sigma:.1f}%)')
axes[0].axhline(y=threshold_95, color='orange', linestyle='--', alpha=0.5, label=f'95th Percentile ({threshold_95:.1f}%)')
axes[0].scatter([anomaly_index+1], [j6_anomaly], color='red', s=100, zorder=5, label=f'Anomaly ({j6_anomaly:.1f}%)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Load Rate (%)')
axes[0].set_title('J6 Axis Load Rate: Real Field Data with Anomaly Detection')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 下图：所有轴的负载率对比
x = range(1, 7)
normal_means = [np.mean(np.delete(load_rate_data[:, j], anomaly_index)) for j in range(6)]
anomaly_vals = load_rate_data[anomaly_index, :]
width = 0.35
axes[1].bar([i - width/2 for i in x], normal_means, width, label='Normal Mean', color='#3498db')
axes[1].bar([i + width/2 for i in x], anomaly_vals, width, label='Anomaly Value', color='#e74c3c')
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([f'J{i}' for i in range(1, 7)])
axes[1].set_ylabel('Load Rate (%)')
axes[1].set_title('All Joints: Normal Mean vs Anomaly Load Rate')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()

images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'signal_health_analysis.png'), dpi=150, bbox_inches='tight')
print(f"\n图表已保存到 {os.path.join(images_dir, 'signal_health_analysis.png')}")
plt.show()

# ======================================================
# 保存结果
# ======================================================
results = {
    "j6_normal_mean": float(j6_mean),
    "j6_normal_std": float(j6_std),
    "j6_anomaly_value": float(j6_anomaly),
    "j6_anomaly_ratio": float(j6_anomaly / j6_mean),
    "threshold_3sigma": float(threshold_3sigma),
    "threshold_95percentile": float(threshold_95),
    "detection_result": "J6 anomaly successfully detected by load rate signal",
    "comparison_with_position_based": "Position-based detection failed to identify J6 anomaly; load rate signal clearly identifies it"
}
with open('D:/VisionBot/robot-anomaly-detection/docs/signal_health_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存到 docs/signal_health_results.json")