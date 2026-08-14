import numpy as np
import matplotlib.pyplot as plt
import os
import json

# ======================================================
# 预测性维护：基于负载率趋势的早期预警系统
# 数据来源：FANUC M-20iD/25 现场30组记录
# ======================================================

# 30组J6负载率数据（完整记录，包含报警组）
j6_loads = np.array([
    3.7, 4.1, 13.2, 13.7, 3.9, 15.8, 13.5, 11.3, 3.8, 18.9,
    127.4, 4.0, 13.4, 13.9, 3.7, 12.9, 3.6, 3.8, 13.3, 15.6,
    4.0, 7.5, 14.7, 15.1, 3.9, 17.6, 14.5, 3.7, 13.6, 3.6
])

# 正常数据（排除报警组）
j6_normal = np.delete(j6_loads, 10)

# 计算正常基线
mean_normal = np.mean(j6_normal)
std_normal = np.std(j6_normal)

# 设置预警阈值
warning_threshold = mean_normal * 2        # 一级预警：均值2倍
critical_threshold = mean_normal + 3 * std_normal  # 二级预警：3σ阈值

print("=" * 70)
print("预测性维护：J6轴负载率早期预警系统")
print("=" * 70)
print(f"\nJ6正常负载率均值: {mean_normal:.2f}%")
print(f"J6正常负载率标准差: {std_normal:.2f}%")
print(f"一级预警阈值（均值2倍）: {warning_threshold:.2f}%")
print(f"二级预警阈值（3σ）: {critical_threshold:.2f}%")

# ======================================================
# 模拟实时监测：逐条处理数据，检测预警触发时机
# ======================================================
print(f"\n--- 实时监测过程 ---")
print(f"{'样本':>4} {'J6负载率':>10} {'状态':>10} {'预警':>20}")
print("-" * 55)

warnings_triggered = []

# 模拟连续监测的滑动窗口
window_size = 3
recent_loads = []

for i, load in enumerate(j6_loads):
    recent_loads.append(load)
    if len(recent_loads) > window_size:
        recent_loads.pop(0)
    
    # 当前状态判断
    if load > critical_threshold:
        status = "异常"
    elif load > warning_threshold:
        status = "偏高"
    else:
        status = "正常"
    
    # 预警逻辑：连续3次超过一级阈值
    window_warning = all(l > warning_threshold for l in recent_loads) if len(recent_loads) == window_size else False
    
    if load > critical_threshold:
        alert = "🚨 二级预警：立即检查"
        warnings_triggered.append((i+1, "critical"))
    elif window_warning:
        alert = "⚠️ 一级预警：加强监测"
        warnings_triggered.append((i+1, "warning"))
    else:
        alert = ""
    
    print(f"{i+1:>4} {load:>10.1f}% {status:>10} {alert:>20}")

# ======================================================
# 分析预警触发时机
# ======================================================
print(f"\n--- 预警分析 ---")
print(f"报警实际发生在: 第11组")
print(f"系统预警触发在:")

for idx, level in warnings_triggered:
    if idx <= 10:
        if level == "warning":
            print(f"  - 第{idx}组触发一级预警（在报警前{11-idx}组）")
        elif level == "critical":
            print(f"  - 第{idx}组触发二级预警（在报警前{11-idx}组）")

# ======================================================
# 可视化
# ======================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 上图：J6负载率全程趋势
axes[0].plot(range(1, 31), j6_loads, 'o-', color='#3498db', markersize=4, alpha=0.7, label='J6 Load Rate')
axes[0].axhline(y=warning_threshold, color='orange', linestyle='--', label=f'Level 1 Warning ({warning_threshold:.1f}%)')
axes[0].axhline(y=critical_threshold, color='red', linestyle='--', label=f'Level 2 Critical ({critical_threshold:.1f}%)')
axes[0].axvspan(8, 11, color='red', alpha=0.1, label='Pre-alarm window')
axes[0].scatter([11], [j6_loads[10]], color='red', s=120, zorder=5, label='Alarm (SRVO-046)')
axes[0].set_xlabel('Sample Index')
axes[0].set_ylabel('Load Rate (%)')
axes[0].set_title('J6 Load Rate: Full Timeline with Warning Thresholds')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 下图：放大报警前的爬升过程
pre_alarm_idx = range(7, 12)
pre_alarm_loads = j6_loads[6:11]
axes[1].plot(pre_alarm_idx, pre_alarm_loads, 'o-', color='#e74c3c', markersize=8, linewidth=2)
axes[1].axhline(y=warning_threshold, color='orange', linestyle='--', label=f'Level 1 Warning ({warning_threshold:.1f}%)')
axes[1].axhline(y=critical_threshold, color='red', linestyle='--', label=f'Level 2 Critical ({critical_threshold:.1f}%)')
axes[1].fill_between(pre_alarm_idx, pre_alarm_loads, warning_threshold, alpha=0.2, color='orange')
axes[1].set_xlabel('Sample Index')
axes[1].set_ylabel('Load Rate (%)')
axes[1].set_title('Pre-Alarm Window: J6 Load Rate Escalation')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()

images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'predictive_maintenance.png'), dpi=150, bbox_inches='tight')
print(f"\n图表已保存到 {os.path.join(images_dir, 'predictive_maintenance.png')}")
plt.show()

# ======================================================
# 保存结果
# ======================================================
results = {
    "j6_normal_mean": float(mean_normal),
    "j6_normal_std": float(std_normal),
    "warning_threshold_2x_mean": float(warning_threshold),
    "critical_threshold_3sigma": float(critical_threshold),
    "alarm_sample_index": 11,
    "warnings_triggered": [(idx, level) for idx, level in warnings_triggered if idx <= 11],
    "early_warning_lead_time_samples": 0
}

# 计算预警提前量
early_warnings = [(idx, level) for idx, level in warnings_triggered if idx < 11]
if early_warnings:
    first_warning_idx = min(idx for idx, _ in early_warnings)
    results["early_warning_lead_time_samples"] = 11 - first_warning_idx
    results["early_warning_lead_time_minutes"] = (11 - first_warning_idx) * 15  # 假设每组间隔15分钟

with open('D:/VisionBot/robot-anomaly-detection/docs/predictive_maintenance_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存到 docs/predictive_maintenance_results.json")