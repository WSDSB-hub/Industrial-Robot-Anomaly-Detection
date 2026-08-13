import numpy as np
import matplotlib.pyplot as plt
import os

# ======================================================
# 故障严重度分级与维护优先级排序
# ======================================================

# 严重度分级定义
# L1 = 安全级（可能导致设备损坏或人员伤害）
# L2 = 功能级（导致焊接质量下降或生产中断）
# L3 = 提示级（需关注但不立即停机）

severity_levels = {
    "L1_Safety": ["Safety Bracket", "Collision Detected", "Welder Power Off"],
    "L2_Functional": ["Arc Loss", "Soft Limit Error", "Motor Overload", "Wire Adhesion",
                       "Coordinate Transform Err", "Singularity Error"],
    "L3_Advisory": ["Welder Comm Fault", "Welder Comm Interrupt", "Motor Current Stall",
                     "Invalid Command", "System Alarm"],
}

# 故障频率（来自方向一的模拟数据）
fault_freq = {
    "Arc Loss": 22,
    "Soft Limit Error": 18,
    "Welder Comm Fault": 15,
    "Welder Comm Interrupt": 12,
    "Safety Bracket": 7,
    "Collision Detected": 6,
    "Coordinate Transform Err": 14,
    "Singularity Error": 11,
    "Wire Adhesion": 10,
    "Welder Power Off": 8,
    "Motor Overload": 9,
    "Motor Current Stall": 13,
    "Invalid Command": 16,
    "System Alarm": 17,
}

# ======================================================
# 维护优先级评分
# 评分 = 严重度权重 × 频率
# 严重度权重：L1=3, L2=2, L3=1
# ======================================================
severity_weight = {"L1_Safety": 3, "L2_Functional": 2, "L3_Advisory": 1}

priority_scores = []
for level, codes in severity_levels.items():
    for code in codes:
        freq = fault_freq.get(code, 0)
        weight = severity_weight[level]
        score = weight * freq
        priority_scores.append({
            "code": code,
            "level": level,
            "frequency": freq,
            "weight": weight,
            "priority_score": score
        })

# 按优先级分数排序
priority_scores.sort(key=lambda x: x["priority_score"], reverse=True)

print("=" * 60)
print("维护优先级排序")
print("=" * 60)
print(f"\n{'故障码':<30} {'等级':<15} {'频率':>5} {'权重':>5} {'优先级分':>8}")
print("-" * 70)
for item in priority_scores:
    print(f"{item['code']:<30} {item['level']:<15} {item['frequency']:>5} {item['weight']:>5} {item['priority_score']:>8}")

# ======================================================
# 可视化：维护优先级柱状图
# ======================================================
codes = [p["code"] for p in priority_scores]
scores = [p["priority_score"] for p in priority_scores]
levels = [p["level"] for p in priority_scores]

# 颜色映射
level_colors = {
    "L1_Safety": "#e74c3c",
    "L2_Functional": "#f39c12",
    "L3_Advisory": "#3498db"
}
colors = [level_colors[l] for l in levels]

fig, ax = plt.subplots(figsize=(14, 7))
bars = ax.bar(range(len(codes)), scores, color=colors, edgecolor='black', linewidth=0.8)

# 添加图例
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#e74c3c", label="L1 Safety (weight=3)"),
    Patch(facecolor="#f39c12", label="L2 Functional (weight=2)"),
    Patch(facecolor="#3498db", label="L3 Advisory (weight=1)"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

ax.set_xticks(range(len(codes)))
ax.set_xticklabels(codes, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Priority Score (severity × frequency)', fontsize=11)
ax.set_title('Maintenance Priority Ranking for Robot Fault Codes', fontsize=13, fontweight='bold')
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'maintenance_priority_ranking.png'), dpi=150, bbox_inches='tight')
print(f"\nMaintenance priority chart saved to {os.path.join(images_dir, 'maintenance_priority_ranking.png')}")
plt.show()