import matplotlib.pyplot as plt
import numpy as np

# 场景测试结果（来自 anomaly_scenario_test.py 的输出）
scenarios = ['Normal State', 'Efficiency Drop', 'TCP Offset', 'Combined Anomaly']
scores = [100.0, 65.0, 40.0, 5.0]
colors = ['#2ecc71', '#f39c12', '#e67e22', '#e74c3c']

# 创建柱状图
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(scenarios, scores, color=colors, edgecolor='black', linewidth=1.2)

# 在每根柱子上方标注分数
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
            f'{score:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# 设置坐标轴和标题
ax.set_ylabel('Health Score (0-100)', fontsize=11)
ax.set_title('Robot Health Score Across Validation Scenarios', fontsize=13, fontweight='bold')
ax.set_ylim(0, 110)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# 添加阈值参考线
ax.axhline(y=80, color='blue', linestyle='--', alpha=0.5, label='Healthy Threshold (80)')
ax.axhline(y=60, color='orange', linestyle='--', alpha=0.5, label='Warning Threshold (60)')
ax.legend(loc='upper right', fontsize=9)

plt.tight_layout()

# 保存图片
import os
images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'health_score_comparison.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存到 {os.path.join(images_dir, 'health_score_comparison.png')}")

# 同时显示
plt.show()