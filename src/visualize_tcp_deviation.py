import matplotlib.pyplot as plt
import numpy as np

# TCP位置偏差数据（来自 anomaly_scenario_test.py 四个场景）
scenarios = ['Normal', 'Efficiency Drop', 'TCP Offset', 'Combined']
dx = [0.04, 0.02, 0.70, 0.40]
dy = [0.03, 0.02, 0.42, 0.22]
dz = [-0.04, -0.03, -0.93, -1.13]

x_pos = np.arange(len(scenarios))
width = 0.25

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_pos - width, dx, width, label='ΔX (mm)', color='#3498db')
ax.bar(x_pos, dy, width, label='ΔY (mm)', color='#2ecc71')
ax.bar(x_pos + width, dz, width, label='ΔZ (mm)', color='#e74c3c')

ax.set_xticks(x_pos)
ax.set_xticklabels(scenarios, fontsize=10)
ax.set_ylabel('Deviation (mm)', fontsize=11)
ax.set_title('TCP Position Deviation by Scenario', fontsize=13, fontweight='bold')
ax.axhline(y=0.2, color='red', linestyle='--', alpha=0.6, label='Threshold (0.2 mm)')
ax.axhline(y=-0.2, color='red', linestyle='--', alpha=0.6)
ax.legend(fontsize=9)
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()

import os
images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'tcp_deviation_by_scenario.png'), dpi=150, bbox_inches='tight')
print(f"图表已保存到 {os.path.join(images_dir, 'tcp_deviation_by_scenario.png')}")
plt.show()