import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

# ======================================================
# 故障关联规则挖掘（简化的Apriori核心思想）
# 无需安装mlxtend，用纯Python实现共现矩阵和关联规则
# ======================================================

# 故障码列表（基于你的现场数据）
fault_codes = [
    "Arc Loss",              # 断弧
    "Soft Limit Error",      # 软限定错误
    "Welder Comm Fault",     # 焊机通信异常
    "Welder Comm Interrupt", # 焊机通信中断
    "Safety Bracket",        # 安全支架动作
    "Collision Detected",    # 检测到碰撞
    "Coordinate Transform Err", # 坐标变换错误
    "Singularity Error",     # 特异点错误
    "Wire Adhesion",         # 粘丝
    "Welder Power Off",      # 焊接电源被关闭
    "Motor Overload",        # 电机过负载
    "Motor Current Stall",   # 电机电流失速
    "Invalid Command",       # 输入指令不正确
    "System Alarm",          # 系统警报
]

# 模拟30个故障事件记录（每个事件包含同时出现的多个故障码）
# 说明：基于现场观察，故障码常成组出现。这是合理模拟，用于方法演示。
np.random.seed(42)

fault_events = []
# 定义一些常见的故障组合模式
patterns = [
    ["Welder Comm Fault", "Arc Loss"],                                          # 通信异常常伴随断弧
    ["Welder Comm Interrupt", "Arc Loss", "Welder Power Off"],                  # 通信中断导致电源关闭
    ["Motor Current Stall", "Motor Overload"],                                  # 电流失速引发过载
    ["Soft Limit Error", "Coordinate Transform Err"],                           # 轨迹和坐标问题
    ["Collision Detected", "Safety Bracket", "Soft Limit Error"],               # 碰撞触发安全+限位
    ["Singularity Error", "Soft Limit Error"],                                  # 奇异点+限位
    ["Wire Adhesion", "Arc Loss"],                                              # 粘丝和断弧
    ["Invalid Command", "System Alarm"],                                        # 指令错误+报警
    ["Welder Comm Fault", "Invalid Command"],                                   # 通信和指令
    ["Welder Comm Interrupt", "System Alarm", "Arc Loss"],                      # 通信中断+报警+断弧
]

# 生成30个事件，每个事件从1-3个故障码中抽取
for i in range(30):
    if np.random.random() < 0.7:
        # 从已知模式中选一个
        pattern = patterns[np.random.randint(len(patterns))]
    else:
        # 随机组合1-2个故障码
        n = np.random.randint(1, 3)
        pattern = list(np.random.choice(fault_codes, size=n, replace=False))
    fault_events.append(pattern)

print(f"Total fault events: {len(fault_events)}")
print(f"Fault codes involved: {len(set([f for event in fault_events for f in event]))}")

# ======================================================
# 计算共现矩阵
# ======================================================
n_codes = len(fault_codes)
cooccurrence = np.zeros((n_codes, n_codes))

for event in fault_events:
    for i, code_i in enumerate(fault_codes):
        if code_i in event:
            for j, code_j in enumerate(fault_codes):
                if code_j in event and i != j:
                    cooccurrence[i][j] += 1

print("\n--- Co-occurrence Matrix (top pairs) ---")
pairs = []
for i in range(n_codes):
    for j in range(i+1, n_codes):
        if cooccurrence[i][j] > 0:
            pairs.append((cooccurrence[i][j], fault_codes[i], fault_codes[j]))

pairs.sort(reverse=True)
for count, code_i, code_j in pairs[:10]:
    print(f"  {code_i} <-> {code_j}: {int(count)} times")

# ======================================================
# 计算单故障码频率
# ======================================================
freq = {}
for event in fault_events:
    for code in event:
        freq[code] = freq.get(code, 0) + 1

print("\n--- Fault Frequency Ranking ---")
freq_sorted = sorted(freq.items(), key=lambda x: x[1], reverse=True)
for code, count in freq_sorted[:10]:
    print(f"  {code}: {count} occurrences")

# ======================================================
# 计算关联规则置信度
# ======================================================
print("\n--- Association Rules (Confidence) ---")
for code_a in fault_codes:
    count_a = freq.get(code_a, 0)
    if count_a == 0:
        continue
    for code_b in fault_codes:
        if code_a != code_b:
            count_ab = 0
            for event in fault_events:
                if code_a in event and code_b in event:
                    count_ab += 1
            if count_ab > 0:
                confidence = count_ab / count_a
                if confidence >= 0.5:  # 只显示置信度>=50%的规则
                    print(f"  {code_a} -> {code_b}: confidence={confidence:.2f}")

# ======================================================
# 可视化：共现矩阵热力图
# ======================================================
fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(cooccurrence, cmap='YlOrRd')
ax.set_xticks(range(n_codes))
ax.set_yticks(range(n_codes))
ax.set_xticklabels(fault_codes, rotation=90, fontsize=9)
ax.set_yticklabels(fault_codes, fontsize=9)
ax.set_title('Fault Co-occurrence Matrix (Simulated Field Data)', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, label='Co-occurrence Count')
plt.tight_layout()

images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
os.makedirs(images_dir, exist_ok=True)
plt.savefig(os.path.join(images_dir, 'fault_cooccurrence_matrix.png'), dpi=150, bbox_inches='tight')
print(f"\nCo-occurrence matrix saved to {os.path.join(images_dir, 'fault_cooccurrence_matrix.png')}")
plt.show()

# ======================================================
# 保存结果到JSON
# ======================================================
import json
results = {
    "fault_frequency": freq_sorted,
    "top_cooccurrence_pairs": [[int(c), a, b] for c, a, b in pairs[:10]],
    "note": "These are simulated fault events based on field observation patterns. Real fault logs would replace this dataset."
}
with open('D:/VisionBot/robot-anomaly-detection/docs/fault_association_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"Results saved to docs/fault_association_results.json")