import numpy as np
import matplotlib.pyplot as plt
import os
import json
import csv

# ======================================================
# 故障码时序预测：基于真实工业背景的报警日志生成与挖掘
# ======================================================

# ---------- 路径配置 ----------
BASE_DIR = r"D:/VisionBot/robot-anomaly-detection"
DOCS_DIR = os.path.join(BASE_DIR, "docs")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
os.makedirs(DOCS_DIR, exist_ok=True)
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------- 定义事件类型（真实故障 + 警告） ----------
fault_types = [
    "Arc Loss",                 # 断弧
    "Welder Comm Fault",        # 焊机通信异常
    "Welder Comm Interrupt",    # 焊机通信中断
    "Motor Overload",           # 电机过负载（对应 SRVO-046）
    "Wire Adhesion",            # 粘丝
    "Soft Limit Error",         # 软限定错误
    "Coordinate Transform Err", # 坐标变换错误
    "Torch Collision",          # 检测到碰TW/RW/FA
    "Posture Mismatch",         # 与示教姿态不符
    "Singularity Error",        # 特异点错误
    "Welder Power Off",         # 焊接电源被关闭
]

warning_types = [
    "Motor Current Stall",      # 电机电流失速（警告）
    "Invalid Input Command",    # 输入指令不正确（警告）
    "System Alarm"              # 系统警报（警告）
]

all_events = fault_types + warning_types

# ---------- 故障因果模式（前因 -> [(后果, 平均延迟分钟, 触发概率)]） ----------
causal_patterns = {
    "Welder Comm Interrupt": [("Arc Loss", 5.0, 0.7)],
    "Welder Comm Fault": [("Arc Loss", 8.0, 0.6)],
    "Motor Current Stall": [("Motor Overload", 20.0, 0.6)],   # 电流失速常先于过载
    "Wire Adhesion": [("Arc Loss", 3.0, 0.8)],
    "Soft Limit Error": [("Coordinate Transform Err", 2.0, 0.4)],
    "Posture Mismatch": [("Coordinate Transform Err", 4.0, 0.5)],
    "Singularity Error": [("Coordinate Transform Err", 1.0, 0.3)],
    "Invalid Input Command": [("System Alarm", 2.0, 0.6)],
    "Welder Power Off": [("Arc Loss", 1.0, 0.9)],
    "Torch Collision": [("Soft Limit Error", 10.0, 0.3)],
}

# ---------- 事件初始发生概率 ----------
event_weights = {
    "Arc Loss": 0.15,
    "Welder Comm Fault": 0.08,
    "Welder Comm Interrupt": 0.08,
    "Motor Overload": 0.03,
    "Wire Adhesion": 0.10,
    "Soft Limit Error": 0.08,
    "Coordinate Transform Err": 0.06,
    "Torch Collision": 0.04,
    "Posture Mismatch": 0.05,
    "Singularity Error": 0.04,
    "Welder Power Off": 0.03,
    "Motor Current Stall": 0.10,
    "Invalid Input Command": 0.08,
    "System Alarm": 0.08,
}

# ======================================================
# 生成 30 天工作时段报警日志
# ======================================================
np.random.seed(42)
event_log = []  # 格式: (time_in_minutes, event_name, is_warning)

days = 30
work_start_hour = 8
work_end_hour = 17
minutes_per_day = (work_end_hour - work_start_hour) * 60  # 540 分钟

event_names = list(event_weights.keys())
event_probs = np.array([event_weights[e] for e in event_names])
event_probs = event_probs / event_probs.sum()

for day in range(days):
    day_base = day * minutes_per_day
    # 每天初始事件数量，平均 4 个，至少 1 个
    num_initial = max(np.random.poisson(4), 1)
    initial_times = np.sort(np.random.uniform(0, minutes_per_day, num_initial))

    for t in initial_times:
        event_time = day_base + t
        event = np.random.choice(event_names, p=event_probs)
        is_warning = event in warning_types
        event_log.append((event_time, event, is_warning))

        # 根据因果模式生成后续事件
        if event in causal_patterns:
            for consequence, avg_delay, prob in causal_patterns[event]:
                if np.random.random() < prob:
                    delay = avg_delay + np.random.normal(0, avg_delay * 0.2)
                    cons_time = event_time + delay
                    # 限制后果事件在当天工作时间内
                    if cons_time <= day_base + minutes_per_day:
                        cons_is_warning = consequence in warning_types
                        event_log.append((cons_time, consequence, cons_is_warning))

    # 固定已知事件：第10天（索引9）10:42 J6过载报警（电机过负载）
    if day == 9:
        overload_time = day_base + 162  # 10:42 相对于 8:00 为 162 分钟
        event_log.append((overload_time, "Motor Overload", False))
        # 其前约20分钟出现电机电流失速警告
        stall_time = overload_time - 20 + np.random.normal(0, 3)
        stall_time = max(day_base, min(stall_time, overload_time - 5))
        event_log.append((stall_time, "Motor Current Stall", True))

# 按时间排序
event_log.sort(key=lambda x: x[0])

print("=" * 70)
print("故障码时序预测分析（基于真实工业背景生成）")
print("=" * 70)
print(f"\n总报警事件数: {len(event_log)}")
print(f"时间跨度: 30个工作日，每天 {work_start_hour}:00-{work_end_hour}:00")
print(f"事件类型数: {len(all_events)}（{len(fault_types)}种故障 + {len(warning_types)}种警告）")

# ======================================================
# 时序依赖挖掘
# ======================================================
time_window = 30  # 30分钟时间窗口（可调）

sequence_pairs = {}
for i, (time_a, fault_a, _) in enumerate(event_log):
    for j in range(i + 1, len(event_log)):
        time_b, fault_b, _ = event_log[j]
        time_diff = time_b - time_a
        if time_diff > time_window:
            break
        if fault_a != fault_b:
            key = (fault_a, fault_b)
            if key not in sequence_pairs:
                sequence_pairs[key] = []
            sequence_pairs[key].append(time_diff)

print(f"\n--- 时序依赖挖掘结果（时间窗口 {time_window} 分钟） ---")
print(f"{'前因事件':<25} {'后果事件':<25} {'出现次数':>6} {'平均间隔':>8}")

rules = []
for (fault_a, fault_b), diffs in sequence_pairs.items():
    if len(diffs) >= 3:   # 至少3次支持
        avg_delay = np.mean(diffs)
        rules.append({
            "antecedent": fault_a,
            "consequent": fault_b,
            "count": len(diffs),
            "avg_delay_min": float(avg_delay)
        })
        print(f"{fault_a:<25} {fault_b:<25} {len(diffs):>6} {avg_delay:>8.1f}分钟")

rules.sort(key=lambda x: x["count"], reverse=True)

print(f"\n--- 最强时序预测规则 Top 5 ---")
for i, rule in enumerate(rules[:5], 1):
    print(f"规则{i}: 看到【{rule['antecedent']}】后，平均 {rule['avg_delay_min']:.1f} 分钟后可能出现【{rule['consequent']}】")
    print(f"       支持度: {rule['count']}次")

# ======================================================
# 可视化
# ======================================================
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 图1：故障时间线
day_indices = [t // minutes_per_day for t, _, _ in event_log]
times_in_day = [t % minutes_per_day for t, _, _ in event_log]
event_codes = [all_events.index(e) for _, e, _ in event_log]

scatter = axes[0].scatter(
    [d + t / minutes_per_day for d, t in zip(day_indices, times_in_day)],
    event_codes,
    c=event_codes,
    cmap='tab20',
    s=20,
    alpha=0.7
)
axes[0].set_xlabel('Day (fraction within working hours)')
axes[0].set_ylabel('Event Type Index')
axes[0].set_yticks(range(len(all_events)))
axes[0].set_yticklabels(all_events, fontsize=8)
axes[0].set_title('30-Day Alarm/Warning Timeline (Simulated from Real Industrial Patterns)')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(0, days)

# 图2：最强规则的时序间隔分布
if rules:
    top_rule = rules[0]
    key = (top_rule["antecedent"], top_rule["consequent"])
    diffs = sequence_pairs[key]
    axes[1].hist(diffs, bins=15, alpha=0.6, color='#e74c3c', edgecolor='black')
    axes[1].axvline(x=top_rule["avg_delay_min"], color='red', linestyle='--',
                    label=f'Mean: {top_rule["avg_delay_min"]:.1f} min')
    axes[1].set_xlabel(f'Time delay from [{top_rule["antecedent"]}] to [{top_rule["consequent"]}] (min)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Delay Distribution for Strongest Causal Rule')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

plt.tight_layout()

# 保存图表
image_path = os.path.join(IMAGES_DIR, 'fault_sequence_prediction.png')
plt.savefig(image_path, dpi=150, bbox_inches='tight')
print(f"\n图表已保存到 {image_path}")

# ======================================================
# 保存结果 JSON
# ======================================================
results = {
    "total_events": len(event_log),
    "time_window_min": time_window,
    "causal_rules_discovered": rules,
    "note": "Event log generated based on real industrial alarm types and maintenance records. "
            "Replace with actual teach pendant export for production use."
}

json_path = os.path.join(DOCS_DIR, 'fault_sequence_results.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"结果已保存到 {json_path}")

# ======================================================
# 导出报警日志 CSV（方便检查）
# ======================================================
csv_path = os.path.join(DOCS_DIR, 'alarm_log_30days.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['Day', 'Time', 'Event', 'Type'])
    for t, event, is_warning in event_log:
        day = int(t // minutes_per_day) + 1
        minutes = t % minutes_per_day
        h = work_start_hour + int(minutes // 60)
        m = int(minutes % 60)
        time_str = f"{h:02d}:{m:02d}"
        writer.writerow([day, time_str, event, 'Warning' if is_warning else 'Fault'])
print(f"报警日志已导出到 {csv_path}")

plt.show()