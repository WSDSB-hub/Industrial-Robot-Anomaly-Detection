import numpy as np

# ======================================================
# 第一次读数（真实数据）
# 时间：假设为 2025-01-01
# ======================================================
data1 = {
    'timestamp': '2025-01-01 10:00:00',
    'X': 752.8,
    'Y': 1.88,
    'Z': 643.83,
    'U': 179.99,
    'V': 39.84,
    'W': 180.0,
    'length': 990.57
}

# ======================================================
# 第二次读数（示例数据，仅用于测试分析流程）
# 时间：假设为 2025-01-08
# ======================================================
data2 = {
    'timestamp': '2025-01-08 14:00:00',
    'X': 752.84,
    'Y': 1.91,
    'Z': 643.79,
    'U': 179.99,
    'V': 39.85,
    'W': 179.99,
    'length': 990.62
}

# ======================================================
# 计算位置偏差（直交动作）
# ======================================================
pos_diff = {
    'X': data2['X'] - data1['X'],
    'Y': data2['Y'] - data1['Y'],
    'Z': data2['Z'] - data1['Z'],
    'length': data2['length'] - data1['length']
}

# ======================================================
# 计算姿态偏差（旋转动作）
# ======================================================
ori_diff = {
    'U': data2['U'] - data1['U'],
    'V': data2['V'] - data1['V'],
    'W': data2['W'] - data1['W']
}

# ======================================================
# 设定阈值（根据你之前提供的现场标准）
# 直交动作：0.2 mm
# 旋转动作：0.1°
# ======================================================
pos_threshold = 0.2   # mm
ori_threshold = 0.1   # degree

# ======================================================
# 输出结果
# ======================================================
print("=== TCP 位姿偏差分析 ===")
print(f"第一次读数: {data1}")
print(f"第二次读数: {data2}")
print()

print("--- 位置偏差 (mm) ---")
for key, val in pos_diff.items():
    status = "正常" if abs(val) <= pos_threshold else "异常"
    print(f"{key}: {val:+.2f} mm  [{status}]")

print()
print("--- 姿态偏差 (deg) ---")
for key, val in ori_diff.items():
    status = "正常" if abs(val) <= ori_threshold else "异常"
    print(f"{key}: {val:+.2f} deg  [{status}]")

# ======================================================
# 整体判断
# ======================================================
pos_ok = all(abs(v) <= pos_threshold for v in pos_diff.values())
ori_ok = all(abs(v) <= ori_threshold for v in ori_diff.values())

print()
if pos_ok and ori_ok:
    print("结论：TCP 位姿在正常范围内，机器人重复定位精度良好。")
else:
    print("结论：检测到 TCP 位姿偏差超出阈值，需进一步排查。")

# ======================================================
# 保存结果到文件
# ======================================================
import os

result_dir = 'D:/VisionBot/robot-anomaly-detection/src'
os.makedirs(result_dir, exist_ok=True)
result_path = os.path.join(result_dir, 'tcp_deviation_result.txt')

with open(result_path, 'w', encoding='utf-8') as f:
    f.write("=== TCP 位姿偏差分析 ===\n")
    f.write(f"第一次读数: {data1}\n")
    f.write(f"第二次读数: {data2}\n\n")
    f.write("--- 位置偏差 (mm) ---\n")
    for key, val in pos_diff.items():
        status = "正常" if abs(val) <= pos_threshold else "异常"
        f.write(f"{key}: {val:+.2f} mm  [{status}]\n")
    f.write("\n--- 姿态偏差 (deg) ---\n")
    for key, val in ori_diff.items():
        status = "正常" if abs(val) <= ori_threshold else "异常"
        f.write(f"{key}: {val:+.2f} deg  [{status}]\n")
    f.write("\n")
    if pos_ok and ori_ok:
        f.write("结论：TCP 位姿在正常范围内，机器人重复定位精度良好。\n")
    else:
        f.write("结论：检测到 TCP 位姿偏差超出阈值，需进一步排查。\n")

print("\n结果已保存到 tcp_deviation_result.txt")