import json

# ==================== 1. 数据录入 ====================
# 当前机器人状态快照（示例数据，用于开发测试）
robot_state = {
    "power_on_time_h": 1283.35,     # 电源闭合时间 (小时)
    "servo_on_time_h": 512.74,      # 伺服闭合时间 (小时)
    "program_run_time_h": 23.21,    # 程序执行时间 (小时)
    "welding_time_h": 1.13,         # 焊接时间 (小时)
    "tcp_length_mm": 990.62,        # TCP 长度
    "tcp_x_mm": 752.84,
    "tcp_y_mm": 1.91,
    "tcp_z_mm": 643.79,
}

# 历史基准数据（用于对比）
baseline = {
    "welding_time_h": 1.08,
    "program_run_time_h": 22.62,
    "tcp_length_mm": 990.57,
    "tcp_x_mm": 752.80,
    "tcp_y_mm": 1.88,
    "tcp_z_mm": 643.83,
}

# 故障码映射（基于你提供的错误历史和警告）
fault_map = {
    "断弧": "焊接过程异常",
    "软限定错误": "轨迹/位置异常",
    "焊机通信异常": "设备通信故障",
    "安全支架动作": "安全事件",
    "检测到碰撞": "碰撞异常",
    "坐标变换错误": "逻辑/配置错误",
    "特异点错误": "运动学异常",
    "焊机通信中断": "设备通信故障",
    "粘丝": "焊接过程异常",
    "焊接电源被关闭": "电源异常",
    "电机过负载错误": "驱动系统异常",
    "电机电流失速": "伺服异常预警",
    "输入指令不正确": "程序/配置预警",
    "系统警报": "系统级预警",
}

# ==================== 2. 特征计算 ====================
# 2.1 焊接效率 = 焊接时间 / 程序执行时间
weld_efficiency = robot_state["welding_time_h"] / robot_state["program_run_time_h"]
baseline_efficiency = baseline["welding_time_h"] / baseline["program_run_time_h"]

# 2.2 TCP 位置偏差（欧氏距离）
tcp_delta = {
    "dx": robot_state["tcp_x_mm"] - baseline["tcp_x_mm"],
    "dy": robot_state["tcp_y_mm"] - baseline["tcp_y_mm"],
    "dz": robot_state["tcp_z_mm"] - baseline["tcp_z_mm"],
    "length_delta": robot_state["tcp_length_mm"] - baseline["tcp_length_mm"],
}
tcp_euclidean = (tcp_delta["dx"]**2 + tcp_delta["dy"]**2 + tcp_delta["dz"]**2) ** 0.5

# 2.3 健康度打分（0-100 分，分数越低表示越健康）
score = 100.0
warnings = []

# 焊接效率异常判断（示例：效率下降超过 1% 扣分）
eff_drop = baseline_efficiency - weld_efficiency
if eff_drop > 0.01:
    score -= 20
    warnings.append(f"焊接效率下降 {eff_drop:.3f} (可能空运行或通信中断)")

# TCP 位置偏差异常判断（示例：长度偏差超过 0.2mm 扣分）
if abs(tcp_delta["length_delta"]) > 0.2:
    score -= 30
    warnings.append(f"TCP 长度偏差 {tcp_delta['length_delta']:+.2f} mm (可能轨迹偏移)")

# 欧氏距离异常判断
if tcp_euclidean > 0.3:
    score -= 30
    warnings.append(f"TCP 欧氏距离偏差 {tcp_euclidean:.3f} mm (可能定位异常)")

# ==================== 3. 结果输出 ====================
print("========== 机器人异常检测与诊断引擎 ==========")
print(f"焊接效率: 当前 {weld_efficiency:.3f}, 基准 {baseline_efficiency:.3f}")
print(f"TCP 位置偏差: ΔX={tcp_delta['dx']:+.2f}mm, ΔY={tcp_delta['dy']:+.2f}mm, ΔZ={tcp_delta['dz']:+.2f}mm")
print(f"TCP 长度偏差: {tcp_delta['length_delta']:+.2f} mm")
print(f"TCP 欧氏距离偏差: {tcp_euclidean:.3f} mm")
print(f"健康分数: {score:.1f}/100")
if warnings:
    print("发现以下异常:")
    for w in warnings:
        print(f"  - {w}")
else:
    print("未发现明显异常。")

# ==================== 4. 保存结果 ====================
result = {
    "weld_efficiency": weld_efficiency,
    "baseline_efficiency": baseline_efficiency,
    "tcp_delta": tcp_delta,
    "tcp_euclidean": tcp_euclidean,
    "health_score": score,
    "warnings": warnings,
}
with open("D:/VisionBot/robot-anomaly-detection/src/anomaly_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n结果已保存到 anomaly_result.json")