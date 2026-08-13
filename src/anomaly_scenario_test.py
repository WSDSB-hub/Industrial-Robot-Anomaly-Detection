import json

# 从刚才的引擎中提取核心逻辑，做成一个函数
def evaluate_robot_state(name, robot_state, baseline, fault_codes=None):
    weld_eff = robot_state["welding_time_h"] / robot_state["program_run_time_h"]
    base_eff = baseline["welding_time_h"] / baseline["program_run_time_h"]
    eff_drop = base_eff - weld_eff

    dx = robot_state["tcp_x_mm"] - baseline["tcp_x_mm"]
    dy = robot_state["tcp_y_mm"] - baseline["tcp_y_mm"]
    dz = robot_state["tcp_z_mm"] - baseline["tcp_z_mm"]
    euclidean = (dx**2 + dy**2 + dz**2) ** 0.5
    length_delta = robot_state["tcp_length_mm"] - baseline["tcp_length_mm"]

    score = 100.0
    warnings = []

    if eff_drop > 0.01:
        score -= 20
        warnings.append(f"焊接效率下降 {eff_drop:.3f}")

    if abs(length_delta) > 0.2:
        score -= 30
        warnings.append(f"TCP 长度偏差 {length_delta:+.2f} mm")

    if euclidean > 0.3:
        score -= 30
        warnings.append(f"TCP 欧氏距离偏差 {euclidean:.3f} mm")

    if fault_codes:
        for code in fault_codes:
            if code in ["断弧", "粘丝", "焊机通信异常", "焊机通信中断", "电机过负载错误"]:
                score -= 15
                warnings.append(f"检测到故障码: {code}")

    print(f"\n===== 场景: {name} =====")
    print(f"焊接效率: {weld_eff:.3f} (基准 {base_eff:.3f})")
    print(f"TCP偏差: ΔX={dx:+.2f}, ΔY={dy:+.2f}, ΔZ={dz:+.2f}, 欧氏={euclidean:.3f} mm")
    print(f"健康分数: {score:.1f}/100")
    if warnings:
        for w in warnings:
            print(f"  ⚠ {w}")
    else:
        print("  未发现异常")
    return score, warnings


# 基准数据
baseline = {
    "welding_time_h": 1.08,
    "program_run_time_h": 22.62,
    "tcp_length_mm": 990.57,
    "tcp_x_mm": 752.80,
    "tcp_y_mm": 1.88,
    "tcp_z_mm": 643.83,
}

# 场景1：正常状态（和刚才一样）
normal_state = {
    "welding_time_h": 1.13,
    "program_run_time_h": 23.21,
    "tcp_length_mm": 990.62,
    "tcp_x_mm": 752.84,
    "tcp_y_mm": 1.91,
    "tcp_z_mm": 643.79,
}

# 场景2：焊接效率显著下降（可能焊机通信中断，大量空运行）
eff_drop_state = {
    "welding_time_h": 0.75,
    "program_run_time_h": 24.5,
    "tcp_length_mm": 990.60,
    "tcp_x_mm": 752.82,
    "tcp_y_mm": 1.90,
    "tcp_z_mm": 643.80,
}
eff_drop_codes = ["焊机通信中断"]

# 场景3：TCP位置明显偏移（可能软限位/碰撞后未复位）
tcp_offset_state = {
    "welding_time_h": 1.12,
    "program_run_time_h": 23.0,
    "tcp_length_mm": 991.80,   # 偏差超过1mm
    "tcp_x_mm": 753.50,
    "tcp_y_mm": 2.30,
    "tcp_z_mm": 642.90,
}
tcp_offset_codes = ["软限定错误"]

# 场景4：同时出现效率下降和位置偏移（综合异常）
combined_state = {
    "welding_time_h": 0.6,
    "program_run_time_h": 25.0,
    "tcp_length_mm": 991.20,
    "tcp_x_mm": 753.20,
    "tcp_y_mm": 2.10,
    "tcp_z_mm": 642.70,
}
combined_codes = ["焊机通信异常", "软限定错误", "检测到碰撞"]

# 执行四个场景
results = {}
results["normal"] = evaluate_robot_state("正常状态", normal_state, baseline)
results["efficiency_drop"] = evaluate_robot_state("焊接效率下降", eff_drop_state, baseline, eff_drop_codes)
results["tcp_offset"] = evaluate_robot_state("TCP位置偏移", tcp_offset_state, baseline, tcp_offset_codes)
results["combined"] = evaluate_robot_state("综合异常", combined_state, baseline, combined_codes)

# 保存结果
with open("D:/VisionBot/robot-anomaly-detection/src/scenario_test_results.json", "w", encoding="utf-8") as f:
    json.dump({k: {"score": v[0], "warnings": v[1]} for k, v in results.items()}, f, ensure_ascii=False, indent=2)

print("\n\n所有场景测试完成，结果已保存。")