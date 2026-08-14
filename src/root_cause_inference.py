import numpy as np
import json
import os

# ======================================================
# 故障诊断因果推理引擎
# 基于专家规则的根因评分系统
# ======================================================

# 根因列表及其特征描述
root_causes = {
    "cable_drag": {
        "name": "Cable Drag (线缆卡滞)",
        "description": "焊枪线缆在关节运动时产生物理阻碍，导致瞬时过载",
        "features": {
            "load_rate_high": 1.0,      # 负载率极高
            "current_high": 0.8,        # 电流较高
            "temp_moderate": 0.6,       # 温度中等
            "tcp_normal": 0.8,          # 位置正常
        }
    },
    "inertia_misconfig": {
        "name": "Inertia Parameter Misconfiguration (惯量参数错误)",
        "description": "工具惯量参数设置偏小，导致伺服系统对负载估计不足",
        "features": {
            "load_rate_high": 0.8,
            "current_high": 0.6,
            "temp_moderate": 0.5,
            "tcp_normal": 0.8,
        }
    },
    "reducer_wear": {
        "name": "Reducer Wear (减速机磨损)",
        "description": "减速机长期磨损导致传动效率下降，负载率缓慢上升",
        "features": {
            "load_rate_high": 0.5,
            "current_high": 0.5,
            "temp_high": 0.9,           # 温度会显著升高
            "tcp_normal": 0.6,
        }
    },
    "bearing_damage": {
        "name": "Bearing Damage (轴承损坏)",
        "description": "轴承损坏导致摩擦增大，温度异常升高并伴随振动",
        "features": {
            "load_rate_high": 0.4,
            "current_high": 0.4,
            "temp_high": 0.95,
            "tcp_normal": 0.5,
        }
    },
    "servo_tuning": {
        "name": "Servo Parameter Mistuning (伺服参数不当)",
        "description": "伺服增益设置不当，导致电流波动大，但温度和负载相对正常",
        "features": {
            "load_rate_high": 0.2,
            "current_high": 0.5,
            "temp_low": 0.8,
            "tcp_normal": 0.8,
        }
    }
}

def compute_feature_scores(load_rate, current, temp, tcp_error):
    """根据实际数据计算特征得分"""
    # 负载率得分
    if load_rate > 100:
        load_high = 1.0
    elif load_rate > 60:
        load_high = 0.7
    elif load_rate > 30:
        load_high = 0.4
    else:
        load_high = 0.1

    # 电流得分
    if current > 2.0:
        current_high = 1.0
    elif current > 1.5:
        current_high = 0.7
    elif current > 1.0:
        current_high = 0.4
    else:
        current_high = 0.1

    # 温度得分
    if temp > 70:
        temp_high = 1.0
        temp_low = 0.0
    elif temp > 60:
        temp_high = 0.7
        temp_low = 0.2
    elif temp > 50:
        temp_high = 0.4
        temp_low = 0.5
    else:
        temp_high = 0.1
        temp_low = 0.9

    # TCP偏差得分（正常表示偏差小）
    if tcp_error < 0.5:
        tcp_normal = 1.0
    elif tcp_error < 2.0:
        tcp_normal = 0.7
    elif tcp_error < 5.0:
        tcp_normal = 0.4
    else:
        tcp_normal = 0.1

    return {
        "load_rate_high": load_high,
        "current_high": current_high,
        "temp_high": temp_high,
        "temp_low": temp_low,
        "tcp_normal": tcp_normal
    }

def compute_root_cause_scores(features):
    """根据特征匹配度计算每个根因的得分"""
    scores = {}
    for cause_id, cause in root_causes.items():
        score = 0.0
        for feat, weight in cause["features"].items():
            if feat in features:
                # 权重与特征值相乘，然后累加
                score += weight * features[feat]
        scores[cause_id] = score
    # 按得分降序排序
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_scores

def main():
    print("=" * 70)
    print("故障诊断因果推理引擎")
    print("=" * 70)

    # 使用J6过载报警时的真实数据
    # J6负载率127.4%，电流2.31A，温度71.8℃，TCP偏差0.5mm（假设）
    load_rate = 127.4
    current = 2.31
    temp = 71.8
    tcp_error = 0.5

    features = compute_feature_scores(load_rate, current, temp, tcp_error)

    print("\n实际信号特征:")
    print(f"  负载率: {load_rate}%")
    print(f"  电流: {current}A")
    print(f"  温度: {temp}℃")
    print(f"  TCP偏差: {tcp_error}mm")
    print(f"\n特征得分:")
    for k, v in features.items():
        print(f"  {k}: {v:.2f}")

    scores = compute_root_cause_scores(features)

    print(f"\n根因可能性排序:")
    for idx, (cause_id, score) in enumerate(scores, 1):
        cause = root_causes[cause_id]
        print(f"  {idx}. {cause['name']} (得分: {score:.3f})")
        print(f"     {cause['description']}")

    # 保存结果
    result = {
        "input_signals": {
            "load_rate": load_rate,
            "current": current,
            "temp": temp,
            "tcp_error": tcp_error
        },
        "feature_scores": features,
        "root_cause_ranking": [{"cause": root_causes[cid]["name"], "score": s} for cid, s in scores]
    }

    with open('D:/VisionBot/robot-anomaly-detection/docs/root_cause_inference_results.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到 docs/root_cause_inference_results.json")

if __name__ == "__main__":
    main()