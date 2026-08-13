import sys
sys.path.insert(0, 'D:/VisionBot/robot-anomaly-detection/src')
import numpy as np
import os
from dh_kinematics import get_tcp_position, numerical_jacobian

# ======================================================
# 真实现场数据验证
# 数据来源：天一焊接 FANUC M-20iD/25 现场记录
# 记录时间：某生产日
# ======================================================

# 正常原点姿态（早班8:15，机器人位于待机原点）
normal_angles = [11.42, -55.76, 121.08, -79.35, -68.24, 98.71]
normal_tcp = [1187.64, 298.37, 312.05]

# 报警时实时姿态（上午10:42，第三条搭接焊缝中段）
alarm_angles = [15.89, -49.23, 114.76, -88.61, -75.32, 112.47]
alarm_tcp = [1324.51, 376.28, 279.63]

# 焊枪工具标定长度
tool_length = 320.00

print("=" * 70)
print("Real Field Data Validation: FANUC M-20iD/25 Anomaly Event")
print("=" * 70)

# ======================================================
# Step 1: 用正常原点数据验证D-H模型并推导工具偏移
# ======================================================
print("\n--- Step 1: D-H Model Validation with Normal Pose ---")

flange_normal = get_tcp_position(normal_angles)
print(f"D-H model flange center at normal pose: ({flange_normal[0]:.2f}, {flange_normal[1]:.2f}, {flange_normal[2]:.2f}) mm")
print(f"Actual TCP at normal pose: ({normal_tcp[0]:.2f}, {normal_tcp[1]:.2f}, {normal_tcp[2]:.2f}) mm")

# 工具偏移向量 = 实际TCP - 法兰盘中心
tool_offset = np.array(normal_tcp) - np.array(flange_normal)
tool_offset_norm = np.linalg.norm(tool_offset)
print(f"Derived tool offset vector: ({tool_offset[0]:.2f}, {tool_offset[1]:.2f}, {tool_offset[2]:.2f}) mm")
print(f"Tool offset vector magnitude: {tool_offset_norm:.2f} mm (expected ~{tool_length} mm)")

# ======================================================
# Step 2: 用报警时数据验证D-H模型预测能力
# ======================================================
print("\n--- Step 2: D-H Model Prediction at Alarm Pose ---")

flange_alarm = get_tcp_position(alarm_angles)
predicted_alarm_tcp = np.array(flange_alarm) + tool_offset

print(f"Predicted TCP at alarm pose: ({predicted_alarm_tcp[0]:.2f}, {predicted_alarm_tcp[1]:.2f}, {predicted_alarm_tcp[2]:.2f}) mm")
print(f"Actual TCP at alarm pose: ({alarm_tcp[0]:.2f}, {alarm_tcp[1]:.2f}, {alarm_tcp[2]:.2f}) mm")

prediction_error = predicted_alarm_tcp - np.array(alarm_tcp)
prediction_error_norm = np.linalg.norm(prediction_error)
print(f"Prediction error: ({prediction_error[0]:+.2f}, {prediction_error[1]:+.2f}, {prediction_error[2]:+.2f}) mm")
print(f"Prediction error magnitude: {prediction_error_norm:.2f} mm")

if prediction_error_norm < 5.0:
    print("=> D-H model is VALID (error < 5 mm). Kinematic model is reliable.")
elif prediction_error_norm < 15.0:
    print("=> D-H model is APPROXIMATE (error 5-15 mm). May contain minor parameter inaccuracy.")
else:
    print("=> D-H model has SIGNIFICANT error (>15 mm). Parameters need revision.")

# ======================================================
# Step 3: 雅可比定位——从TCP偏差反推异常关节
# ======================================================
print("\n--- Step 3: Jacobian-Based Anomaly Joint Localization ---")

tcp_delta = np.array(alarm_tcp) - np.array(normal_tcp)
print(f"TCP deviation (alarm - normal): ({tcp_delta[0]:+.2f}, {tcp_delta[1]:+.2f}, {tcp_delta[2]:+.2f}) mm")

J = numerical_jacobian(normal_angles)
J_pinv = np.linalg.pinv(J)
joint_delta = J_pinv @ tcp_delta

abs_contrib = np.abs(joint_delta)
total = np.sum(abs_contrib)
if total > 0:
    contribution = abs_contrib / total
else:
    contribution = np.zeros(6)

print("\nRecovered joint deviations from TCP error:")
for i, d in enumerate(joint_delta):
    print(f"  J{i+1}: {d:+.4f} deg")

print("\nJoint contribution ratios:")
for i, c in enumerate(contribution):
    print(f"  J{i+1}: {c*100:.1f}%")

detected_joint = int(np.argmax(contribution)) + 1
print(f"\nDetected anomalous joint: J{detected_joint}")
print(f"Actual root cause: J6 (SRVO-046 J6 axis overload)")

if detected_joint == 6:
    print("=> LOCALIZATION SUCCESS: Detected joint matches real fault.")
else:
    print(f"=> LOCALIZATION MISMATCH: Detected J{detected_joint}, actual J6.")
    print(f"   Note: This is expected because the TCP deviation includes")
    print(f"   normal path motion, not pure joint anomaly.")

# ======================================================
# Step 4: 报警时的关键判断——D-H预测误差
# ======================================================
print("\n--- Step 4: D-H Prediction Error Analysis ---")
print(f"If the D-H model is accurate and the robot is operating normally,")
print(f"the prediction error at the alarm pose should be small (<5 mm).")
print(f"Actual prediction error: {prediction_error_norm:.2f} mm")

if prediction_error_norm < 5.0:
    print("\nInterpretation: The robot's TCP position was consistent with its joint angles.")
    print("The J6 overload was caused by cable resistance (torque overload), not by position deviation.")
    print("This is consistent with the field diagnosis: cable drag + incorrect inertia parameter.")
elif prediction_error_norm >= 5.0:
    print("\nInterpretation: The prediction error suggests possible mechanical deviation or")
    print("D-H parameter inaccuracy. The J6 cable drag may have pulled the TCP off its")
    print("commanded trajectory before the overload protection triggered.")