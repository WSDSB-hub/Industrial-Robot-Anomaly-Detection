import numpy as np
import os

# ======================================================
# FANUC M-20iD/25 D-H参数（基于公开资料）
# 每个关节包含: [a(mm), alpha(deg), d(mm), theta_offset(deg)]
# 注意：theta_offset用于处理零位姿态的初始角度
# 精确值需从FANUC官方技术手册获取
# ======================================================
DH_PARAMS = [
    # [a, alpha, d, theta_offset]
    [150, -90, 670, 0],    # J1 底座旋转轴
    [780,   0,   0, -90],  # J2 大臂俯仰轴
    [200, -90,   0, 0],    # J3 小臂俯仰轴
    [0,    90, 735, 0],    # J4 手腕旋转轴
    [0,   -90,   0, 0],    # J5 手腕摆动轴
    [0,     0, 100, 0],    # J6 手腕末端旋转轴
]

def dh_transform(a, alpha, d, theta):
    """计算单个关节的D-H变换矩阵
    a: 连杆长度(mm)
    alpha: 扭转角(deg)
    d: 偏移量(mm)
    theta: 关节角(deg)
    """
    alpha_rad = np.deg2rad(alpha)
    theta_rad = np.deg2rad(theta)

    T = np.array([
        [np.cos(theta_rad), -np.sin(theta_rad)*np.cos(alpha_rad),  np.sin(theta_rad)*np.sin(alpha_rad), a*np.cos(theta_rad)],
        [np.sin(theta_rad),  np.cos(theta_rad)*np.cos(alpha_rad), -np.cos(theta_rad)*np.sin(alpha_rad), a*np.sin(theta_rad)],
        [0,                 np.sin(alpha_rad),                    np.cos(alpha_rad),                    d],
        [0,                 0,                                    0,                                    1]
    ])
    return T

def forward_kinematics(joint_angles_deg):
    """正向运动学：给定六个关节角，计算TCP位姿
    joint_angles_deg: 六维数组，单位度（示教器显示值）
    返回: 4x4齐次变换矩阵
    """
    T = np.eye(4)
    for i, theta_display in enumerate(joint_angles_deg):
        a = DH_PARAMS[i][0]
        alpha = DH_PARAMS[i][1]
        d = DH_PARAMS[i][2]
        theta_offset = DH_PARAMS[i][3]
        # 实际物理关节角 = 示教器显示值 + 零位偏移
        theta_actual = theta_display + theta_offset
        T_i = dh_transform(a, alpha, d, theta_actual)
        T = T @ T_i
    return T

def get_tcp_position(joint_angles_deg):
    """获取TCP位置（X, Y, Z）"""
    T = forward_kinematics(joint_angles_deg)
    return T[0:3, 3]

def numerical_jacobian(joint_angles_deg, delta=0.01):
    """数值法计算雅可比矩阵（仅位置部分）
    雅可比矩阵将关节速度映射为TCP末端速度
    """
    joint_angles = np.array(joint_angles_deg, dtype=float)
    base_pos = get_tcp_position(joint_angles)
    J = np.zeros((3, 6))
    for j in range(6):
        joint_angles_perturbed = joint_angles.copy()
        joint_angles_perturbed[j] += delta
        perturbed_pos = get_tcp_position(joint_angles_perturbed)
        J[:, j] = (perturbed_pos - base_pos) / delta
    return J

def joint_contribution_from_tcp_error(nominal_joint_angles, tcp_error_mm):
    """从TCP位置偏差反推各关节贡献度
    nominal_joint_angles: 名义关节角（正常状态，示教器显示值）
    tcp_error_mm: TCP位置偏差 [dx, dy, dz]，单位mm
    返回: 各关节的偏差贡献(deg)和贡献率(%)
    """
    J = numerical_jacobian(nominal_joint_angles)
    J_pinv = np.linalg.pinv(J)  # 伪逆
    joint_errors_deg = J_pinv @ tcp_error_mm
    # 注意：这里的偏差已经是真实关节角偏差，不需要再转换
    # 但为了统一，我们仍然以度数表示
    abs_contrib = np.abs(joint_errors_deg)
    total = np.sum(abs_contrib)
    if total > 0:
        contribution_ratio = abs_contrib / total
    else:
        contribution_ratio = np.zeros(6)
    return joint_errors_deg, contribution_ratio

# ======================================================
# 测试：验证正向运动学的正确性
# ======================================================
if __name__ == "__main__":
    print("=" * 60)
    print("FANUC M-20iD/25 正向运动学测试（修正版）")
    print("=" * 60)

    # 零位姿态（示教器显示值全部为0）
    zero_pose = [0, 0, 0, 0, 0, 0]
    tcp_zero = get_tcp_position(zero_pose)
    print(f"\n零位姿态 TCP位置: ({tcp_zero[0]:.2f}, {tcp_zero[1]:.2f}, {tcp_zero[2]:.2f}) mm")
    print(f"预期值范围: Z轴应为正，表示机器人向上伸展")

    # 单轴运动测试：只转动J1
    pose_j1 = [30, 0, 0, 0, 0, 0]
    tcp_j1 = get_tcp_position(pose_j1)
    print(f"\nJ1=30°时 TCP位置: ({tcp_j1[0]:.2f}, {tcp_j1[1]:.2f}, {tcp_j1[2]:.2f}) mm")
    print(f"预期：X/Y应改变，Z应基本不变")

    # 单轴运动测试：只转动J2
    pose_j2 = [0, 30, 0, 0, 0, 0]
    tcp_j2 = get_tcp_position(pose_j2)
    print(f"\nJ2=30°时 TCP位置: ({tcp_j2[0]:.2f}, {tcp_j2[1]:.2f}, {tcp_j2[2]:.2f}) mm")
    print(f"预期：X/Z应改变，Y应基本不变")