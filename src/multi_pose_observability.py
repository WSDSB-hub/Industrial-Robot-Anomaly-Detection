import numpy as np
import matplotlib.pyplot as plt
import os
from dh_kinematics import get_tcp_position, numerical_jacobian

# ======================================================
# Multi-Pose Observability Enhancement
# 通过多个名义姿态联合观测，增强对弱关节的异常定位能力
# ======================================================

def stack_jacobians(pose_list):
    """
    将多个姿态下的雅可比矩阵堆叠，形成扩展雅可比矩阵。
    pose_list: 多个名义姿态（六轴角度）
    """
    J_list = []
    for pose in pose_list:
        J = numerical_jacobian(pose)
        J_list.append(J)
    return np.vstack(J_list)  # 垂直堆叠，形状 (3*N, 6)

def multi_pose_pseudoinverse(nominal_poses, tcp_errors):
    """
    多姿态伪逆定位
    nominal_poses: 多个名义姿态列表
    tcp_errors: 每个姿态对应的TCP偏差列表（形状相同）
    """
    J_stacked = stack_jacobians(nominal_poses)
    errors_stacked = np.hstack(tcp_errors)  # 拼接所有误差向量
    J_pinv = np.linalg.pinv(J_stacked)
    joint_delta = J_pinv @ errors_stacked
    abs_contrib = np.abs(joint_delta)
    total = np.sum(abs_contrib)
    if total > 0:
        contribution = abs_contrib / total
    else:
        contribution = np.zeros(6)
    return joint_delta, contribution

def simulate_pose_error(pose, faulty_joint, error_deg):
    """模拟某个关节在某个姿态下产生异常后的TCP偏差"""
    faulty_pose = pose.copy()
    faulty_pose[faulty_joint] += error_deg
    nominal_tcp = get_tcp_position(pose)
    faulty_tcp = get_tcp_position(faulty_pose)
    return faulty_tcp - nominal_tcp

def main():
    print("=" * 70)
    print("Multi-Pose Observability Analysis for Weak Joint Localization")
    print("=" * 70)

    # 定义5个不同的名义姿态（模拟机器人处于不同的焊接工位）
    nominal_poses = [
        [15.0, 42.0, 25.0, 0.0, 15.0, 0.0],       # Pose 1
        [30.0, 35.0, 18.0, 10.0, 5.0, 20.0],       # Pose 2
        [45.0, 20.0, 30.0, -10.0, -5.0, -20.0],    # Pose 3
        [10.0, 55.0, 40.0, 15.0, 25.0, 30.0],      # Pose 4
        [25.0, 10.0, 15.0, -5.0, -15.0, 10.0],     # Pose 5
    ]

    # 模拟J5关节异常 (+0.3°)
    faulty_joint = 4  # J5 (zero-indexed)
    error_deg = 0.3

    # 生成每个姿态下的TCP误差
    tcp_errors = []
    for pose in nominal_poses:
        err = simulate_pose_error(pose, faulty_joint, error_deg)
        tcp_errors.append(err)

    # ---- 单姿态伪逆定位（使用第一个姿态）----
    print("\n--- Single-Pose Pseudoinverse (Pose 1) ---")
    J_single = numerical_jacobian(nominal_poses[0])
    J_single_pinv = np.linalg.pinv(J_single)
    joint_delta_single = J_single_pinv @ tcp_errors[0]
    abs_single = np.abs(joint_delta_single)
    total_single = np.sum(abs_single)
    contrib_single = abs_single / total_single if total_single > 0 else np.zeros(6)
    detected_single = int(np.argmax(contrib_single)) + 1
    print(f"Recovered joint errors: ", end="")
    for i, d in enumerate(joint_delta_single):
        print(f"J{i+1}={d:+.4f} deg", end="  ")
    print()
    print(f"Contribution ratios: ", end="")
    for i, c in enumerate(contrib_single):
        print(f"J{i+1}={c*100:.1f}%", end="  ")
    print()
    print(f"Detected joint: J{detected_single} (injected: J{faulty_joint+1})")
    if detected_single == faulty_joint + 1:
        print("=> Success")
    else:
        print("=> Misclassification (expected: J5, detected: J%d)" % detected_single)

    # ---- 多姿态伪逆定位 ----
    print("\n--- Multi-Pose Pseudoinverse (5 poses combined) ---")
    joint_delta_multi, contrib_multi = multi_pose_pseudoinverse(nominal_poses, tcp_errors)
    detected_multi = int(np.argmax(contrib_multi)) + 1
    print(f"Recovered joint errors: ", end="")
    for i, d in enumerate(joint_delta_multi):
        print(f"J{i+1}={d:+.4f} deg", end="  ")
    print()
    print(f"Contribution ratios: ", end="")
    for i, c in enumerate(contrib_multi):
        print(f"J{i+1}={c*100:.1f}%", end="  ")
    print()
    print(f"Detected joint: J{detected_multi} (injected: J{faulty_joint+1})")
    if detected_multi == faulty_joint + 1:
        print("=> Success: multi-pose observability resolves the weak-joint ambiguity")
    else:
        print("=> Still misclassified. More poses or different configurations may be needed.")

    # ---- 可视化对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    joints = [f'J{i+1}' for i in range(6)]

    # 单姿态结果
    colors_single = ['#e74c3c' if j == detected_single-1 else '#3498db' for j in range(6)]
    axes[0].bar(joints, contrib_single*100, color=colors_single, edgecolor='black', linewidth=1)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel('Contribution (%)')
    axes[0].set_title(f'Single-Pose: Detected J{detected_single}\n(Injected J{faulty_joint+1})')
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')

    # 多姿态结果
    colors_multi = ['#e74c3c' if j == detected_multi-1 else '#3498db' for j in range(6)]
    axes[1].bar(joints, contrib_multi*100, color=colors_multi, edgecolor='black', linewidth=1)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel('Contribution (%)')
    axes[1].set_title(f'Multi-Pose: Detected J{detected_multi}\n(Injected J{faulty_joint+1})')
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'multi_pose_observability_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved to {os.path.join(images_dir, 'multi_pose_observability_comparison.png')}")
    plt.show()

if __name__ == "__main__":
    main()