import numpy as np
import matplotlib.pyplot as plt
import os
from dh_kinematics import get_tcp_position, numerical_jacobian

def dls_joint_contribution(nominal_pose, tcp_error_mm, damping_lambda=0.05):
    """
    阻尼最小二乘法（Damped Least Squares）反推关节偏差
    相比伪逆，增加正则化项，避免误差过度集中在高灵敏度关节。
    """
    J = numerical_jacobian(nominal_pose)
    # DLS公式：delta_theta = J^T (J J^T + lambda^2 I)^-1 delta_x
    JJt = J @ J.T
    reg = damping_lambda**2 * np.eye(3)
    delta_theta = J.T @ np.linalg.inv(JJt + reg) @ tcp_error_mm
    abs_contrib = np.abs(delta_theta)
    total = np.sum(abs_contrib)
    if total > 0:
        contribution = abs_contrib / total
    else:
        contribution = np.zeros(6)
    return delta_theta, contribution

def main():
    nominal_pose = [15.0, 42.0, 25.0, 0.0, 15.0, 0.0]

    scenarios = [
        ("J3 Anomaly (+0.3 deg)", 2, 0.3),
        ("J5 Anomaly (+0.5 deg)", 4, 0.5),
        ("J2 Anomaly (+0.25 deg)", 1, 0.25),
    ]

    print("=" * 60)
    print("DLS Joint-Level Anomaly Localization")
    print("=" * 60)

    results = []
    for name, joint_idx, error_deg in scenarios:
        # simulate anomaly
        nominal_tcp = get_tcp_position(nominal_pose)
        faulty_pose = nominal_pose.copy()
        faulty_pose[joint_idx] += error_deg
        faulty_tcp = get_tcp_position(faulty_pose)
        tcp_error = faulty_tcp - nominal_tcp

        delta_theta, contribution = dls_joint_contribution(nominal_pose, tcp_error)

        print(f"\n===== {name} =====")
        print(f"Recovered joint errors: ", end="")
        for i, e in enumerate(delta_theta):
            print(f"J{i+1}={e:+.4f} deg", end="  ")
        print()
        print(f"Contribution ratios: ", end="")
        for i, c in enumerate(contribution):
            print(f"J{i+1}={c*100:.1f}%", end="  ")
        print()
        print(f"Detected joint: J{np.argmax(contribution)+1} (injected: J{joint_idx+1})")

        results.append({'scenario': name, 'detected': int(np.argmax(contribution))+1, 'injected': joint_idx+1, 'contribution': contribution})

    # visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, r in enumerate(results):
        ax = axes[idx]
        joints = [f'J{i+1}' for i in range(6)]
        contrib = r['contribution'] * 100
        colors = ['#e74c3c' if j == r['detected']-1 else '#3498db' for j in range(6)]
        ax.bar(joints, contrib, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_title(f"{r['scenario']}\nInjected J{r['injected']} -> Detected J{r['detected']} (DLS)", fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'dls_joint_localization.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved to {os.path.join(images_dir, 'dls_joint_localization.png')}")
    plt.show()

if __name__ == "__main__":
    main()