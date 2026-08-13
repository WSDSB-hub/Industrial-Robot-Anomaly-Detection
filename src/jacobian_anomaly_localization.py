import numpy as np
import matplotlib.pyplot as plt
import os
from dh_kinematics import get_tcp_position, numerical_jacobian, joint_contribution_from_tcp_error

def simulate_joint_anomaly(nominal_pose, faulty_joint_index, joint_error_deg):
    """Simulate an anomaly in one joint and compute the resulting TCP error."""
    faulty_pose = nominal_pose.copy()
    faulty_pose[faulty_joint_index] += joint_error_deg
    nominal_tcp = get_tcp_position(nominal_pose)
    faulty_tcp = get_tcp_position(faulty_pose)
    tcp_error = faulty_tcp - nominal_tcp
    return tcp_error

def main():
    nominal_pose = [15.0, 42.0, 25.0, 0.0, 15.0, 0.0]

    print("=" * 60)
    print("Joint-Level Anomaly Localization Demo")
    print("=" * 60)

    scenarios = [
        ("J3 Anomaly (+0.3 deg)", 2, 0.3),
        ("J5 Anomaly (+0.5 deg)", 4, 0.5),   # Increased from 0.2 to 0.5
        ("J2 Anomaly (+0.25 deg)", 1, 0.25),
    ]

    results = []
    for name, joint_idx, error_deg in scenarios:
        tcp_error = simulate_joint_anomaly(nominal_pose, joint_idx, error_deg)
        joint_errors, contribution = joint_contribution_from_tcp_error(nominal_pose, tcp_error)

        print(f"\n===== {name} =====")
        print(f"Injected TCP error: ({tcp_error[0]:+.3f}, {tcp_error[1]:+.3f}, {tcp_error[2]:+.3f}) mm")
        print("Recovered joint errors: ", end="")
        for i, e in enumerate(joint_errors):
            print(f"J{i+1}={e:+.4f} deg", end="  ")
        print()
        print("Contribution ratios: ", end="")
        for i, c in enumerate(contribution):
            print(f"J{i+1}={c*100:.1f}%", end="  ")
        print()
        print(f"Detected joint: J{np.argmax(contribution)+1} (injected: J{joint_idx+1})")

        results.append({
            'scenario': name,
            'injected_joint': joint_idx+1,
            'detected_joint': int(np.argmax(contribution))+1,
            'tcp_error': tcp_error,
            'contribution': contribution
        })

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for idx, result in enumerate(results):
        ax = axes[idx]
        joints = [f'J{i+1}' for i in range(6)]
        contrib = result['contribution'] * 100
        colors = ['#e74c3c' if j == result['detected_joint']-1 else '#3498db' for j in range(6)]
        ax.bar(joints, contrib, color=colors, edgecolor='black', linewidth=1)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Contribution (%)', fontsize=10)
        ax.set_title(f"{result['scenario']}\nInjected J{result['injected_joint']} -> Detected J{result['detected_joint']}",
                     fontsize=10, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    images_dir = 'D:/VisionBot/robot-anomaly-detection/images'
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'joint_anomaly_localization.png'), dpi=150, bbox_inches='tight')
    print(f"\nChart saved to {os.path.join(images_dir, 'joint_anomaly_localization.png')}")
    plt.show()

if __name__ == "__main__":
    main()