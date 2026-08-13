from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='robot_health_monitor',
            executable='state_publisher',
            name='state_publisher',
        ),
        Node(
            package='robot_health_monitor',
            executable='anomaly_detector',
            name='anomaly_detector',
        ),
        Node(
            package='robot_health_monitor',
            executable='alert_manager',
            name='alert_manager',
        ),
    ])
