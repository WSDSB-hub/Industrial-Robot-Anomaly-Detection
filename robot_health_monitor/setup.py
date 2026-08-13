from setuptools import setup
import os
from glob import glob

package_name = 'robot_health_monitor'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='your@email.com',
    description='ROS2 package for robot health monitoring',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            'state_publisher = robot_health_monitor.state_publisher_node:main',
            'anomaly_detector = robot_health_monitor.anomaly_detector_node:main',
            'alert_manager = robot_health_monitor.alert_manager_node:main',
        ],
    },
)
