#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import random

class StatePublisher(Node):
    def __init__(self):
        super().__init__('state_publisher')
        self.publisher = self.create_publisher(Float32, '/robot_health/state', 10)
        self.timer = self.create_timer(1.0, self.publish_state)  # 1Hz
        self.weld_efficiency = 0.048
        self.tcp_deviation = 0.05
        self.fault_code_present = 0

    def publish_state(self):
        # 模拟状态：大部分正常，偶尔异常
        if random.random() < 0.1:  # 10%概率注入异常
            self.weld_efficiency = 0.020
            self.tcp_deviation = 0.8
            self.fault_code_present = 1
        else:
            self.weld_efficiency = 0.048
            self.tcp_deviation = 0.05
            self.fault_code_present = 0

        # 发布健康分数基准信号（此处简化为效率值）
        msg = Float32()
        msg.data = self.weld_efficiency
        self.publisher.publish(msg)
        self.get_logger().info(f'Published efficiency: {msg.data:.3f}')

def main(args=None):
    rclpy.init(args=args)
    node = StatePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
