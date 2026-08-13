#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

class AnomalyDetector(Node):
    def __init__(self):
        super().__init__('anomaly_detector')
        self.subscription = self.create_subscription(
            Float32, '/robot_health/state', self.state_callback, 10)
        self.publisher = self.create_publisher(Float32, '/robot_health/score', 10)
        self.threshold = 0.03  # 效率低于此值视为异常

    def state_callback(self, msg):
        efficiency = msg.data
        # 简单的规则：效率低于阈值扣分
        if efficiency < self.threshold:
            score = 40.0
            self.get_logger().warn(f'Low efficiency detected: {efficiency:.3f}, score={score}')
        else:
            score = 100.0
            self.get_logger().info(f'Normal efficiency: {efficiency:.3f}, score={score}')
        out_msg = Float32()
        out_msg.data = score
        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = AnomalyDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
