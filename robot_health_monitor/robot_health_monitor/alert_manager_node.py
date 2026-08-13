#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, String

class AlertManager(Node):
    def __init__(self):
        super().__init__('alert_manager')
        self.subscription = self.create_subscription(
            Float32, '/robot_health/score', self.score_callback, 10)
        self.publisher = self.create_publisher(String, '/robot_health/alert', 10)
        self.alert_threshold = 60.0

    def score_callback(self, msg):
        score = msg.data
        if score < self.alert_threshold:
            alert_msg = String()
            alert_msg.data = f'ALERT: Health score {score:.1f} below threshold'
            self.publisher.publish(alert_msg)
            self.get_logger().error(alert_msg.data)
        else:
            self.get_logger().info(f'Score {score:.1f} is normal')

def main(args=None):
    rclpy.init(args=args)
    node = AlertManager()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
