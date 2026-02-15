#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class YoloListener(Node):
    def __init__(self):
        super().__init__('yolo_listener')
        self.subscription = self.create_subscription(
            String,
            'yolo_detections',
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        self.get_logger().info(f"I heard: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloListener()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
