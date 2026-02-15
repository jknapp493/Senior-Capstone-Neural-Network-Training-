#!/usr/bin/env python3

import sys
print(">>> Using Python:", sys.executable)

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
from ultralytics import YOLO

class YoloTalker(Node):
    def __init__(self):
        super().__init__('yolo_talker')
        self.publisher_ = self.create_publisher(String, 'yolo_detections', 10)

        # Load YOLOv8n model
        self.model = YOLO("yolov8n.pt")

        # Open webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Webcam not found!")
            exit(1)

        # Timer callback to read frames
        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Failed to grab frame")
            return

        # Run YOLOv8 inference
        results = self.model(frame)

        # Collect detected class names
        detected_classes = set()
        for r in results:
            for c in r.boxes.cls:
                detected_classes.add(self.model.names[int(c)])

        if detected_classes:
            msg = String()
            msg.data = ", ".join(detected_classes)
            self.publisher_.publish(msg)
            self.get_logger().info(f"Published detections: {msg.data}")

    def destroy_node(self):
        self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = YoloTalker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
