from transformers import pipeline
from PIL import Image
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as rosimg
from std_msgs.msg import String
import cv_bridge

class ros2aiNode(Node):

    def __init__(self):
        self.image_to_text = pipeline("image-to-text", model="nlpconnect/vit-gpt2-image-captioning")
        super().__init__('vit_gpt2_image_captioning')
        self.subscription = self.create_subscription(
            rosimg,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.subscription
        self.bridge = cv_bridge.CvBridge()
        self.publisher = self.create_publisher(String, 'image_caption', 0)

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        pil_image = Image.fromarray(np.uint8(cv_image)).convert('RGB')
        out = self.image_to_text(pil_image)
        text_output = out[0]['generated_text'] if out else "No caption generated"
        self.get_logger().info(text_output)
        self.publisher.publish(String(data=text_output))


def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = ros2aiNode()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()