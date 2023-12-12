# Copyright 2023 Taiga Takano
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image as rosimg
from std_msgs.msg import String
import cv_bridge

class ros2aiNode(Node):

    def __init__(self):
        self.processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        self.model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50", torch_dtype=torch.float16).to("cuda")

        super().__init__('resnet_50')
        self.subscription = self.create_subscription(
            rosimg,
            '/camera/color/image_raw',
            self.listener_callback,0)
        self.subscription
        self.bridge = cv_bridge.CvBridge()
        self.publisher = self.create_publisher(String, 'image_caption', 0)

    def listener_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        inputs = self.processor(cv_image, return_tensors="pt").to("cuda", torch.float16)
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        caption = self.model.config.id2label[predicted_label].replace(',', ' ')
        self.get_logger().info(caption)
        self.publisher.publish(String(data=caption))

def main(args=None):
    rclpy.init(args=args)
    minimal_subscriber = ros2aiNode()
    rclpy.spin(minimal_subscriber)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
