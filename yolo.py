import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloDetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector_node')

        # Load YOLO model
        self.get_logger().info('Loading YOLO model...')
        self.model = YOLO('yolov8n.pt')

        self.bridge = CvBridge()

        # Subscribe to camera topic (e.g., /image_raw from v4l2_camera or Gazebo)
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        # Publisher for the annotated detection feed
        self.publisher = self.create_publisher(Image, '/yolo/detections_image', 10)
        self.get_logger().info('YOLO Detector Node initialized and listening on /image_raw')

    def image_callback(self, msg: Image):
        try:
            # Convert ROS 2 Image -> OpenCV format (BGR8)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Run inference
        results = self.model(cv_image, verbose=False)

        # Plot bounding boxes onto the frame
        annotated_image = results[0].plot()

        # Convert OpenCV format -> ROS 2 Image msg
        out_msg = self.bridge.cv2_to_imgmsg(annotated_image, encoding='bgr8')
        out_msg.header = msg.header

        self.publisher.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()