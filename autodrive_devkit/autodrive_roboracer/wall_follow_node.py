"""ROS 2 entrypoint for the LiDAR wall-following controller."""
import time
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32
from autodrive_roboracer.wall_core import Config, WallFollower


class WallFollowNode(Node):
    def __init__(self):
        super().__init__("wall_follow_node")
        defaults = Config()
        for name in defaults.__dataclass_fields__:
            self.declare_parameter(name, getattr(defaults, name))
        cfg = Config(**{name: self.get_parameter(name).value
                        for name in defaults.__dataclass_fields__})
        self.controller = WallFollower(cfg)
        self.steer_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.scan_sub = self.create_subscription(
            LaserScan, "/autodrive/roboracer_1/lidar", self.on_scan, qos)
        self.last_scan_at = None
        self.last_log_at = 0.0
        self.create_timer(0.10, self.watchdog)
        self.get_logger().info(
            f"BIDIRECTIONAL_V3 active: side={cfg.wall_side}, target={cfg.desired_distance:.2f} m, max_throttle={cfg.cruise_throttle:.3f}, corner_direction={cfg.corner_direction}")

    def publish(self, throttle, steering):
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.steer_pub.publish(Float32(data=float(steering)))

    def on_scan(self, msg):
        now = time.monotonic()
        dt = 0.05 if self.last_scan_at is None else now - self.last_scan_at
        self.last_scan_at = now
        scan = (msg.ranges, msg.angle_min, msg.angle_increment)
        throttle, steering, d = self.controller.compute(scan, dt)
        self.publish(throttle, steering)
        if now - self.last_log_at > 2.0:
            self.get_logger().info(
                f"mode={d['mode']} front={d.get('front')} "
                f"throttle={throttle:.2f} steer={steering:.2f} "
                f"left_perp={d.get('left_perpendicular')} "
                f"right_perp={d.get('right_perpendicular')} "
                f"path_clear={d.get('path_clear')} "
                f"heading={d.get('target_heading_deg')} "
                f"P={d.get('p_term')} D={d.get('d_term')}")
            self.last_log_at = now

    def watchdog(self):
        if self.last_scan_at is None or time.monotonic() - self.last_scan_at > 0.35:
            self.controller.reset()
            self.publish(0.0, 0.0)

    def stop(self):
        self.publish(0.0, 0.0)


def main(args=None):
    rclpy.init(args=args)
    node = WallFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()