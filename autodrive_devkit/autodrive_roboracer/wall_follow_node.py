import json
import math
import os
import statistics
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32


def bound(x, lo, hi):
    return min(hi, max(lo, x))


def angle_diff(a, b):
    return math.atan2(math.sin(a - b), math.cos(a - b))


def sector(scan, degrees, width=6):
    values, start, step = scan
    if step <= 0 or not math.isfinite(step):
        return None
    samples = []
    for offset in (-1, -0.5, 0, 0.5, 1):
        idx = round((math.radians(degrees + offset * width) - start) / step)
        if 0 <= idx < len(values):
            r = values[idx]
            if math.isfinite(r) and 0.06 <= r <= 10:
                samples.append(float(r))
    return statistics.median(samples) if samples else None


class Circuit:
    """Closed polyline with continuous projection and precomputed segment headings."""

    def __init__(self, points, left_offset=0.10):
        raw = [(float(x), float(y)) for x, y in points]
        if len(raw) < 3:
            raise ValueError('At least three path points are required')
        if left_offset:
            shifted = []
            for i, (x, y) in enumerate(raw):
                prev, nxt = raw[i - 1], raw[(i + 1) % len(raw)]
                dx, dy = nxt[0] - prev[0], nxt[1] - prev[1]
                length = math.hypot(dx, dy)
                shifted.append((x - left_offset * dy / length,
                                y + left_offset * dx / length) if length > 1e-8 else (x, y))
            raw = shifted
        self.points = raw
        self.length = len(raw)
        self.segments = []
        for i, p in enumerate(raw):
            q = raw[(i + 1) % self.length]
            dx, dy = q[0] - p[0], q[1] - p[1]
            size = math.hypot(dx, dy)
            self.segments.append((dx, dy, max(size, 1e-8), math.atan2(dy, dx)))

    @classmethod
    def load(cls, filename, offset):
        with open(filename, encoding='utf-8') as stream:
            obj = json.load(stream)
        if obj.get('version') != 1:
            raise ValueError('Unsupported racing line version')
        return cls(obj['points'], offset)

    def project(self, position, hint=None):
        """Return segment, lateral error, and tangent heading at closest projection."""
        px, py = position
        if hint is None:
            indices = range(self.length)
        else:
            indices = ((hint + delta) % self.length for delta in range(-12, 36))
        best = None
        for i in indices:
            ax, ay = self.points[i]
            dx, dy, length, heading = self.segments[i]
            t = bound(((px - ax) * dx + (py - ay) * dy) / (length * length), 0, 1)
            cx, cy = ax + t * dx, ay + t * dy
            sq = (px - cx) ** 2 + (py - cy) ** 2
            if best is None or sq < best[0]:
                # Positive lateral error means the vehicle is left of the path.
                lateral = (dx * (py - cy) - dy * (px - cx)) / length
                best = (sq, i, lateral, heading)
        return best[1:]

    def upcoming_bend(self, index, distance=1.2):
        """Maximum heading change per metre across the upcoming preview arc."""
        traversed = 0.0
        current = index
        heading = self.segments[current][3]
        maximum = 0.0
        while traversed < distance:
            following = (current + 1) % self.length
            step = self.segments[current][2]
            turn = abs(angle_diff(self.segments[following][3], heading))
            # Average over several short segments to reduce sampling noise.
            maximum = max(maximum, turn / max(step, 0.08))
            traversed += step
            current = following
            heading = self.segments[current][3]
            if current == index:
                break
        return maximum


class StanleyRacer:
    """Heading + cross-track controller, with preview-based speed scheduling."""

    def __init__(self, circuit, heading_gain=0.55, lateral_gain=0.55,
                 softening=0.65, cruise=0.34, minimum=0.12,
                 curve_penalty=1.8, steering_limit=0.78):
        self.path = circuit
        self.heading_gain = heading_gain
        self.lateral_gain = lateral_gain
        self.softening = softening
        self.cruise = cruise
        self.minimum = minimum
        self.curve_penalty = curve_penalty
        self.steering_limit = steering_limit
        self.segment = None
        self.speed_estimate = 0.0
        self.previous_position = None
        self.previous_time = None

    def compute(self, position, yaw, now):
        if self.previous_position is not None and self.previous_time is not None:
            dt = now - self.previous_time
            if 0.005 < dt < 0.5:
                measured = math.dist(position, self.previous_position) / dt
                self.speed_estimate += 0.2 * (measured - self.speed_estimate)
        self.previous_position, self.previous_time = position, now
        idx, lateral, heading = self.path.project(position, self.segment)
        self.segment = idx
        heading_term = self.heading_gain * angle_diff(heading, yaw)
        # If the vehicle is left of the reference, turn right (negative).
        lateral_term = math.atan2(-self.lateral_gain * lateral,
                                  self.softening + self.speed_estimate)
        steering = bound(heading_term + lateral_term,
                         -self.steering_limit, self.steering_limit)
        bend = self.path.upcoming_bend(idx)
        throttle = self.cruise / (1 + self.curve_penalty * bend)
        throttle *= 1 - 0.20 * abs(steering) / self.steering_limit
        throttle = bound(throttle, self.minimum, self.cruise)
        return throttle, steering


class SafetyEnvelope:
    def __init__(self, emergency=0.42, side_limit=0.38, slow_distance=2.10,
                 minimum_throttle=0.12):
        self.emergency = emergency
        self.side_limit = side_limit
        self.slow_distance = slow_distance
        self.minimum_throttle = minimum_throttle

    def apply(self, scan, throttle, steering):
        front = sector(scan, 0, 9)
        if front is None or front <= self.emergency:
            return 0.0, 0.0
        left, right = sector(scan, 90), sector(scan, -90)
        if left is not None and left < self.side_limit:
            steering = min(steering, -0.35)
            throttle = min(throttle, self.minimum_throttle)
        if right is not None and right < self.side_limit:
            steering = max(steering, 0.35)
            throttle = min(throttle, self.minimum_throttle)
        if front < self.slow_distance:
            throttle *= bound((front - self.emergency) / 1.2, 0, 1)
        return throttle, steering


class RoboRacerNode(Node):
    def __init__(self):
        super().__init__('wall_follow_node')
        settings = {
            'map_file': '/home/autodrive_devkit/src/autodrive_devkit/racing_line.json',
            'path_left_offset': -0.20,
            'right_wall_target': 0.75,
            'right_wall_gain': 0.30,
            'right_wall_max_correction': 0.18,
            'heading_gain': 0.90,
            'lateral_gain': 1.10,
            'stanley_softening': 0.35,
            'race_throttle': 0.24,
            'min_throttle': 0.09,
            'curvature_slowdown': 1.8,
            'max_steer': 0.78,
            'emergency_distance': 0.42,
            'side_safety_distance': 0.38,
            'front_slow_distance': 2.10,
            'stop_after_racing_laps': 0,
        }
        for name, default in settings.items():
            self.declare_parameter(name, default)
        p = {key: self.get_parameter(key).value for key in settings}
        if not os.path.isfile(p['map_file']):
            raise FileNotFoundError('This experimental controller requires a saved racing_line.json: '
                                    + p['map_file'])
        circuit = Circuit.load(p['map_file'], p['path_left_offset'])
        self.racer = StanleyRacer(circuit, p['heading_gain'], p['lateral_gain'],
                                 p['stanley_softening'], p['race_throttle'],
                                 p['min_throttle'], p['curvature_slowdown'], p['max_steer'])
        self.right_wall_target = float(p['right_wall_target'])
        self.right_wall_gain = float(p['right_wall_gain'])
        self.right_wall_max_correction = float(p['right_wall_max_correction'])
        self.safety = SafetyEnvelope(p['emergency_distance'], p['side_safety_distance'],
                                     p['front_slow_distance'], p['min_throttle'])
        self.throttle_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/throttle_command', 10)
        self.steer_pub = self.create_publisher(Float32, '/autodrive/roboracer_1/steering_command', 10)
        self.create_subscription(Odometry, '/autodrive/roboracer_1/odom', self.on_odom, 10)
        self.create_subscription(LaserScan, '/autodrive/roboracer_1/lidar', self.on_scan, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/lap_count', self.on_lap, 10)
        self.create_subscription(Int32, '/autodrive/roboracer_1/collision_count', self.on_collision, 10)
        self.create_timer(0.1, self.watchdog)
        self.pose = None
        self.yaw = None
        self.last_scan = None
        self.collision_start = None
        self.faulted = False
        self.start_lap = None
        self.lap_limit = int(p['stop_after_racing_laps'])
        self.get_logger().info('Experimental Stanley racing controller ready')

    def publish(self, throttle, steering):
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.steer_pub.publish(Float32(data=float(steering)))

    def stop(self):
        self.publish(0, 0)

    def on_odom(self, msg):
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self.pose = (float(p.x), float(p.y))
        self.yaw = math.atan2(2 * (q.w * q.z + q.x * q.y),
                              1 - 2 * (q.y * q.y + q.z * q.z))

    def on_collision(self, msg):
        count = int(msg.data)
        if self.collision_start is None:
            self.collision_start = count
        elif count > self.collision_start:
            self.faulted = True
            self.get_logger().error('Collision detected; stopping')
            self.stop()

    def on_lap(self, msg):
        lap = int(msg.data)
        if self.start_lap is None:
            self.start_lap = lap
        elif self.lap_limit > 0 and lap - self.start_lap >= self.lap_limit:
            self.faulted = True
            self.stop()

    def on_scan(self, msg):
        self.last_scan = time.monotonic()
        if self.faulted or self.pose is None or self.yaw is None or self.collision_start is None:
            self.stop()
            return
        scan = (msg.ranges, msg.angle_min, msg.angle_increment)
        throttle, steering = self.racer.compute(self.pose, self.yaw, self.last_scan)

        right_distance = sector(scan, -90, 8)
        if right_distance is not None:
            error = right_distance - self.right_wall_target
            correction = bound(
                -self.right_wall_gain * error,
                -self.right_wall_max_correction,
                self.right_wall_max_correction
            )
            steering = bound(
                steering + correction,
                -self.racer.steering_limit,
                self.racer.steering_limit
            )

        throttle, steering = self.safety.apply(scan, throttle, steering)
        self.publish(throttle, steering)

    def watchdog(self):
        if self.last_scan is None or time.monotonic() - self.last_scan > 0.35:
            self.stop()


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = RoboRacerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.stop()
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
