"""Single-file version of the user-provided 8.5-second map racing controller.
Combines wall_follow_core, map_racing_core, and map_racing_node.
The standalone wall_follow_node was a separate entrypoint, not used by map racer.
"""

import json
import math
import os
import statistics
import time
from dataclasses import dataclass

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, Int32




# From wall_follow_core(1).py
@dataclass
class Config:
    wall_side: str = 'right'
    desired_distance: float = 0.85
    lookahead: float = 0.75
    kp: float = 1.5
    kd: float = 0.20
    max_steer: float = 0.78
    cruise_throttle: float = 0.28
    min_throttle: float = 0.10
    corner_throttle: float = 0.13
    front_slow_distance: float = 2.2
    front_stop_distance: float = 0.65
    emergency_distance: float = 0.42
    max_throttle_slew: float = 0.8  # normalized command per second
    min_valid_range: float = 0.06
    max_valid_range: float = 10.0

def robust_sector(scan, center_degrees, half_width_degrees=4.0):
    """Return median finite measured range in a scan sector, or None."""
    ranges, angle_min, increment = scan
    if not ranges or not math.isfinite(increment) or increment <= 0:
        return None
    indices = [round((math.radians(a) - angle_min) / increment)
               for a in (center_degrees - half_width_degrees,
                         center_degrees - half_width_degrees / 2,
                         center_degrees,
                         center_degrees + half_width_degrees / 2,
                         center_degrees + half_width_degrees)]
    vals = []
    for i in indices:
        if 0 <= i < len(ranges):
            x = ranges[i]
            if math.isfinite(x) and x >= 0.06 and x <= 10.0:
                vals.append(float(x))
    return statistics.median(vals) if vals else None

class WallFollower:
    def __init__(self, config=None):
        self.cfg = config or Config()
        if self.cfg.wall_side not in ('left', 'right'):
            raise ValueError('wall_side must be left or right')
        self.previous_error = None
        self.previous_throttle = 0.0

    def reset(self):
        self.previous_error = None
        self.previous_throttle = 0.0

    def compute(self, scan, dt=0.05):
        """Compute (throttle, steering, diagnostic dict). Fail closed on bad data."""
        c = self.cfg
        dt = clamp(float(dt), 0.01, 0.20)
        direction = 1 if c.wall_side == 'left' else -1
        side = robust_sector(scan, direction * 90)
        forward_side = robust_sector(scan, direction * 45)
        front = robust_sector(scan, 0, 9)
        front_left = robust_sector(scan, 25, 7)
        front_right = robust_sector(scan, -25, 7)
        diag = {'wall_side': c.wall_side, 'side': side, 'side_forward': forward_side,
                'front': front, 'front_left': front_left, 'front_right': front_right,
                'mode': 'tracking'}

        # Do not drive when crucial measurements are absent.
        if front is None:
            self.reset()
            diag['mode'] = 'no_front_scan'
            return 0.0, 0.0, diag
        if front < c.emergency_distance:
            self.reset()
            diag['mode'] = 'emergency_stop'
            return 0.0, 0.0, diag

        if side is not None and forward_side is not None:
            # Alpha is the angle between wall and car heading.
            theta = math.radians(45)
            alpha = math.atan2(forward_side * math.cos(theta) - side,
                               forward_side * math.sin(theta))
            projected = side * math.cos(alpha) + c.lookahead * math.sin(alpha)
            error = c.desired_distance - projected
            derivative = 0.0 if self.previous_error is None else (error - self.previous_error) / dt
            self.previous_error = error
            # For right wall, being too far from wall means steer right (negative).
            steering = -direction * (c.kp * error + c.kd * derivative)
            diag.update(error=error, projected_distance=projected, alpha=alpha)
        else:
            self.previous_error = None
            diag['mode'] = 'wall_missing'
            # Search gently for wall; if ahead is constrained, steer toward open space.
            steering = direction * 0.24

        # Forward obstacle avoidance: steer toward side with more clearance.
        # Front sectors suppress risk when wall geometry is misleading at corners.
        if front < c.front_slow_distance:
            left_clear = front_left if front_left is not None else 0.0
            right_clear = front_right if front_right is not None else 0.0
            avoid_sign = 1.0 if left_clear > right_clear else -1.0
            if abs(left_clear - right_clear) < 0.12:
                avoid_sign = -direction  # Prefer to turn away from followed wall.
            strength = clamp((c.front_slow_distance - front) /
                             max(0.01, c.front_slow_distance - c.front_stop_distance), 0, 1)
            steering = steering * (1 - strength) + avoid_sign * c.max_steer * strength
            diag['mode'] = 'obstacle_avoidance'

        steering = clamp(steering, -c.max_steer, c.max_steer)
        # Corner and obstacle speed scheduling; conservative baseline.
        steer_factor = 1 - 0.65 * abs(steering) / max(c.max_steer, 0.01)
        target_throttle = max(c.min_throttle, c.cruise_throttle * steer_factor)
        if front < c.front_slow_distance:
            clearance = clamp((front - c.front_stop_distance) /
                              max(0.01, c.front_slow_distance - c.front_stop_distance), 0, 1)
            target_throttle = min(target_throttle, c.corner_throttle * clearance)
        if front <= c.front_stop_distance:
            target_throttle = 0.0
            diag['mode'] = 'front_stop'
        # Allow immediate braking; only limit acceleration.
        throttle = min(target_throttle, self.previous_throttle + c.max_throttle_slew * dt)
        self.previous_throttle = throttle
        return throttle, steering, diag



# From map_racing_core(1).py
def clamp(value, low, high):
    return max(low, min(high, value))

def wrap_angle(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi

class PathRecorder:
    def __init__(self, spacing=0.08):
        self.spacing = spacing
        self.points = []

    def add(self, x, y):
        point = (float(x), float(y))
        if not self.points or math.dist(self.points[-1], point) >= self.spacing:
            self.points.append(point)
            return True
        return False

    def build(self, smoothing_radius=2):
        if len(self.points) < 20:
            raise ValueError("too few trajectory points for a closed racing path")
        points = self.points[:]
        radius = max(0, int(smoothing_radius))
        if radius:
            smoothed = []
            count = len(points)
            for index in range(count):
                window = [points[(index + offset) % count]
                          for offset in range(-radius, radius + 1)]
                smoothed.append((sum(p[0] for p in window) / len(window),
                                 sum(p[1] for p in window) / len(window)))
            points = smoothed
        return ClosedPath(points)

class ClosedPath:
    def __init__(self, points):
        self.points = [(float(x), float(y)) for x, y in points]
        if len(self.points) < 3:
            raise ValueError("closed path needs at least three points")

    def nearest_index(self, position, hint=None, backward=12, forward=35):
        x, y = position
        if hint is None:
            candidates = range(len(self.points))
        else:
            candidates = [(hint + offset) % len(self.points)
                          for offset in range(-backward, forward + 1)]
        return min(candidates,
                   key=lambda i: (self.points[i][0] - x) ** 2
                   + (self.points[i][1] - y) ** 2)

    def offset_left(self, distance):
        shifted = []
        count = len(self.points)
        for index, point in enumerate(self.points):
            previous = self.points[(index - 1) % count]
            following = self.points[(index + 1) % count]
            tx, ty = following[0] - previous[0], following[1] - previous[1]
            length = math.hypot(tx, ty)
            if length <= 1e-9:
                shifted.append(point)
            else:
                shifted.append((point[0] - distance * ty / length,
                                point[1] + distance * tx / length))
        return ClosedPath(shifted)

    def target_index(self, start, distance):
        travelled = 0.0
        index = start
        for _ in range(len(self.points)):
            nxt = (index + 1) % len(self.points)
            travelled += math.dist(self.points[index], self.points[nxt])
            index = nxt
            if travelled >= distance:
                break
        return index

    def curvature_ahead(self, start, distance=1.2):
        middle = self.target_index(start, distance * 0.5)
        end = self.target_index(start, distance)
        p0, p1, p2 = self.points[start], self.points[middle], self.points[end]
        heading_a = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        heading_b = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
        return abs(wrap_angle(heading_b - heading_a)) / max(distance, 0.01)

    def save(self, path, metadata=None):
        with open(path, "w") as handle:
            json.dump({"version": 1, "points": self.points,
                       "metadata": metadata or {}}, handle, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as handle:
            data = json.load(handle)
        if data.get("version") != 1:
            raise ValueError("unsupported racing path version")
        return cls(data["points"])

class PathFollower:
    def __init__(self, path, lookahead=0.65, steering_gain=0.55,
                 max_steer=0.78, cruise_throttle=0.34,
                 min_throttle=0.12, curvature_slowdown=1.8):
        self.path = path
        self.lookahead = lookahead
        self.steering_gain = steering_gain
        self.max_steer = max_steer
        self.cruise_throttle = cruise_throttle
        self.min_throttle = min_throttle
        self.curvature_slowdown = curvature_slowdown
        self.last_index = None
        self.bias_filtered = 0.0

    def compute(self, position, yaw, right_clearance=None, bias_max=0.0,
                min_right_clearance=0.85, full_curvature=0.45):
        nearest = self.path.nearest_index(position, self.last_index)
        self.last_index = nearest
        target_index = self.path.target_index(nearest, self.lookahead)
        target = self.path.points[target_index]
        curvature = self.path.curvature_ahead(nearest)
        # Only move the pursuit target, never distort the saved closed path.
        # If the right-side scan is absent or narrow, return to the baseline.
        clearance_factor = 0.0
        if right_clearance is not None and math.isfinite(right_clearance):
            clearance_factor = clamp(
                (right_clearance - min_right_clearance) / 0.40, 0.0, 1.0)
        corner_factor = clamp(curvature / max(full_curvature, 0.01), 0.0, 1.0)
        requested_bias = max(0.0, bias_max) * corner_factor * clearance_factor
        # Smooth the target adjustment to prevent a step in steering.
        self.bias_filtered += 0.12 * (requested_bias - self.bias_filtered)
        before = self.path.points[(target_index - 1) % len(self.path.points)]
        after = self.path.points[(target_index + 1) % len(self.path.points)]
        tx, ty = after[0] - before[0], after[1] - before[1]
        norm = math.hypot(tx, ty)
        if norm > 1e-8:
            target = (target[0] + self.bias_filtered * ty / norm,
                      target[1] - self.bias_filtered * tx / norm)
        desired = math.atan2(target[1] - position[1], target[0] - position[0])
        heading_error = wrap_angle(desired - yaw)
        steering = clamp(self.steering_gain * (2.0 * math.sin(heading_error)
                                                / max(self.lookahead, 0.05)),
                         -self.max_steer, self.max_steer)
        throttle = self.cruise_throttle / (1.0 + self.curvature_slowdown * curvature)
        throttle *= 1.0 - 0.45 * abs(steering) / max(self.max_steer, 0.01)
        throttle = clamp(throttle, self.min_throttle, self.cruise_throttle)
        return throttle, steering, {"nearest": nearest, "target": target_index,
                                    "curvature": curvature,
                                    "heading_error": heading_error}



# From map_racing_node(1).py
class MapRacingNode(Node):
    def __init__(self):
        super().__init__("map_racing_node")
        parameters = {
            "map_file": "/home/autodrive_devkit/src/autodrive_devkit/racing_line.json",
            "force_discovery": False,
            "acquisition_lap": True,
            "stop_after_racing_laps": 0,
            "record_spacing": 0.08,
            "smoothing_radius": 2,
            "discovery_lookahead": 0.70,
            "discovery_front_slow_distance": 2.10,
            "discovery_throttle": 0.25,
            "path_lookahead": 0.65,
            "path_left_offset": 0.10,  # Restore the proven baseline racing line
            "corner_right_bias": 0.035,  # Maximum additional rightward target shift (m)
            "corner_bias_min_right_clearance": 0.85,
            "corner_bias_full_curvature": 0.45,
            "steering_gain": 0.55,
            "max_steer": 0.78,
            "race_throttle": 0.34,
            "min_throttle": 0.12,
            "curvature_slowdown": 1.8,
            "emergency_distance": 0.42,
            "side_safety_distance": 0.38,
        }
        for name, value in parameters.items():
            self.declare_parameter(name, value)
        self.p = {name: self.get_parameter(name).value for name in parameters}

        config = Config(lookahead=self.p["discovery_lookahead"],
                        front_slow_distance=self.p["discovery_front_slow_distance"],
                        cruise_throttle=self.p["discovery_throttle"])
        self.wall_follower = WallFollower(config)
        self.recorder = PathRecorder(self.p["record_spacing"])
        self.path_follower = None
        self.pending_path = None
        self.mode = "discovery"
        self.position = None
        self.yaw = None
        self.lap_count = None
        self.start_lap = None
        self.recording_full_lap = False
        self.racing_start_lap = None
        self.finished = False
        self.last_reported_lap = None
        self.collision_start = None
        self.collision_count = None
        self.last_scan_at = None
        self.last_control_at = None
        self.faulted = False

        map_file = self.p["map_file"]
        if os.path.exists(map_file) and not self.p["force_discovery"]:
            loaded_path = ClosedPath.load(map_file)
            if self.p["acquisition_lap"]:
                self.pending_path = loaded_path
                self.mode = "acquisition"
            else:
                self._activate_path(loaded_path)

        self.throttle_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/throttle_command", 10)
        self.steering_pub = self.create_publisher(
            Float32, "/autodrive/roboracer_1/steering_command", 10)
        self.create_subscription(Odometry, "/autodrive/roboracer_1/odom",
                                 self.on_odom, 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/lap_count",
                                 self.on_lap, 10)
        self.create_subscription(Int32, "/autodrive/roboracer_1/collision_count",
                                 self.on_collision, 10)
        self.create_subscription(LaserScan, "/autodrive/roboracer_1/lidar",
                                 self.on_scan, 10)
        self.create_timer(0.1, self.watchdog)
        self.get_logger().info(f"map racer ready in {self.mode} mode")

    def _activate_path(self, path):
        path = path.offset_left(self.p["path_left_offset"])
        self.path_follower = PathFollower(
            path, lookahead=self.p["path_lookahead"],
            steering_gain=self.p["steering_gain"], max_steer=self.p["max_steer"],
            cruise_throttle=self.p["race_throttle"],
            min_throttle=self.p["min_throttle"],
            curvature_slowdown=self.p["curvature_slowdown"])
        self.mode = "racing"
        self.racing_start_lap = self.lap_count

    def publish(self, throttle, steering):
        self.throttle_pub.publish(Float32(data=float(throttle)))
        self.steering_pub.publish(Float32(data=float(steering)))

    def stop(self):
        self.publish(0.0, 0.0)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.position = (float(p.x), float(p.y))
        self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                              1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        if self.mode == "discovery":
            self.recorder.add(p.x, p.y)

    def on_lap(self, msg):
        self.lap_count = int(msg.data)
        if self.start_lap is None:
            self.start_lap = self.lap_count
            return
        if self.mode == "acquisition" and self.lap_count > self.start_lap:
            if (self.lap_count != self.start_lap + 1
                    or self.collision_count != self.collision_start):
                self.get_logger().error("acquisition lap was not clean; stopping")
                self.faulted = True
                self.stop()
                return
            self._activate_path(self.pending_path)
            self.get_logger().info("clean acquisition lap complete; switching to racing mode")
            return
        if (self.mode == "racing" and self.racing_start_lap is not None
                and self.lap_count > self.racing_start_lap
                and self.lap_count != self.last_reported_lap):
            self.last_reported_lap = self.lap_count
            completed = self.lap_count - self.racing_start_lap
            self.get_logger().info(f"completed mapped racing lap {completed}")
            limit = int(self.p["stop_after_racing_laps"])
            if limit > 0 and completed >= limit:
                self.finished = True
                self.stop()
                self.get_logger().info("mapped racing validation complete; vehicle stopped")
            return
        if self.mode != "discovery" or self.lap_count <= self.start_lap:
            return
        if self.lap_count != self.start_lap + 1:
            self.get_logger().error("lap counter skipped; refusing discovered path")
            self.faulted = True
            self.stop()
            return
        if self.collision_count != self.collision_start:
            self.get_logger().error("discovery lap had a collision; refusing path")
            self.faulted = True
            self.stop()
            return
        if not self.recording_full_lap:
            # The node may have started after a checkpoint respawn, so the
            # first finish crossing can represent only a partial lap. Begin a
            # fresh recording at that crossing and accept only the next lap.
            self.recorder = PathRecorder(self.p["record_spacing"])
            self.start_lap = self.lap_count
            self.collision_start = self.collision_count
            self.recording_full_lap = True
            self.get_logger().info("finish line acquired; recording one complete clean lap")
            return
        try:
            path = self.recorder.build(self.p["smoothing_radius"])
            path.save(self.p["map_file"], {"source": "clean discovery lap"})
            self._activate_path(path)
            self.get_logger().info(
                f"saved {len(path.points)} path points; switching to racing mode")
        except (OSError, ValueError) as error:
            self.get_logger().error(f"could not create racing path: {error}")
            self.faulted = True
            self.stop()

    def on_collision(self, msg):
        self.collision_count = int(msg.data)
        if self.collision_start is None:
            self.collision_start = self.collision_count
        elif self.collision_count > self.collision_start and not self.faulted:
            self.faulted = True
            self.get_logger().error("collision detected; stopping map racer")
            self.stop()

    def on_scan(self, msg):
        self.last_scan_at = time.monotonic()
        if self.faulted or self.finished:
            self.stop()
            return
        if self.position is None or self.yaw is None or self.collision_count is None:
            self.stop()
            return
        if self.collision_count > self.collision_start:
            self.stop()
            return
        scan = (msg.ranges, msg.angle_min, msg.angle_increment)
        front = robust_sector(scan, 0, 9)
        if front is None or front <= self.p["emergency_distance"]:
            self.stop()
            return
        if self.mode in ("discovery", "acquisition"):
            now = time.monotonic()
            dt = 0.025 if self.last_control_at is None else now - self.last_control_at
            self.last_control_at = now
            throttle, steering, _ = self.wall_follower.compute(scan, dt)
        else:
            left = robust_sector(scan, 90, 6)
            right = robust_sector(scan, -90, 6)
            throttle, steering, _ = self.path_follower.compute(
                self.position, self.yaw,
                right_clearance=right,
                bias_max=self.p["corner_right_bias"],
                min_right_clearance=self.p["corner_bias_min_right_clearance"],
                full_curvature=self.p["corner_bias_full_curvature"])
            safety = self.p["side_safety_distance"]
            if left is not None and left < safety:
                steering = min(steering, -0.35)
                throttle = min(throttle, self.p["min_throttle"])
            if right is not None and right < safety:
                steering = max(steering, 0.35)
                throttle = min(throttle, self.p["min_throttle"])
            if front < self.p["discovery_front_slow_distance"]:
                throttle *= clamp((front - self.p["emergency_distance"]) / 1.2, 0.0, 1.0)
        self.publish(throttle, steering)

    def watchdog(self):
        if self.last_scan_at is None or time.monotonic() - self.last_scan_at > 0.35:
            self.stop()

def main(args=None):
    rclpy.init(args=args)
    node = MapRacingNode()
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
