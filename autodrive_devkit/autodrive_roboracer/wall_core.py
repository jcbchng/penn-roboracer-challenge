"""Bidirectional LiDAR gap follower with multi-beam wall-distance estimates.

Steering positive = left. ROS-independent; no external dependencies.
Tune vehicle_half_width and clearance_margin for the simulator vehicle.
"""
from dataclasses import dataclass
import math
import statistics


def clamp(x, a, b):
    return max(a, min(b, x))


@dataclass
class Config:
    wall_side: str = 'right'  # retained for compatibility; used only for tie-breaking
    desired_distance: float = 0.80
    lookahead: float = 1.15
    kp: float = 1.75
    kd: float = 0.18
    max_steer: float = 1.0
    cruise_throttle: float = 0.12
    min_throttle: float = 0.035
    front_slow_distance: float = 4.0
    front_stop_distance: float = 0.85
    emergency_distance: float = 0.40
    preview_slow_distance: float = 3.5
    side_safety_distance: float = 0.40
    min_valid_range: float = 0.06
    max_valid_range: float = 10.0
    vehicle_half_width: float = 0.18  # verify from vehicle dimensions
    clearance_margin: float = 0.28
    steering_gain: float = 2.8
    derivative_gain: float = 0.10
    derivative_filter_tau: float = 0.20
    heading_filter_tau: float = 0.07
    steering_rate_limit: float = 5.0
    heading_switch_penalty: float = 0.30
    forward_cone_degrees: float = 80.0
    bubble_distance: float = 2.2
    turn_start_distance: float = 3.5
    corner_direction: str = 'auto'  # allow both left and right bends
    wrong_way_penalty: float = 4.0
    corner_left_limit: float = 0.12
    # V3: conservative anticipation and nonzero propulsion only in safe corridors.
    side_warning_distance: float = 0.55
    side_critical_distance: float = 0.30
    turn_preview_distance: float = 3.0
    max_turn_throttle: float = 0.065
    safe_creep_clearance: float = 0.75
    creep_throttle: float = 0.022


class WallFollower:
    def __init__(self, config=None):
        self.cfg = config or Config()
        if self.cfg.wall_side not in ('left', 'right'):
            raise ValueError('wall_side must be left or right')
        self.previous_steering = 0.0
        self.filtered_heading = None
        self.previous_filtered_heading = None
        self.filtered_derivative = 0.0

    def reset(self):
        self.previous_steering = 0.0
        self.filtered_heading = None
        self.previous_filtered_heading = None
        self.filtered_derivative = 0.0

    @staticmethod
    def _wall_fit(points):
        """Fit x = a*y+b in local coordinates; b is lateral intercept at x=0.
        Return perpendicular distance to the fitted line and its heading.
        Points are (forward_x, lateral_y); only local approximately straight walls.
        """
        if len(points) < 5:
            return None
        mx = statistics.mean(p[0] for p in points)
        my = statistics.mean(p[1] for p in points)
        sxx = sum((p[0]-mx)**2 for p in points)
        if sxx < 0.03:
            return None
        slope = sum((x-mx)*(y-my) for x,y in points)/sxx
        intercept = my-slope*mx
        residuals = [abs(y-slope*x-intercept)/math.hypot(slope,1) for x,y in points]
        if statistics.median(residuals) > 0.22:
            return None
        return abs(intercept)/math.hypot(slope,1), math.atan(slope)

    def compute(self, scan, dt=0.05):
        c = self.cfg
        dt = clamp(float(dt), 0.01, 0.20)
        ranges, angle_min, inc = scan
        diag = {'mode': 'gap_follow', 'left_perpendicular': None,
                'right_perpendicular': None, 'front': None}
        if not ranges or not math.isfinite(inc) or inc <= 0:
            self.reset()
            return 0.0, 0.0, {**diag, 'mode': 'invalid_scan'}
        # Each range is a ray length, NOT a perpendicular distance.
        beams = []
        walls = {'left': [], 'right': []}
        for i, raw in enumerate(ranges):
            angle = angle_min + i*inc
            if abs(angle) > math.radians(135):
                continue
            r = float(raw)
            if not math.isfinite(r) or r < c.min_valid_range:
                continue
            r = min(r, c.max_valid_range)
            x, y = r*math.cos(angle), r*math.sin(angle)
            beams.append((angle, r, x, y))
            if 0.1 <= x <= 2.3 and abs(y) <= 3.0:
                if math.radians(35) <= angle <= math.radians(110):
                    walls['left'].append((x,y))
                if -math.radians(110) <= angle <= -math.radians(35):
                    walls['right'].append((x,y))
        for side in ('left','right'):
            fitted = self._wall_fit(walls[side])
            if fitted:
                diag[side+'_perpendicular'] = round(fitted[0],3)
                diag[side+'_wall_heading'] = round(fitted[1],3)
        forward = [r for a,r,x,y in beams if abs(a) <= math.radians(8)]
        if len(forward) < 3:
            self.reset()
            return 0.0, 0.0, {**diag, 'mode': 'invalid_front_scan'}
        front = statistics.median(forward)
        diag['front'] = round(front,3)
        # Angular free-space scoring with a physical-width safety corridor.
        # A candidate heading is blocked if an obstacle is too near its swept ray.
        obstacles = [(x,y) for a,r,x,y in beams if r < c.max_valid_range-0.01]
        candidates = []
        clearance = c.vehicle_half_width + c.clearance_margin
        for deg in range(-int(c.forward_cone_degrees), int(c.forward_cone_degrees)+1, 3):
            a = math.radians(deg)
            ux,uy = math.cos(a), math.sin(a)
            free = c.max_valid_range
            for x,y in obstacles:
                along = x*ux+y*uy
                if along <= 0 or along >= free:
                    continue
                lateral = abs(-x*uy+y*ux)
                if lateral < clearance:
                    free = along
            # Prefer farther clear paths while retaining a modest forward bias.
            # Prefer openings, but discourage straight headings that approach
            # a nearby side wall. This matters before the front beam is blocked.
            score = min(free, 5.0) - 0.48*abs(a)
            if front < c.turn_preview_distance:
                score -= 0.12 * abs(a)
            if self.filtered_heading is not None:
                score -= c.heading_switch_penalty * abs(a-self.filtered_heading)
            # Avoid choosing the wrong branch at a right-hand hairpin.
            # This is a track-specific preference, NOT a universal safety rule.
            if front < c.turn_start_distance and c.corner_direction == 'right' and a > 0:
                score -= c.wrong_way_penalty * a
            candidates.append((score, free, a))
        _, path_clear, heading = max(candidates, key=lambda p: p[0])
        # When already close to a side wall, the straight-ray winner can be
        # nearly straight even though the car must steer away immediately.
        # Add a bounded avoidance heading before filtering the target.
        left_dist = diag['left_perpendicular']
        right_dist = diag['right_perpendicular']
        if left_dist is not None and left_dist < c.side_warning_distance:
            urgency = clamp((c.side_warning_distance-left_dist) /
                            max(c.side_warning_distance-c.side_critical_distance, 0.01), 0.0, 1.0)
            heading = min(heading, -math.radians(12.0)*urgency)
        if right_dist is not None and right_dist < c.side_warning_distance:
            urgency = clamp((c.side_warning_distance-right_dist) /
                            max(c.side_warning_distance-c.side_critical_distance, 0.01), 0.0, 1.0)
            heading = max(heading, math.radians(12.0)*urgency)
        # Filter the chosen heading before differentiating: raw LiDAR target
        # switches otherwise produce large derivative spikes.
        urgent = path_clear < 2.5 or front < c.turn_start_distance
        tau = c.heading_filter_tau * (0.25 if urgent else 1.0)
        alpha = dt / (tau + dt)
        if self.filtered_heading is None:
            self.filtered_heading = heading
            self.previous_filtered_heading = heading
        else:
            self.previous_filtered_heading = self.filtered_heading
            self.filtered_heading += alpha * (heading - self.filtered_heading)
        raw_derivative = (self.filtered_heading - self.previous_filtered_heading) / dt
        beta = dt / (c.derivative_filter_tau + dt)
        self.filtered_derivative += beta * (raw_derivative - self.filtered_derivative)

        # PD-like target-change damping: oppose rapid heading-target changes.
        # The derivative of target heading is not vehicle yaw rate; this
        # reduces target jitter but cannot replace a true yaw-rate damper.
        p_term = c.steering_gain * self.filtered_heading
        d_term = -c.derivative_gain * self.filtered_derivative
        if urgent:
            # Do not let target-change damping cancel a needed sharp turn.
            d_term = clamp(d_term, -0.12, 0.12)
        steering = clamp(p_term + d_term, -c.max_steer, c.max_steer)
        if front < c.turn_start_distance and c.corner_direction == 'right':
            steering = min(steering, c.corner_left_limit)
        # Rate limiting stops left/right command chatter. Relax it near hazards.
        max_change = (20.0 if urgent else c.steering_rate_limit) * dt
        steering = clamp(steering, self.previous_steering-max_change,
                         self.previous_steering+max_change)
        steering = clamp(steering, -c.max_steer, c.max_steer)
        if front < c.turn_start_distance and c.corner_direction == 'right':
            steering = min(steering, c.corner_left_limit)
        self.previous_steering = steering
        # Don't stop merely because the wall is ahead: hairpins require turning.
        # But do stop if the chosen path is too short to be safe.
        if path_clear <= max(c.emergency_distance, clearance+0.10):
            diag.update(mode='blocked_stop', target_heading_deg=round(math.degrees(heading),1), path_clear=round(path_clear,2))
            return 0.0, steering, diag
        # A bend is not an obstacle: slow moderately for steering, and use
        # the clearance of the selected turning corridor rather than the
        # straight-ahead wall to decide how much forward power is safe.
        corner_scale = 1.0 - 0.55 * abs(steering) / max(c.max_steer, 0.01)
        range_scale = clamp((path_clear - 0.60) / 2.1, 0.0, 1.0)
        throttle = c.cruise_throttle * corner_scale * range_scale
        # Slow BEFORE reaching a wall, even when the selected gap appears open.
        # A sharp turn at full throttle caused the V2 second-lap deadlock.
        if front < c.turn_preview_distance:
            throttle = min(throttle, c.max_turn_throttle)
        if ((left_dist is not None and left_dist < c.side_warning_distance) or
                (right_dist is not None and right_dist < c.side_warning_distance)):
            throttle = min(throttle, c.max_turn_throttle)
        # Avoid a mathematical near-zero throttle deadlock when the selected
        # corridor is wider than the stop threshold. Do not override blocked_stop.
        if path_clear >= c.safe_creep_clearance:
            throttle = max(throttle, c.creep_throttle)
        # Maintain propulsion through a clear turn; never override the
        # emergency stop above or force a minimum in a tight corridor.
        if path_clear >= 1.4:
            throttle = max(throttle, c.min_throttle)
        diag.update(corner_direction=c.corner_direction, target_heading_deg=round(math.degrees(heading),1),
                    path_clear=round(path_clear,2), steering=round(steering,3),
                    p_term=round(p_term,3), d_term=round(d_term,3),
                    filtered_heading_deg=round(math.degrees(self.filtered_heading),1))
        return clamp(throttle,0.0,c.cruise_throttle), steering, diag
