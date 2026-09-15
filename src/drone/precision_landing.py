"""Precision landing system using ArUco markers.

The RVR+ has a large ArUco marker printed on its top plate.
The drone uses its downward camera to detect this marker and
execute a PID-controlled descent to land precisely on the cradle.

Supports:
- DJI Tello (via djitellopy + OpenCV)
- Any drone with a downward-facing camera + position control
"""

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ArUco marker config
MARKER_ID = 42  # ID of ArUco marker on RVR+ top plate
MARKER_SIZE_CM = 15.0  # Physical size of the marker

# PID gains for landing controller
PID_X = {"kp": 0.4, "ki": 0.01, "kd": 0.2}
PID_Y = {"kp": 0.4, "ki": 0.01, "kd": 0.2}
PID_Z = {"kp": 0.3, "ki": 0.005, "kd": 0.15}
PID_YAW = {"kp": 0.5, "ki": 0.0, "kd": 0.1}


@dataclass
class MarkerDetection:
    """Detected ArUco marker position relative to drone."""

    marker_id: int
    center_x: float  # -1.0 to 1.0 (normalized image coords)
    center_y: float
    distance_cm: float  # estimated distance from marker
    yaw_offset: float  # rotation offset in degrees
    timestamp: float


class PIDController:
    """Simple PID controller for one axis."""

    def __init__(self, kp: float, ki: float, kd: float, output_limit: float = 50.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()

    def update(self, error: float) -> float:
        now = time.time()
        dt = now - self._last_time
        if dt <= 0:
            dt = 0.01

        self._integral += error * dt
        # Anti-windup
        self._integral = np.clip(self._integral, -self.output_limit, self.output_limit)

        derivative = (error - self._last_error) / dt

        output = self.kp * error + self.ki * self._integral + self.kd * derivative
        output = np.clip(output, -self.output_limit, self.output_limit)

        self._last_error = error
        self._last_time = now

        return float(output)

    def reset(self):
        self._integral = 0.0
        self._last_error = 0.0
        self._last_time = time.time()


class PrecisionLandingSystem:
    """Manages precision landing using ArUco marker detection + PID control."""

    def __init__(self, target_marker_id: int = MARKER_ID):
        self.target_marker_id = target_marker_id
        # Built once. Constructing the dictionary, the parameter block, and
        # the detector per frame is pure overhead inside a 20 Hz control loop.
        self._cached_detector = None
        self._detector_unavailable = False
        self.pid_x = PIDController(**PID_X)
        self.pid_y = PIDController(**PID_Y)
        self.pid_z = PIDController(**PID_Z, output_limit=30.0)
        self.pid_yaw = PIDController(**PID_YAW, output_limit=40.0)
        self._landing_complete = False
        self._marker_lost_count = 0

    def _detector(self):
        """Lazily build and cache the ArUco detector."""
        if self._cached_detector is not None:
            return self._cached_detector
        if self._detector_unavailable:
            return None
        try:
            from cv2 import aruco
        except ImportError:
            logger.error("precision landing needs opencv-python with the aruco module")
            self._detector_unavailable = True
            return None
        self._cached_detector = aruco.ArucoDetector(
            aruco.getPredefinedDictionary(aruco.DICT_4X4_50),
            aruco.DetectorParameters(),
        )
        return self._cached_detector

    def detect_marker(self, frame: np.ndarray) -> MarkerDetection | None:
        """Detect ArUco marker in camera frame.

        Args:
            frame: BGR image from drone camera (numpy array)

        Returns:
            MarkerDetection if target marker found, None otherwise.
        """
        detector = self._detector()
        if detector is None:
            return None

        import cv2

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _rejected = detector.detectMarkers(gray)

        if ids is None:
            return None

        # Find our target marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == self.target_marker_id:
                # Get marker center in normalized coordinates
                marker_corners = corners[i][0]
                center_x_px = np.mean(marker_corners[:, 0])
                center_y_px = np.mean(marker_corners[:, 1])

                h, w = frame.shape[:2]
                center_x_norm = (center_x_px - w / 2) / (w / 2)  # -1 to 1
                center_y_norm = (center_y_px - h / 2) / (h / 2)  # -1 to 1

                # Estimate distance from marker size in pixels
                marker_width_px = np.linalg.norm(marker_corners[0] - marker_corners[1])
                # focal_length approximation for Tello (720p)
                focal_length_px = 600.0
                distance_cm = (MARKER_SIZE_CM * focal_length_px) / marker_width_px

                # Estimate yaw from marker rotation
                vec = marker_corners[1] - marker_corners[0]
                yaw_offset = float(np.degrees(np.arctan2(vec[1], vec[0])))

                return MarkerDetection(
                    marker_id=int(marker_id),
                    center_x=float(center_x_norm),
                    center_y=float(center_y_norm),
                    distance_cm=float(distance_cm),
                    yaw_offset=yaw_offset,
                    timestamp=time.time(),
                )

        return None

    def compute_landing_commands(self, detection: MarkerDetection) -> dict:
        """Compute RC commands to align with marker and descend.

        Returns dict with rc commands: lr, fb, ud, yaw (each -100 to 100).
        """
        # PID on X axis (left-right)
        lr = self.pid_x.update(detection.center_x)

        # PID on Y axis (forward-back, inverted)
        fb = -self.pid_y.update(detection.center_y)

        # PID on Z axis (descent — target distance decreases over time)
        # Descend faster when far, slower when close
        target_distance = 20.0  # target: 20cm above marker before final drop
        z_error = detection.distance_cm - target_distance
        ud = -self.pid_z.update(z_error)  # negative = descend

        # PID on yaw (align with marker orientation)
        yaw = self.pid_yaw.update(detection.yaw_offset)

        # Check if close enough to land
        if (
            detection.distance_cm < 30.0
            and abs(detection.center_x) < 0.1
            and abs(detection.center_y) < 0.1
        ):
            self._landing_complete = True

        return {
            "lr": int(np.clip(lr, -100, 100)),
            "fb": int(np.clip(fb, -100, 100)),
            "ud": int(np.clip(ud, -100, 100)),
            "yaw": int(np.clip(yaw, -100, 100)),
        }

    @property
    def landing_complete(self) -> bool:
        return self._landing_complete

    def reset(self):
        """Reset for a new landing attempt."""
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()
        self._landing_complete = False
        self._marker_lost_count = 0

    async def execute_landing(self, drone, frame_source) -> bool:
        """Execute the full precision landing sequence.

        Args:
            drone: anything exposing async ``send_rc(lr, fb, ud, yaw)`` and
                ``land()`` -- normally a :class:`DroneController`.
            frame_source: callable returning the current camera frame, or None.
                It is run off the event loop because grabbing a frame from a
                Tello blocks on a UDP read.

        Returns:
            True if the landing completed, False if it was aborted.
        """
        self.reset()
        logger.info("[LAND] Starting precision landing sequence...")
        max_iterations = 200  # ~10 seconds at 20Hz
        iteration = 0

        while iteration < max_iterations and not self._landing_complete:
            iteration += 1
            frame = await asyncio.to_thread(frame_source)
            detection = self.detect_marker(frame) if frame is not None else None

            if detection is None:
                self._marker_lost_count += 1
                if self._marker_lost_count > 40:  # lost for ~2 seconds
                    logger.warning("[LAND] Marker lost for too long, aborting")
                    # Hold position rather than drifting on the last command.
                    await drone.send_rc(0, 0, 0, 0)
                    return False
                await asyncio.sleep(0.05)
                continue

            self._marker_lost_count = 0
            commands = self.compute_landing_commands(detection)
            await drone.send_rc(commands["lr"], commands["fb"], commands["ud"], commands["yaw"])

            if iteration % 20 == 0:
                logger.info(
                    "[LAND] Dist=%.0fcm X=%.2f Y=%.2f | RC: lr=%d fb=%d ud=%d",
                    detection.distance_cm,
                    detection.center_x,
                    detection.center_y,
                    commands["lr"],
                    commands["fb"],
                    commands["ud"],
                )

            await asyncio.sleep(0.05)

        if not self._landing_complete:
            logger.warning("[LAND] Did not converge within %d iterations", max_iterations)
            await drone.send_rc(0, 0, 0, 0)
            return False

        logger.info("[LAND] Aligned with cradle, executing final land")
        await drone.send_rc(0, 0, 0, 0)
        await asyncio.sleep(0.5)
        await drone.land()
        return True
