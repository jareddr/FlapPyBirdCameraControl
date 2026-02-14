"""
Webcam pose detection with arm flap gesture recognition.
Uses MediaPipe Pose to track arms and detect a "flap" (arms moving from high to low, semi-outstretched).
"""

import collections
import logging
import sys
import threading
import time
from typing import Optional

import cv2

logger = logging.getLogger(__name__)
import mediapipe as mp
import numpy as np

# MediaPipe Pose landmark indices (see mediapipe.solutions.pose.PoseLandmark)
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_ELBOW = 13
LEFT_WRIST = 14
RIGHT_ELBOW = 16
RIGHT_WRIST = 17

# Flap detection tuning
HISTORY_LEN = 6  # frames to consider for downward motion
FLAP_COOLDOWN_FRAMES = 10  # ~0.33s at 30 fps
ARM_EXTENSION_MIN = 0.25  # min shoulder-wrist distance (normalized) to count as "outstretched"
DOWNWARD_THRESHOLD = 0.03  # min Y increase (in normalized coords) to count as downward motion

# Webcam startup: bounded warmup so we don't stall indefinitely
WARMUP_TIMEOUT_S = 10.0  # max seconds to wait for first valid frame
WARMUP_POLL_INTERVAL_S = 0.2  # interval between read attempts during warmup


class WebcamPose:
    """Captures webcam, runs pose detection, draws skeleton, detects flap gestures."""

    def __init__(self, width: int = 256, height: int = 192, camera_index: int = 0):
        self.width = width
        self.height = height
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None  # BGR, with skeleton drawn
        self._flap_requested = False  # set True when flap detected; main thread reads and clears
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Flap state
        self._left_wrist_y_history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self._right_wrist_y_history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self._left_shoulder_y_history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self._right_shoulder_y_history: collections.deque = collections.deque(maxlen=HISTORY_LEN)
        self._cooldown_remaining = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Returns latest BGR frame with skeleton drawn, or None if not ready."""
        with self._lock:
            if self._frame is None:
                return None
            return self._frame.copy()

    def consume_flap(self) -> bool:
        """Returns True if a flap was detected since last consume; clears the flag."""
        with self._lock:
            out = self._flap_requested
            self._flap_requested = False
            return out

    def _open_camera(self):
        """Open camera: try CAP_DSHOW on Windows first for faster startup, else default."""
        t0 = time.perf_counter()
        backends = []
        if sys.platform == "win32":
            if hasattr(cv2, "CAP_DSHOW"):
                backends.append((cv2.CAP_DSHOW, "CAP_DSHOW"))
            backends.append((None, "default"))
        else:
            backends.append((None, "default"))
        cap = None
        for api, name in backends:
            if not self._running:
                return None
            try:
                cap = (
                    cv2.VideoCapture(self.camera_index, api)
                    if api is not None
                    else cv2.VideoCapture(self.camera_index)
                )
                if cap.isOpened():
                    elapsed = time.perf_counter() - t0
                    logger.info("Webcam opened with %s in %.2fs", name, elapsed)
                    return cap
            except Exception as e:
                logger.debug("Webcam open with %s failed: %s", name, e)
            if cap is not None:
                cap.release()
                cap = None
        logger.warning("Webcam could not be opened with any backend after %.2fs", time.perf_counter() - t0)
        return None

    def _run_loop(self) -> None:
        t_start = time.perf_counter()
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        logger.info("Pose model created in %.2fs", time.perf_counter() - t_start)

        cap = self._open_camera()
        if cap is None:
            with self._lock:
                self._frame = self._blank_frame()
            return
        t_after_open = time.perf_counter() - t_start
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Bounded warmup: wait for first valid frame so we don't stall in main loop
        warmup_deadline = time.perf_counter() + WARMUP_TIMEOUT_S
        first_frame_at = None
        first_pose_at = None
        while self._running and cap.isOpened() and time.perf_counter() < warmup_deadline:
            ok, bgr = cap.read()
            if ok and bgr is not None:
                first_frame_at = time.perf_counter()
                logger.info(
                    "First webcam frame in %.2fs (open: %.2fs)",
                    first_frame_at - t_start,
                    t_after_open,
                )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                first_pose_at = time.perf_counter()
                logger.info(
                    "First pose inference in %.2fs",
                    first_pose_at - t_start,
                )
                frame_out = bgr.copy()
                if results.pose_landmarks:
                    self._draw_skeleton(frame_out, results.pose_landmarks)
                frame_resized = cv2.resize(frame_out, (self.width, self.height))
                with self._lock:
                    self._frame = frame_resized
                break
            time.sleep(WARMUP_POLL_INTERVAL_S)
        if first_frame_at is None:
            logger.warning(
                "No webcam frame within %.1fs; showing placeholder",
                WARMUP_TIMEOUT_S,
            )
            with self._lock:
                self._frame = self._blank_frame()
            cap.release()
            pose.close()
            return

        try:
            while self._running and cap.isOpened():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                frame_out = bgr.copy()
                if results.pose_landmarks:
                    self._draw_skeleton(frame_out, results.pose_landmarks)
                    self._update_flap_detection(results.pose_landmarks)
                else:
                    self._clear_arm_history()
                if self._cooldown_remaining > 0:
                    self._cooldown_remaining -= 1
                frame_resized = cv2.resize(frame_out, (self.width, self.height))
                with self._lock:
                    self._frame = frame_resized
        finally:
            cap.release()
            pose.close()
            with self._lock:
                if self._frame is None:
                    self._frame = self._blank_frame()

    def _blank_frame(self) -> np.ndarray:
        out = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        out[:] = (40, 40, 40)
        cv2.putText(
            out,
            "No webcam",
            (self.width // 4, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        return out

    def _draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks,
    ) -> None:
        h, w = frame.shape[:2]
        connections = mp.solutions.pose.POSE_CONNECTIONS
        for conn in connections:
            i, j = conn
            if i >= len(landmarks.landmark) or j >= len(landmarks.landmark):
                continue
            a = landmarks.landmark[i]
            b = landmarks.landmark[j]
            x1, y1 = int(a.x * w), int(a.y * h)
            x2, y2 = int(b.x * w), int(b.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 3, (0, 200, 255), -1)

    def _clear_arm_history(self) -> None:
        self._left_wrist_y_history.clear()
        self._right_wrist_y_history.clear()
        self._left_shoulder_y_history.clear()
        self._right_shoulder_y_history.clear()

    def _update_flap_detection(self, landmarks) -> None:
        ls = landmarks.landmark[LEFT_SHOULDER]
        rs = landmarks.landmark[RIGHT_SHOULDER]
        lw = landmarks.landmark[LEFT_WRIST]
        rw = landmarks.landmark[RIGHT_WRIST]
        ly_s, ry_s = ls.y, rs.y
        ly_w, ry_w = lw.y, rw.y
        self._left_shoulder_y_history.append(ly_s)
        self._right_shoulder_y_history.append(ry_s)
        self._left_wrist_y_history.append(ly_w)
        self._right_wrist_y_history.append(ry_w)

        if self._cooldown_remaining > 0:
            return
        if len(self._left_wrist_y_history) < HISTORY_LEN:
            return

        # Arm extension: shoulder-to-wrist distance (normalized). Use simple Y/X span.
        def extended(sw_x, sw_y, ww_x, ww_y):
            d = ((ww_x - sw_x) ** 2 + (ww_y - sw_y) ** 2) ** 0.5
            return d >= ARM_EXTENSION_MIN

        left_ext = extended(ls.x, ls.y, lw.x, lw.y)
        right_ext = extended(rs.x, rs.y, rw.x, rw.y)

        # Downward motion: wrist Y increased (in image coords, higher Y = lower position)
        left_down = self._left_wrist_y_history[-1] - self._left_wrist_y_history[0] >= DOWNWARD_THRESHOLD
        right_down = self._right_wrist_y_history[-1] - self._right_wrist_y_history[0] >= DOWNWARD_THRESHOLD

        # Both arms moved down and were somewhat extended → flap
        if left_ext and right_ext and left_down and right_down:
            with self._lock:
                self._flap_requested = True
            self._cooldown_remaining = FLAP_COOLDOWN_FRAMES
            self._clear_arm_history()
