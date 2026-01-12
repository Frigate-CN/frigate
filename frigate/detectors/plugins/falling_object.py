"""Falling object detector plugin for Frigate.

This module implements a specialized detector for falling objects
based on motion analysis, trajectory tracking, and velocity calculation.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np

from frigate.detectors.detection_api import DetectionApi
from frigate.detectors.detector_config import BaseDetectorConfig

logger = logging.getLogger(__name__)


class FallingObjectDetectorConfig(BaseDetectorConfig):
    """Configuration for falling object detector."""

    type_key: str = "falling_object"


class FallingObjectDetection:
    """Represents a falling object detection."""

    def __init__(
        self,
        label: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        velocity_y: float,
        trajectory: List[Tuple[float, float]],
        start_height: float,
        current_height: float,
        fall_distance: float,
        estimated_time_to_impact: Optional[float] = None,
    ):
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # (x1, y1, x2, y2)
        self.velocity_y = velocity_y  # vertical velocity (positive = downward)
        self.trajectory = trajectory  # list of (x, y) positions
        self.start_height = start_height
        self.current_height = current_height
        self.fall_distance = fall_distance
        self.estimated_time_to_impact = estimated_time_to_impact

    def to_array(self) -> np.ndarray:
        """Convert to detection array format."""
        return np.array(
            [
                0,  # label index (will be mapped to falling_object)
                self.confidence,
                self.bbox[0],  # x1
                self.bbox[1],  # y1
                self.bbox[2],  # x2
                self.bbox[3],  # y2
            ]
        )


class FallingObjectDetector(DetectionApi):
    """
    Falling object detector based on motion analysis and trajectory tracking.

    Key features:
    - Detects objects with downward velocity exceeding threshold
    - Tracks trajectory to confirm falling pattern
    - Filters out objects that are not actually falling (e.g., swaying)
    - Supports multiple object types (person, bottle, etc.)
    """

    type_key: str = "falling_object"
    supported_models = []

    # Minimum vertical velocity (pixels/frame) to consider as falling
    MIN_VELOCITY_Y: float = 5.0

    # Maximum horizontal velocity to filter out moving objects
    MAX_VELOCITY_X: float = 3.0

    # Minimum fall distance (pixels) to trigger detection
    MIN_FALL_DISTANCE: float = 30.0

    # Number of trajectory points to track
    TRAJECTORY_LENGTH: int = 10

    # Confidence thresholds for different object types
    CONFIDENCE_THRESHOLDS = {
        "person": 0.7,
        "bottle": 0.6,
        "cup": 0.6,
        "backpack": 0.6,
        "handbag": 0.6,
        "default": 0.5,
    }

    def __init__(self, detector_config: FallingObjectDetectorConfig):
        super().__init__(detector_config)
        self.height = detector_config.model.height
        self.width = detector_config.model.width

        # Trajectory tracking: {object_id: {'trajectory': [], 'velocity': [], 'last_seen': 0}}
        self.tracked_trajectories: dict = {}

        # Object detection from base detector
        self.base_detector = None
        self._init_base_detector(detector_config)

        # Frame counter for trajectory cleanup
        self.frame_count = 0

    def _init_base_detector(self, detector_config: BaseDetectorConfig):
        """Initialize base object detector for identifying objects."""
        from frigate.detectors import create_detector

        # Use CPU detector as base for object identification
        base_config = detector_config.model_copy()
        base_config.type = "cpu"
        self.base_detector = create_detector(base_config)

    def detect_raw(self, tensor_input: np.ndarray) -> np.ndarray:
        """
        Detect falling objects from video frame.

        Args:
            tensor_input: Input tensor (1, height, width, 3)

        Returns:
            Array of detections in format [[label_idx, confidence, x1, y1, x2, y2], ...]
        """
        self.frame_count += 1

        # Get base object detections
        base_detections = self._get_base_detections(tensor_input)

        # Analyze trajectories for falling objects
        falling_detections = self._detect_falling_objects(base_detections)

        # Clean up old trajectories
        self._cleanup_trajectories()

        return falling_detections

    def _get_base_detections(self, tensor_input: np.ndarray) -> List[dict]:
        """
        Get base object detections from the underlying detector.

        Args:
            tensor_input: Input tensor

        Returns:
            List of detection dictionaries with 'label', 'confidence', 'bbox'
        """
        if self.base_detector is None:
            return []

        # Get raw detections from base detector
        raw_detections = self.base_detector.detect(tensor_input, threshold=0.3)

        detections = []
        for label, confidence, bbox in raw_detections:
            center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            detections.append(
                {
                    "label": label,
                    "confidence": confidence,
                    "bbox": bbox,
                    "center": center,
                }
            )

        return detections

    def _detect_falling_objects(self, detections: List[dict]) -> np.ndarray:
        """
        Analyze trajectories to detect falling objects.

        Args:
            detections: List of current frame detections

        Returns:
            Array of falling object detections
        """
        falling_objects = []

        # Update trajectories for each detection
        for det in detections:
            object_id = self._get_object_id(det)

            # Update or create trajectory
            if object_id not in self.tracked_trajectories:
                self.tracked_trajectories[object_id] = {
                    "trajectory": [],
                    "velocity_y": [],
                    "velocity_x": [],
                    "first_seen": self.frame_count,
                }

            trajectory = self.tracked_trajectories[object_id]
            trajectory["trajectory"].append(det["center"])

            # Keep trajectory at fixed length
            if len(trajectory["trajectory"]) > self.TRAJECTORY_LENGTH:
                trajectory["trajectory"].pop(0)

            # Calculate velocity if we have enough history
            if len(trajectory["trajectory"]) >= 2:
                prev_pos = trajectory["trajectory"][-2]
                curr_pos = trajectory["trajectory"][-1]

                vel_y = curr_pos[1] - prev_pos[1]  # y increases downward
                vel_x = curr_pos[0] - prev_pos[0]

                trajectory["velocity_y"].append(vel_y)
                trajectory["velocity_x"].append(vel_x)

                # Keep velocity history limited
                if len(trajectory["velocity_y"]) > 5:
                    trajectory["velocity_y"].pop(0)
                    trajectory["velocity_x"].pop(0)

                # Check if this is a falling object
                if self._is_falling_object(
                    trajectory,
                    det["label"],
                    det["confidence"],
                ):
                    # Calculate fall metrics
                    start_pos = trajectory["trajectory"][0]
                    current_pos = trajectory["trajectory"][-1]

                    fall_distance = current_pos[1] - start_pos[1]
                    avg_velocity_y = np.mean(trajectory["velocity_y"])

                    # Estimate time to impact (assuming constant velocity)
                    estimated_time = None
                    if avg_velocity_y > 0:
                        distance_to_bottom = self.height - current_pos[1]
                        estimated_time = distance_to_bottom / avg_velocity_y

                    falling_detection = FallingObjectDetection(
                        label=det["label"],
                        confidence=det["confidence"],
                        bbox=det["bbox"],
                        velocity_y=avg_velocity_y,
                        trajectory=trajectory["trajectory"].copy(),
                        start_height=start_pos[1],
                        current_height=current_pos[1],
                        fall_distance=fall_distance,
                        estimated_time_to_impact=estimated_time,
                    )

                    falling_objects.append(falling_detection.to_array())

        # Convert to numpy array
        if falling_objects:
            return np.array(falling_objects, dtype=np.float32)
        else:
            return np.array([], dtype=np.float32).reshape(0, 6)

    def _is_falling_object(
        self,
        trajectory: dict,
        label: str,
        confidence: float,
    ) -> bool:
        """
        Determine if an object is falling based on trajectory analysis.

        Args:
            trajectory: Trajectory tracking data
            label: Object label
            confidence: Detection confidence

        Returns:
            True if object is falling, False otherwise
        """
        # Check confidence threshold
        threshold = self.CONFIDENCE_THRESHOLDS.get(
            label, self.CONFIDENCE_THRESHOLDS["default"]
        )
        if confidence < threshold:
            return False

        # Need minimum trajectory length
        if len(trajectory["trajectory"]) < 3:
            return False

        # Check vertical velocity
        if len(trajectory["velocity_y"]) < 2:
            return False

        avg_velocity_y = np.mean(trajectory["velocity_y"])
        if avg_velocity_y < self.MIN_VELOCITY_Y:
            return False

        # Check horizontal velocity (should be relatively low for falling objects)
        avg_velocity_x = np.mean(np.abs(trajectory["velocity_x"]))
        if avg_velocity_x > self.MAX_VELOCITY_X:
            return False

        # Check fall distance
        start_pos = trajectory["trajectory"][0]
        current_pos = trajectory["trajectory"][-1]
        fall_distance = current_pos[1] - start_pos[1]

        if fall_distance < self.MIN_FALL_DISTANCE:
            return False

        # Check trajectory consistency (velocity should be relatively stable)
        velocity_std = np.std(trajectory["velocity_y"])
        if velocity_std > avg_velocity_y * 0.5:
            return False  # Too much variation, likely not falling

        # Additional check: trajectory should show consistent downward movement
        # Count downward movements vs upward movements
        downward_count = sum(1 for v in trajectory["velocity_y"] if v > 0)
        total_count = len(trajectory["velocity_y"])

        if downward_count / total_count < 0.7:
            return False  # Less than 70% of movements are downward

        logger.debug(
            f"Falling object detected: label={label}, "
            f"velocity_y={avg_velocity_y:.2f}, "
            f"velocity_x={avg_velocity_x:.2f}, "
            f"fall_distance={fall_distance:.2f}"
        )

        return True

    def _get_object_id(self, detection: dict) -> str:
        """
        Generate a unique ID for tracking the object.

        Uses position and size to create a stable ID.
        """
        x, y = detection["center"]
        bbox = detection["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Quantize position for stability
        x_q = int(x / 10)
        y_q = int(y / 10)
        w_q = int(width / 10)
        h_q = int(height / 10)

        return f"{detection['label']}_{x_q}_{y_q}_{w_q}_{h_q}"

    def _cleanup_trajectories(self):
        """Remove old trajectories to prevent memory leaks."""
        max_age = 30  # frames

        to_remove = []
        for obj_id, trajectory in self.tracked_trajectories.items():
            age = self.frame_count - trajectory["first_seen"]
            if age > max_age:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            del self.tracked_trajectories[obj_id]

    def set_stop_event(self, stop_event):
        """Set stop event for graceful shutdown."""
        if hasattr(self.base_detector, "set_stop_event"):
            self.base_detector.set_stop_event(stop_event)
