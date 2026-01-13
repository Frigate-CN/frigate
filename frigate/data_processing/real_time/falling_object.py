"""Real-time falling object detection processor."""

import datetime
import logging
import random
import string
from typing import Any

import cv2
import numpy as np

from frigate.comms.event_metadata_updater import (
    EventMetadataPublisher,
    EventMetadataTypeEnum,
)
from frigate.config import FrigateConfig

from ..types import DataProcessorMetrics
from .api import RealTimeProcessorApi

logger = logging.getLogger(__name__)


class FallingObjectRealTimeProcessor(RealTimeProcessorApi):
    """Real-time processor for detecting falling objects using motion analysis."""

    def __init__(
        self,
        config: FrigateConfig,
        requestor,
        event_metadata_publisher: EventMetadataPublisher,
        metrics: DataProcessorMetrics,
    ):
        self.requestor = requestor
        self.event_metadata_publisher = event_metadata_publisher
        self.config = config
        self.metrics = metrics

        logger.info(
            f"FallingObjectRealTimeProcessor initialized with {len(config.cameras)} cameras"
        )

        # Initialize background subtraction model for each camera
        self.bg_models: dict[str, cv2.BackgroundSubtractorMOG2] = {}
        self.tracked_trajectories: dict[
            str, dict
        ] = {}  # {camera_id: {object_id: {...}}}
        self.consecutive_detections: dict[
            str, dict
        ] = {}  # {camera_id: {object_id: count}}
        self.frame_counts: dict[str, int] = {}  # {camera_id: frame_count}

        # Configuration parameters
        self.BG_HISTORY = 1000
        self.BG_VAR_THRESHOLD = 25
        self.BG_DETECT_SHADOWS = (
            False  # Disable shadow detection to reduce false positives
        )
        self.WARMUP_FRAMES = 30
        self.MIN_VELOCITY_Y = 2.0  # Reduced to be less selective for falling objects
        self.MAX_VELOCITY_X = 5.0  # Allow slightly more horizontal movement
        self.MIN_FALL_DISTANCE = 15.0  # Reduced minimum fall distance
        self.TRAJECTORY_LENGTH = 10
        self.MIN_CONTOUR_AREA = 50  # Increased minimum area to filter small noise
        self.MIN_CONSECUTIVE_FRAMES = 2  # Reduced from 3 to 2 for easier detection
        self.EDGE_MARGIN = 20
        self.MAX_AGE = 30
        self.MAX_TRAJECTORY_STDDEV_RATIO = (
            0.6  # Increased to allow more velocity variation
        )

        super().__init__(config, metrics)

    def _get_bg_model(self, camera: str) -> cv2.BackgroundSubtractorMOG2:
        """Get or create background model for camera."""
        if camera not in self.bg_models:
            self.bg_models[camera] = cv2.createBackgroundSubtractorMOG2(
                history=self.BG_HISTORY,
                varThreshold=self.BG_VAR_THRESHOLD,
                detectShadows=self.BG_DETECT_SHADOWS,
            )
            self.tracked_trajectories[camera] = {}
            self.consecutive_detections[camera] = {}
            self.frame_counts[camera] = 0
        return self.bg_models[camera]

    def _get_object_id(self, center: tuple, bbox: tuple) -> str:
        """Generate a unique ID for tracking the object."""
        x, y = center
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        # Quantize position for stability
        x_q = int(x / 10)
        y_q = int(y / 10)
        w_q = int(width / 10)
        h_q = int(height / 10)

        return f"{x_q}_{y_q}_{w_q}_{h_q}"

    def _detect_motion(self, frame: np.ndarray, camera: str) -> list[dict[str, Any]]:
        """Detect moving objects using background subtraction."""
        # Convert YUV to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)

        # Convert to grayscale
        gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

        # Apply background subtraction
        bg_model = self._get_bg_model(camera)
        fg_mask = bg_model.apply(gray_frame, learningRate=-1)

        # Remove shadows
        fg_mask[fg_mask == 127] = 0

        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # Larger kernel for better noise reduction
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(
            fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort contours by area in descending order to prioritize larger objects
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Get camera config
        camera_config = self.config.cameras[camera]
        width = camera_config.detect.width or 320
        height = camera_config.detect.height or 320

        # Filter out camera shake/vibration effects by analyzing motion distribution
        # If motion is distributed evenly across the frame, it's likely camera shake
        total_fg_pixels = np.count_nonzero(fg_mask)
        if total_fg_pixels > (width * height * 0.3):  # If more than 30% of frame is moving
            # Likely camera shake, return empty detections
            return []

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.MIN_CONTOUR_AREA:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Exclude detections near frame edges
            if (
                x < self.EDGE_MARGIN
                or y < self.EDGE_MARGIN
                or x + w > width - self.EDGE_MARGIN
                or y + h > height - self.EDGE_MARGIN
            ):
                continue

            # Additional filtering: check aspect ratio to filter out very thin or wide detections
            aspect_ratio = w / h
            if aspect_ratio < 0.1 or aspect_ratio > 10:  # Very narrow or very wide
                continue

            # Additional filtering: check solidity (convexity) to filter out very irregular shapes
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area == 0:
                continue
            solidity = float(area) / hull_area
            if solidity < 0.2:  # Too irregular shape
                continue

            # Additional filtering: reject detections that cover too much of the frame
            # This helps filter out global motion due to camera shake
            if area > (width * height * 0.1):  # If detection is more than 10% of frame
                continue

            bbox = (x, y, x + w, y + h)
            center = ((x + x + w) / 2, (y + y + h) / 2)

            # Calculate confidence based on mask density and shape properties
            mask_roi = fg_mask[y : y + h, x : x + w]
            mask_density = np.count_nonzero(mask_roi) / (w * h)
            # Combine density with solidity for better confidence estimation
            confidence = mask_density * solidity

            detections.append(
                {
                    "center": center,
                    "bbox": bbox,
                    "confidence": confidence,
                }
            )

        return detections

    def _is_falling_object(
        self, trajectory: dict, confidence: float
    ) -> tuple[bool, dict[str, Any]]:
        """Determine if an object is falling based on trajectory analysis."""
        # Need minimum trajectory length
        if len(trajectory["trajectory"]) < 3:
            return False, {}

        # Check vertical velocity
        if len(trajectory["velocity_y"]) < 2:
            return False, {}

        avg_velocity_y = np.mean(trajectory["velocity_y"])
        if avg_velocity_y < self.MIN_VELOCITY_Y:
            return False, {}

        # Check horizontal velocity
        avg_velocity_x = np.mean(np.abs(trajectory["velocity_x"]))
        if avg_velocity_x > self.MAX_VELOCITY_X:
            return False, {}

        # Check fall distance
        start_pos = trajectory["trajectory"][0]
        current_pos = trajectory["trajectory"][-1]
        fall_distance = current_pos[1] - start_pos[1]

        if fall_distance < self.MIN_FALL_DISTANCE:
            return False, {}

        # Check trajectory consistency
        velocity_std = np.std(trajectory["velocity_y"])
        if velocity_std > avg_velocity_y * self.MAX_TRAJECTORY_STDDEV_RATIO:
            return False, {}

        # Check downward movement consistency
        downward_count = sum(1 for v in trajectory["velocity_y"] if v > 0)
        total_count = len(trajectory["velocity_y"])

        if downward_count / total_count < 0.7:
            return False, {}

        # Return True with metadata
        return True, {
            "velocity_y": float(avg_velocity_y),
            "velocity_x": float(avg_velocity_x),
            "fall_distance": float(fall_distance),
            "confidence": float(confidence),
            "trajectory": trajectory["trajectory"].copy(),
        }

    def process_frame(self, obj_data: dict[str, Any], frame: np.ndarray) -> None:
        """Process frame to detect falling objects."""
        camera = obj_data.get("camera")
        if not camera or camera not in self.config.cameras:
            return

        camera_config = self.config.cameras[camera]
        if not camera_config.falling_object.enabled:
            return

        # Increment frame count
        if camera not in self.frame_counts:
            self.frame_counts[camera] = 0
        self.frame_counts[camera] += 1

        frame_count = self.frame_counts[camera]
        bg_model = self._get_bg_model(camera)

        # Warmup period
        if frame_count < self.WARMUP_FRAMES:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_I420)
            gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
            bg_model.apply(gray_frame, learningRate=1.0)
            return

        # Get motion detections
        motion_detections = self._detect_motion(frame, camera)

        # Analyze trajectories
        for det in motion_detections:
            # Try to match with existing trajectories based on proximity
            matched_obj_id = None
            det_center = det["center"]

            # Look for existing objects that are close to this detection
            for existing_obj_id, existing_trajectory in self.tracked_trajectories[camera].items():
                if existing_trajectory["trajectory"]:  # If there are previous points
                    last_pos = existing_trajectory["trajectory"][-1]
                    distance = np.sqrt((det_center[0] - last_pos[0])**2 + (det_center[1] - last_pos[1])**2)

                    # If the detection is close to an existing trajectory, consider it the same object
                    # Use a distance threshold based on expected movement
                    if distance < 50:  # Adjust this threshold as needed
                        matched_obj_id = existing_obj_id
                        break

            if matched_obj_id is not None:
                object_id = matched_obj_id
            else:
                # Generate new object ID if no match found
                object_id = self._get_object_id(det["center"], det["bbox"])

            # Initialize tracking for new object
            if object_id not in self.tracked_trajectories[camera]:
                self.tracked_trajectories[camera][object_id] = {
                    "trajectory": [],
                    "velocity_y": [],
                    "velocity_x": [],
                    "first_seen": frame_count,
                }
                self.consecutive_detections[camera][object_id] = 1  # Start with 1 since we just detected it
            else:
                # Increment consecutive detection count for existing object
                self.consecutive_detections[camera][object_id] += 1

            trajectory = self.tracked_trajectories[camera][object_id]
            trajectory["trajectory"].append(det["center"])

            # Keep trajectory at fixed length
            if len(trajectory["trajectory"]) > self.TRAJECTORY_LENGTH:
                trajectory["trajectory"].pop(0)

            # Require minimum consecutive detections
            if (
                self.consecutive_detections[camera][object_id]
                < self.MIN_CONSECUTIVE_FRAMES
            ):
                # Only log when we have enough detections to almost meet the requirement
                if (
                    self.consecutive_detections[camera][object_id]
                    >= self.MIN_CONSECUTIVE_FRAMES - 1
                ):
                    logger.debug(
                        f"Object {object_id} has insufficient consecutive detections: {self.consecutive_detections[camera][object_id]}/{self.MIN_CONSECUTIVE_FRAMES}"
                    )
                continue

            # Calculate velocity if we have enough history
            if len(trajectory["trajectory"]) >= 2:
                prev_pos = trajectory["trajectory"][-2]
                curr_pos = trajectory["trajectory"][-1]

                vel_y = curr_pos[1] - prev_pos[1]
                vel_x = curr_pos[0] - prev_pos[0]

                trajectory["velocity_y"].append(vel_y)
                trajectory["velocity_x"].append(vel_x)

                # Keep velocity history limited
                if len(trajectory["velocity_y"]) > 5:
                    trajectory["velocity_y"].pop(0)
                    trajectory["velocity_x"].pop(0)

                # Check if falling object
                is_falling, metadata = self._is_falling_object(
                    trajectory, det["confidence"]
                )

                if is_falling:
                    logger.info(
                        f"Falling object detected on camera {camera}: "
                        f"velocity_y={metadata['velocity_y']:.2f}, "
                        f"fall_distance={metadata['fall_distance']:.2f}"
                    )

                    # Generate unique event ID
                    now = datetime.datetime.now().timestamp()
                    rand_id = "".join(
                        random.choices(string.ascii_lowercase + string.digits, k=6)
                    )
                    event_id = f"{now}-{rand_id}"

                    # Publish event metadata
                    self.event_metadata_publisher.publish(
                        (
                            now,
                            camera,
                            event_id,
                            det["bbox"],
                            det["confidence"],
                            metadata,
                        ),
                        EventMetadataTypeEnum.falling_object_event_create.value,
                    )
                else:
                    # Only log detailed failure reasons occasionally to avoid spam
                    if frame_count % 30 == 0:  # Log every 30 frames
                        avg_velocity_y = (
                            np.mean(trajectory["velocity_y"])
                            if trajectory["velocity_y"]
                            else 0
                        )
                        avg_velocity_x = (
                            np.mean(np.abs(trajectory["velocity_x"]))
                            if trajectory["velocity_x"]
                            else 0
                        )
                        start_pos = (
                            trajectory["trajectory"][0]
                            if trajectory["trajectory"]
                            else (0, 0)
                        )
                        current_pos = (
                            trajectory["trajectory"][-1]
                            if trajectory["trajectory"]
                            else (0, 0)
                        )
                        fall_distance = current_pos[1] - start_pos[1]

                        logger.debug(
                            f"Falling object check failed for {object_id}: "
                            f"vel_y={avg_velocity_y:.2f} (need>{self.MIN_VELOCITY_Y:.1f}), "
                            f"vel_x={avg_velocity_x:.2f} (max<{self.MAX_VELOCITY_X:.1f}), "
                            f"fall_dist={fall_distance:.2f} (need>{self.MIN_FALL_DISTANCE:.1f})"
                        )

        # Reset consecutive detection counter for objects not detected in this frame
        detected_object_ids = {
            self._get_object_id(det["center"], det["bbox"]) for det in motion_detections
        }

        # Decrement consecutive detection counter for objects not detected in this frame
        for obj_id in list(self.consecutive_detections[camera].keys()):
            if obj_id not in detected_object_ids:
                # If object not detected in this frame, decrement the counter
                # but don't let it go below 1 to allow for brief occlusions
                # This allows objects to maintain some "memory" even when briefly not detected
                self.consecutive_detections[camera][obj_id] = max(
                    1, self.consecutive_detections[camera][obj_id] - 1
                )

                # If counter reaches 1 and stays there for a while, consider removing old trajectories
                # Only remove very old trajectories that have minimal activity
                trajectory = self.tracked_trajectories[camera].get(obj_id)
                if trajectory:
                    age = frame_count - trajectory["first_seen"]
                    # Remove if very old and has minimal consecutive detections
                    if age > self.MAX_AGE * 2 and self.consecutive_detections[camera][obj_id] <= 1:
                        if obj_id in self.tracked_trajectories[camera]:
                            del self.tracked_trajectories[camera][obj_id]
                        if obj_id in self.consecutive_detections[camera]:
                            del self.consecutive_detections[camera][obj_id]

        # Cleanup old trajectories
        self._cleanup_trajectories(camera)

    def _cleanup_trajectories(self, camera: str):
        """Remove old trajectories."""
        frame_count = self.frame_counts.get(camera, 0)
        to_remove = []

        for obj_id, trajectory in self.tracked_trajectories[camera].items():
            age = frame_count - trajectory["first_seen"]
            # Only remove if the trajectory is old AND has low consecutive detections
            # This allows newer trajectories to persist even if they don't have many consecutive detections yet
            if age > self.MAX_AGE:
                to_remove.append(obj_id)

        for obj_id in to_remove:
            if obj_id in self.tracked_trajectories[camera]:
                del self.tracked_trajectories[camera][obj_id]
            if obj_id in self.consecutive_detections[camera]:
                del self.consecutive_detections[camera][obj_id]

    def handle_request(
        self, topic: str, request_data: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Handle metadata requests."""
        if topic == "get_falling_object_trajectories":
            camera = request_data.get("camera")
            if camera:
                return self.get_trajectories(camera)
        return None

    def get_trajectories(self, camera: str) -> list[dict[str, Any]]:
        """Get current trajectories for a camera."""
        if camera not in self.tracked_trajectories:
            return []

        trajectories = []
        current_time = datetime.datetime.now().timestamp()

        for obj_id, trajectory_data in self.tracked_trajectories[camera].items():
            # Convert trajectory points to the format expected by frontend
            trajectory_points = [
                {
                    "x": point[0],
                    "y": point[1],
                    "timestamp": current_time
                    - (len(trajectory_data["trajectory"]) - i - 1)
                    * 0.1,  # Approximate timestamps
                }
                for i, point in enumerate(trajectory_data["trajectory"])
            ]

            if (
                len(trajectory_points) >= 3
            ):  # Only include trajectories with minimum points
                trajectory_entry = {
                    "objectId": obj_id,
                    "trajectory": trajectory_points,
                    "velocity_y": float(
                        np.mean(trajectory_data["velocity_y"])
                        if trajectory_data["velocity_y"]
                        else 0
                    ),
                    "velocity_x": float(
                        np.mean(trajectory_data["velocity_x"])
                        if trajectory_data["velocity_x"]
                        else 0
                    ),
                    "fall_distance": float(
                        max(trajectory_data["trajectory"], key=lambda p: p[1])[1]
                        - min(trajectory_data["trajectory"], key=lambda p: p[1])[1]
                    ),
                    "confidence": 0.8,  # Default confidence
                    "start_time": trajectory_data["first_seen"] * 0.1
                    + current_time
                    - len(trajectory_data["trajectory"]) * 0.1,
                }
                trajectories.append(trajectory_entry)

        return trajectories

    def expire_object(self, object_id: str, camera: str) -> None:
        """Handle objects that are no longer detected."""
        if (
            camera in self.tracked_trajectories
            and object_id in self.tracked_trajectories[camera]
        ):
            del self.tracked_trajectories[camera][object_id]
        if (
            camera in self.consecutive_detections
            and object_id in self.consecutive_detections[camera]
        ):
            del self.consecutive_detections[camera][object_id]
