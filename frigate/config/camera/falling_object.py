"""Configuration for falling object detection."""

from typing import Optional

from pydantic import Field, field_validator

from ..base import FrigateBaseModel

__all__ = [
    "FallingObjectConfig",
    "FallingObjectFilterConfig",
    "FallingObjectZoneConfig",
]


class FallingObjectFilterConfig(FrigateBaseModel):
    """Filter configuration for falling object detection."""

    enabled: bool = Field(
        default=True,
        title="Enable falling object detection.",
    )
    min_confidence: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        title="Minimum confidence threshold for falling object detection.",
    )
    min_velocity_y: float = Field(
        default=5.0,
        ge=0.0,
        title="Minimum vertical velocity (pixels/frame) to consider as falling.",
    )
    max_velocity_x: float = Field(
        default=3.0,
        ge=0.0,
        title="Maximum horizontal velocity to filter out moving objects.",
    )
    min_fall_distance: float = Field(
        default=30.0,
        ge=0.0,
        title="Minimum fall distance (pixels) to trigger detection.",
    )
    trajectory_length: int = Field(
        default=10,
        ge=3,
        le=50,
        title="Number of trajectory points to track.",
    )
    objects: list[str] = Field(
        default_factory=lambda: ["person", "bottle", "cup", "backpack", "handbag"],
        title="Object types to monitor for falling.",
    )

    @field_validator("objects", mode="before")
    @classmethod
    def validate_objects(cls, v):
        if isinstance(v, str):
            return [obj.strip() for obj in v.split(",")]
        return v


class FallingObjectZoneConfig(FrigateBaseModel):
    """Zone configuration for falling object detection."""

    enabled: bool = Field(
        default=False,
        title="Enable falling object detection in this zone.",
    )
    trigger_alert: bool = Field(
        default=True,
        title="Trigger alert when falling object detected in this zone.",
    )
    severity: Optional[str] = Field(
        default="high",
        title="Severity level for falling object events.",
    )
    min_height_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        title="Minimum height (0-1, relative to frame) to trigger detection.",
    )


class FallingObjectConfig(FrigateBaseModel):
    """Main configuration for falling object detection."""

    enabled: bool = Field(
        default=False,
        title="Enable falling object detection for camera.",
    )
    global_filter: FallingObjectFilterConfig = Field(
        default_factory=FallingObjectFilterConfig,
        title="Global filter configuration for falling objects.",
    )
    alert_on_all_detections: bool = Field(
        default=True,
        title="Alert on all falling object detections, not just those in zones.",
    )
    save_snapshots: bool = Field(
        default=True,
        title="Save snapshots when falling object detected.",
    )
    save_clips: bool = Field(
        default=True,
        title="Save video clips when falling object detected.",
    )
    cooldown: int = Field(
        default=10,
        ge=0,
        title="Cooldown period in seconds before triggering another alert.",
    )
    severity_threshold: str = Field(
        default="medium",
        title="Minimum severity level to trigger alert.",
    )
    estimated_impact_warning: bool = Field(
        default=True,
        title="Include estimated time to impact in alerts.",
    )
