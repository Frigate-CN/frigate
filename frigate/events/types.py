"""Types for event management."""

from enum import Enum


class EventTypeEnum(str, Enum):
    api = "api"
    tracked_object = "tracked_object"
    falling_object = "falling_object"


class EventStateEnum(str, Enum):
    start = "start"
    update = "update"
    end = "end"


class FallingObjectSeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class RegenerateDescriptionEnum(str, Enum):
    thumbnails = "thumbnails"
    snapshot = "snapshot"
