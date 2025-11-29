"""
Geometric utilities for bounding box operations.

All functions work with boxes in [x_center, y_center, width, height] format
unless otherwise specified.
"""

from typing import List, Tuple

import numpy as np

# Re-export IoU functions from metrics for convenience
from .metrics import batch_iou, calculate_diou, calculate_giou, calculate_iou


def xywh_to_xyxy(box: List[float]) -> List[float]:
    """
    Convert box from center format to corner format.

    Args:
        box: Box in [x_center, y_center, width, height] format

    Returns:
        Box in [x1, y1, x2, y2] format (top-left and bottom-right corners)
    """
    x, y, w, h = box
    return [x - w / 2, y - h / 2, x + w / 2, y + h / 2]


def xyxy_to_xywh(box: List[float]) -> List[float]:
    """
    Convert box from corner format to center format.

    Args:
        box: Box in [x1, y1, x2, y2] format

    Returns:
        Box in [x_center, y_center, width, height] format
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return [x1 + w / 2, y1 + h / 2, w, h]


def box_center(box: List[float]) -> Tuple[float, float]:
    """
    Get center point of a box.

    Args:
        box: Box in [x_center, y_center, width, height] format

    Returns:
        Tuple of (x_center, y_center)
    """
    return (box[0], box[1])


def box_area(box: List[float]) -> float:
    """
    Calculate area of a box.

    Args:
        box: Box in [x_center, y_center, width, height] format

    Returns:
        Area of the box
    """
    return box[2] * box[3]


def box_intersection(box1: List[float], box2: List[float]) -> float:
    """
    Calculate intersection area between two boxes.

    Args:
        box1: First box in [x_center, y_center, width, height] format
        box2: Second box in [x_center, y_center, width, height] format

    Returns:
        Intersection area (0 if boxes don't overlap)
    """
    # Convert to xyxy
    b1 = xywh_to_xyxy(box1)
    b2 = xywh_to_xyxy(box2)

    # Calculate intersection
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def center_distance(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Euclidean distance between box centers.

    Args:
        box1: First box in [x_center, y_center, width, height] format
        box2: Second box in [x_center, y_center, width, height] format

    Returns:
        Distance between centers
    """
    return np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)


def scale_box(box: List[float], scale: float) -> List[float]:
    """
    Scale a box around its center.

    Args:
        box: Box in [x_center, y_center, width, height] format
        scale: Scale factor (1.0 = no change)

    Returns:
        Scaled box (center unchanged, width/height scaled)
    """
    return [box[0], box[1], box[2] * scale, box[3] * scale]


def boxes_overlap(box1: List[float], box2: List[float]) -> bool:
    """
    Check if two boxes have any overlap.

    Args:
        box1: First box in [x_center, y_center, width, height] format
        box2: Second box in [x_center, y_center, width, height] format

    Returns:
        True if boxes overlap, False otherwise
    """
    return box_intersection(box1, box2) > 0


__all__ = [
    # Coordinate conversion
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    # Box properties
    "box_center",
    "box_area",
    "box_intersection",
    "center_distance",
    "scale_box",
    "boxes_overlap",
    # Re-exported from metrics
    "calculate_iou",
    "calculate_giou",
    "calculate_diou",
    "batch_iou",
]
