from typing import List

import numpy as np
import torch


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1: First box in [x, y, w, h] format
        box2: Second box in [x, y, w, h] format

    Returns:
        IoU value between 0 and 1
    """
    # Convert to xyxy format
    box1_xyxy = [
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2,
    ]
    box2_xyxy = [
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2,
    ]

    # Calculate intersection
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])

    if x2 < x1 or y2 < y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def calculate_giou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Generalized IoU (GIoU) between two bounding boxes.

    Args:
        box1: First box in [x, y, w, h] format
        box2: Second box in [x, y, w, h] format

    Returns:
        GIoU value between -1 and 1
    """
    iou = calculate_iou(box1, box2)

    # Convert to xyxy format
    box1_xyxy = [
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2,
    ]
    box2_xyxy = [
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2,
    ]

    # Calculate enclosing box
    enc_x1 = min(box1_xyxy[0], box2_xyxy[0])
    enc_y1 = min(box1_xyxy[1], box2_xyxy[1])
    enc_x2 = max(box1_xyxy[2], box2_xyxy[2])
    enc_y2 = max(box1_xyxy[3], box2_xyxy[3])

    enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)

    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union = area1 + area2 - (iou * (area1 + area2) / (1 + iou))

    return iou - (enc_area - union) / enc_area if enc_area > 0 else iou


def calculate_diou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Distance IoU (DIoU) between two bounding boxes.

    Args:
        box1: First box in [x, y, w, h] format
        box2: Second box in [x, y, w, h] format

    Returns:
        DIoU value
    """
    iou = calculate_iou(box1, box2)

    # Center distance
    center_distance = np.sqrt((box1[0] - box2[0]) ** 2 + (box1[1] - box2[1]) ** 2)

    # Convert to xyxy format
    box1_xyxy = [
        box1[0] - box1[2] / 2,
        box1[1] - box1[3] / 2,
        box1[0] + box1[2] / 2,
        box1[1] + box1[3] / 2,
    ]
    box2_xyxy = [
        box2[0] - box2[2] / 2,
        box2[1] - box2[3] / 2,
        box2[0] + box2[2] / 2,
        box2[1] + box2[3] / 2,
    ]

    # Diagonal of enclosing box
    enc_x1 = min(box1_xyxy[0], box2_xyxy[0])
    enc_y1 = min(box1_xyxy[1], box2_xyxy[1])
    enc_x2 = max(box1_xyxy[2], box2_xyxy[2])
    enc_y2 = max(box1_xyxy[3], box2_xyxy[3])

    diagonal = np.sqrt((enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2)

    return iou - (center_distance**2) / (diagonal**2) if diagonal > 0 else iou


def batch_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate IoU between all pairs of boxes from two sets.

    Args:
        boxes1: Tensor of shape (N, 4) in xyxy format
        boxes2: Tensor of shape (M, 4) in xyxy format

    Returns:
        IoU matrix of shape (N, M)
    """
    import torchvision.ops as ops

    return ops.box_iou(boxes1, boxes2)
