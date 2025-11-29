"""
Detection matching utilities for comparing prediction sets.

This module provides the canonical matching algorithm used by analyzer,
evaluation, and gt_rectify modules, replacing 3+ duplicate implementations.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Set, Tuple

from .metrics import calculate_iou

if TYPE_CHECKING:
    from ..core.detection import Detection


@dataclass
class MatchResult:
    """
    Result of matching two detection sets.

    Attributes:
        matches: List of (pred_idx, gt_idx, iou) tuples for matched pairs
        unmatched_predictions: Set of prediction indices with no GT match
        unmatched_ground_truth: Set of GT indices with no prediction match
    """

    matches: List[Tuple[int, int, float]] = field(default_factory=list)
    unmatched_predictions: Set[int] = field(default_factory=set)
    unmatched_ground_truth: Set[int] = field(default_factory=set)

    @property
    def true_positives(self) -> int:
        """Number of matched predictions (true positives)."""
        return len(self.matches)

    @property
    def false_positives(self) -> int:
        """Number of unmatched predictions (false positives)."""
        return len(self.unmatched_predictions)

    @property
    def false_negatives(self) -> int:
        """Number of unmatched ground truth (false negatives)."""
        return len(self.unmatched_ground_truth)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)."""
        total_preds = self.true_positives + self.false_positives
        return self.true_positives / total_preds if total_preds > 0 else 0.0

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)."""
        total_gt = self.true_positives + self.false_negatives
        return self.true_positives / total_gt if total_gt > 0 else 0.0

    @property
    def f1_score(self) -> float:
        """F1 = 2 * precision * recall / (precision + recall)."""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def match_detections(
    predictions: "List[Detection]",
    ground_truth: "List[Detection]",
    iou_threshold: float = 0.5,
    match_class: bool = True,
    confidence_sorted: bool = True,
) -> MatchResult:
    """
    Match predictions to ground truth using greedy IoU matching.

    This is the canonical matching algorithm used throughout the codebase.
    Predictions are matched to ground truth in order of confidence (if enabled),
    with each ground truth only matching once.

    Args:
        predictions: List of predicted Detection objects
        ground_truth: List of ground truth Detection objects
        iou_threshold: Minimum IoU for a valid match
        match_class: If True, only match detections of the same class
        confidence_sorted: If True, sort predictions by confidence descending

    Returns:
        MatchResult containing matches and unmatched indices
    """
    if not predictions or not ground_truth:
        return MatchResult(
            matches=[],
            unmatched_predictions=set(range(len(predictions))),
            unmatched_ground_truth=set(range(len(ground_truth))),
        )

    # Sort predictions by confidence if requested
    if confidence_sorted:
        pred_order = sorted(
            range(len(predictions)), key=lambda i: predictions[i].confidence, reverse=True
        )
    else:
        pred_order = list(range(len(predictions)))

    matches = []
    matched_gt: Set[int] = set()
    matched_pred: Set[int] = set()

    for pred_idx in pred_order:
        pred = predictions[pred_idx]
        best_iou = iou_threshold
        best_gt_idx = -1

        for gt_idx, gt in enumerate(ground_truth):
            if gt_idx in matched_gt:
                continue

            if match_class and pred.class_id != gt.class_id:
                continue

            iou = calculate_iou(pred.bbox, gt.bbox)
            if iou >= best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0:
            matches.append((pred_idx, best_gt_idx, best_iou))
            matched_gt.add(best_gt_idx)
            matched_pred.add(pred_idx)

    return MatchResult(
        matches=matches,
        unmatched_predictions=set(range(len(predictions))) - matched_pred,
        unmatched_ground_truth=set(range(len(ground_truth))) - matched_gt,
    )


def compute_pairwise_iou_matrix(
    detections1: "List[Detection]", detections2: "List[Detection]"
) -> "List[List[float]]":
    """
    Compute NxM IoU matrix between two detection lists.

    Args:
        detections1: First list of N detections
        detections2: Second list of M detections

    Returns:
        NxM matrix where element [i][j] is IoU between detections1[i] and detections2[j]
    """
    n = len(detections1)
    m = len(detections2)
    matrix = [[0.0] * m for _ in range(n)]

    for i, det1 in enumerate(detections1):
        for j, det2 in enumerate(detections2):
            matrix[i][j] = calculate_iou(det1.bbox, det2.bbox)

    return matrix


def find_best_matches(
    detections1: "List[Detection]",
    detections2: "List[Detection]",
    iou_threshold: float = 0.5,
    match_class: bool = True,
) -> List[Tuple[int, int, float]]:
    """
    Find best 1-to-1 matches between two detection sets.

    Similar to match_detections but treats both sets symmetrically
    (no prediction vs GT distinction). Uses greedy matching by highest IoU.

    Args:
        detections1: First list of detections
        detections2: Second list of detections
        iou_threshold: Minimum IoU for a valid match
        match_class: If True, only match detections of the same class

    Returns:
        List of (idx1, idx2, iou) tuples for matched pairs
    """
    if not detections1 or not detections2:
        return []

    # Build list of all candidate pairs with their IoU
    candidates = []
    for i, det1 in enumerate(detections1):
        for j, det2 in enumerate(detections2):
            if match_class and det1.class_id != det2.class_id:
                continue
            iou = calculate_iou(det1.bbox, det2.bbox)
            if iou >= iou_threshold:
                candidates.append((i, j, iou))

    # Sort by IoU descending
    candidates.sort(key=lambda x: x[2], reverse=True)

    # Greedy matching
    matches = []
    used1: Set[int] = set()
    used2: Set[int] = set()

    for i, j, iou in candidates:
        if i not in used1 and j not in used2:
            matches.append((i, j, iou))
            used1.add(i)
            used2.add(j)

    return matches


__all__ = [
    "MatchResult",
    "match_detections",
    "compute_pairwise_iou_matrix",
    "find_best_matches",
]
