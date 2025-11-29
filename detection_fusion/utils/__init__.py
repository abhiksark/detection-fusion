from .aggregation import (
    calculate_model_weights,
    get_detection_weight,
    merge_cluster_simple,
    merge_cluster_weighted,
    vote_class_majority,
    vote_class_weighted,
)
from .clustering import (
    cluster_by_center_distance,
    cluster_by_iou,
    flatten_detections,
)
from .geometry import (
    box_area,
    box_center,
    box_intersection,
    boxes_overlap,
    center_distance,
    scale_box,
    xywh_to_xyxy,
    xyxy_to_xywh,
)
from .matching import (
    MatchResult,
    compute_pairwise_iou_matrix,
    find_best_matches,
    match_detections,
)
from .metrics import batch_iou, calculate_diou, calculate_giou, calculate_iou

__all__ = [
    # Metrics
    "calculate_iou",
    "calculate_giou",
    "calculate_diou",
    "batch_iou",
    # Geometry
    "xywh_to_xyxy",
    "xyxy_to_xywh",
    "box_center",
    "box_area",
    "box_intersection",
    "center_distance",
    "scale_box",
    "boxes_overlap",
    # Clustering
    "cluster_by_iou",
    "cluster_by_center_distance",
    "flatten_detections",
    # Matching
    "MatchResult",
    "match_detections",
    "compute_pairwise_iou_matrix",
    "find_best_matches",
    # Aggregation
    "calculate_model_weights",
    "merge_cluster_weighted",
    "merge_cluster_simple",
    "vote_class_majority",
    "vote_class_weighted",
    "get_detection_weight",
]
