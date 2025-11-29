"""Pytest configuration and shared fixtures."""

from pathlib import Path
from typing import Dict, List

import pytest

from detection_fusion import Detection, DetectionSet


@pytest.fixture
def sample_detection() -> Detection:
    """Create a sample detection."""
    return Detection(
        class_id=0,
        x=0.5,
        y=0.5,
        w=0.1,
        h=0.1,
        confidence=0.9,
        model_source="test_model",
        image_name="test_image",
    )


@pytest.fixture
def sample_detections() -> List[Detection]:
    """Create a list of sample detections."""
    return [
        Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1, confidence=0.9, image_name="img1"),
        Detection(class_id=0, x=0.51, y=0.51, w=0.1, h=0.1, confidence=0.85, image_name="img1"),
        Detection(class_id=1, x=0.3, y=0.3, w=0.15, h=0.15, confidence=0.8, image_name="img1"),
    ]


@pytest.fixture
def multi_model_detections() -> Dict[str, List[Detection]]:
    """Create detections from multiple models."""
    return {
        "model1": [
            Detection(class_id=0, x=0.5, y=0.5, w=0.1, h=0.1, confidence=0.9, image_name="img1"),
            Detection(class_id=1, x=0.3, y=0.3, w=0.15, h=0.15, confidence=0.85, image_name="img1"),
        ],
        "model2": [
            Detection(class_id=0, x=0.51, y=0.51, w=0.1, h=0.1, confidence=0.88, image_name="img1"),
            Detection(
                class_id=1, x=0.31, y=0.31, w=0.14, h=0.14, confidence=0.82, image_name="img1"
            ),
        ],
        "model3": [
            Detection(
                class_id=0, x=0.49, y=0.49, w=0.11, h=0.11, confidence=0.87, image_name="img1"
            ),
        ],
    }


@pytest.fixture
def detection_set(multi_model_detections) -> DetectionSet:
    """Create a DetectionSet from multi-model detections."""
    return DetectionSet(multi_model_detections)


@pytest.fixture
def tmp_labels_dir(tmp_path: Path) -> Path:
    """Create a temporary labels directory with YOLO format annotations."""
    labels_dir = tmp_path / "labels"

    # Create model directories
    for model in ["model1", "model2"]:
        model_dir = labels_dir / model
        model_dir.mkdir(parents=True)

        # Create annotation files
        for img in ["img1", "img2"]:
            ann_file = model_dir / f"{img}.txt"
            ann_file.write_text("0 0.5 0.5 0.1 0.1 0.9\n1 0.3 0.3 0.15 0.15 0.85\n")

    return labels_dir


@pytest.fixture
def tmp_gt_dir(tmp_path: Path) -> Path:
    """Create a temporary ground truth directory."""
    gt_dir = tmp_path / "GT"
    gt_dir.mkdir(parents=True)

    for img in ["img1", "img2"]:
        ann_file = gt_dir / f"{img}.txt"
        ann_file.write_text("0 0.5 0.5 0.1 0.1\n1 0.3 0.3 0.15 0.15\n")

    return gt_dir


@pytest.fixture
def yolo_labels_dir(tmp_path: Path) -> Path:
    """Create a temporary YOLO labels directory with model subdirectories."""
    labels_dir = tmp_path / "labels"

    # Create multiple model directories with YOLO annotations
    models = ["yolov8n", "yolov8s", "yolov8m"]
    for model in models:
        model_dir = labels_dir / model
        model_dir.mkdir(parents=True)

        # Create annotation files for multiple images
        for img_idx in range(3):
            ann_file = model_dir / f"image_{img_idx:04d}.txt"
            # Write varied detections per model
            lines = []
            lines.append(f"0 0.5 0.5 0.1 0.1 0.{90 - img_idx}")
            if img_idx % 2 == 0:
                lines.append(f"1 0.3 0.3 0.15 0.15 0.{85 - img_idx}")
            ann_file.write_text("\n".join(lines) + "\n")

    return labels_dir
