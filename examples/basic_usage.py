#!/usr/bin/env python3
"""
Basic Usage Example for DetectionFusion v1.0

This example demonstrates the core v1.0 API patterns:
- merge_detections() convenience function
- Detection class with keyword arguments
- StrategyRegistry for creating strategies
- DetectionPipeline for fluent workflows
"""


def example_merge_detections():
    """Example 1: Simple merge using convenience function."""
    print("=== Example 1: merge_detections() ===")

    from detection_fusion import StrategyRegistry, merge_detections

    # List all available strategies
    strategies = StrategyRegistry.list_all()
    print(f"Available strategies: {len(strategies)}")
    print(f"  {', '.join(strategies[:5])}...")

    # Merge detections using weighted voting
    # This loads from labels/ directory and merges all model subdirectories
    try:
        results = merge_detections(path="labels", strategy="weighted_vote", iou_threshold=0.5)
        print(f"Merged result: {len(results)} detections")
    except FileNotFoundError:
        print("  (Skipped - labels/ directory not found)")

    print()


def example_detection_class():
    """Example 2: Working with Detection objects."""
    print("=== Example 2: Detection Class ===")

    from detection_fusion import Detection

    # Create detections using keyword arguments (required in v1.0)
    det1 = Detection(
        class_id=0,
        x=0.5,
        y=0.5,  # Center coordinates (normalized)
        w=0.2,
        h=0.3,  # Width and height (normalized)
        confidence=0.95,
        model_source="yolov8",
    )
    print(
        f"Detection 1: class={det1.class_id}, conf={det1.confidence:.2f}, source={det1.model_source}"
    )

    det2 = Detection(
        class_id=0, x=0.52, y=0.48, w=0.21, h=0.29, confidence=0.87, model_source="yolov9"
    )
    print(
        f"Detection 2: class={det2.class_id}, conf={det2.confidence:.2f}, source={det2.model_source}"
    )

    # Detection is immutable (frozen Pydantic model)
    # Use with_* methods to create modified copies
    det3 = det1.with_confidence(0.99)
    print(f"Modified copy: conf={det3.confidence:.2f} (original unchanged: {det1.confidence:.2f})")

    # Calculate IoU between detections
    iou = det1.iou_with(det2)
    print(f"IoU between det1 and det2: {iou:.3f}")

    print()


def example_strategy_registry():
    """Example 3: Using StrategyRegistry."""
    print("=== Example 3: Strategy Registry ===")

    from detection_fusion import Detection
    from detection_fusion.strategies import StrategyRegistry, create_strategy

    # Create strategy using the registry
    strategy = create_strategy("weighted_vote", iou_threshold=0.5)
    print(f"Created strategy: {strategy.metadata.name}")
    print(f"  Category: {strategy.metadata.category}")
    print(f"  Description: {strategy.metadata.description[:60]}...")

    # Create sample detections from multiple models
    model1_dets = [
        Detection(class_id=0, x=0.5, y=0.5, w=0.2, h=0.3, confidence=0.9, model_source="model1"),
        Detection(class_id=1, x=0.3, y=0.7, w=0.15, h=0.2, confidence=0.85, model_source="model1"),
    ]
    model2_dets = [
        Detection(
            class_id=0, x=0.51, y=0.49, w=0.21, h=0.29, confidence=0.88, model_source="model2"
        ),
        Detection(
            class_id=1, x=0.31, y=0.69, w=0.14, h=0.21, confidence=0.82, model_source="model2"
        ),
    ]

    # Merge detections
    detections_by_model = {
        "model1": model1_dets,
        "model2": model2_dets,
    }

    merged = strategy.merge(detections_by_model)
    print(f"Merged {len(model1_dets) + len(model2_dets)} detections -> {len(merged)} results")

    # List strategies by category
    print("\nStrategies by category:")
    categories = {}
    for name in StrategyRegistry.list_all():
        s = StrategyRegistry.create(name)
        cat = s.metadata.category
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(name)

    for cat, names in sorted(categories.items()):
        print(f"  {cat}: {', '.join(names[:3])}{'...' if len(names) > 3 else ''}")

    print()


def example_pipeline():
    """Example 4: Using DetectionPipeline for fluent workflows."""
    print("=== Example 4: DetectionPipeline ===")

    from detection_fusion import DetectionPipeline

    print("Pipeline pattern (fluent interface):")
    print("""
    result = (
        DetectionPipeline()
        .load("labels/", format="yolo")
        .ensemble("weighted_vote", iou_threshold=0.5)
        .evaluate("labels/GT/")
        .run()
    )

    print(f"Merged: {len(result.ensemble_result)} detections")
    print(f"Precision: {result.evaluation_result.precision:.3f}")
    print(f"Recall: {result.evaluation_result.recall:.3f}")
    """)

    # Demo without actual files
    pipeline = DetectionPipeline()
    print(f"Pipeline created with {len(pipeline._stages)} stages")

    print()


def example_format_conversion():
    """Example 5: Format conversion."""
    print("=== Example 5: Format Conversion ===")

    from detection_fusion import FormatRegistry

    formats = FormatRegistry.list_formats()
    print(f"Supported readers: {', '.join(formats['readers'])}")
    print(f"Supported writers: {', '.join(formats['writers'])}")

    print("\nConversion example:")
    print("""
    from detection_fusion import convert_annotations

    # Convert VOC XML to YOLO
    convert_annotations(
        input_path="annotations.xml",
        output_path="labels.txt",
        input_format="voc_xml",
        output_format="yolo"
    )

    # Or via CLI:
    # detection-fusion convert -i annotations.xml -o labels.txt --from voc_xml --to yolo
    """)

    print()


def main():
    """Run all basic usage examples."""
    print("DetectionFusion v1.0 - Basic Usage Examples")
    print("=" * 50)
    print()

    example_merge_detections()
    example_detection_class()
    example_strategy_registry()
    example_pipeline()
    example_format_conversion()

    print("=" * 50)
    print("CLI Quick Reference:")
    print("-" * 50)
    print("""
# List available strategies
detection-fusion list-strategies
detection-fusion list-strategies --category voting -v

# Merge detections from multiple models
detection-fusion merge --input labels/ --strategy weighted_vote --output unified/

# Validate against ground truth
detection-fusion validate --input labels/ --gt GT/ --strategy weighted_vote

# Convert annotation formats
detection-fusion convert --input annotations.xml --output labels/ \\
    --input-format voc_xml --output-format yolo

# GT rectification
detection-fusion rectify --config configs/gt_rectification/balanced.yaml
""")


if __name__ == "__main__":
    main()
