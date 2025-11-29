#!/usr/bin/env python3
"""
GT Rectification Configuration Example

This example demonstrates how to use different configuration files
for GT rectification with various modes and parameters using the v1.0 API.
"""

from pathlib import Path


def run_rectification_example():
    """Run GT rectification examples with different configurations."""

    print("DetectionFusion GT Rectification Configuration Examples (v1.0)")
    print("=" * 65)

    # Check if config files exist
    config_dir = Path("configs")
    if not config_dir.exists():
        print("configs/ directory not found. Please run from project root.")
        return

    configs = [
        (
            "gt_rectification/conservative.yaml",
            "Conservative mode - high precision error detection",
        ),
        ("gt_rectification/aggressive.yaml", "Aggressive mode - comprehensive error detection"),
        ("gt_rectification/balanced.yaml", "Balanced mode - moderate precision/recall"),
        ("gt_rectification/custom.yaml", "Custom template - modify for your needs"),
    ]

    print("\nAvailable GT rectification configurations:")
    print()

    for config_file, description in configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"  {config_file}")
            print(f"    {description}")
            print(f"    Usage: detection-fusion rectify --config {config_path}")
            print()
        else:
            print(f"  {config_file} - Not found")
            print()

    print("Configuration File Examples:")
    print("-" * 50)

    print("\n1. Conservative Mode (High Precision):")
    print("   detection-fusion rectify --config configs/gt_rectification/conservative.yaml")
    print("   - Only flags high-confidence errors")
    print("   - Uses strict thresholds")
    print("   - Good for critical datasets where precision is key")

    print("\n2. Aggressive Mode (High Recall):")
    print("   detection-fusion rectify --config configs/gt_rectification/aggressive.yaml")
    print("   - Comprehensive error detection")
    print("   - Lower thresholds, more candidates")
    print("   - Good for initial dataset quality assessment")

    print("\n3. Balanced Mode (Moderate Precision/Recall):")
    print("   detection-fusion rectify --config configs/gt_rectification/balanced.yaml")
    print("   - Balanced approach between precision and recall")
    print("   - Good general-purpose configuration")
    print("   - Recommended for most use cases")

    print("\n4. Custom Configuration:")
    print("   # Copy and customize the template")
    print("   cp configs/gt_rectification/custom.yaml my_config.yaml")
    print("   # Edit my_config.yaml with your parameters")
    print("   detection-fusion rectify --config my_config.yaml")

    print("\n" + "=" * 50)
    print("Configuration Override Examples:")
    print("-" * 50)

    print("\n# Use config but override output directory")
    print("detection-fusion rectify \\")
    print("    --config configs/gt_rectification/balanced.yaml \\")
    print("    --output custom_output/")

    print("\n# Override paths via CLI")
    print("detection-fusion rectify \\")
    print("    --config configs/gt_rectification/conservative.yaml \\")
    print("    --labels-dir my_labels/ \\")
    print("    --gt-dir my_gt/ \\")
    print("    --images-dir my_images/")

    print("\n" + "=" * 50)
    print("Python API Usage:")
    print("-" * 50)

    print("""
from detection_fusion.config import ConfigLoader, RectificationConfig

# Load from YAML file
config = ConfigLoader.load_rectification("configs/gt_rectification/balanced.yaml")
print(f"Mode: {config.mode}")
print(f"IoU threshold: {config.thresholds.iou}")
print(f"Output dir: {config.paths.output_dir}")

# Or build programmatically with builder pattern
config = (
    RectificationConfig()
    .with_paths(labels_dir="my_labels", output_dir="my_output")
    .with_thresholds(iou=0.6, min_agreement=4)
    .with_output(most_correct=100, most_incorrect=100)
)
""")

    print("Expected Directory Structure:")
    print("-" * 50)
    print("""
your_dataset/
├── labels/
│   ├── model1/
│   │   ├── image1.txt
│   │   ├── image2.txt
│   │   └── ...
│   ├── model2/
│   │   └── ...
│   └── model3/
│       └── ...
├── GT/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
└── images/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
""")

    print("Quick Start:")
    print("-" * 50)
    print("1. Choose a configuration file based on your needs")
    print("2. Run GT rectification:")
    print("   detection-fusion rectify --config configs/gt_rectification/balanced.yaml")
    print("3. Review results in the output directory")
    print("4. Use 'most_incorrect' images for annotation review/correction")
    print("5. Use 'most_correct' images as reference examples")


if __name__ == "__main__":
    run_rectification_example()
