#!/usr/bin/env python3
"""
GT Rectification Configuration Example

This example demonstrates how to use different configuration files
for GT rectification with various modes and parameters.
"""

from pathlib import Path

def run_rectification_example():
    """Run GT rectification examples with different configurations."""
    
    print("🔍 DetectionFusion GT Rectification Configuration Examples")
    print("=" * 60)
    
    # Check if config files exist
    config_dir = Path("configs")
    if not config_dir.exists():
        print("❌ configs/ directory not found. Please run from project root.")
        return
    
    configs = [
        ("gt_rectification/gt_rectify_conservative.yaml", "Conservative mode - high precision error detection"),
        ("gt_rectification/gt_rectify_aggressive.yaml", "Aggressive mode - comprehensive error detection"),
        ("gt_rectification/gt_rectify_balanced.yaml", "Balanced mode - moderate precision/recall"),
        ("gt_rectification/gt_rectify_custom.yaml", "Custom template - modify for your needs")
    ]
    
    print("Available GT rectification configurations:")
    print()
    
    for config_file, description in configs:
        config_path = config_dir / config_file
        if config_path.exists():
            print(f"✅ {config_file}")
            print(f"   {description}")
            print(f"   Usage: python gt_rectify.py --config {config_path}")
            print()
        else:
            print(f"❌ {config_file} - Not found")
            print()
    
    print("Configuration File Examples:")
    print("-" * 40)
    
    print("1. Conservative Mode (High Precision):")
    print("   python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml")
    print("   - Only flags high-confidence errors")
    print("   - Uses strict thresholds and proven strategies")
    print("   - Good for critical datasets where precision is key")
    print()
    
    print("2. Aggressive Mode (High Recall):")
    print("   python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml")
    print("   - Comprehensive error detection")
    print("   - Uses all available strategies")
    print("   - Good for initial dataset quality assessment")
    print()
    
    print("3. Balanced Mode (Moderate Precision/Recall):")
    print("   python gt_rectify.py --config configs/gt_rectification/gt_rectify_balanced.yaml")
    print("   - Balanced approach between precision and recall")
    print("   - Good general-purpose configuration")
    print("   - Recommended for most use cases")
    print()
    
    print("4. Custom Configuration:")
    print("   # Copy and customize the template")
    print("   cp configs/gt_rectification/gt_rectify_custom.yaml my_config.yaml")
    print("   # Edit my_config.yaml with your parameters")
    print("   python gt_rectify.py --config my_config.yaml")
    print()
    
    print("Configuration Override Examples:")
    print("-" * 40)
    print("# Use conservative config but switch to aggressive mode")
    print("python gt_rectify.py --config configs/gt_rectification/gt_rectify_conservative.yaml --mode maximize_error")
    print()
    print("# Use aggressive config but reduce output images")
    print("python gt_rectify.py --config configs/gt_rectification/gt_rectify_aggressive.yaml --most-incorrect 25")
    print()
    print("# Use custom config with different output directory")
    print("python gt_rectify.py --config configs/gt_rectification/gt_rectify_custom.yaml --output-dir my_rectified_dataset")
    print()
    
    print("Expected Directory Structure:")
    print("-" * 40)
    print("""
    your_dataset/
    ├── labels/
    │   ├── model1/
    │   │   └── detections.txt
    │   ├── model2/
    │   │   └── detections.txt
    │   └── model3/
    │       └── detections.txt
    ├── gt/
    │   └── detections.txt
    └── images/
        ├── image1.jpg
        ├── image2.jpg
        └── ...
    """)
    
    print("Quick Start:")
    print("-" * 40)
    print("1. Choose a configuration file based on your needs")
    print("2. Update the paths in the config file or use command line overrides")
    print("3. Run GT rectification:")
    print("   python gt_rectify.py --config configs/gt_rectification/gt_rectify_balanced.yaml")
    print("4. Review results in the output directory")
    print("5. Use 'most_incorrect' images for annotation review/correction")
    print("6. Use 'most_correct' images as reference examples")

if __name__ == "__main__":
    run_rectification_example()