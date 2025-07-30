#!/usr/bin/env python3
"""
Example demonstrating YAML configuration usage.
"""

from detection_fusion import AdvancedEnsemble
from detection_fusion.utils import load_yaml_config, save_yaml_config


def config_example():
    """Demonstrate YAML configuration usage."""
    
    # Load configuration from YAML file
    config = load_yaml_config("configs/my_config.yaml")
    
    # Setup ensemble with config
    ensemble = AdvancedEnsemble(
        config["ensemble"]["labels_dir"],
        config["ensemble"]["output_dir"]
    )
    
    # Load detections
    ensemble.load_detections("detections.txt")
    
    # Apply strategy parameters from config
    for strategy_name, params in config["ensemble"]["strategies"].items():
        if strategy_name in ensemble.strategies:
            ensemble.set_strategy_params(strategy_name, **params)
    
    # Run strategies
    results = {}
    for strategy_name in config["ensemble"]["strategies"].keys():
        if strategy_name in ensemble.strategies:
            result = ensemble.run_strategy(strategy_name)
            results[strategy_name] = result
            print(f"{strategy_name}: {len(result)} detections")
    
    return results


def create_custom_config():
    """Create a custom YAML configuration."""
    
    custom_config = {
        "ensemble": {
            "labels_dir": "my_models/labels",
            "output_dir": "results/ensemble",
            "iou_threshold": 0.6,
            "strategies": {
                "majority_vote": {
                    "min_votes": 3
                },
                "dbscan": {
                    "eps": 0.05,
                    "min_samples": 2
                },
                "soft_voting": {
                    "temperature": 0.8
                },
                "bayesian": {
                    "class_priors": {
                        0: 0.4,  # person
                        1: 0.3,  # car
                        2: 0.2,  # bike
                        3: 0.1   # other
                    }
                }
            }
        },
        "analysis": {
            "iou_threshold": 0.6,
            "top_classes": 15,
            "generate_plots": True,
            "plot_formats": ["png", "svg"]
        },
        "visualization": {
            "figure_size": [16, 10],
            "dpi": 300,
            "style": "whitegrid",
            "color_palette": "husl"
        }
    }
    
    # Save custom configuration
    save_yaml_config(custom_config, "configs/my_config.yaml")
    print("Custom configuration saved to configs/my_config.yaml")
    
    return custom_config


def advanced_config_usage():
    """Advanced configuration usage with environment-specific settings."""
    
    # Load base configuration
    base_config = load_yaml_config("configs/default_config.yaml")
    
    # Environment-specific overrides
    if os.getenv("ENVIRONMENT") == "production":
        # Production settings
        base_config["ensemble"]["strategies"]["majority_vote"]["min_votes"] = 3
        base_config["ensemble"]["iou_threshold"] = 0.7
        base_config["analysis"]["generate_plots"] = False
    elif os.getenv("ENVIRONMENT") == "development":
        # Development settings
        base_config["ensemble"]["strategies"]["majority_vote"]["min_votes"] = 2
        base_config["analysis"]["generate_plots"] = True
        base_config["visualization"]["dpi"] = 150  # Lower DPI for faster plots
    
    # Dataset-specific settings
    dataset_type = os.getenv("DATASET_TYPE", "general")
    if dataset_type == "small_objects":
        base_config["ensemble"]["iou_threshold"] = 0.3
        base_config["ensemble"]["strategies"]["dbscan"]["eps"] = 0.05
    elif dataset_type == "large_objects":
        base_config["ensemble"]["iou_threshold"] = 0.7
        base_config["ensemble"]["strategies"]["dbscan"]["eps"] = 0.2
    
    return base_config


def batch_processing_with_config():
    """Process multiple experiments with different configurations."""
    
    experiments = [
        {
            "name": "conservative",
            "config": {
                "ensemble": {
                    "strategies": {
                        "majority_vote": {"min_votes": 4},
                        "affirmative_nms": {"min_models": 3}
                    }
                }
            }
        },
        {
            "name": "balanced",
            "config": {
                "ensemble": {
                    "strategies": {
                        "majority_vote": {"min_votes": 2},
                        "weighted_vote": {"use_model_weights": True},
                        "soft_voting": {"temperature": 1.0}
                    }
                }
            }
        },
        {
            "name": "permissive",
            "config": {
                "ensemble": {
                    "strategies": {
                        "nms": {"score_threshold": 0.05},
                        "dbscan": {"eps": 0.15, "min_samples": 2}
                    }
                }
            }
        }
    ]
    
    results = {}
    base_config = load_yaml_config("configs/default_config.yaml")
    
    for experiment in experiments:
        print(f"\nRunning {experiment['name']} experiment...")
        
        # Merge configurations
        config = base_config.copy()
        config.update(experiment["config"])
        
        # Run ensemble
        ensemble = AdvancedEnsemble("labels", f"results/{experiment['name']}")
        ensemble.load_detections("detections.txt")
        
        exp_results = {}
        for strategy_name, params in config["ensemble"]["strategies"].items():
            if strategy_name in ensemble.strategies:
                ensemble.set_strategy_params(strategy_name, **params)
                result = ensemble.run_strategy(strategy_name)
                exp_results[strategy_name] = len(result)
        
        results[experiment["name"]] = exp_results
        print(f"Results: {exp_results}")
    
    return results


if __name__ == "__main__":
    import os
    
    print("=== YAML Configuration Examples ===")
    
    # Example 1: Create custom config
    print("\n1. Creating custom configuration...")
    create_custom_config()
    
    # Example 2: Use configuration
    print("\n2. Using configuration...")
    try:
        results = config_example()
        print("Configuration-based processing completed!")
    except FileNotFoundError:
        print("Config file not found, skipping...")
    
    # Example 3: Advanced usage
    print("\n3. Advanced configuration usage...")
    advanced_config = advanced_config_usage()
    print(f"Loaded configuration for {os.getenv('ENVIRONMENT', 'default')} environment")
    
    # Example 4: Batch processing
    print("\n4. Batch processing with different configs...")
    try:
        batch_results = batch_processing_with_config()
        print("Batch processing completed!")
        
        # Summary
        print("\nExperiment Summary:")
        for exp_name, exp_results in batch_results.items():
            total_detections = sum(exp_results.values())
            print(f"  {exp_name}: {total_detections} total detections")
            
    except FileNotFoundError:
        print("Labels directory not found, skipping batch processing...")