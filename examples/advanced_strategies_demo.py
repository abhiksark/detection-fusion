#!/usr/bin/env python3
"""
Demonstration of all advanced ensemble strategies.
"""

from detection_fusion import AdvancedEnsemble
from detection_fusion.utils import load_yaml_config
import numpy as np


def demo_distance_based_strategies():
    """Demonstrate distance-based strategies."""
    print("\n=== Distance-Based Strategies ===")
    
    ensemble = AdvancedEnsemble("labels")
    ensemble.load_detections("detections.txt")
    
    # Distance-weighted voting
    print("\n1. Distance-Weighted Voting:")
    print("   - Weights detections by spatial distance to cluster centroid")
    print("   - Closer to centroid = higher weight")
    
    results = ensemble.run_strategy("distance_weighted")
    print(f"   Results: {len(results)} detections")
    
    # Centroid clustering
    print("\n2. Centroid Clustering:")
    print("   - Agglomerative clustering based on detection centers")
    print("   - Groups nearby detections regardless of IoU")
    
    results = ensemble.run_strategy("centroid_clustering")
    print(f"   Results: {len(results)} detections")


def demo_confidence_based_strategies():
    """Demonstrate confidence-based strategies."""
    print("\n=== Confidence-Based Strategies ===")
    
    ensemble = AdvancedEnsemble("labels")
    ensemble.load_detections("detections.txt")
    
    # Confidence threshold voting
    print("\n1. Confidence Threshold Voting:")
    print("   - Adaptive confidence thresholds per model")
    print("   - Filters low-confidence detections before voting")
    
    results = ensemble.run_strategy("confidence_threshold")
    print(f"   Results: {len(results)} detections")
    
    # Confidence-weighted NMS
    print("\n2. Confidence-Weighted NMS:")
    print("   - NMS with box regression weighted by confidence")
    print("   - Higher confidence detections have more influence")
    
    results = ensemble.run_strategy("confidence_weighted_nms")
    print(f"   Results: {len(results)} detections")
    
    # High confidence first
    print("\n3. High Confidence First:")
    print("   - Prioritizes high-confidence detections")
    print("   - Adds medium-confidence only if no spatial overlap")
    
    results = ensemble.run_strategy("high_confidence_first")
    print(f"   Results: {len(results)} detections")


def demo_adaptive_strategies():
    """Demonstrate adaptive strategies."""
    print("\n=== Adaptive Strategies ===")
    
    ensemble = AdvancedEnsemble("labels")
    ensemble.load_detections("detections.txt")
    
    # Adaptive threshold
    print("\n1. Adaptive Threshold Strategy:")
    print("   - Different IoU thresholds for small vs large objects")
    print("   - Small objects: more permissive matching")
    
    results = ensemble.run_strategy("adaptive_threshold")
    print(f"   Results: {len(results)} detections")
    
    # Density adaptive
    print("\n2. Density Adaptive Strategy:")
    print("   - Different strategies for high/low density regions")
    print("   - High density: aggressive NMS, Low density: conservative voting")
    
    results = ensemble.run_strategy("density_adaptive")
    print(f"   Results: {len(results)} detections")
    
    # Multi-scale
    print("\n3. Multi-Scale Strategy:")
    print("   - Scale-specific processing (tiny/small/medium/large)")
    print("   - Each scale uses optimized parameters")
    
    results = ensemble.run_strategy("multi_scale")
    print(f"   Results: {len(results)} detections")
    
    # Consensus ranking
    print("\n4. Consensus Ranking Strategy:")
    print("   - Combines model ranking with confidence scores")
    print("   - Higher-ranked detections get more weight")
    
    results = ensemble.run_strategy("consensus_ranking")
    print(f"   Results: {len(results)} detections")


def compare_all_strategies():
    """Compare all strategies side by side."""
    print("\n=== Strategy Comparison ===")
    
    ensemble = AdvancedEnsemble("labels")
    ensemble.load_detections("detections.txt")
    
    # List all available strategies
    strategies = [
        "majority_vote_2", "weighted_vote", "nms",
        "dbscan", "soft_voting", "bayesian",
        "distance_weighted", "centroid_clustering",
        "confidence_threshold", "confidence_weighted_nms", "high_confidence_first",
        "adaptive_threshold", "density_adaptive", "multi_scale", "consensus_ranking",
        "affirmative_nms"
    ]
    
    print(f"\nComparing {len(strategies)} strategies:")
    print("-" * 60)
    print(f"{'Strategy':<25} {'Detections':<12} {'Avg Confidence':<15}")
    print("-" * 60)
    
    results_summary = {}
    
    for strategy in strategies:
        if strategy in ensemble.strategies:
            try:
                results = ensemble.run_strategy(strategy)
                avg_conf = np.mean([d.confidence for d in results]) if results else 0.0
                
                results_summary[strategy] = {
                    'count': len(results),
                    'avg_confidence': avg_conf
                }
                
                print(f"{strategy:<25} {len(results):<12} {avg_conf:<15.3f}")
                
            except Exception as e:
                print(f"{strategy:<25} {'ERROR':<12} {str(e)[:15]:<15}")
    
    return results_summary


def demo_config_based_strategies():
    """Demonstrate using strategies with configuration files."""
    print("\n=== Configuration-Based Strategy Usage ===")
    
    # Load configuration
    try:
        config = load_yaml_config("configs/advanced_strategies_config.yaml")
        print("Loaded advanced strategies configuration")
        
        ensemble = AdvancedEnsemble(
            config["ensemble"]["labels_dir"],
            config["ensemble"]["output_dir"]
        )
        ensemble.load_detections("detections.txt")
        
        # Apply configuration to strategies
        configured_strategies = []
        for strategy_name, params in config["ensemble"]["strategies"].items():
            if strategy_name in ensemble.strategies:
                ensemble.set_strategy_params(strategy_name, **params)
                configured_strategies.append(strategy_name)
        
        print(f"Configured {len(configured_strategies)} strategies from YAML")
        
        # Run a few configured strategies
        sample_strategies = configured_strategies[:5]  # First 5
        print(f"\nRunning sample strategies: {sample_strategies}")
        
        for strategy in sample_strategies:
            results = ensemble.run_strategy(strategy)
            print(f"  {strategy}: {len(results)} detections")
            
    except FileNotFoundError:
        print("Configuration file not found. Using default parameters.")


def analyze_strategy_characteristics():
    """Analyze characteristics of different strategies."""
    print("\n=== Strategy Characteristics Analysis ===")
    
    ensemble = AdvancedEnsemble("labels")
    ensemble.load_detections("detections.txt")
    
    # Categorize strategies
    strategy_categories = {
        "Conservative (High Precision)": [
            "affirmative_nms", "bayesian", "high_confidence_first"
        ],
        "Balanced": [
            "majority_vote_2", "weighted_vote", "soft_voting"
        ],
        "Permissive (High Recall)": [
            "nms", "distance_weighted", "confidence_threshold"
        ],
        "Adaptive": [
            "adaptive_threshold", "density_adaptive", "multi_scale"
        ],
        "Clustering-Based": [
            "dbscan", "centroid_clustering", "consensus_ranking"
        ]
    }
    
    print("\nStrategy Categories and Results:")
    for category, strategies in strategy_categories.items():
        print(f"\n{category}:")
        
        category_results = []
        for strategy in strategies:
            if strategy in ensemble.strategies:
                try:
                    results = ensemble.run_strategy(strategy)
                    category_results.append(len(results))
                    print(f"  {strategy}: {len(results)} detections")
                except Exception as e:
                    print(f"  {strategy}: ERROR - {e}")
        
        if category_results:
            avg_detections = np.mean(category_results)
            print(f"  → Average: {avg_detections:.1f} detections")


def main():
    """Run all strategy demonstrations."""
    print("🎯 Advanced Ensemble Strategies Demonstration")
    print("=" * 50)
    
    try:
        # Demo different strategy categories
        demo_distance_based_strategies()
        demo_confidence_based_strategies() 
        demo_adaptive_strategies()
        
        # Compare all strategies
        results_summary = compare_all_strategies()
        
        # Configuration-based usage
        demo_config_based_strategies()
        
        # Analyze strategy characteristics
        analyze_strategy_characteristics()
        
        print("\n" + "=" * 50)
        print("✅ Advanced strategies demonstration complete!")
        
        # Summary insights
        if results_summary:
            print("\n📊 Key Insights:")
            
            # Most/least detections
            max_strategy = max(results_summary, key=lambda x: results_summary[x]['count'])
            min_strategy = min(results_summary, key=lambda x: results_summary[x]['count'])
            
            print(f"• Most detections: {max_strategy} ({results_summary[max_strategy]['count']})")
            print(f"• Fewest detections: {min_strategy} ({results_summary[min_strategy]['count']})")
            
            # Highest confidence
            high_conf_strategy = max(results_summary, 
                                   key=lambda x: results_summary[x]['avg_confidence'])
            print(f"• Highest avg confidence: {high_conf_strategy} "
                  f"({results_summary[high_conf_strategy]['avg_confidence']:.3f})")
        
    except FileNotFoundError:
        print("\n⚠️  Labels directory 'labels' not found.")
        print("Please create the labels directory structure:")
        print("labels/")
        print("├── model1/")
        print("│   └── detections.txt")
        print("├── model2/")
        print("│   └── detections.txt")
        print("└── model3/")
        print("    └── detections.txt")
    
    except Exception as e:
        print(f"\n❌ Error running demonstration: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()