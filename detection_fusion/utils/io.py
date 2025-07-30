from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json
import yaml

from ..core.detection import Detection


def read_detections(file_path: str, model_name: str = "") -> List[Detection]:
    """
    Read detections from a text file.
    
    Args:
        file_path: Path to detection file
        model_name: Name of the model that produced these detections
        
    Returns:
        List of Detection objects
    """
    detections = []
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 6:
                det = Detection(
                    class_id=int(parts[0]),
                    x=float(parts[1]),
                    y=float(parts[2]),
                    w=float(parts[3]),
                    h=float(parts[4]),
                    confidence=float(parts[5]),
                    model_source=model_name
                )
                detections.append(det)
    return detections


def save_detections(detections: List[Detection], output_path: str):
    """
    Save detections to a text file.
    
    Args:
        detections: List of Detection objects
        output_path: Path to save file
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for det in detections:
            f.write(f"{det.class_id} {det.x:.6f} {det.y:.6f} "
                   f"{det.w:.6f} {det.h:.6f} {det.confidence:.6f}\n")


def load_class_names(class_names_file: Optional[str] = None) -> Dict[int, str]:
    """
    Load class names from file or generate default names.
    
    Args:
        class_names_file: Optional path to class names file
        
    Returns:
        Dictionary mapping class IDs to names
    """
    if class_names_file and Path(class_names_file).exists():
        with open(class_names_file, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f)}
    else:
        return {i: f"Class_{i}" for i in range(100)}


def save_json_results(data: dict, output_path: str):
    """Save results as JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_yaml_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def load_json_config(config_path: str) -> dict:
    """Load configuration from JSON file (deprecated, use YAML)."""
    with open(config_path, 'r') as f:
        return json.load(f)

def save_yaml_config(data: dict, output_path: str):
    """Save configuration as YAML file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)


# Ground Truth Loading Utilities

def load_ground_truth(
    gt_dir: str, 
    gt_file: str = "detections.txt",
    cache: Optional[Dict] = None
) -> List[Detection]:
    """
    Load ground truth detections with caching support.
    
    Args:
        gt_dir: Directory containing ground truth files
        gt_file: Name of ground truth file
        cache: Optional cache dictionary for storing loaded GT data
        
    Returns:
        List of ground truth Detection objects
        
    Raises:
        FileNotFoundError: If ground truth file doesn't exist
    """
    gt_path = Path(gt_dir) / gt_file
    
    # Check cache first
    if cache is not None and str(gt_path) in cache:
        return cache[str(gt_path)]
    
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    # Load ground truth detections
    gt_detections = read_detections(str(gt_path), model_source="GT")
    
    # Cache if cache provided
    if cache is not None:
        cache[str(gt_path)] = gt_detections
    
    return gt_detections


def discover_ground_truth_files(gt_dir: str) -> List[str]:
    """
    Discover all ground truth files in a directory.
    
    Args:
        gt_dir: Ground truth directory path
        
    Returns:
        List of ground truth file names found
    """
    gt_path = Path(gt_dir)
    if not gt_path.exists():
        return []
    
    # Look for common ground truth file patterns
    gt_files = []
    patterns = ["detections.txt", "*.txt", "ground_truth.txt", "gt.txt"]
    
    for pattern in patterns:
        for file_path in gt_path.glob(pattern):
            if file_path.is_file():
                gt_files.append(file_path.name)
    
    return sorted(list(set(gt_files)))


def validate_ground_truth_structure(labels_dir: str) -> Dict[str, bool]:
    """
    Validate the ground truth directory structure.
    
    Args:
        labels_dir: Base labels directory
        
    Returns:
        Dictionary with validation results
    """
    labels_path = Path(labels_dir)
    validation = {
        'labels_dir_exists': labels_path.exists(),
        'gt_dir_exists': False,
        'gt_files_found': [],
        'model_dirs_found': [],
        'structure_valid': False
    }
    
    if not validation['labels_dir_exists']:
        return validation
    
    # Check for GT directory
    gt_dir = labels_path / "GT"
    validation['gt_dir_exists'] = gt_dir.exists()
    
    if validation['gt_dir_exists']:
        validation['gt_files_found'] = discover_ground_truth_files(str(gt_dir))
    
    # Find model directories
    for item in labels_path.iterdir():
        if item.is_dir() and item.name != "GT":
            if (item / "detections.txt").exists():
                validation['model_dirs_found'].append(item.name)
    
    # Structure is valid if we have GT dir with files and at least one model dir
    validation['structure_valid'] = (
        validation['gt_dir_exists'] and 
        len(validation['gt_files_found']) > 0 and 
        len(validation['model_dirs_found']) > 0
    )
    
    return validation


def load_detection_batch(
    detection_files: Dict[str, str],
    model_names: Optional[List[str]] = None
) -> Dict[str, List[Detection]]:
    """
    Load multiple detection files in batch.
    
    Args:
        detection_files: Dictionary mapping source names to file paths
        model_names: Optional list to filter which models to load
        
    Returns:
        Dictionary mapping source names to detection lists
    """
    detections = {}
    
    for source_name, file_path in detection_files.items():
        if model_names is not None and source_name not in model_names:
            continue
        
        try:
            detections[source_name] = read_detections(file_path, source_name)
        except Exception as e:
            print(f"Error loading detections from {file_path}: {e}")
            detections[source_name] = []
    
    return detections


def save_evaluation_results(
    results: Dict,
    output_path: str,
    format: str = 'json',
    include_metadata: bool = True
):
    """
    Save evaluation results with optional metadata.
    
    Args:
        results: Evaluation results dictionary
        output_path: Output file path
        format: Output format ('json', 'yaml', or 'txt')
        include_metadata: Whether to include timestamp and version info
    """
    # Add metadata if requested
    if include_metadata:
        from datetime import datetime
        results = results.copy()  # Don't modify original
        results['_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'detection_fusion_version': '0.2.0',
            'evaluation_framework': 'detection_fusion.evaluation'
        }
    
    # Save in requested format
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format.lower() == 'json':
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    elif format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(results, f, default_flow_style=False, indent=2, default=str)
    
    elif format.lower() == 'txt':
        _save_text_evaluation_report(results, output_path)
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def _save_text_evaluation_report(results: Dict, output_path: Path):
    """Save evaluation results as formatted text."""
    with open(output_path, 'w') as f:
        f.write("GROUND TRUTH EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall metrics
        if 'overall_metrics' in results:
            metrics = results['overall_metrics']
            f.write("OVERALL PERFORMANCE:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Precision: {metrics.get('precision', 0):.4f}\n")
            f.write(f"Recall: {metrics.get('recall', 0):.4f}\n")
            f.write(f"F1-Score: {metrics.get('f1_score', 0):.4f}\n")
            f.write(f"mAP@0.5: {metrics.get('map_50', 0):.4f}\n")
            f.write(f"mAP@0.5:0.95: {metrics.get('map_50_95', 0):.4f}\n\n")
        
        # Error analysis
        if 'error_analysis' in results:
            error = results['error_analysis']
            if 'summary' in error:
                summary = error['summary']
                f.write("ERROR BREAKDOWN:\n")
                f.write("-" * 20 + "\n")
                f.write(f"True Positives: {summary.get('true_positives', 0)}\n")
                f.write(f"False Positives: {summary.get('false_positives', 0)}\n")
                f.write(f"False Negatives: {summary.get('false_negatives', 0)}\n")
                f.write(f"Localization Errors: {summary.get('localization_errors', 0)}\n")
                f.write(f"Classification Errors: {summary.get('classification_errors', 0)}\n\n")


def create_sample_ground_truth(
    output_dir: str,
    num_images: int = 10,
    num_objects_per_image: Tuple[int, int] = (1, 5),
    num_classes: int = 3
):
    """
    Create sample ground truth data for testing.
    
    Args:
        output_dir: Directory to create sample GT data
        num_images: Number of sample images to create GT for
        num_objects_per_image: Range of objects per image (min, max)
        num_classes: Number of different classes
    """
    import random
    
    gt_dir = Path(output_dir) / "GT"
    gt_dir.mkdir(parents=True, exist_ok=True)
    
    # Create main detections.txt file
    detections = []
    
    for image_idx in range(num_images):
        num_objects = random.randint(num_objects_per_image[0], num_objects_per_image[1])
        
        for obj_idx in range(num_objects):
            # Random object properties
            class_id = random.randint(0, num_classes - 1)
            x = random.uniform(0.1, 0.9)  # Center x
            y = random.uniform(0.1, 0.9)  # Center y
            w = random.uniform(0.05, 0.3)  # Width
            h = random.uniform(0.05, 0.3)  # Height
            confidence = 1.0  # GT has perfect confidence
            
            detection = Detection(class_id, x, y, w, h, confidence, "GT")
            detections.append(detection)
    
    # Save sample ground truth
    save_detections(detections, str(gt_dir / "detections.txt"))
    
    # Create class names file
    class_names = [f"class_{i}" for i in range(num_classes)]
    with open(gt_dir / "classes.txt", 'w') as f:
        for class_name in class_names:
            f.write(f"{class_name}\n")
    
    print(f"Created sample ground truth in {gt_dir}")
    print(f"Generated {len(detections)} objects across {num_images} images")
    print(f"Classes: {class_names}")