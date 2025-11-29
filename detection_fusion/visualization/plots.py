from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_class_distribution(analyzer, top_n: int = 20, save_path: Optional[str] = None):
    """Plot detection count distribution across classes."""
    class_stats = analyzer.get_class_statistics()
    top_classes = class_stats.nlargest(top_n, "total")

    fig, ax = plt.subplots(figsize=(14, 8))

    x = np.arange(len(top_classes))
    width = 0.8 / len(analyzer.models)

    for i, model in enumerate(analyzer.models):
        offset = (i - len(analyzer.models) / 2 + 0.5) * width
        ax.bar(x + offset, top_classes[model], width, label=model)

    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Detections")
    ax.set_title(f"Top {top_n} Classes by Detection Count")
    ax.set_xticks(x)
    ax.set_xticklabels(top_classes["class_name"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_confidence_distribution(analyzer, save_path: Optional[str] = None):
    """Plot confidence score distributions for each model."""
    n_models = len(analyzer.models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5))

    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, analyzer.models):
        confidences = [det.confidence for det in analyzer.detections[model]]
        if confidences:
            ax.hist(confidences, bins=50, alpha=0.7, edgecolor="black")
            ax.axvline(
                np.mean(confidences),
                color="red",
                linestyle="--",
                label=f"Mean: {np.mean(confidences):.3f}",
            )
            ax.set_xlabel("Confidence Score")
            ax.set_ylabel("Frequency")
            ax.set_title(f"{model} Confidence Distribution")
            ax.legend()
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_model_comparison_heatmap(analyzer, save_path: Optional[str] = None):
    """Plot heatmap showing similarity between models."""
    comparison_df = analyzer.compare_all_models()

    models = analyzer.models
    matrix = np.zeros((len(models), len(models)))

    for _, row in comparison_df.iterrows():
        i = models.index(row["model1"])
        j = models.index(row["model2"])

        total = row["total_matches"] + row["model1_unique"] + row["model2_unique"]
        if total > 0:
            value = row["total_matches"] / total
        else:
            value = 0

        matrix[i, j] = value
        matrix[j, i] = value

    np.fill_diagonal(matrix, 1.0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        xticklabels=models,
        yticklabels=models,
        cbar_kws={"label": "Similarity Score"},
        vmin=0,
        vmax=1,
    )
    plt.title("Model Detection Similarity Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_variance_analysis(analyzer, top_n: int = 15, save_path: Optional[str] = None):
    """Plot classes with highest variance between models."""
    class_stats = analyzer.get_class_statistics()
    high_variance = class_stats.nlargest(top_n, "variance")

    fig, ax = plt.subplots(figsize=(12, 8))

    x = np.arange(len(high_variance))
    width = 0.35

    ax.bar(
        x - width / 2,
        high_variance["mean"],
        width,
        label="Mean Count",
        yerr=high_variance["std"],
        capsize=5,
    )
    ax.bar(x + width / 2, high_variance["variance"], width, label="Variance")

    ax.set_xlabel("Class")
    ax.set_ylabel("Value")
    ax.set_title(f"Top {top_n} Classes by Detection Variance")
    ax.set_xticks(x)
    ax.set_xticklabels(high_variance["class_name"], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_all_plots(analyzer, output_dir: str = "plots", top_n: int = 20):
    """Generate all visualization plots."""
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Generate plots
    plot_class_distribution(analyzer, top_n, save_path=str(output_path / "class_distribution.png"))

    plot_confidence_distribution(
        analyzer, save_path=str(output_path / "confidence_distribution.png")
    )

    plot_model_comparison_heatmap(analyzer, save_path=str(output_path / "model_similarity.png"))

    plot_variance_analysis(analyzer, top_n, save_path=str(output_path / "variance_analysis.png"))

    print(f"All plots saved to: {output_dir}/")
