import click

from detection_fusion._version import __version__


@click.group()
@click.version_option(version=__version__)
def cli():
    """DetectionFusion - Object Detection Ensemble Toolkit."""
    pass


@cli.command("merge")
@click.option(
    "--models-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing model detection outputs",
)
@click.option(
    "--strategy",
    "-s",
    default="weighted_vote",
    help="Ensemble strategy to use",
)
@click.option(
    "--output",
    "-o",
    default="ensemble_results",
    help="Output directory for merged results",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.5,
    help="IoU threshold for detection matching",
)
@click.option(
    "--format",
    "-f",
    default="auto",
    help="Input annotation format (auto, yolo, voc_xml, coco)",
)
def merge(models_dir, strategy, output, iou_threshold, format):
    """Merge detections from multiple models."""
    from detection_fusion._compat import has_rich

    if has_rich():
        from rich.console import Console

        console = Console()
    else:
        console = None

    from detection_fusion.data import FileDetectionLoader
    from detection_fusion.data.formats import FormatRegistry
    from detection_fusion.strategies import StrategyRegistry

    if console:
        console.print(f"[bold]Loading detections from:[/bold] {models_dir}")

    loader = FileDetectionLoader(models_dir, format=format)
    models = loader.find_all_models()

    if console:
        console.print(f"[green]Found {len(models)} models:[/green] {', '.join(models)}")

    detections = loader.load_all()
    total = sum(len(d) for d in detections.values())

    if console:
        console.print(f"[green]Loaded {total} total detections[/green]")
        console.print(f"[bold]Running strategy:[/bold] {strategy}")

    strategy_obj = StrategyRegistry.create(strategy, iou_threshold=iou_threshold)
    result = strategy_obj.merge(detections)

    if console:
        console.print(f"[green]Merged into {len(result)} detections[/green]")

    # Save results
    from pathlib import Path

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)

    writer = FormatRegistry.get_writer("yolo")

    # Group by image
    by_image = {}
    for det in result:
        if det.image_name not in by_image:
            by_image[det.image_name] = []
        by_image[det.image_name].append(det)

    writer.write_directory(by_image, output_path)

    if console:
        console.print(f"[green]Results saved to:[/green] {output_path}")
    else:
        click.echo(f"Merged {total} detections into {len(result)} using {strategy}")
        click.echo(f"Results saved to: {output_path}")


@cli.command("validate")
@click.option(
    "--models-dir",
    "-d",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing model detection outputs",
)
@click.option(
    "--gt-dir",
    "-g",
    type=click.Path(exists=True),
    help="Directory containing ground truth annotations",
)
@click.option(
    "--strategy",
    "-s",
    default="weighted_vote",
    help="Ensemble strategy to use",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=0.5,
    help="IoU threshold for evaluation",
)
def validate(models_dir, gt_dir, strategy, iou_threshold):
    """Validate ensemble results against ground truth."""
    from detection_fusion.pipeline import DetectionPipeline

    pipeline = DetectionPipeline()
    pipeline.load(models_dir)
    pipeline.ensemble(strategy, iou_threshold=iou_threshold)

    if gt_dir:
        pipeline.evaluate(gt_dir, iou_threshold=iou_threshold)

    result = pipeline.run()

    from detection_fusion._compat import has_rich

    if has_rich():
        from rich.console import Console
        from rich.table import Table

        console = Console()

        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if result.detections:
            table.add_row("Models", str(len(result.detections.model_names)))
            table.add_row("Total Detections", str(result.detections.total_count))

        if result.ensemble_result:
            table.add_row("Merged Detections", str(len(result.ensemble_result)))

        if result.evaluation_result:
            ev = result.evaluation_result
            table.add_row("Precision", f"{ev.precision:.4f}")
            table.add_row("Recall", f"{ev.recall:.4f}")
            table.add_row("F1 Score", f"{ev.f1_score:.4f}")
            table.add_row("True Positives", str(ev.true_positives))
            table.add_row("False Positives", str(ev.false_positives))
            table.add_row("False Negatives", str(ev.false_negatives))

        console.print(table)
    else:
        if result.detections:
            click.echo(f"Models: {len(result.detections.model_names)}")
            click.echo(f"Total Detections: {result.detections.total_count}")

        if result.ensemble_result:
            click.echo(f"Merged Detections: {len(result.ensemble_result)}")

        if result.evaluation_result:
            ev = result.evaluation_result
            click.echo(f"Precision: {ev.precision:.4f}")
            click.echo(f"Recall: {ev.recall:.4f}")
            click.echo(f"F1 Score: {ev.f1_score:.4f}")


@cli.command("list-strategies")
@click.option(
    "--category",
    "-c",
    type=str,
    default=None,
    help="Filter by category (voting, nms, clustering, probabilistic, adaptive)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Show detailed information including parameters",
)
def list_strategies(category, verbose):
    """List all available ensemble strategies."""
    from detection_fusion._compat import has_rich
    from detection_fusion.strategies import StrategyRegistry

    if category:
        strategies = StrategyRegistry.list_by_category(category)
    else:
        strategies = StrategyRegistry.list_all()

    if has_rich():
        from rich.console import Console
        from rich.table import Table

        console = Console()

        if verbose:
            table = Table(title="Available Strategies (Detailed)")
            table.add_column("Name", style="cyan")
            table.add_column("Category", style="yellow")
            table.add_column("Description", style="white")
            table.add_column("Parameters", style="dim")

            for name in sorted(strategies):
                info = StrategyRegistry.get_info(name)
                param_str = ", ".join(
                    f"{p}={v.get('default', '?')}"
                    for p, v in info.parameters.items()
                    if p not in ("config",)
                )
                table.add_row(name, info.category, info.description, param_str or "-")
        else:
            table = Table(title="Available Strategies")
            table.add_column("Name", style="cyan")
            table.add_column("Category", style="yellow")
            table.add_column("Description", style="white")

            for name in sorted(strategies):
                metadata = StrategyRegistry.get_metadata(name)
                cat = metadata.category if metadata else "unknown"
                desc = metadata.description if metadata else ""
                table.add_row(name, cat, desc)

        console.print(table)
        console.print(f"\n[dim]Total: {len(strategies)} strategies[/dim]")
    else:
        click.echo("Available strategies:")
        for name in sorted(strategies):
            metadata = StrategyRegistry.get_metadata(name)
            cat = metadata.category if metadata else "unknown"
            desc = metadata.description if metadata else ""
            if verbose:
                click.echo(f"  {name} [{cat}]: {desc}")
            else:
                click.echo(f"  - {name}")


@cli.command("list-formats")
def list_formats():
    """List all supported annotation formats."""
    from detection_fusion.data.formats import FormatRegistry

    formats = FormatRegistry.list_formats()

    from detection_fusion._compat import has_rich

    if has_rich():
        from rich.console import Console

        console = Console()
        console.print("[bold]Supported Formats:[/bold]")
        console.print(f"  [cyan]Readers:[/cyan] {', '.join(formats['readers'])}")
        console.print(f"  [cyan]Writers:[/cyan] {', '.join(formats['writers'])}")
    else:
        click.echo("Supported Formats:")
        click.echo(f"  Readers: {', '.join(formats['readers'])}")
        click.echo(f"  Writers: {', '.join(formats['writers'])}")


@cli.command("convert")
@click.option(
    "--input",
    "-i",
    "input_path",
    type=click.Path(exists=True),
    required=True,
    help="Input file or directory",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=click.Path(),
    required=True,
    help="Output file or directory",
)
@click.option(
    "--from",
    "input_format",
    type=str,
    default="auto",
    help="Input format (auto, yolo, voc_xml, coco)",
)
@click.option(
    "--to",
    "output_format",
    type=str,
    default="yolo",
    help="Output format (yolo, voc_xml, coco)",
)
@click.option(
    "--image-size",
    type=(int, int),
    default=None,
    help="Image size (width height) for format conversion",
)
def convert(input_path, output_path, input_format, output_format, image_size):
    """Convert annotations between formats."""
    from pathlib import Path

    from detection_fusion.data.formats import FormatRegistry

    if input_format == "auto":
        reader = FormatRegistry.auto_detect_reader(Path(input_path))
    else:
        reader = FormatRegistry.get_reader(input_format)

    writer = FormatRegistry.get_writer(output_format)

    input_path = Path(input_path)
    output_path = Path(output_path)

    if input_path.is_file():
        detections = reader.read_file(input_path, image_size)
        writer.write_file(detections, output_path, image_size)
    else:
        result = reader.read_directory(input_path)
        output_path.mkdir(parents=True, exist_ok=True)
        writer.write_directory(result, output_path)

    from detection_fusion._compat import has_rich

    if has_rich():
        from rich.console import Console

        console = Console()
        console.print(f"[green]Converted {input_path} to {output_path}[/green]")
    else:
        click.echo(f"Converted {input_path} to {output_path}")


@cli.command("rectify")
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    default=None,
    help="Configuration file (YAML) - CLI options override config values",
)
@click.option(
    "--labels-dir",
    "-l",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing model prediction subdirectories",
)
@click.option(
    "--gt-dir",
    "-g",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing ground truth labels",
)
@click.option(
    "--images-dir",
    "-i",
    type=click.Path(exists=True),
    default=None,
    help="Directory containing images",
)
@click.option(
    "--output",
    "-o",
    default=None,
    help="Output directory for rectified dataset",
)
@click.option(
    "--iou-threshold",
    type=float,
    default=None,
    help="IoU threshold for matching detections",
)
@click.option(
    "--confidence-threshold",
    type=float,
    default=None,
    help="Minimum confidence threshold",
)
@click.option(
    "--min-agreement",
    type=int,
    default=None,
    help="Minimum strategies that must agree",
)
@click.option(
    "--mode",
    type=click.Choice(["minimize_error", "maximize_error"]),
    default=None,
    help="Rectification mode (conservative or aggressive)",
)
@click.option(
    "--most-correct",
    type=int,
    default=None,
    help="Number of most correct images to include",
)
@click.option(
    "--most-incorrect",
    type=int,
    default=None,
    help="Number of most incorrect images to include",
)
@click.option(
    "--skip-dataset",
    is_flag=True,
    help="Only run analysis without creating dataset",
)
def rectify(
    config,
    labels_dir,
    gt_dir,
    images_dir,
    output,
    iou_threshold,
    confidence_threshold,
    min_agreement,
    mode,
    most_correct,
    most_incorrect,
    skip_dataset,
):
    """Analyze and rectify ground truth annotations.

    Identifies potential annotation errors by comparing ensemble consensus
    with ground truth labels across all strategies. Creates a rectified
    dataset with most reliable and most problematic images for review.

    Use --config to load settings from a YAML file. CLI options override
    config file values.
    """
    from pathlib import Path

    from detection_fusion._compat import has_rich
    from detection_fusion.config import ConfigLoader, RectificationConfig

    if has_rich():
        from rich.console import Console
        from rich.table import Table

        console = Console()
    else:
        console = None

    # Load config file if provided, otherwise use defaults
    if config:
        cfg = ConfigLoader.load_rectification(Path(config))
        if console:
            console.print(f"[dim]Loaded config from: {config}[/dim]")
    else:
        cfg = RectificationConfig()

    # CLI options override config values
    final_labels_dir = labels_dir or cfg.paths.labels_dir
    final_gt_dir = gt_dir or cfg.paths.gt_dir
    final_images_dir = images_dir or cfg.paths.images_dir
    final_output = output or cfg.paths.output_dir
    final_iou = iou_threshold if iou_threshold is not None else cfg.thresholds.iou
    final_confidence = (
        confidence_threshold if confidence_threshold is not None else cfg.thresholds.confidence
    )
    final_min_agreement = (
        min_agreement if min_agreement is not None else cfg.thresholds.min_agreement
    )
    final_mode = mode or cfg.mode
    final_most_correct = most_correct if most_correct is not None else cfg.output.most_correct
    final_most_incorrect = (
        most_incorrect if most_incorrect is not None else cfg.output.most_incorrect
    )

    # Validate required paths exist
    if not Path(final_labels_dir).exists():
        click.echo(f"Error: Labels directory does not exist: {final_labels_dir}", err=True)
        raise click.Abort()
    if not Path(final_gt_dir).exists():
        click.echo(f"Error: GT directory does not exist: {final_gt_dir}", err=True)
        raise click.Abort()
    if not Path(final_images_dir).exists():
        click.echo(f"Error: Images directory does not exist: {final_images_dir}", err=True)
        raise click.Abort()

    try:
        from detection_fusion.rectification import GTRectifier
    except ImportError:
        if console:
            console.print("[red]Rectification module not available.[/red]")
            console.print("Ensure gt_rectify.py is in the project root directory.")
        else:
            click.echo("Error: Rectification module not available.")
        return

    if GTRectifier is None:
        if console:
            console.print("[red]GTRectifier could not be imported.[/red]")
            console.print("Check dependencies: numpy, tqdm")
        else:
            click.echo("Error: GTRectifier could not be imported.")
        return

    if console:
        console.print("[bold]GT Rectification Analysis[/bold]")
        console.print(f"Labels directory: {final_labels_dir}")
        console.print(f"Ground truth: {final_gt_dir}")
        console.print(f"Images: {final_images_dir}")
        console.print(f"Mode: {final_mode}")
    else:
        click.echo("GT Rectification Analysis")
        click.echo(f"Labels: {final_labels_dir}")
        click.echo(f"GT: {final_gt_dir}")

    try:
        rectifier = GTRectifier(
            final_labels_dir,
            final_gt_dir,
            final_images_dir,
            final_output,
            final_iou,
            final_confidence,
            final_min_agreement,
            final_mode,
        )

        if console:
            console.print("[yellow]Running analysis...[/yellow]")

        results = rectifier.run_full_analysis()

        if console:
            table = Table(title="Rectification Results")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")

            table.add_row("Total Images", str(results["total_images"]))
            table.add_row("Errors Found", str(results["total_errors_found"]))

            for error_type, count in results["error_types"].items():
                table.add_row(f"  {error_type}", str(count))

            console.print(table)

            if results["most_problematic_images"]:
                console.print("\n[bold]Most Problematic Images:[/bold]")
                for img, score in results["most_problematic_images"][:5]:
                    console.print(f"  {img}: {score:.3f}")

            if results["most_reliable_images"]:
                console.print("\n[bold]Most Reliable Images:[/bold]")
                for img, score in results["most_reliable_images"][:5]:
                    console.print(f"  {img}: {score:.3f}")
        else:
            click.echo(f"Total images: {results['total_images']}")
            click.echo(f"Errors found: {results['total_errors_found']}")

        if not skip_dataset:
            if console:
                console.print("\n[yellow]Creating rectified dataset...[/yellow]")

            rectifier.create_rectified_dataset(
                final_output,
                final_most_correct,
                final_most_incorrect,
            )

            if console:
                console.print(f"[green]Dataset saved to: {final_output}[/green]")
            else:
                click.echo(f"Dataset saved to: {final_output}")

    except Exception as e:
        if console:
            console.print(f"[red]Error: {e}[/red]")
        else:
            click.echo(f"Error: {e}")
        raise click.Abort()


if __name__ == "__main__":
    cli()
