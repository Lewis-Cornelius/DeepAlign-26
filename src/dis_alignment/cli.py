"""DeepAlign-26 command-line interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def _print_swd_summary(summary: dict[str, Any]) -> None:
    click.echo(f"SWD root: {summary['root']}")
    click.echo(f"Audio files: {summary['audio_files']}")
    click.echo(f"Measure annotations: {summary['annotations']}")
    click.echo(f"Lieder: {summary['lieder']}")
    click.echo(f"Available pairs: {summary['pairs']}")
    click.echo("Performances:")
    for performance_id, count in summary["performances"].items():
        click.echo(f"  - {performance_id}: {count} recordings")


def _coalesce(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


@click.group()
@click.version_option()
def main() -> None:
    """DeepAlign-26 dissertation workflow.

    Recommended path:
      1. `deepalign prepare-swd`
      2. `deepalign train`
      3. `deepalign evaluate-swd`
      4. `deepalign visualize`
      5. `deepalign analyze`
    """


@main.command(name="prepare-swd")
@click.argument("output_dir", type=click.Path())
@click.option("--verify-only", is_flag=True, help="Only verify an existing SWD download")
@click.option("--cleanup-zip/--keep-zip", default=False, help="Delete the archive after extraction")
def prepare_swd(output_dir: str, verify_only: bool, cleanup_zip: bool) -> None:
    """Download or verify the Schubert Winterreise Dataset used by DeepAlign-26."""
    from dis_alignment.data import EXPECTED_SIZE_MB, download_swd_dataset, verify_swd_dataset

    output = Path(output_dir)

    if verify_only:
        try:
            summary = verify_swd_dataset(output)
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        _print_swd_summary(summary)
        return

    click.echo(f"SWD archive size: ~{EXPECTED_SIZE_MB}MB")
    click.echo(f"Target directory: {output.resolve()}")
    if not click.confirm("Continue with SWD download / extraction?", default=True):
        return

    with click.progressbar(length=100, label="Downloading") as progress:
        last_progress = [0]

        def reporthook(count: int, block_size: int, total_size: int) -> None:
            if total_size <= 0:
                return
            percent = int(count * block_size * 100 / total_size)
            increment = max(percent - last_progress[0], 0)
            if increment > 0:
                progress.update(increment)
                last_progress[0] = percent

        try:
            extracted_path = download_swd_dataset(
                output,
                cleanup_zip=cleanup_zip,
                reporthook=reporthook,
            )
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc

    click.echo(f"SWD ready at {extracted_path}")
    summary = verify_swd_dataset(extracted_path)
    _print_swd_summary(summary)


@main.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="YAML training config")
@click.option("--swd-path", type=click.Path(exists=True), default=None, help="Path to the SWD root")
@click.option("--output-dir", type=click.Path(), default=None, help="Checkpoint output directory")
@click.option("--epochs", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--device", type=str, default=None, help="Explicit device, e.g. cpu or cuda")
@click.option("--start-gamma", type=float, default=None)
@click.option("--end-gamma", type=float, default=None)
@click.option("--no-augment", is_flag=True, help="Disable waveform augmentation")
@click.option("--dry-run", is_flag=True, help="Run the synthetic dry-run pipeline")
def train(
    config: str | None,
    swd_path: str | None,
    output_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    device: str | None,
    start_gamma: float | None,
    end_gamma: float | None,
    no_augment: bool,
    dry_run: bool,
) -> None:
    """Train DeepAlign-26 on SWD or validate the stack with a dry run."""
    from dis_alignment.model.train import _load_training_config, train as train_model

    training_config = _load_training_config(config)
    dataset_cfg = training_config.get("dataset", {})
    training_cfg = training_config.get("training", {})
    soft_dtw_cfg = training_config.get("soft_dtw", {})
    output_cfg = training_config.get("output", {})
    aug_cfg = training_config.get("augmentation", {})

    resolved_swd_path = _coalesce(swd_path, dataset_cfg.get("swd_path"), None)
    if not dry_run and not resolved_swd_path:
        raise click.ClickException("--swd-path is required unless using --dry-run")

    augment_enabled = False if no_augment else _coalesce(None, aug_cfg.get("enabled"), True)
    augmentor_kwargs = {
        "time_stretch_range": tuple(aug_cfg["time_stretch_range"]),
        "pitch_shift_range": tuple(aug_cfg["pitch_shift_range"]),
        "noise_snr_range": tuple(aug_cfg["noise_snr_range"]),
        "prob": aug_cfg["augment_prob"],
        "sr": training_config.get("audio", {}).get("sr", 22050),
    } if augment_enabled and aug_cfg else {}

    train_model(
        swd_path=resolved_swd_path,
        output_dir=_coalesce(output_dir, output_cfg.get("checkpoint_dir"), "checkpoints"),
        epochs=_coalesce(epochs, training_cfg.get("epochs"), 50),
        batch_size=_coalesce(batch_size, training_cfg.get("batch_size"), 4),
        lr=_coalesce(None, training_cfg.get("lr"), 1e-3),
        embed_dim=_coalesce(None, training_config.get("model", {}).get("embed_dim"), 64),
        n_freq_bins=_coalesce(None, training_config.get("audio", {}).get("n_bins"), 84),
        num_conv_channels=training_config.get("model", {}).get("conv_channels"),
        gru_hidden_size=_coalesce(None, training_config.get("model", {}).get("gru_hidden_size"), 128),
        num_gru_layers=_coalesce(None, training_config.get("model", {}).get("num_gru_layers"), 2),
        dropout=_coalesce(None, training_config.get("model", {}).get("dropout"), 0.1),
        max_length_sec=_coalesce(None, dataset_cfg.get("max_length_sec"), 30.0),
        start_gamma=_coalesce(start_gamma, soft_dtw_cfg.get("start_gamma"), 1.0),
        end_gamma=_coalesce(end_gamma, soft_dtw_cfg.get("end_gamma"), 0.01),
        normalize_loss=_coalesce(None, soft_dtw_cfg.get("normalize"), True),
        dist_func=_coalesce(None, soft_dtw_cfg.get("dist_func"), "sqeuclidean"),
        gradient_clip=_coalesce(None, training_cfg.get("gradient_clip"), 1.0),
        weight_decay=_coalesce(None, training_cfg.get("weight_decay"), 1e-4),
        augment=augment_enabled,
        augmentor_kwargs=augmentor_kwargs,
        device=device,
        val_split=_coalesce(None, dataset_cfg.get("val_split"), 0.2),
        save_every_n_epochs=_coalesce(None, output_cfg.get("save_every_n_epochs"), 10),
        dry_run=dry_run,
    )


@main.command(name="evaluate-swd")
@click.argument("swd_path", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), default=None, help="Optional DeepAlign checkpoint")
@click.option("--output", "-o", default="results/swd_evaluation.csv", help="Output CSV path")
@click.option("--device", type=str, default=None)
@click.option("--performance", "performances", multiple=True, help="Limit to one or more performance ids")
@click.option("--lied", "lieder", multiple=True, help="Limit to one or more lied ids")
@click.option("--quiet", is_flag=True, help="Disable the progress bar")
def evaluate_swd(
    swd_path: str,
    checkpoint: str | None,
    output: str,
    device: str | None,
    performances: tuple[str, ...],
    lieder: tuple[str, ...],
    quiet: bool,
) -> None:
    """Evaluate SWD pairs with chroma DTW and optional DeepAlign features."""
    from dis_alignment.data import SWDDataset
    from dis_alignment.evaluation import (
        check_success_criteria,
        evaluate_swd_dataset,
        save_evaluation_results,
        summarize_evaluation,
    )

    dataset = SWDDataset(swd_path)
    results = evaluate_swd_dataset(
        dataset,
        checkpoint_path=checkpoint,
        device=device,
        performances=list(performances) or None,
        lieder=list(lieder) or None,
        show_progress=not quiet,
    )
    if results.empty:
        raise click.ClickException("No SWD evaluation rows were produced. Check annotations and filters.")

    output_path = save_evaluation_results(results, output)
    click.echo(f"Saved evaluation results to {output_path}")

    click.echo("\n=== SWD Summary ===")
    for method, stats in summarize_evaluation(results).items():
        click.echo(f"\n{method}:")
        click.echo(f"  pairs: {int(stats['pairs'])}")
        click.echo(f"  MAE: {stats['mae_ms']:.1f} ms")
        click.echo(f"  Median AE: {stats['median_ae_ms']:.1f} ms")
        click.echo(f"  AR@50ms: {stats['ar_50ms_pct']:.1f}%")
        click.echo(f"  Runtime: {stats['runtime_s']:.2f}s")

    success = check_success_criteria(results)
    if success["available"]:
        click.echo("\n=== Success Criteria ===")
        click.echo(
            f"Criterion 1 (MAE < 50ms): {'PASS' if success['criterion_1_pass'] else 'FAIL'} "
            f"[{success['mae_seconds'] * 1000:.1f} ms]"
        )
        click.echo(
            f"Criterion 2 (MAE < 20ms): {'PASS' if success['criterion_2_mae_pass'] else 'FAIL'}"
        )
        click.echo(
            f"Criterion 2 (AR@50ms > 98%): {'PASS' if success['criterion_2_ar_pass'] else 'FAIL'} "
            f"[{success['ar_50ms'] * 100:.1f}%]"
        )


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="figures/", help="Output directory for figures")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "pdf", "svg"]))
def visualize(results_path: str, output_dir: str, fmt: str) -> None:
    """Generate figures from DeepAlign evaluation results."""
    import pandas as pd

    from dis_alignment.analysis.visualize import (
        create_summary_table,
        plot_error_vs_length,
        plot_metric_boxplot,
        plot_runtime_comparison,
    )

    results = pd.read_csv(results_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    click.echo("Generating figures...")
    plot_error_vs_length(results, output_path=output / f"error_vs_length.{fmt}")
    click.echo("  - error_vs_length")
    plot_runtime_comparison(results, output_path=output / f"runtime_comparison.{fmt}")
    click.echo("  - runtime_comparison")
    plot_metric_boxplot(results, metric="mae", output_path=output / f"mae_boxplot.{fmt}")
    click.echo("  - mae_boxplot")
    plot_metric_boxplot(results, metric="ar_50ms", output_path=output / f"ar50_boxplot.{fmt}")
    click.echo("  - ar50_boxplot")
    create_summary_table(results, output_path=output / "summary.csv")
    click.echo("  - summary.csv")
    click.echo(f"\nFigures saved to {output.resolve()}")


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--baseline", default="chroma_dtw", help="Baseline method name")
@click.option("--candidate", default="deepalign", help="Candidate method name")
def analyze(results_path: str, baseline: str, candidate: str) -> None:
    """Run statistical analysis on DeepAlign evaluation results."""
    import pandas as pd

    from dis_alignment.analysis.statistics import (
        compute_significance,
        failure_analysis,
        memory_analysis,
        runtime_efficiency_analysis,
    )

    results = pd.read_csv(results_path)

    click.echo("=== Statistical Significance ===")
    significance = compute_significance(results, baseline, candidate, "mae")
    if "error" in significance:
        click.echo(f"Unable to compare {baseline} vs {candidate}: {significance['error']}")
    else:
        click.echo(f"Comparing {baseline} vs {candidate} (MAE):")
        click.echo(f"  Baseline mean: {significance['baseline_mean']:.4f} s")
        click.echo(f"  Candidate mean: {significance['candidate_mean']:.4f} s")
        click.echo(f"  Improvement: {significance['improvement_pct']:.1f}%")
        click.echo(f"  p-value: {significance['p_value_twosided']:.4f}")
        click.echo(f"  Significant: {significance['is_significant']}")
        click.echo(
            f"  Effect size: {significance['effect_size']:.3f} "
            f"({significance['effect_interpretation']})"
        )

    click.echo("\n=== Runtime Efficiency ===")
    for method, stats in runtime_efficiency_analysis(results).items():
        click.echo(f"\n{method}:")
        click.echo(f"  Complexity: {stats['complexity_estimate']}")
        click.echo(f"  Mean runtime: {stats['mean_runtime']:.2f} s")

    click.echo("\n=== Memory ===")
    for method, stats in memory_analysis(results).items():
        click.echo(f"\n{method}:")
        click.echo(f"  Mean memory: {stats['mean_memory_mb']:.2f} MB")
        click.echo(f"  Max memory: {stats['max_memory_mb']:.2f} MB")

    failures = failure_analysis(results)
    click.echo("\n=== Failure Analysis ===")
    click.echo(f"Failure rate: {failures['failure_rate']:.1%}")
    click.echo(f"Failures by method: {failures['failures_by_algorithm']}")


@main.command(name="inspect-features")
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Optional output path for the feature comparison plot")
def inspect_features(audio_path: str, output: str | None) -> None:
    """Inspect the hand-crafted baseline features used in DeepAlign comparisons."""
    import librosa
    import matplotlib.pyplot as plt

    from dis_alignment.analysis.visualize import plot_feature_comparison
    from dis_alignment.features.chroma import extract_chroma_cqt
    from dis_alignment.features.dlnco import extract_dlnco

    audio, sr = librosa.load(audio_path, sr=22050)
    chroma = extract_chroma_cqt(audio, sr=sr)
    dlnco = extract_dlnco(audio, sr=sr)

    click.echo(f"Chroma shape: {chroma.shape}")
    click.echo(f"DLNCO shape: {dlnco.shape}")

    if output:
        plot_feature_comparison(chroma, dlnco, sr=sr, output_path=output)
        click.echo(f"Feature comparison saved to {output}")
        return

    plot_feature_comparison(chroma, dlnco, sr=sr)
    plt.show()


@main.command(name="legacy-maestro-benchmark")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default="test", help="Dataset split: train/validation/test")
@click.option("--algorithms", default="global_dtw,mrmsdtw", help="Comma-separated algorithms")
@click.option("--features", default="chroma", type=click.Choice(["chroma", "dlnco", "combined"]))
@click.option("--limit", default=None, type=int, help="Limit number of pieces")
@click.option("--output", "-o", default="results/legacy_maestro_results.csv", help="Output CSV path")
@click.option("--memory-limit", default=500, type=int, help="Memory limit for MrMsDTW (MB)")
def legacy_maestro_benchmark(
    dataset_path: str,
    split: str,
    algorithms: str,
    features: str,
    limit: int | None,
    output: str,
    memory_limit: int,
) -> None:
    """Legacy MAESTRO benchmark kept only for baseline-oriented experiments."""
    from dis_alignment.data.maestro import MAESTRODataset
    from dis_alignment.evaluation.runner import BenchmarkConfig, BenchmarkRunner

    dataset = MAESTRODataset(dataset_path)
    algo_list = [algorithm.strip() for algorithm in algorithms.split(",")]
    config = BenchmarkConfig(
        feature_type=features,
        algorithms=algo_list,
        memory_limit_mb=memory_limit,
    )
    runner = BenchmarkRunner(config)
    results = runner.run_on_dataset(dataset, split=split, limit=limit)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner.save_results(output_path)
    click.echo(f"Legacy benchmark results saved to {output_path}")

    click.echo("\n=== Summary ===")
    for algorithm in algo_list:
        algo_results = results[results["algorithm"] == algorithm]
        click.echo(f"\n{algorithm}:")
        click.echo(f"  MAE: {algo_results['mae'].mean():.4f} ± {algo_results['mae'].std():.4f} s")
        click.echo(f"  AR@50ms: {algo_results['ar_50ms'].mean():.2%}")
        click.echo(f"  Runtime: {algo_results['runtime_s'].mean():.2f} s")


if __name__ == "__main__":
    main()
