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

DEEP_DECODE_CHOICES = [
    "unconstrained",
    "diagonal_band",
    "chroma_guided_band",
    "deepalign_transcription_fused",
    "deepalign_transcription_fused_refined",
    "deepalign_transcription_guided",
    "deepalign_score_guided_refined",
]


def _print_swd_summary(summary: dict[str, Any]) -> None:
    click.echo(f"SWD root: {summary['root']}")
    click.echo(f"Audio files: {summary['audio_files']}")
    click.echo(f"Measure annotations: {summary['annotations']}")
    click.echo(f"Lieder: {summary['lieder']}")
    click.echo(f"Available pairs: {summary['pairs']}")
    click.echo("Performances:")
    for performance_id, count in summary["performances"].items():
        click.echo(f"  - {performance_id}: {count} recordings")


def _print_mazurka_summary(summary: dict[str, Any]) -> None:
    click.echo(f"Mazurka root: {summary['root']}")
    click.echo(f"Performances: {summary['performances']}")
    click.echo(f"Works: {summary['works']}")
    click.echo(f"Pairs: {summary['pairs']}")
    click.echo(f"With score files: {summary['performances_with_scores']}")
    click.echo(f"With annotations: {summary['performances_with_annotations']}")


def _coalesce(cli_value: Any, config_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if config_value is not None:
        return config_value
    return default


def _print_evaluation_summary(results, candidate_method: str = "deepalign") -> None:
    from dis_alignment.evaluation import check_success_criteria, summarize_evaluation

    click.echo("\n=== Summary ===")
    for method, stats in summarize_evaluation(results).items():
        click.echo(f"\n{method}:")
        click.echo(f"  pairs: {int(stats['pairs'])}")
        click.echo(f"  MAE: {stats['mae_ms']:.1f} ms")
        click.echo(f"  Median AE: {stats['median_ae_ms']:.1f} ms")
        click.echo(f"  AR@50ms: {stats['ar_50ms_pct']:.1f}%")
        click.echo(f"  Runtime: {stats['runtime_s']:.2f}s")

    success = check_success_criteria(results, candidate_method=candidate_method)
    if success["available"]:
        click.echo("\n=== Success Criteria ===")
        click.echo(
            f"{candidate_method}: Criterion 1 (MAE < 50ms): "
            f"{'PASS' if success['criterion_1_pass'] else 'FAIL'} "
            f"[{success['mae_seconds'] * 1000:.1f} ms]"
        )
        click.echo(
            f"{candidate_method}: Criterion 2 (MAE < 20ms): "
            f"{'PASS' if success['criterion_2_mae_pass'] else 'FAIL'}"
        )
        click.echo(
            f"{candidate_method}: Criterion 2 (AR@50ms > 98%): "
            f"{'PASS' if success['criterion_2_ar_pass'] else 'FAIL'} "
            f"[{success['ar_50ms'] * 100:.1f}%]"
        )
    elif "error" in success:
        click.echo(f"\nSuccess criteria not aggregated: {success['error']}")


@click.group()
@click.version_option()
def main() -> None:
    """DeepAlign-26 dissertation workflow.

    Recommended path:
      1. `deepalign prepare-swd`
      2. `deepalign train`
      3. `deepalign evaluate-swd --methods chroma_dtw,mrmsdtw,deepalign,matchmaker`
      4. `deepalign evaluate-mazurka`
      5. `deepalign visualize`
      6. `deepalign analyze`
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
    _print_swd_summary(verify_swd_dataset(extracted_path))


@main.command(name="verify-mazurka")
@click.argument("dataset_path", type=click.Path(exists=True))
def verify_mazurka(dataset_path: str) -> None:
    """Verify a MazurkaBL-style checkout before robustness evaluation."""
    from dis_alignment.data import verify_mazurka_dataset

    _print_mazurka_summary(verify_mazurka_dataset(dataset_path))


@main.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="YAML training config")
@click.option("--swd-path", type=click.Path(exists=True), default=None, help="Path to the SWD root")
@click.option("--output-dir", type=click.Path(), default=None, help="Checkpoint output directory")
@click.option("--epochs", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--max-length-sec", type=float, default=None, help="Maximum aligned segment duration in seconds")
@click.option("--temporal-attention-heads", type=int, default=None, help="Optional temporal attention heads after the GRU stack")
@click.option("--device", type=str, default=None, help="Explicit device, e.g. cpu or cuda")
@click.option("--resume-from", type=click.Path(exists=True), default=None, help="Resume or warm-start from a checkpoint")
@click.option(
    "--selection-metric",
    type=click.Choice(["debug_mae", "debug_ar50", "debug_balanced", "val_loss", "val_mae"]),
    default=None,
    help="Checkpoint selection metric",
)
@click.option(
    "--segment-sampling",
    type=click.Choice(["aligned_measures", "independent_random"]),
    default=None,
    help="Segment sampling strategy for SWD training pairs",
)
@click.option("--samples-per-epoch", type=int, default=None, help="Effective training samples per epoch")
@click.option("--cache-spectrograms/--no-cache-spectrograms", default=None, help="Cache full-song CQTs for training/eval gates")
@click.option("--cache-root", type=click.Path(), default=None, help="Directory for cached full-song CQTs")
@click.option(
    "--deep-decode",
    type=click.Choice(DEEP_DECODE_CHOICES),
    default=None,
    help="Decode mode for debug/validation alignment gates during training",
)
@click.option("--band-radius-frames", type=int, default=None, help="Band radius for constrained DeepAlign decoding")
@click.option("--start-gamma", type=float, default=None)
@click.option("--end-gamma", type=float, default=None)
@click.option("--anchor-loss-weight", type=float, default=None, help="Optional anchor contrastive loss weight")
@click.option("--anchor-temperature", type=float, default=None, help="Anchor contrastive loss temperature")
@click.option("--anchor-min-anchor-gap", type=int, default=None, help="Minimum event gap for anchor negatives")
@click.option("--no-augment", is_flag=True, help="Disable waveform augmentation")
@click.option("--dry-run", is_flag=True, help="Run the synthetic dry-run pipeline")
def train(
    config: str | None,
    swd_path: str | None,
    output_dir: str | None,
    epochs: int | None,
    batch_size: int | None,
    max_length_sec: float | None,
    temporal_attention_heads: int | None,
    device: str | None,
    resume_from: str | None,
    selection_metric: str | None,
    segment_sampling: str | None,
    samples_per_epoch: int | None,
    cache_spectrograms: bool | None,
    cache_root: str | None,
    deep_decode: str | None,
    band_radius_frames: int | None,
    start_gamma: float | None,
    end_gamma: float | None,
    anchor_loss_weight: float | None,
    anchor_temperature: float | None,
    anchor_min_anchor_gap: int | None,
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
    augmentor_kwargs = (
        {
            "time_stretch_range": tuple(aug_cfg["time_stretch_range"]),
            "pitch_shift_range": tuple(aug_cfg["pitch_shift_range"]),
            "noise_snr_range": tuple(aug_cfg["noise_snr_range"]),
            "prob": aug_cfg["augment_prob"],
            "sr": training_config.get("audio", {}).get("sr", 22050),
        }
        if augment_enabled and aug_cfg
        else {}
    )

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
        temporal_attention_heads=_coalesce(temporal_attention_heads, training_config.get("model", {}).get("temporal_attention_heads"), 0),
        max_length_sec=_coalesce(max_length_sec, dataset_cfg.get("max_length_sec"), 30.0),
        segment_sampling=_coalesce(segment_sampling, dataset_cfg.get("segment_sampling"), "aligned_measures"),
        samples_per_epoch=_coalesce(samples_per_epoch, dataset_cfg.get("samples_per_epoch"), None),
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
        cache_spectrograms=_coalesce(cache_spectrograms, dataset_cfg.get("cache_spectrograms"), False),
        cache_root=_coalesce(cache_root, dataset_cfg.get("cache_root"), None),
        resume_from=_coalesce(resume_from, training_cfg.get("resume_from"), None),
        selection_metric=_coalesce(selection_metric, training_cfg.get("selection_metric"), "debug_ar50"),
        debug_subset_lieder=dataset_cfg.get("debug_subset_lieder"),
        deep_decode=_coalesce(deep_decode, training_config.get("evaluation", {}).get("deep_decode"), "unconstrained"),
        band_radius_frames=_coalesce(band_radius_frames, training_config.get("evaluation", {}).get("band_radius_frames"), None),
        anchor_loss_weight=_coalesce(anchor_loss_weight, training_cfg.get("anchor_loss_weight"), 0.0),
        anchor_temperature=_coalesce(anchor_temperature, training_cfg.get("anchor_temperature"), 0.1),
        anchor_min_anchor_gap=_coalesce(anchor_min_anchor_gap, training_cfg.get("anchor_min_anchor_gap"), 1),
        dry_run=dry_run,
    )


@main.command(name="evaluate-swd")
@click.argument("swd_path", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), default=None, help="Optional DeepAlign checkpoint")
@click.option("--methods", default=None, help="Comma-separated methods: chroma_dtw,mrmsdtw,deepalign,matchmaker")
@click.option("--output", "-o", default="results/swd_evaluation.csv", help="Output CSV path")
@click.option("--device", type=str, default=None)
@click.option("--cache-root", type=click.Path(), default=None, help="Directory for cached full-song CQTs")
@click.option("--pool-size", type=int, default=2, help="Temporal pooling factor for DeepAlign features")
@click.option("--deep-distance", type=click.Choice(["sqeuclidean", "cosine"]), default="sqeuclidean", help="DTW distance for DeepAlign feature comparisons")
@click.option(
    "--deep-decode",
    default="unconstrained",
    type=click.Choice(DEEP_DECODE_CHOICES),
    help="DeepAlign decoding mode",
)
@click.option("--band-radius-frames", type=int, default=None, help="Band radius for constrained DeepAlign decoding")
@click.option("--transcription-cache-root", type=click.Path(), default=None, help="Directory for Basic Pitch feature cache")
@click.option("--fusion-deep-weight", type=float, default=1.0, help="DeepAlign feature weight for transcription fusion")
@click.option("--fusion-onset-weight", type=float, default=1.0, help="Basic Pitch onset weight for transcription fusion")
@click.option("--fusion-note-weight", type=float, default=0.75, help="Basic Pitch note/contour weight for transcription fusion")
@click.option("--fusion-dlnco-weight", type=float, default=0.5, help="DLNCO weight for transcription fusion")
@click.option("--fusion-chroma-weight", type=float, default=0.25, help="Chroma weight for transcription fusion")
@click.option("--refine-window-sec", type=float, default=8.0, help="Guided transcription refinement window in seconds")
@click.option("--score-refine-radius-sec", type=float, default=0.5, help="Local score-guided anchor refinement radius")
@click.option("--performance", "performances", multiple=True, help="Limit to one or more performance ids")
@click.option("--lied", "lieder", multiple=True, help="Limit to one or more lied ids")
@click.option("--quiet", is_flag=True, help="Disable the progress bar")
@click.option("--matchmaker-method", default="arzt", help="Matchmaker score-following method")
@click.option("--matchmaker-feature-type", default="chroma", help="Matchmaker feature type")
@click.option("--matchmaker-frame-rate", default=100, type=int, help="Matchmaker frame rate")
def evaluate_swd(
    swd_path: str,
    checkpoint: str | None,
    methods: str | None,
    output: str,
    device: str | None,
    cache_root: str | None,
    pool_size: int,
    deep_distance: str,
    deep_decode: str,
    band_radius_frames: int | None,
    transcription_cache_root: str | None,
    fusion_deep_weight: float,
    fusion_onset_weight: float,
    fusion_note_weight: float,
    fusion_dlnco_weight: float,
    fusion_chroma_weight: float,
    refine_window_sec: float,
    score_refine_radius_sec: float,
    performances: tuple[str, ...],
    lieder: tuple[str, ...],
    quiet: bool,
    matchmaker_method: str,
    matchmaker_feature_type: str,
    matchmaker_frame_rate: int,
) -> None:
    """Evaluate SWD pairs with offline baselines and optional DeepAlign checkpoint."""
    from dis_alignment.data import SWDDataset
    from dis_alignment.evaluation import evaluate_swd_dataset, save_evaluation_results

    dataset = SWDDataset(swd_path)
    results = evaluate_swd_dataset(
        dataset,
        checkpoint_path=checkpoint,
        device=device,
        methods=methods,
        cache_root=cache_root,
        pool_size=pool_size,
        deep_distance=deep_distance,
        deep_decode=deep_decode,
        band_radius_frames=band_radius_frames,
        transcription_cache_root=transcription_cache_root,
        fusion_deep_weight=fusion_deep_weight,
        fusion_onset_weight=fusion_onset_weight,
        fusion_note_weight=fusion_note_weight,
        fusion_dlnco_weight=fusion_dlnco_weight,
        fusion_chroma_weight=fusion_chroma_weight,
        refine_window_sec=refine_window_sec,
        score_refine_radius_sec=score_refine_radius_sec,
        performances=list(performances) or None,
        lieder=list(lieder) or None,
        show_progress=not quiet,
        matchmaker_method=matchmaker_method,
        matchmaker_feature_type=matchmaker_feature_type,
        matchmaker_frame_rate=matchmaker_frame_rate,
    )
    if results.empty:
        raise click.ClickException("No SWD evaluation rows were produced. Check annotations, score files, and filters.")

    output_path = save_evaluation_results(results, output)
    click.echo(f"Saved evaluation results to {output_path}")
    _print_evaluation_summary(results)


@main.command(name="evaluate-mazurka")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), default=None, help="Optional DeepAlign checkpoint")
@click.option("--methods", default=None, help="Comma-separated methods: chroma_dtw,mrmsdtw,deepalign,matchmaker")
@click.option("--output", "-o", default="results/mazurka_evaluation.csv", help="Output CSV path")
@click.option("--device", type=str, default=None)
@click.option("--cache-root", type=click.Path(), default=None, help="Directory for cached full-song CQTs")
@click.option("--pool-size", type=int, default=2, help="Temporal pooling factor for DeepAlign features")
@click.option("--deep-distance", type=click.Choice(["sqeuclidean", "cosine"]), default="sqeuclidean", help="DTW distance for DeepAlign feature comparisons")
@click.option(
    "--deep-decode",
    default="unconstrained",
    type=click.Choice(DEEP_DECODE_CHOICES),
    help="DeepAlign decoding mode",
)
@click.option("--band-radius-frames", type=int, default=None, help="Band radius for constrained DeepAlign decoding")
@click.option("--transcription-cache-root", type=click.Path(), default=None, help="Directory for Basic Pitch feature cache")
@click.option("--fusion-deep-weight", type=float, default=1.0, help="DeepAlign feature weight for transcription fusion")
@click.option("--fusion-onset-weight", type=float, default=1.0, help="Basic Pitch onset weight for transcription fusion")
@click.option("--fusion-note-weight", type=float, default=0.75, help="Basic Pitch note/contour weight for transcription fusion")
@click.option("--fusion-dlnco-weight", type=float, default=0.5, help="DLNCO weight for transcription fusion")
@click.option("--fusion-chroma-weight", type=float, default=0.25, help="Chroma weight for transcription fusion")
@click.option("--refine-window-sec", type=float, default=8.0, help="Guided transcription refinement window in seconds")
@click.option("--score-refine-radius-sec", type=float, default=0.5, help="Local score-guided anchor refinement radius")
@click.option("--work", "works", multiple=True, help="Limit to one or more work ids")
@click.option("--quiet", is_flag=True, help="Disable the progress bar")
@click.option("--matchmaker-method", default="arzt", help="Matchmaker score-following method")
@click.option("--matchmaker-feature-type", default="chroma", help="Matchmaker feature type")
@click.option("--matchmaker-frame-rate", default=100, type=int, help="Matchmaker frame rate")
def evaluate_mazurka(
    dataset_path: str,
    checkpoint: str | None,
    methods: str | None,
    output: str,
    device: str | None,
    cache_root: str | None,
    pool_size: int,
    deep_distance: str,
    deep_decode: str,
    band_radius_frames: int | None,
    transcription_cache_root: str | None,
    fusion_deep_weight: float,
    fusion_onset_weight: float,
    fusion_note_weight: float,
    fusion_dlnco_weight: float,
    fusion_chroma_weight: float,
    refine_window_sec: float,
    score_refine_radius_sec: float,
    works: tuple[str, ...],
    quiet: bool,
    matchmaker_method: str,
    matchmaker_feature_type: str,
    matchmaker_frame_rate: int,
) -> None:
    """Evaluate MazurkaBL-style pairs for robustness and rubato stress testing."""
    from dis_alignment.data import MazurkaDataset
    from dis_alignment.evaluation import evaluate_mazurka_dataset, save_evaluation_results

    dataset = MazurkaDataset(dataset_path)
    results = evaluate_mazurka_dataset(
        dataset,
        checkpoint_path=checkpoint,
        device=device,
        methods=methods,
        cache_root=cache_root,
        pool_size=pool_size,
        deep_distance=deep_distance,
        deep_decode=deep_decode,
        band_radius_frames=band_radius_frames,
        transcription_cache_root=transcription_cache_root,
        fusion_deep_weight=fusion_deep_weight,
        fusion_onset_weight=fusion_onset_weight,
        fusion_note_weight=fusion_note_weight,
        fusion_dlnco_weight=fusion_dlnco_weight,
        fusion_chroma_weight=fusion_chroma_weight,
        refine_window_sec=refine_window_sec,
        score_refine_radius_sec=score_refine_radius_sec,
        works=list(works) or None,
        show_progress=not quiet,
        matchmaker_method=matchmaker_method,
        matchmaker_feature_type=matchmaker_feature_type,
        matchmaker_frame_rate=matchmaker_frame_rate,
    )
    if results.empty:
        raise click.ClickException(
            "No Mazurka evaluation rows were produced. Check annotations, score files, and filters."
        )

    output_path = save_evaluation_results(results, output)
    click.echo(f"Saved evaluation results to {output_path}")
    _print_evaluation_summary(results)


@main.command(name="merge-results")
@click.argument("results_paths", nargs=-1, type=click.Path(exists=True))
@click.option("--output", "-o", default="results/merged_evaluation.csv", help="Output CSV path")
def merge_results(results_paths: tuple[str, ...], output: str) -> None:
    """Merge evaluation CSVs produced in separate environments."""
    from dis_alignment.evaluation import add_method_variant_column, merge_evaluation_results, save_evaluation_results

    if not results_paths:
        raise click.ClickException("Provide at least one results CSV to merge.")

    try:
        merged = merge_evaluation_results(list(results_paths))
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    if merged.empty:
        raise click.ClickException("The provided result files were empty, so nothing was merged.")

    output_path = save_evaluation_results(merged, output)
    click.echo(f"Saved merged evaluation results to {output_path}")
    click.echo(f"Merged rows: {len(merged)}")
    click.echo(f"Methods: {', '.join(sorted(merged['method'].astype(str).unique()))}")
    labelled = add_method_variant_column(merged)
    if "method_variant" in labelled.columns:
        click.echo(f"Report labels: {', '.join(sorted(labelled['method_variant'].astype(str).unique()))}")
    click.echo(f"Datasets: {', '.join(sorted(merged['dataset'].astype(str).unique()))}")


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="figures/", help="Output directory for figures")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "pdf", "svg"]))
def visualize(results_path: str, output_dir: str, fmt: str) -> None:
    """Generate dissertation-ready figures from evaluation results."""
    import pandas as pd

    from dis_alignment.analysis.visualize import (
        create_success_criteria_table,
        create_summary_table,
        plot_error_histogram,
        plot_error_vs_length,
        plot_metric_boxplot,
        plot_runtime_comparison,
        plot_success_criteria_summary,
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
    plot_error_histogram(results, output_path=output / f"mae_histogram.{fmt}")
    click.echo("  - mae_histogram")
    plot_success_criteria_summary(results, output_path=output / f"success_criteria.{fmt}")
    click.echo("  - success_criteria")
    create_summary_table(results, output_path=output / "summary.csv")
    click.echo("  - summary.csv")
    create_success_criteria_table(results, output_path=output / "success_criteria.csv")
    click.echo("  - success_criteria.csv")
    click.echo(f"\nFigures saved to {output.resolve()}")


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--baseline", default="chroma_dtw", help="Baseline method name")
@click.option("--candidate", default="deepalign", help="Candidate method name")
@click.option("--metric", default="mae", help="Metric to analyze")
def analyze(results_path: str, baseline: str, candidate: str, metric: str) -> None:
    """Run statistical analysis on evaluation results."""
    import pandas as pd

    from dis_alignment.analysis.statistics import (
        compute_significance,
        failure_analysis,
        friedman_nemenyi_analysis,
        memory_analysis,
        runtime_efficiency_analysis,
    )

    from dis_alignment.evaluation import add_method_variant_column

    results = add_method_variant_column(pd.read_csv(results_path))
    method_col = "method_variant" if "method_variant" in results.columns else "method"
    available_methods = sorted(results[method_col].unique()) if method_col in results.columns else []

    if len(available_methods) >= 3:
        click.echo("=== Friedman / Nemenyi ===")
        omnibus = friedman_nemenyi_analysis(results, metric=metric)
        if not omnibus["available"]:
            click.echo(f"Unable to run omnibus analysis: {omnibus['error']}")
        else:
            click.echo(f"Metric: {metric}")
            click.echo(f"Items: {omnibus['n_items']}")
            click.echo(f"Methods: {omnibus['n_methods']}")
            click.echo(f"Friedman statistic: {omnibus['friedman_statistic']:.4f}")
            click.echo(f"Friedman p-value: {omnibus['friedman_p_value']:.4f}")
            click.echo(f"Critical difference: {omnibus['critical_difference']:.4f}")
            click.echo("Average ranks:")
            for method, rank in omnibus["average_ranks"].items():
                click.echo(f"  {method}: {rank:.3f}")

            significant_pairs = [row for row in omnibus["nemenyi"] if row["significant"]]
            click.echo("Nemenyi significant comparisons:")
            if significant_pairs:
                for row in significant_pairs:
                    click.echo(
                        f"  {row['method_a']} vs {row['method_b']}: "
                        f"rank diff={row['rank_diff']:.3f}, p={row['p_value']:.4f}"
                    )
            else:
                click.echo("  none")

    click.echo("\n=== Pairwise Significance ===")
    significance = compute_significance(results, baseline, candidate, metric)
    if "error" in significance:
        click.echo(f"Unable to compare {baseline} vs {candidate}: {significance['error']}")
    else:
        click.echo(f"Comparing {baseline} vs {candidate} ({metric}):")
        click.echo(f"  Baseline mean: {significance['baseline_mean']:.4f}")
        click.echo(f"  Candidate mean: {significance['candidate_mean']:.4f}")
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
