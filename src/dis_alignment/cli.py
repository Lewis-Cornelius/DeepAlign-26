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


def _write_failure_report(results: Any, output_path: Path) -> None:
    failures = getattr(results, "attrs", {}).get("failures")
    if failures is None or failures.empty:
        return
    failure_path = output_path.with_name(f"{output_path.stem}_failures.csv")
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_csv(failure_path, index=False)
    click.echo(f"Skipped pairs written to {failure_path}")


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


@main.command(name="generate-teacher-paths")
@click.argument("swd_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="results/teacher_audio_only_sota", help="Teacher artifact directory")
@click.option("--sr", type=int, default=22050, help="Teacher audio sample rate")
@click.option("--hop-length", type=int, default=110, help="Teacher feature hop length")
@click.option("--backend", type=click.Choice(["mrmsdtw", "full"]), default="mrmsdtw", help="Teacher DTW backend")
@click.option("--distance", type=click.Choice(["cosine", "sqeuclidean"]), default="cosine", help="Full-DTW distance")
@click.option("--memory-limit-mb", type=int, default=500, help="MrMsDTW memory limit")
@click.option("--chroma-weight", type=float, default=1.0, help="Teacher chroma feature weight")
@click.option("--dlnco-weight", type=float, default=1.0, help="Teacher DLNCO feature weight")
@click.option("--spectral-flux-weight", type=float, default=0.5, help="Teacher spectral-flux feature weight")
@click.option(
    "--trim-top-db",
    type=float,
    default=40.0,
    help="Audio-only active-region trim threshold; use a negative value to disable",
)
@click.option(
    "--estimate-chroma-shift/--no-estimate-chroma-shift",
    default=True,
    help="Estimate a chroma roll for the first recording before teacher DTW",
)
@click.option(
    "--chroma-shift-max-frames",
    type=int,
    default=1500,
    help="Maximum pooled frames used when estimating teacher chroma shift",
)
@click.option(
    "--confidence-local-radius-frames",
    type=int,
    default=24,
    help="Local reference radius for audio-only teacher confidence scoring",
)
@click.option(
    "--confidence-exclusion-radius-frames",
    type=int,
    default=2,
    help="Exclude near-identical frames when scoring teacher confidence",
)
@click.option(
    "--anchor-calibrate/--no-anchor-calibrate",
    default=False,
    help="Use SWD measure annotations to calibrate teacher paths for supervised training",
)
@click.option("--performance", "performances", multiple=True, help="Limit to one or more performance ids")
@click.option("--lied", "lieder", multiple=True, help="Limit to one or more lied ids")
@click.option("--min-ar50", type=float, default=0.95, help="Gate threshold for teacher AR@50")
@click.option("--max-mae-ms", type=float, default=None, help="Optional gate threshold for teacher MAE in ms")
@click.option("--fail-below-gate/--no-fail-below-gate", default=True, help="Fail if teacher misses --min-ar50")
@click.option("--quiet", is_flag=True, help="Disable progress bar")
def generate_teacher_paths(
    swd_path: str,
    output_dir: str,
    sr: int,
    hop_length: int,
    backend: str,
    distance: str,
    memory_limit_mb: int,
    chroma_weight: float,
    dlnco_weight: float,
    spectral_flux_weight: float,
    trim_top_db: float,
    estimate_chroma_shift: bool,
    chroma_shift_max_frames: int,
    confidence_local_radius_frames: int,
    confidence_exclusion_radius_frames: int,
    anchor_calibrate: bool,
    performances: tuple[str, ...],
    lieder: tuple[str, ...],
    min_ar50: float,
    max_mae_ms: float | None,
    fail_below_gate: bool,
    quiet: bool,
) -> None:
    """Generate audio-only teacher paths and strict validation CSVs for SWD."""
    from dis_alignment.alignment.teacher import generate_swd_teacher_paths, summarize_teacher_results
    from dis_alignment.data import SWDDataset

    dataset = SWDDataset(swd_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    teacher_results, oracle_results = generate_swd_teacher_paths(
        dataset,
        output_dir=output,
        sr=sr,
        hop_length=hop_length,
        performances=list(performances) or None,
        lieder=list(lieder) or None,
        backend=backend,
        distance=distance,
        memory_limit_mb=memory_limit_mb,
        chroma_weight=chroma_weight,
        dlnco_weight=dlnco_weight,
        spectral_flux_weight=spectral_flux_weight,
        trim_top_db=trim_top_db if trim_top_db >= 0 else None,
        estimate_chroma_shift=estimate_chroma_shift,
        chroma_shift_max_frames=chroma_shift_max_frames,
        confidence_local_radius_frames=confidence_local_radius_frames,
        confidence_exclusion_radius_frames=confidence_exclusion_radius_frames,
        anchor_calibrate=anchor_calibrate,
        show_progress=not quiet,
    )
    teacher_csv = output / "teacher_results.csv"
    oracle_csv = output / "oracle_results.csv"
    teacher_results.to_csv(teacher_csv, index=False)
    oracle_results.to_csv(oracle_csv, index=False)

    teacher_summary = summarize_teacher_results(teacher_results)
    oracle_summary = summarize_teacher_results(oracle_results)
    click.echo(f"Teacher paths saved under {output / 'paths'}")
    click.echo(f"Teacher validation CSV: {teacher_csv}")
    click.echo(f"Oracle sanity CSV: {oracle_csv}")
    click.echo(
        "Oracle: "
        f"pairs={int(oracle_summary['pairs'])}, "
        f"MAE={oracle_summary['mae'] * 1000:.3f} ms, "
        f"AR@50={oracle_summary['ar_50ms'] * 100:.3f}%"
    )
    click.echo(
        "Teacher: "
        f"pairs={int(teacher_summary['pairs'])}, "
        f"MAE={teacher_summary['mae'] * 1000:.3f} ms, "
        f"AR@50={teacher_summary['ar_50ms'] * 100:.3f}%"
    )
    if anchor_calibrate:
        click.echo(
            "WARNING: --anchor-calibrate uses SWD measure annotations and is diagnostic-only; "
            "do not report these artifacts as the headline audio-only teacher."
        )
    gate_failures = []
    if teacher_summary["ar_50ms"] < min_ar50:
        gate_failures.append(
            f"AR@50 {teacher_summary['ar_50ms'] * 100:.2f}% is below {min_ar50 * 100:.2f}%"
        )
    if max_mae_ms is not None and teacher_summary["mae"] * 1000.0 >= max_mae_ms:
        gate_failures.append(
            f"MAE {teacher_summary['mae'] * 1000.0:.2f} ms is not below {max_mae_ms:.2f} ms"
        )
    if fail_below_gate and gate_failures:
        raise click.ClickException("Teacher gate failed: " + "; ".join(gate_failures) + ".")


@main.command(name="mine-pseudo-teacher-paths")
@click.argument("swd_path", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), required=True, help="Audio-only encoder checkpoint")
@click.option("--output-dir", "-o", default="results/pseudo_teacher_audio_only", help="Pseudo-teacher artifact directory")
@click.option("--sr", type=int, default=22050, help="Audio sample rate")
@click.option("--hop-length", type=int, default=440, help="Feature hop length used for mining")
@click.option("--device", type=str, default=None)
@click.option("--cache-root", type=click.Path(), default=None, help="Directory for cached mining CQTs")
@click.option("--min-confidence", type=float, default=0.55, help="Minimum reciprocal-match confidence")
@click.option("--min-gap-frames", type=int, default=8, help="Minimum spacing between kept pseudo anchors")
@click.option("--max-anchors", type=int, default=2048, help="Maximum pseudo anchors per pair")
@click.option("--band-radius-sec", type=float, default=8.0, help="Audio-duration band radius for mining; negative disables")
@click.option(
    "--coarse-prior",
    type=click.Choice(["none", "full", "mrmsdtw"]),
    default="none",
    help="Audio-only coarse DTW path used as a local mining corridor",
)
@click.option("--coarse-local-radius-sec", type=float, default=2.0, help="Local radius around the coarse path; negative disables")
@click.option("--coarse-distance", type=click.Choice(["cosine", "sqeuclidean"]), default="cosine")
@click.option(
    "--use-coarse-path-only",
    is_flag=True,
    help="Use sampled points from the audio-only coarse DTW path instead of learned reciprocal matches",
)
@click.option("--chunk-size", type=int, default=1024, help="Nearest-neighbor chunk size")
@click.option("--performance", "performances", multiple=True, help="Limit to one or more performance ids")
@click.option("--lied", "lieder", multiple=True, help="Limit to one or more lied ids")
@click.option("--min-anchors", type=int, default=16, help="Fail if any pair has fewer pseudo anchors")
@click.option("--quiet", is_flag=True, help="Disable progress bar")
def mine_pseudo_teacher_paths(
    swd_path: str,
    checkpoint: str,
    output_dir: str,
    sr: int,
    hop_length: int,
    device: str | None,
    cache_root: str | None,
    min_confidence: float,
    min_gap_frames: int,
    max_anchors: int,
    band_radius_sec: float,
    coarse_prior: str,
    coarse_local_radius_sec: float,
    coarse_distance: str,
    use_coarse_path_only: bool,
    chunk_size: int,
    performances: tuple[str, ...],
    lieder: tuple[str, ...],
    min_anchors: int,
    quiet: bool,
) -> None:
    """Mine audio-only pseudo teacher paths from reciprocal learned features."""
    from dis_alignment.alignment.teacher import mine_swd_pseudo_teacher_paths, summarize_teacher_results
    from dis_alignment.data import SWDDataset

    dataset = SWDDataset(swd_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    pseudo_results, oracle_results = mine_swd_pseudo_teacher_paths(
        dataset,
        checkpoint_path=checkpoint,
        output_dir=output,
        sr=sr,
        hop_length=hop_length,
        performances=list(performances) or None,
        lieder=list(lieder) or None,
        device=device,
        cache_root=cache_root,
        min_confidence=min_confidence,
        min_gap_frames=min_gap_frames,
        max_anchors=max_anchors,
        band_radius_frames=(
            None
            if band_radius_sec < 0
            else max(0, int(round(band_radius_sec * sr / hop_length)))
        ),
        coarse_prior=coarse_prior,
        coarse_local_radius_frames=(
            None
            if coarse_local_radius_sec < 0
            else max(0, int(round(coarse_local_radius_sec * sr / hop_length)))
        ),
        coarse_distance=coarse_distance,
        use_coarse_path_only=use_coarse_path_only,
        chunk_size=chunk_size,
        show_progress=not quiet,
    )
    pseudo_csv = output / "teacher_results.csv"
    oracle_csv = output / "oracle_results.csv"
    pseudo_results.to_csv(pseudo_csv, index=False)
    oracle_results.to_csv(oracle_csv, index=False)
    summary = summarize_teacher_results(pseudo_results)
    click.echo(f"Pseudo-teacher paths saved under {output / 'paths'}")
    click.echo(f"Pseudo-teacher validation CSV: {pseudo_csv}")
    click.echo(
        "Pseudo teacher: "
        f"pairs={int(summary['pairs'])}, "
        f"MAE={summary['mae'] * 1000:.3f} ms, "
        f"AR@50={summary['ar_50ms'] * 100:.3f}%"
    )
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        import json

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        weak = [row for row in manifest if int(row.get("anchors", 0)) < min_anchors]
        if weak:
            labels = ", ".join(f"{row['pair_id']}={row.get('anchors', 0)}" for row in weak)
            raise click.ClickException(
                f"Pseudo-anchor mining produced too few anchors for: {labels}"
            )


@main.command()
@click.option("--config", type=click.Path(exists=True), default=None, help="YAML training config")
@click.option("--swd-path", type=click.Path(exists=True), default=None, help="Path to the SWD root")
@click.option("--output-dir", type=click.Path(), default=None, help="Checkpoint output directory")
@click.option("--epochs", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--sr", type=int, default=None, help="Training audio sample rate")
@click.option("--hop-length", type=int, default=None, help="Training CQT/deep feature hop length")
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
    type=click.Choice(["aligned_measures", "independent_random", "teacher_path", "self_audio"]),
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
@click.option("--soft-dtw-loss-weight", type=float, default=None, help="Weight for the Soft-DTW loss term")
@click.option("--anchor-loss-weight", type=float, default=None, help="Optional anchor contrastive loss weight")
@click.option("--dense-anchor-loss-weight", type=float, default=None, help="Dense measure-anchor contrastive loss weight")
@click.option("--path-distill-loss-weight", type=float, default=None, help="Teacher-path distillation loss weight")
@click.option("--teacher-path-root", type=click.Path(), default=None, help="Directory containing teacher path NPZ files")
@click.option("--num-anchor-samples", type=int, default=None, help="Teacher path samples per aligned crop")
@click.option("--teacher-min-confidence", type=float, default=None, help="Minimum teacher confidence for distillation samples")
@click.option(
    "--disable-time-stretch-for-anchors/--allow-time-stretch-for-anchors",
    default=None,
    help="Disable time-stretch augmentation when anchor/teacher times are used",
)
@click.option("--eval-pool-size", type=int, default=None, help="DeepAlign temporal pool size for training gates")
@click.option("--alignment-eval-every-n-epochs", type=int, default=None, help="Run alignment gates every N epochs; 0 disables")
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
    sr: int | None,
    hop_length: int | None,
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
    soft_dtw_loss_weight: float | None,
    anchor_loss_weight: float | None,
    dense_anchor_loss_weight: float | None,
    path_distill_loss_weight: float | None,
    teacher_path_root: str | None,
    num_anchor_samples: int | None,
    teacher_min_confidence: float | None,
    disable_time_stretch_for_anchors: bool | None,
    eval_pool_size: int | None,
    alignment_eval_every_n_epochs: int | None,
    anchor_temperature: float | None,
    anchor_min_anchor_gap: int | None,
    no_augment: bool,
    dry_run: bool,
) -> None:
    """Train DeepAlign-26 on SWD or validate the stack with a dry run."""
    from dis_alignment.model.train import (
        _is_headline_claim_config,
        _load_training_config,
        _validate_headline_claim_config,
        train as train_model,
    )

    training_config = _load_training_config(config)
    if _is_headline_claim_config(training_config):
        _validate_headline_claim_config(training_config, path=config)
    dataset_cfg = training_config.get("dataset", {})
    training_cfg = training_config.get("training", {})
    soft_dtw_cfg = training_config.get("soft_dtw", {})
    output_cfg = training_config.get("output", {})
    aug_cfg = training_config.get("augmentation", {})
    audio_cfg = training_config.get("audio", {})

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
            "sr": _coalesce(sr, audio_cfg.get("sr"), 22050),
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
        sr=_coalesce(sr, audio_cfg.get("sr"), 22050),
        hop_length=_coalesce(hop_length, audio_cfg.get("hop_length"), 220),
        n_freq_bins=_coalesce(None, audio_cfg.get("n_bins"), 84),
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
        soft_dtw_loss_weight=_coalesce(soft_dtw_loss_weight, soft_dtw_cfg.get("loss_weight"), 1.0),
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
        dense_anchor_loss_weight=_coalesce(dense_anchor_loss_weight, training_cfg.get("dense_anchor_loss_weight"), 0.0),
        path_distill_loss_weight=_coalesce(path_distill_loss_weight, training_cfg.get("path_distill_loss_weight"), 0.0),
        teacher_path_root=_coalesce(teacher_path_root, training_cfg.get("teacher_path_root"), None),
        num_anchor_samples=_coalesce(num_anchor_samples, training_cfg.get("num_anchor_samples"), 64),
        teacher_min_confidence=_coalesce(teacher_min_confidence, training_cfg.get("teacher_min_confidence"), 0.0),
        disable_time_stretch_for_anchors=_coalesce(
            disable_time_stretch_for_anchors,
            training_cfg.get("disable_time_stretch_for_anchors"),
            True,
        ),
        eval_pool_size=_coalesce(eval_pool_size, training_config.get("evaluation", {}).get("pool_size"), 2),
        alignment_eval_every_n_epochs=_coalesce(
            alignment_eval_every_n_epochs,
            training_config.get("evaluation", {}).get("alignment_eval_every_n_epochs"),
            1,
        ),
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
@click.option("--deep-hop", type=int, default=220, help="DeepAlign feature hop length in samples")
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
@click.option("--allow-skips", is_flag=True, help="Write a partial CSV when some pairs fail")
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
    deep_hop: int,
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
    allow_skips: bool,
    matchmaker_method: str,
    matchmaker_feature_type: str,
    matchmaker_frame_rate: int,
) -> None:
    """Evaluate SWD pairs with offline baselines and optional DeepAlign checkpoint."""
    from dis_alignment.data import SWDDataset
    from dis_alignment.evaluation import evaluate_swd_dataset, save_evaluation_results

    dataset = SWDDataset(swd_path)
    try:
        results = evaluate_swd_dataset(
            dataset,
            checkpoint_path=checkpoint,
            device=device,
            methods=methods,
            cache_root=cache_root,
            deep_hop=deep_hop,
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
            allow_skips=allow_skips,
            matchmaker_method=matchmaker_method,
            matchmaker_feature_type=matchmaker_feature_type,
            matchmaker_frame_rate=matchmaker_frame_rate,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if results.empty:
        _write_failure_report(results, Path(output))
        raise click.ClickException("No SWD evaluation rows were produced. Check annotations, score files, and filters.")

    output_path = save_evaluation_results(results, output)
    click.echo(f"Saved evaluation results to {output_path}")
    _write_failure_report(results, output_path)
    _print_evaluation_summary(results)


@main.command(name="evaluate-mazurka")
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--checkpoint", type=click.Path(exists=True), default=None, help="Optional DeepAlign checkpoint")
@click.option("--methods", default=None, help="Comma-separated methods: chroma_dtw,mrmsdtw,deepalign,matchmaker")
@click.option("--output", "-o", default="results/mazurka_evaluation.csv", help="Output CSV path")
@click.option("--device", type=str, default=None)
@click.option("--cache-root", type=click.Path(), default=None, help="Directory for cached full-song CQTs")
@click.option("--deep-hop", type=int, default=220, help="DeepAlign feature hop length in samples")
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
@click.option("--allow-skips", is_flag=True, help="Write a partial CSV when some pairs fail")
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
    deep_hop: int,
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
    allow_skips: bool,
    matchmaker_method: str,
    matchmaker_feature_type: str,
    matchmaker_frame_rate: int,
) -> None:
    """Evaluate MazurkaBL-style pairs for robustness and rubato stress testing."""
    from dis_alignment.data import MazurkaDataset
    from dis_alignment.evaluation import evaluate_mazurka_dataset, save_evaluation_results

    dataset = MazurkaDataset(dataset_path)
    try:
        results = evaluate_mazurka_dataset(
            dataset,
            checkpoint_path=checkpoint,
            device=device,
            methods=methods,
            cache_root=cache_root,
            deep_hop=deep_hop,
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
            allow_skips=allow_skips,
            matchmaker_method=matchmaker_method,
            matchmaker_feature_type=matchmaker_feature_type,
            matchmaker_frame_rate=matchmaker_frame_rate,
        )
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    if results.empty:
        _write_failure_report(results, Path(output))
        raise click.ClickException(
            "No Mazurka evaluation rows were produced. Check annotations, score files, and filters."
        )

    output_path = save_evaluation_results(results, output)
    click.echo(f"Saved evaluation results to {output_path}")
    _write_failure_report(results, output_path)
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
