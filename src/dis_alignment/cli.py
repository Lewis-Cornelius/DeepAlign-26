"""Command-line interface for dis-alignment benchmarking."""

import logging
from pathlib import Path

import click

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


@click.group()
@click.version_option()
def main():
    """Dis-Alignment: Audio-to-Score Alignment Benchmarking Tool."""
    pass


@main.command()
@click.argument("dataset_path", type=click.Path(exists=True))
@click.option("--split", default="test", help="Dataset split: train/validation/test")
@click.option("--algorithms", default="global_dtw,mrmsdtw", help="Comma-separated algorithms")
@click.option("--features", default="chroma", type=click.Choice(["chroma", "dlnco", "combined"]))
@click.option("--limit", default=None, type=int, help="Limit number of pieces")
@click.option("--output", "-o", default="results/benchmark_results.csv", help="Output CSV path")
@click.option("--memory-limit", default=500, type=int, help="Memory limit for MrMsDTW (MB)")
def benchmark(dataset_path, split, algorithms, features, limit, output, memory_limit):
    """
    Run alignment benchmark on MAESTRO dataset.
    
    DATASET_PATH: Path to the MAESTRO dataset directory.
    """
    from dis_alignment.data.maestro import MAESTRODataset
    from dis_alignment.evaluation.runner import BenchmarkConfig, BenchmarkRunner
    
    click.echo(f"Loading dataset from {dataset_path}...")
    dataset = MAESTRODataset(dataset_path)
    
    algo_list = [a.strip() for a in algorithms.split(",")]
    
    config = BenchmarkConfig(
        feature_type=features,
        algorithms=algo_list,
        memory_limit_mb=memory_limit,
    )
    
    runner = BenchmarkRunner(config)
    
    click.echo(f"Running benchmark on {split} split...")
    results = runner.run_on_dataset(dataset, split=split, limit=limit)
    
    # Ensure output directory exists
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    runner.save_results(output_path)
    click.echo(f"Results saved to {output_path}")
    
    # Print summary
    click.echo("\n=== Summary ===")
    for algo in algo_list:
        algo_results = results[results["algorithm"] == algo]
        click.echo(f"\n{algo}:")
        click.echo(f"  MAE: {algo_results['mae'].mean():.4f} ± {algo_results['mae'].std():.4f} s")
        click.echo(f"  AR@50ms: {algo_results['ar_50ms'].mean():.2%}")
        click.echo(f"  Runtime: {algo_results['runtime_s'].mean():.2f} s (mean)")


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output-dir", "-o", default="figures/", help="Output directory for figures")
@click.option("--format", "fmt", default="png", type=click.Choice(["png", "pdf", "svg"]))
def visualize(results_path, output_dir, fmt):
    """
    Generate visualization figures from benchmark results.
    
    RESULTS_PATH: Path to benchmark results CSV.
    """
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
    
    plot_metric_boxplot(results, metric="ar_50ms", output_path=output / f"ar_boxplot.{fmt}")
    click.echo("  - ar_boxplot")
    
    summary = create_summary_table(results, output_path=output / "summary.csv")
    click.echo("  - summary.csv")
    
    click.echo(f"\nAll figures saved to {output}/")


@main.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--baseline", default="global_dtw", help="Baseline algorithm name")
@click.option("--candidate", default="mrmsdtw", help="Candidate algorithm to compare")
def analyze(results_path, baseline, candidate):
    """
    Run statistical analysis on benchmark results.
    
    RESULTS_PATH: Path to benchmark results CSV.
    """
    import pandas as pd
    
    from dis_alignment.analysis.statistics import (
        compute_significance,
        failure_analysis,
        memory_analysis,
        runtime_efficiency_analysis,
    )
    
    results = pd.read_csv(results_path)
    
    click.echo("=== Statistical Significance ===")
    sig = compute_significance(results, baseline, candidate, "mae")
    click.echo(f"Comparing {baseline} vs {candidate} (MAE):")
    click.echo(f"  Baseline mean: {sig['baseline_mean']:.4f} s")
    click.echo(f"  Candidate mean: {sig['candidate_mean']:.4f} s")
    click.echo(f"  Improvement: {sig['improvement_pct']:.1f}%")
    click.echo(f"  p-value: {sig['p_value_twosided']:.4f}")
    click.echo(f"  Significant: {sig['is_significant']}")
    click.echo(f"  Effect size: {sig['effect_size']:.3f} ({sig['effect_interpretation']})")
    
    click.echo("\n=== Runtime Efficiency ===")
    runtime = runtime_efficiency_analysis(results)
    for algo, stats in runtime.items():
        click.echo(f"\n{algo}:")
        click.echo(f"  Complexity: {stats['complexity_estimate']}")
        click.echo(f"  Mean runtime: {stats['mean_runtime']:.2f} s")
    
    click.echo("\n=== Failure Analysis ===")
    failures = failure_analysis(results)
    click.echo(f"Failure rate: {failures['failure_rate']:.1%}")
    click.echo(f"Failures by algorithm: {failures['failures_by_algorithm']}")


@main.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--output", "-o", default=None, help="Output path for feature plot")
def extract_features(audio_path, output):
    """
    Extract and visualize features from an audio file.
    
    AUDIO_PATH: Path to audio file (WAV, MP3, etc.)
    """
    import librosa
    
    from dis_alignment.analysis.visualize import plot_feature_comparison
    from dis_alignment.features.chroma import extract_chroma_cqt
    from dis_alignment.features.dlnco import extract_dlnco
    
    click.echo(f"Loading audio from {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=22050)
    
    click.echo("Extracting features...")
    chroma = extract_chroma_cqt(audio, sr=sr)
    dlnco = extract_dlnco(audio, sr=sr)
    
    click.echo(f"Chroma shape: {chroma.shape}")
    click.echo(f"DLNCO shape: {dlnco.shape}")
    
    if output:
        plot_feature_comparison(chroma, dlnco, sr=sr, output_path=output)
        click.echo(f"Feature comparison saved to {output}")
    else:
        import matplotlib.pyplot as plt
        plot_feature_comparison(chroma, dlnco, sr=sr)
        plt.show()


if __name__ == "__main__":
    main()
