#!/usr/bin/env python
"""Download and prepare the Schubert Winterreise Dataset (SWD).

Downloads from Zenodo: https://zenodo.org/records/3968389
The free version includes 2 of 9 performances (AL98 + SC06).
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import click


SWD_URL = "https://zenodo.org/records/3968389/files/Schubert_Winterreise_Dataset_v1-0.zip"
EXPECTED_SIZE_MB = 506


@click.command()
@click.argument("output_dir", type=click.Path())
@click.option("--verify-only", is_flag=True, help="Only verify existing download")
def main(output_dir: str, verify_only: bool):
    """
    Download the Schubert Winterreise Dataset (SWD) v1.0.

    OUTPUT_DIR: Directory to download and extract the dataset to.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    zip_path = output / "Schubert_Winterreise_Dataset_v1-0.zip"
    # The zip extracts into a subdirectory
    extracted_path = output / "Schubert_Winterreise_Dataset_v1-0"

    # Check if already extracted
    if extracted_path.exists():
        raw_data = extracted_path / "01_RawData"
        if raw_data.exists():
            click.echo(f"✓ Dataset already extracted at {extracted_path}")
            if verify_only:
                verify_dataset(extracted_path)
            return

    if verify_only:
        click.echo("✗ Dataset not found. Run without --verify-only to download.")
        sys.exit(1)

    click.echo(f"SWD dataset is approximately {EXPECTED_SIZE_MB}MB")
    click.echo(f"Download location: {output}")

    if not click.confirm("Continue with download?"):
        sys.exit(0)

    # Download
    if not zip_path.exists():
        click.echo(f"Downloading from Zenodo...")
        download_with_progress(SWD_URL, zip_path)
    else:
        click.echo(f"Using existing download: {zip_path}")

    # Extract
    click.echo("Extracting archive...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(output)

    click.echo(f"✓ Dataset ready at {extracted_path}")

    # Verify
    verify_dataset(extracted_path)

    # Cleanup
    if click.confirm("Delete zip file to save space?", default=True):
        zip_path.unlink()
        click.echo("Zip file deleted.")


def download_with_progress(url: str, dest: Path):
    """Download file with progress bar."""
    with click.progressbar(length=100, label="Downloading") as bar:
        last_progress = [0]

        def reporthook(count, block_size, total_size):
            if total_size > 0:
                progress = int(count * block_size * 100 / total_size)
                increment = progress - last_progress[0]
                if increment > 0:
                    bar.update(increment)
                    last_progress[0] = progress

        urllib.request.urlretrieve(url, dest, reporthook)


def verify_dataset(path: Path):
    """Verify SWD dataset structure and count available files."""
    click.echo("\nVerifying dataset...")

    audio_dir = path / "01_RawData" / "audio_wav"
    ann_dir = path / "02_Annotations" / "ann_audio_measure"

    if not audio_dir.exists():
        click.echo("✗ Missing audio directory: 01_RawData/audio_wav")
        return

    # Count audio files by performance
    wav_files = list(audio_dir.glob("*.wav"))
    performances: dict[str, int] = {}
    for f in wav_files:
        # Parse performance ID from filename
        name = f.stem
        parts = name.split("_")
        if len(parts) >= 3:
            perf_id = parts[-1]
            performances[perf_id] = performances.get(perf_id, 0) + 1

    click.echo(f"\nAudio files: {len(wav_files)} total")
    for perf, count in sorted(performances.items()):
        free_tag = " (free)" if perf in ("AL98", "SC06") else ""
        click.echo(f"  {perf}: {count} lieder{free_tag}")

    # Count annotations
    if ann_dir.exists():
        ann_files = list(ann_dir.glob("*.csv"))
        click.echo(f"\nMeasure annotations: {len(ann_files)} files")
    else:
        click.echo("\n✗ Missing annotation directory: 02_Annotations/ann_audio_measure")

    # Count available pairs
    pair_count = 0
    for lied in range(1, 25):
        lied_perf = set()
        for f in wav_files:
            if f"D911-{lied:02d}" in f.name:
                parts = f.stem.split("_")
                if parts:
                    lied_perf.add(parts[-1])
        n = len(lied_perf)
        pair_count += n * (n - 1) // 2

    click.echo(f"\nAvailable alignment pairs: {pair_count}")
    click.echo("✓ Dataset verification complete")


if __name__ == "__main__":
    main()
