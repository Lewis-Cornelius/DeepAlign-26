#!/usr/bin/env python
"""Download and prepare the MAESTRO dataset."""

import os
import sys
import urllib.request
import zipfile
from pathlib import Path

import click


MAESTRO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"
EXPECTED_SIZE_GB = 120


@click.command()
@click.argument("output_dir", type=click.Path())
@click.option("--verify-only", is_flag=True, help="Only verify existing download")
def main(output_dir: str, verify_only: bool):
    """
    Download the MAESTRO v3.0.0 dataset.
    
    OUTPUT_DIR: Directory to download and extract the dataset to.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    zip_path = output / "maestro-v3.0.0.zip"
    extracted_path = output / "maestro-v3.0.0"
    
    # Check if already extracted
    if extracted_path.exists():
        json_file = extracted_path / "maestro-v3.0.0.json"
        if json_file.exists():
            click.echo(f"✓ Dataset already extracted at {extracted_path}")
            if verify_only:
                verify_dataset(extracted_path)
            return
    
    if verify_only:
        click.echo("✗ Dataset not found. Run without --verify-only to download.")
        sys.exit(1)
    
    # Check disk space
    click.echo(f"⚠ Note: MAESTRO dataset is approximately {EXPECTED_SIZE_GB}GB")
    click.echo(f"  Download location: {output}")
    
    if not click.confirm("Continue with download?"):
        sys.exit(0)
    
    # Download
    if not zip_path.exists():
        click.echo(f"Downloading from {MAESTRO_URL}...")
        download_with_progress(MAESTRO_URL, zip_path)
    else:
        click.echo(f"Using existing download: {zip_path}")
    
    # Extract
    click.echo("Extracting archive...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output)
    
    click.echo(f"✓ Dataset ready at {extracted_path}")
    
    # Cleanup zip to save space
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
    """Verify dataset integrity."""
    click.echo("Verifying dataset...")
    
    json_file = path / "maestro-v3.0.0.json"
    if not json_file.exists():
        click.echo("✗ Missing metadata file")
        sys.exit(1)
    
    import json
    with open(json_file) as f:
        metadata = json.load(f)
    
    # Count files
    audio_count = 0
    midi_count = 0
    missing = []
    
    for item in metadata:
        audio_path = path / item["audio_filename"]
        midi_path = path / item["midi_filename"]
        
        if audio_path.exists():
            audio_count += 1
        else:
            missing.append(str(audio_path))
        
        if midi_path.exists():
            midi_count += 1
        else:
            missing.append(str(midi_path))
    
    click.echo(f"Audio files: {audio_count}/{len(metadata)}")
    click.echo(f"MIDI files: {midi_count}/{len(metadata)}")
    
    if missing:
        click.echo(f"✗ {len(missing)} missing files")
        for m in missing[:5]:
            click.echo(f"  - {m}")
        if len(missing) > 5:
            click.echo(f"  ... and {len(missing) - 5} more")
    else:
        click.echo("✓ All files present")


if __name__ == "__main__":
    main()
