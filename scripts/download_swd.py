#!/usr/bin/env python
"""Thin wrapper around the package SWD downloader / verifier."""

from __future__ import annotations

import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@click.command()
@click.argument("output_dir", type=click.Path())
@click.option("--verify-only", is_flag=True, help="Only verify an existing download")
@click.option("--cleanup-zip/--keep-zip", default=False, help="Delete the archive after extraction")
def main(output_dir: str, verify_only: bool, cleanup_zip: bool) -> None:
    from dis_alignment.data import EXPECTED_SIZE_MB, download_swd_dataset, verify_swd_dataset

    output = Path(output_dir)

    if verify_only:
        try:
            _print_summary(verify_swd_dataset(output))
        except FileNotFoundError as exc:
            raise click.ClickException(str(exc)) from exc
        return

    click.echo(f"SWD dataset is approximately {EXPECTED_SIZE_MB}MB")
    click.echo(f"Download location: {output.resolve()}")
    if not click.confirm("Continue with download?", default=True):
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

    click.echo(f"Dataset ready at {extracted_path}")
    _print_summary(verify_swd_dataset(extracted_path))


def _print_summary(summary: dict[str, object]) -> None:
    click.echo("\nSWD verification summary")
    click.echo(f"  root: {summary['root']}")
    click.echo(f"  audio files: {summary['audio_files']}")
    click.echo(f"  annotations: {summary['annotations']}")
    click.echo(f"  lieder: {summary['lieder']}")
    click.echo(f"  pairs: {summary['pairs']}")
    click.echo("  performances:")
    for performance_id, count in summary["performances"].items():
        click.echo(f"    - {performance_id}: {count}")


if __name__ == "__main__":
    main()
