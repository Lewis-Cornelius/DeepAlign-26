"""Dataset access helpers for the DeepAlign-26 project."""

from dis_alignment.data.maestro import MAESTRODataset, MAESTROPiece, load_audio
from dis_alignment.data.swd import (
    EXPECTED_SIZE_MB,
    SWD_URL,
    SWDDataset,
    SWDPair,
    SWDPiece,
    compute_ground_truth_alignment,
    download_swd_dataset,
    load_swd_audio,
    verify_swd_dataset,
)

__all__ = [
    "EXPECTED_SIZE_MB",
    "MAESTRODataset",
    "MAESTROPiece",
    "SWDDataset",
    "SWDPair",
    "SWDPiece",
    "SWD_URL",
    "compute_ground_truth_alignment",
    "download_swd_dataset",
    "load_audio",
    "load_swd_audio",
    "verify_swd_dataset",
]
