"""Dataset access helpers for the DeepAlign-26 project."""

from dis_alignment.data.maestro import MAESTRODataset, MAESTROPiece, load_audio
from dis_alignment.data.mazurka import (
    MazurkaDataset,
    MazurkaPair,
    MazurkaPerformance,
    compute_ground_truth_alignment as compute_mazurka_ground_truth_alignment,
    load_mazurka_annotations,
    load_mazurka_audio,
    verify_mazurka_dataset,
)
from dis_alignment.data.swd import (
    EXPECTED_SIZE_MB,
    SWD_URL,
    SWDDataset,
    SWDPair,
    SWDPiece,
    compute_ground_truth_measure_alignment,
    compute_ground_truth_alignment,
    download_swd_dataset,
    load_swd_measure_annotations,
    load_swd_audio,
    verify_swd_dataset,
)

__all__ = [
    "EXPECTED_SIZE_MB",
    "MAESTRODataset",
    "MAESTROPiece",
    "MazurkaDataset",
    "MazurkaPair",
    "MazurkaPerformance",
    "SWDDataset",
    "SWDPair",
    "SWDPiece",
    "SWD_URL",
    "compute_ground_truth_measure_alignment",
    "compute_mazurka_ground_truth_alignment",
    "compute_ground_truth_alignment",
    "download_swd_dataset",
    "load_audio",
    "load_mazurka_annotations",
    "load_mazurka_audio",
    "load_swd_measure_annotations",
    "load_swd_audio",
    "verify_mazurka_dataset",
    "verify_swd_dataset",
]
