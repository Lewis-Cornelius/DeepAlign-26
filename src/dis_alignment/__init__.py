"""DeepAlign-26 dissertation toolkit with classical DTW baselines."""

from __future__ import annotations

import importlib

__version__ = "0.1.0"
__author__ = "Lewis"

_EXPORTS = {
    "MazurkaDataset": ("dis_alignment.data", "MazurkaDataset"),
    "SWDDataset": ("dis_alignment.data", "SWDDataset"),
    "align_global_dtw": ("dis_alignment.alignment.baseline_dtw", "align_global_dtw"),
    "align_mrmsdtw": ("dis_alignment.alignment.multiscale_dtw", "align_mrmsdtw"),
    "align_with_deep_features": ("dis_alignment.model.inference", "align_with_deep_features"),
    "evaluate_mazurka_dataset": ("dis_alignment.evaluation", "evaluate_mazurka_dataset"),
    "evaluate_swd_dataset": ("dis_alignment.evaluation", "evaluate_swd_dataset"),
    "compute_ground_truth_alignment": ("dis_alignment.data", "compute_ground_truth_alignment"),
    "extract_chroma_cqt": ("dis_alignment.features.chroma", "extract_chroma_cqt"),
    "extract_deep_features": ("dis_alignment.model.inference", "extract_deep_features"),
    "extract_dlnco": ("dis_alignment.features.dlnco", "extract_dlnco"),
    "load_trained_encoder": ("dis_alignment.model.inference", "load_trained_encoder"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
