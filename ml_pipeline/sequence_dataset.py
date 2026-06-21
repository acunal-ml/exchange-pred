"""Windowed sequence Dataset for the LSTM.

docs/03: use a streaming/windowed DataLoader so full tensors aren't
materialized in RAM. Each window is sliced from the shared feature
array on demand in __getitem__ rather than pre-stacked into one big
[n_samples, seq_len, n_features] tensor up front — the array we window
over here is small for a single asset/timeframe, but the access pattern
is what matters for the 4GB-VRAM dev box once batch sizes/seq lengths
grow for heavier runs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def valid_sample_positions(label_end_idx: np.ndarray, seq_len: int) -> np.ndarray:
    """Positions usable as an LSTM sample: must have `seq_len` bars of
    history (i >= seq_len - 1) AND a valid label (label_end_idx >= 0)."""
    n = len(label_end_idx)
    positions = np.arange(n)
    has_history = positions >= seq_len - 1
    has_label = label_end_idx >= 0
    return positions[has_history & has_label]


class SequenceWindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray, sample_positions: np.ndarray, seq_len: int) -> None:
        self.features = features  # [n, n_features], already scaled
        self.labels = labels  # [n], int
        self.sample_positions = sample_positions
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.sample_positions)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i = int(self.sample_positions[idx])
        window = self.features[i - self.seq_len + 1 : i + 1]
        x = torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
        y = torch.tensor(int(self.labels[i]), dtype=torch.long)
        return x, y
