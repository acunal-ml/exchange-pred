import numpy as np

from ml_pipeline.sequence_dataset import SequenceWindowDataset, valid_sample_positions


def test_valid_sample_positions_requires_history_and_label():
    label_end_idx = np.array([-1, -1, 5, 6, -1, 8])
    positions = valid_sample_positions(label_end_idx, seq_len=3)
    # position 0,1 lack a label; position 2 lacks full history (needs i>=2, ok actually)
    # i>=seq_len-1=2 required; label_end_idx>=0 required.
    assert set(positions.tolist()) == {2, 3, 5}


def test_dataset_window_shape_and_label():
    n, n_features, seq_len = 10, 4, 3
    features = np.arange(n * n_features, dtype=np.float32).reshape(n, n_features)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    positions = np.array([2, 5, 9])

    ds = SequenceWindowDataset(features, labels, positions, seq_len)
    assert len(ds) == 3

    x, y = ds[1]  # position 5
    assert x.shape == (seq_len, n_features)
    assert np.allclose(x.numpy(), features[3:6])
    assert y.item() == labels[5]


def test_dataset_window_is_sliced_not_precomputed():
    # Regression-style guard: __getitem__ slices from the shared array
    # rather than the constructor building a stacked [n, seq_len, feat]
    # tensor up front (docs/03's "streaming/windowed DataLoader" rule).
    n, n_features, seq_len = 5, 2, 2
    features = np.zeros((n, n_features), dtype=np.float32)
    ds = SequenceWindowDataset(features, np.zeros(n, dtype=int), np.array([1, 2, 3]), seq_len)
    assert ds.features is features  # no copy/stack at construction time
