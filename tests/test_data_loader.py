import h5py
import numpy as np
import pytest
import torch

from data_loader.data_loader import get_batch, load_tokens


def test_load_tokens_reads_hdf5_stream(tmp_path) -> None:
    """Check the HDF5 token dataset loads as a one-dimensional tensor."""
    path = tmp_path / "tokens.h5"
    expected = np.arange(20, dtype=np.int32)

    with h5py.File(path, "w") as data_file:
        data_file.create_dataset("tokens", data=expected)

    loaded = load_tokens(path)

    torch.testing.assert_close(loaded, torch.from_numpy(expected))


def test_get_batch_targets_are_shifted_inputs() -> None:
    """Check each target sequence is the corresponding input shifted by one token."""
    tokens = torch.arange(50)

    inputs, targets = get_batch(
        tokens,
        batch_size=3,
        context_length=5,
        device=torch.device("cpu"),
    )

    assert inputs.shape == (3, 5)
    assert targets.shape == (3, 5)
    torch.testing.assert_close(inputs[:, 1:], targets[:, :-1])


def test_get_batch_rejects_short_stream() -> None:
    """Check a stream too short for one input/target window raises a clear error."""
    with pytest.raises(ValueError, match="longer than context_length"):
        get_batch(
            torch.arange(5),
            batch_size=1,
            context_length=5,
            device=torch.device("cpu"),
        )