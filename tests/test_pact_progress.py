import math

import h5py
import numpy as np
import torch

from pact.progress import ProgressModel, demonstration_targets, progress_loss
from pact.segments import correction_weights, select_segment
from pact.train_progress import progress_dataset
from rlpd.env import OBS_KEYS


def test_demonstration_supervision_and_gradient():
    labels, returns = demonstration_targets(4)
    torch.testing.assert_close(labels, torch.tensor([0, 0.25, 0.5, 0.75]))
    torch.testing.assert_close(returns, torch.tensor([0.98**3, 0.98**2, 0.98, 1]))
    prediction = ProgressModel(19)(torch.zeros(4, 19))
    assert prediction.shape == (4,)
    progress_loss(prediction, labels, returns, torch.tensor([False, False, True, True])).backward()


def test_progress_regression_and_recovery():
    assert select_segment([0.5, 0.3, 0.31, 0.32, 0.33, 0.4], 5) == (0, 1)
    assert select_segment([0.5, 0.3, 0.31], 2) == (0, 1)
    assert select_segment([0.1, 0.2, 0.3], 2) is None
    assert select_segment([0.5, 0.3, 0.31], 2, start=1) is None
    assert select_segment([0.5, 0.3, 0.7, 0.4, 0.42], 4) == (2, 3)
    assert correction_weights(2, 3) == (math.exp(-3), 1.0)
    assert correction_weights(3, 3) == (1.0,)


def test_dataset_selects_twenty_successful_demos_and_cuts_at_success(tmp_path):
    path = tmp_path / "demos.hdf5"
    with h5py.File(path, "w") as file:
        for index in range(21):
            demo = file.create_group(f"data/demo_{index}")
            demo["rewards"] = [0, int(index > 0), 0]
            obs = demo.create_group("obs")
            for key in OBS_KEYS:
                obs[key] = np.zeros((3, 1), dtype=np.float32)
    dataset, names = progress_dataset(path, seed=42, terminal_steps=1)
    assert len(names) == 20 and "demo_0" not in names
    assert len(dataset) == 40
    torch.testing.assert_close(dataset.tensors[3].reshape(-1, 2),
                               torch.tensor([[False, True]]).expand(20, 2))
