# SPDX-License-Identifier: LGPL-3.0-or-later
import copy
import json
import unittest
from typing import (
    List,
    Optional,
)

import numpy as np
import torch

from deepmd.pt.model.model import (
    get_model,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.dataloader import (
    DpLoaderSet,
)
from deepmd.pt.utils.dataset import (
    DeepmdDataSystem,
)
from deepmd.pt.utils.stat import (
    make_stat_input,
)


class CheckSymmetry(DeepmdDataSystem):
    def __init__(
        self,
        sys_path: str,
        rcut,
        sec,
        type_map: Optional[List[str]] = None,
        type_split=True,
    ):
        super().__init__(sys_path, rcut, sec, type_map, type_split)

    def get_disturb(self, index, atom_index, axis_index, delta):
        for i in range(
            0, len(self._dirs) + 1
        ):  # note: if different sets can be merged, prefix sum is unused to calculate
            if index < self.prefix_sum[i]:
                break
        frames = self._load_set(self._dirs[i - 1])
        tmp = copy.deepcopy(frames["coord"].reshape(self.nframes, -1, 3))
        tmp[:, atom_index, axis_index] += delta
        frames["coord"] = tmp
        frame = self.single_preprocess(frames, index - self.prefix_sum[i - 1])
        return frame


def get_data(batch):
    inputs = {}
    for key in ["coord", "atype", "box"]:
        inputs[key] = batch[key].unsqueeze(0).to(env.DEVICE)
    return inputs


class TestForceGrad(unittest.TestCase):
    def setUp(self):
        with open(env.TEST_CONFIG) as fin:
            self.config = json.load(fin)
        self.system_index = 0
        self.batch_index = 0
        self.get_dataset(self.system_index, self.batch_index)
        self.get_model()

    def get_model(self):
        training_systems = self.config["training"]["training_data"]["systems"]
        model_params = self.config["model"]
        data_stat_nbatch = model_params.get("data_stat_nbatch", 10)
        train_data = DpLoaderSet(
            training_systems,
            self.config["training"]["training_data"]["batch_size"],
            model_params,
        )
        sampled = make_stat_input(
            train_data.systems, train_data.dataloaders, data_stat_nbatch
        )
        self.model = get_model(self.config["model"], sampled).to(env.DEVICE)

    def get_dataset(self, system_index=0, batch_index=0):
        systems = self.config["training"]["training_data"]["systems"]
        rcut = self.config["model"]["descriptor"]["rcut"]
        sel = self.config["model"]["descriptor"]["sel"]
        sec = torch.cumsum(torch.tensor(sel), dim=0)
        type_map = self.config["model"]["type_map"]
        self.dpdatasystem = CheckSymmetry(
            sys_path=systems[system_index], rcut=rcut, sec=sec, type_map=type_map
        )
        self.origin_batch = self.dpdatasystem._get_item(batch_index)

    def test_force_grad(self, threshold=1e-3, delta0=1e-6, seed=20):
        result0 = self.model(**get_data(self.origin_batch))
        np.random.default_rng(seed)
        errors = np.zeros((self.dpdatasystem._natoms, 3))
        for atom_index in range(self.dpdatasystem._natoms):
            for axis_index in range(3):
                delta = np.random.random() * delta0
                disturb_batch = self.dpdatasystem.get_disturb(
                    self.batch_index, atom_index, axis_index, delta
                )
                disturb_result = self.model(**get_data(disturb_batch))
                disturb_force = -(disturb_result["energy"] - result0["energy"]) / delta
                disturb_error = (
                    result0["force"][0, atom_index, axis_index] - disturb_force
                )
                errors[atom_index, axis_index] = disturb_error.detach().cpu().numpy()
        self.assertTrue(np.abs(errors).max() < threshold)


if __name__ == "__main__":
    unittest.main()