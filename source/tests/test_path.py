# run this test with DP_SCRATCH=`pwd` to test cache
import os
import unittest
from pathlib import (
    Path,
)

import h5py
import numpy as np
from common import (
    tests_path,
)

from deepmd.utils.path import (
    DPPath,
)

tested_numpy_path = os.path.join(tests_path, "init_frz_model", "data")
tested_h5_path = os.path.join(tests_path, "test.hdf5")


class TestDPPath(unittest.TestCase):
    def test_os_path(self):
        a = (DPPath(tested_numpy_path) / "set.000" / "coord.npy").load_numpy()
        b = np.load(Path(tested_numpy_path) / "set.000" / "coord.npy")
        np.testing.assert_allclose(a, b)

    def test_h5_path(self):
        a = (DPPath(tested_h5_path) / "set.000" / "coord.npy").load_numpy()
        with h5py.File(tested_h5_path) as f:
            b = f["/set.000/coord.npy"][:]
        np.testing.assert_allclose(a, b)
