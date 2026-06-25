# SPDX-License-Identifier: LGPL-3.0-or-later
import unittest

import numpy as np


class TestModelStatMixedSystems(unittest.TestCase):
    class _FakeSystem:
        def __init__(self, natoms: int, atype: list[int]) -> None:
            self.natoms = natoms
            self.atype = np.asarray(atype, dtype=np.int64)
            self.calls = 0

        def get_batch(self, batch_size: int) -> dict:
            self.calls += 1
            coord = np.full(
                (batch_size, self.natoms * 3),
                float(self.natoms),
                dtype=np.float64,
            )
            return {
                "type": np.tile(self.atype.reshape(1, -1), (batch_size, 1)),
                "coord": coord,
                "energy": np.full((batch_size, 1), float(self.natoms)),
                "force": -coord,
                "find_energy": np.float32(1.0),
                "find_force": np.float32(1.0),
            }

    class _FakeMixedData:
        mixed_systems = True

        def __init__(self) -> None:
            self.data_systems = [
                TestModelStatMixedSystems._FakeSystem(4, [0, 1, 1, 1]),
                TestModelStatMixedSystems._FakeSystem(7, [0, 0, 1, 1, 1, 1, 1]),
            ]
            self.batch_size = np.asarray([2, 2], dtype=np.int64)
            self.natoms_vec = [
                np.asarray([4, 4, 1, 3], dtype=np.int32),
                np.asarray([7, 7, 2, 5], dtype=np.int32),
            ]
            self.default_mesh = [
                np.zeros(6, dtype=np.int32),
                np.zeros(6, dtype=np.int32),
            ]
            self.fallback_get_batch_calls = 0

        def get_nsystems(self) -> int:
            return len(self.data_systems)

        def get_batch(self, sys_idx=None) -> dict:
            self.fallback_get_batch_calls += 1
            raise AssertionError("mixed-system stat collection must not use get_batch")

    def test_make_stat_input_keeps_mixed_systems_separate(self) -> None:
        from deepmd.utils.model_stat import (
            make_stat_input,
        )

        data = self._FakeMixedData()
        all_stat = make_stat_input(data, nbatches=2, merge_sys=False)

        self.assertEqual(data.fallback_get_batch_calls, 0)
        self.assertEqual([sys.calls for sys in data.data_systems], [2, 2])
        self.assertEqual(len(all_stat["coord"]), 2)
        self.assertEqual([batch.shape for batch in all_stat["coord"][0]], [(2, 12)] * 2)
        self.assertEqual([batch.shape for batch in all_stat["coord"][1]], [(2, 21)] * 2)
        self.assertEqual([batch.shape for batch in all_stat["force"][0]], [(2, 12)] * 2)
        self.assertEqual([batch.shape for batch in all_stat["force"][1]], [(2, 21)] * 2)
        self.assertEqual(
            [batch.shape for batch in all_stat["real_natoms_vec"][0]],
            [(2, 4)] * 2,
        )
        self.assertEqual(
            [batch.shape for batch in all_stat["real_natoms_vec"][1]],
            [(2, 4)] * 2,
        )
        np.testing.assert_array_equal(
            all_stat["real_natoms_vec"][0][0],
            np.tile(np.asarray([4, 4, 1, 3], dtype=np.int32), (2, 1)),
        )
        np.testing.assert_array_equal(
            all_stat["real_natoms_vec"][1][0],
            np.tile(np.asarray([7, 7, 2, 5], dtype=np.int32), (2, 1)),
        )
        self.assertFalse(np.any(all_stat["type"][0][0] < 0))
        self.assertFalse(np.any(all_stat["type"][1][0] < 0))


if __name__ == "__main__":
    unittest.main()
