# SPDX-License-Identifier: LGPL-3.0-or-later
import json
import os
import shutil
import sys
import tempfile
import unittest
from copy import (
    deepcopy,
)
from pathlib import (
    Path,
)
from types import (
    SimpleNamespace,
)
from unittest.mock import (
    Mock,
    patch,
)

import numpy as np

from deepmd.dpmodel.descriptor.repflows import (
    _maybe_apply_jax_placeholder_sharding,
)
from deepmd.jax.entrypoints import freeze as jax_freeze_entrypoint
from deepmd.jax.entrypoints import train as jax_train_entrypoint
from deepmd.jax.model.model import (
    get_model_for_wrapper,
)
from deepmd.jax.model.multitask import (
    ModelWrapper,
)
from deepmd.jax.train.trainer import (
    _compute_multitask_data_stat,
)
from deepmd.jax.utils.multi_task import (
    get_case_embd_config,
    preprocess_shared_params,
)
from deepmd.jax.utils.serialization import (
    select_model_branch,
    serialize_from_file,
)

from ..pt.model.test_permutation import (
    model_dpa3,
    model_se_e2_a,
)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXMultiTaskHelpers(unittest.TestCase):
    def test_preprocess_shared_params_and_case_embd(self) -> None:
        model_config = {
            "shared_dict": {
                "shared_tm": ["O", "H"],
                "shared_desc": {"type": "se_e2_a", "sel": [4, 4], "rcut": 6.0},
                "shared_fit": {"type": "ener", "neuron": [16], "dim_case_embd": 2},
            },
            "model_dict": {
                "task_b": {
                    "type_map": "shared_tm",
                    "descriptor": "shared_desc",
                    "fitting_net": "shared_fit",
                },
                "task_a": {
                    "type_map": "shared_tm",
                    "descriptor": "shared_desc",
                    "fitting_net": "shared_fit",
                },
            },
        }
        updated_model_config, shared_links = preprocess_shared_params(
            deepcopy(model_config)
        )
        self.assertIn("shared_desc", shared_links)
        self.assertIn("shared_fit", shared_links)
        enabled, case_embd_index = get_case_embd_config(updated_model_config)
        self.assertTrue(enabled)
        self.assertEqual(case_embd_index, {"task_a": 0, "task_b": 1})

    def test_partial_shared_level_shares_only_type_embedding(self) -> None:
        model_config = {
            "shared_dict": {
                "shared_tm": ["O", "H"],
                "shared_desc": deepcopy(model_dpa3["descriptor"]),
            },
            "model_dict": {
                "task_a": {
                    "type_map": "shared_tm",
                    "descriptor": "shared_desc",
                    "fitting_net": {
                        "type": "ener",
                        **deepcopy(model_dpa3["fitting_net"]),
                    },
                },
                "task_b": {
                    "type_map": "shared_tm",
                    "descriptor": "shared_desc:1",
                    "fitting_net": {
                        "type": "ener",
                        **deepcopy(model_dpa3["fitting_net"]),
                    },
                },
            },
        }
        updated_model_config, shared_links = preprocess_shared_params(
            deepcopy(model_config)
        )
        model = get_model_for_wrapper(updated_model_config, shared_links=shared_links)
        desc_a = model["task_a"].atomic_model.descriptor
        desc_b = model["task_b"].atomic_model.descriptor
        self.assertIs(desc_a.type_embedding, desc_b.type_embedding)
        self.assertIsNot(desc_a, desc_b)
        self.assertIsNot(desc_a.repflows, desc_b.repflows)

    def test_hybrid_subcomponent_full_sharing_is_supported(self) -> None:
        hybrid_descriptor = {
            "type": "hybrid",
            "list": [
                "shared_desc",
                deepcopy(model_se_e2_a["descriptor"]),
            ],
        }
        model_config = {
            "shared_dict": {
                "shared_tm": ["O", "H", "B"],
                "shared_desc": deepcopy(model_se_e2_a["descriptor"]),
            },
            "model_dict": {
                "task_a": {
                    "type_map": "shared_tm",
                    "descriptor": deepcopy(hybrid_descriptor),
                    "fitting_net": {
                        "type": "ener",
                        **deepcopy(model_se_e2_a["fitting_net"]),
                    },
                },
                "task_b": {
                    "type_map": "shared_tm",
                    "descriptor": deepcopy(hybrid_descriptor),
                    "fitting_net": {
                        "type": "ener",
                        **deepcopy(model_se_e2_a["fitting_net"]),
                    },
                },
            },
        }
        updated_model_config, shared_links = preprocess_shared_params(
            deepcopy(model_config)
        )
        model = get_model_for_wrapper(updated_model_config, shared_links=shared_links)
        desc_a = model["task_a"].atomic_model.descriptor.descrpt_list[0]
        desc_b = model["task_b"].atomic_model.descriptor.descrpt_list[0]
        self.assertIs(desc_a, desc_b)

    def test_multitask_data_stat_merges_shared_components_once(self) -> None:
        class FakeWrapper:
            def __init__(self, model_dict: dict[str, SimpleNamespace]) -> None:
                self._model_dict = model_dict

            def keys(self) -> list[str]:
                return list(self._model_dict)

            def __getitem__(self, key: str) -> SimpleNamespace:
                return self._model_dict[key]

        shared_descriptor = Mock()
        shared_nets = object()
        fitting_a = SimpleNamespace(
            numb_fparam=0,
            numb_aparam=0,
            nets=shared_nets,
            compute_output_stats=Mock(),
        )
        fitting_b = SimpleNamespace(
            numb_fparam=0,
            numb_aparam=0,
            nets=shared_nets,
            compute_output_stats=Mock(),
        )
        model_dict = {
            "task_a": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=fitting_a,
                )
            ),
            "task_b": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=fitting_b,
                )
            ),
        }
        wrapper = FakeWrapper(model_dict)

        train_data = {
            "task_a": SimpleNamespace(mixed_type=False),
            "task_b": SimpleNamespace(mixed_type=False),
        }
        with patch(
            "deepmd.jax.train.trainer._build_single_data_stat",
            side_effect=[
                (
                    [{"coord": "a"}],
                    {
                        "energy": [[[np.array([2.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 1.0, 0.0])]],
                    },
                ),
                (
                    [{"coord": "b"}],
                    {
                        "energy": [[[np.array([4.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 0.0, 1.0])]],
                    },
                ),
            ],
        ):
            _compute_multitask_data_stat(
                wrapper,
                train_data,
                {"task_a": 0.2, "task_b": 0.8},
                {"task_a": 1e-2, "task_b": 1e-2},
            )
        shared_descriptor.compute_input_stats.assert_called_once_with(
            [{"coord": "a"}, {"coord": "b"}]
        )
        fitting_a.compute_output_stats.assert_called_once()
        fitting_b.compute_output_stats.assert_called_once()
        fitting_stat_a = fitting_a.compute_output_stats.call_args.args[0]
        fitting_stat_b = fitting_b.compute_output_stats.call_args.args[0]
        np.testing.assert_allclose(fitting_stat_a["energy"][0][0][0], np.array([2.0]))
        np.testing.assert_allclose(fitting_stat_b["energy"][0][0][0], np.array([4.0]))
        np.testing.assert_allclose(
            fitting_stat_a["natoms_vec"][0][0], np.array([0.0, 0.0, 1.0, 0.0])
        )
        np.testing.assert_allclose(
            fitting_stat_b["natoms_vec"][0][0], np.array([0.0, 0.0, 0.0, 1.0])
        )
        self.assertFalse(fitting_a.compute_output_stats.call_args.kwargs["mixed_type"])
        self.assertFalse(fitting_b.compute_output_stats.call_args.kwargs["mixed_type"])

    def test_multitask_shared_fitting_input_stats_follow_weights_and_protection(
        self,
    ) -> None:
        class FakeWrapper:
            def __init__(self, model_dict: dict[str, SimpleNamespace]) -> None:
                self._model_dict = model_dict

            def keys(self) -> list[str]:
                return list(self._model_dict)

            def __getitem__(self, key: str) -> SimpleNamespace:
                return self._model_dict[key]

        shared_descriptor = Mock()
        shared_fitting = SimpleNamespace(
            numb_fparam=1,
            numb_aparam=1,
            nets=object(),
            fparam_avg=np.zeros(1, dtype=np.float64),
            fparam_inv_std=np.ones(1, dtype=np.float64),
            aparam_avg=np.zeros(1, dtype=np.float64),
            aparam_inv_std=np.ones(1, dtype=np.float64),
            compute_output_stats=Mock(),
        )
        model_dict = {
            "task_a": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=shared_fitting,
                )
            ),
            "task_b": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=shared_fitting,
                )
            ),
        }
        wrapper = FakeWrapper(model_dict)
        train_data = {
            "task_a": SimpleNamespace(mixed_type=False),
            "task_b": SimpleNamespace(mixed_type=False),
        }
        with patch(
            "deepmd.jax.train.trainer._build_single_data_stat",
            side_effect=[
                (
                    [
                        {
                            "coord": "a",
                            "fparam": np.array([[1.0], [3.0]]),
                            "aparam": np.array([[[2.0]], [[4.0]]]),
                        }
                    ],
                    {
                        "energy": [[[np.array([2.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 1.0, 0.0])]],
                    },
                ),
                (
                    [
                        {
                            "coord": "b",
                            "fparam": np.array([[10.0], [14.0]]),
                            "aparam": np.array([[[12.0]], [[16.0]]]),
                        }
                    ],
                    {
                        "energy": [[[np.array([4.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 0.0, 1.0])]],
                    },
                ),
            ],
        ):
            _compute_multitask_data_stat(
                wrapper,
                train_data,
                {"task_a": 0.25, "task_b": 0.75},
                {"task_a": 0.5, "task_b": 0.5},
            )
        np.testing.assert_allclose(shared_fitting.fparam_avg, np.array([9.5]))
        np.testing.assert_allclose(
            shared_fitting.fparam_inv_std,
            np.array([1.0 / np.sqrt(22.0)]),
        )
        np.testing.assert_allclose(shared_fitting.aparam_avg, np.array([11.25]))
        np.testing.assert_allclose(
            shared_fitting.aparam_inv_std,
            np.array([1.0 / np.sqrt(25.9375)]),
        )

    def test_multitask_shared_fitting_requires_same_data_stat_protect(self) -> None:
        class FakeWrapper:
            def __init__(self, model_dict: dict[str, SimpleNamespace]) -> None:
                self._model_dict = model_dict

            def keys(self) -> list[str]:
                return list(self._model_dict)

            def __getitem__(self, key: str) -> SimpleNamespace:
                return self._model_dict[key]

        shared_descriptor = Mock()
        shared_fitting = SimpleNamespace(
            numb_fparam=0,
            numb_aparam=0,
            nets=object(),
            compute_output_stats=Mock(),
        )
        model_dict = {
            "task_a": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=shared_fitting,
                )
            ),
            "task_b": SimpleNamespace(
                atomic_model=SimpleNamespace(
                    descriptor=shared_descriptor,
                    fitting=shared_fitting,
                )
            ),
        }
        wrapper = FakeWrapper(model_dict)
        train_data = {
            "task_a": SimpleNamespace(mixed_type=False),
            "task_b": SimpleNamespace(mixed_type=False),
        }
        with patch(
            "deepmd.jax.train.trainer._build_single_data_stat",
            side_effect=[
                (
                    [{"coord": "a"}],
                    {
                        "energy": [[[np.array([1.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 1.0, 0.0])]],
                    },
                ),
                (
                    [{"coord": "b"}],
                    {
                        "energy": [[[np.array([1.0])]]],
                        "natoms_vec": [[np.array([0.0, 0.0, 1.0, 0.0])]],
                    },
                ),
            ],
        ):
            with self.assertRaisesRegex(ValueError, "data_stat_protect"):
                _compute_multitask_data_stat(
                    wrapper,
                    train_data,
                    {"task_a": 0.5, "task_b": 0.5},
                    {"task_a": 1e-2, "task_b": 1e-1},
                )

    def test_select_model_branch_resolves_alias(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {
                        "type": "standard",
                        "model_branch_alias": ["A"],
                    },
                    "branch_b": {
                        "type": "standard",
                    },
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                    "branch_b": {"value": "b"},
                }
            },
        }
        selected = select_model_branch(deepcopy(data), "A")
        self.assertEqual(selected["model_def_script"]["type"], "standard")
        self.assertEqual(selected["model"]["value"], "a")

    def test_select_model_branch_requires_explicit_head_for_multitask(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {"type": "standard"},
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                }
            },
        }
        with self.assertRaisesRegex(ValueError, "--head/--model-branch"):
            select_model_branch(deepcopy(data), None)

    def test_placeholder_sharding_skips_missing_mesh_context(self) -> None:
        placeholder = object()
        with patch(
            "deepmd.jax.env.jax.lax.with_sharding_constraint",
            side_effect=RuntimeError("requires a non-empty mesh in context"),
        ):
            self.assertIs(
                _maybe_apply_jax_placeholder_sharding(placeholder),
                placeholder,
            )


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXMultiTaskTraining(unittest.TestCase):
    def setUp(self) -> None:
        self.repo_root = Path(__file__).resolve().parents[3]
        self.pt_water_dir = self.repo_root / "source/tests/pt/model/water"
        self.data_dir = self.repo_root / "source/tests/pt/water/data/data_0"
        self.tmpdir = Path(tempfile.mkdtemp(prefix="jax_mt_test_"))
        self.prev_cwd = Path.cwd()
        os.chdir(self.tmpdir)

    def tearDown(self) -> None:
        os.chdir(self.prev_cwd)
        shutil.rmtree(self.tmpdir)

    def _load_template(self, filename: str) -> dict:
        with open(self.pt_water_dir / filename) as fp:
            return json.load(fp)

    def _write_config(self, filename: str, config: dict) -> Path:
        path = self.tmpdir / filename
        with open(path, "w") as fp:
            json.dump(config, fp)
        return path

    def _base_multitask_config(
        self, *, descriptor: dict, sharefit: bool = False
    ) -> dict:
        template_name = "multitask_sharefit.json" if sharefit else "multitask.json"
        config = self._load_template(template_name)
        config["model"]["shared_dict"]["my_descriptor"] = deepcopy(descriptor)
        if sharefit:
            config["model"]["shared_dict"]["my_fitting"]["seed"] = 1
        else:
            for model_key in config["model"]["model_dict"]:
                config["model"]["model_dict"][model_key]["fitting_net"]["seed"] = 1
        for model_key in config["model"]["model_dict"]:
            config["model"]["model_dict"][model_key]["data_stat_nbatch"] = 1
            config["training"]["data_dict"][model_key]["training_data"]["systems"] = [
                str(self.data_dir)
            ]
            config["training"]["data_dict"][model_key]["validation_data"]["systems"] = [
                str(self.data_dir)
            ]
            config["training"]["data_dict"][model_key]["stat_file"] = str(
                self.tmpdir / f"{model_key}.hdf5"
            )
        config["training"]["numb_steps"] = 1
        config["training"]["save_freq"] = 1
        config["training"]["disp_freq"] = 1
        config["training"]["save_ckpt"] = "model.ckpt"
        config["training"]["disp_file"] = "lcurve.out"
        return config

    def _run_entrypoint(self, input_path: Path, *, restart: str | None = None) -> None:
        jax_train_entrypoint.train(
            INPUT=str(input_path),
            init_model=None,
            restart=restart,
            output="out.json",
            init_frz_model="",
            mpi_log="master",
            log_level=2,
            log_path=None,
            skip_neighbor_stat=True,
            finetune=None,
            use_pretrain_script=False,
        )

    def _load_wrapper(self, ckpt_path: Path) -> tuple[dict, ModelWrapper]:
        serialized = serialize_from_file(str(ckpt_path))
        model_def_script = serialized["model_def_script"]
        shared_links = model_def_script.get("shared_links", {})
        _, case_embd_index = get_case_embd_config(model_def_script)
        wrapper = ModelWrapper.deserialize(
            serialized["model"],
            shared_links=shared_links,
            case_embd_index=case_embd_index,
        )
        return serialized, wrapper

    def test_entrypoint_multitask_train_and_restart_se_e2_a(self) -> None:
        config = self._base_multitask_config(
            descriptor=deepcopy(model_se_e2_a["descriptor"])
        )
        input_path = self._write_config("multitask_se_e2_a.json", config)

        self._run_entrypoint(input_path)
        ckpt_path = self.tmpdir / "model.ckpt-1.jax"
        self.assertTrue(ckpt_path.is_dir())

        serialized, wrapper = self._load_wrapper(ckpt_path)
        self.assertEqual(
            set(serialized["model_def_script"]["model_dict"].keys()),
            {"model_1", "model_2"},
        )
        self.assertIn("shared_links", serialized["model_def_script"])
        self.assertIs(
            wrapper["model_1"].atomic_model.descriptor,
            wrapper["model_2"].atomic_model.descriptor,
        )

        self._run_entrypoint(input_path, restart=str(ckpt_path))
        restarted = serialize_from_file(str(ckpt_path))
        self.assertEqual(restarted["@variables"]["current_step"], 1)

    def test_sharefit_case_embd_checkpoint_restores_shared_objects(self) -> None:
        config = self._base_multitask_config(
            descriptor=deepcopy(model_se_e2_a["descriptor"]),
            sharefit=True,
        )
        input_path = self._write_config("multitask_sharefit.json", config)

        self._run_entrypoint(input_path)
        serialized, wrapper = self._load_wrapper(self.tmpdir / "model.ckpt-1.jax")

        self.assertIs(
            wrapper["model_1"].atomic_model.descriptor,
            wrapper["model_2"].atomic_model.descriptor,
        )
        self.assertIsNot(
            wrapper["model_1"].atomic_model.fitting,
            wrapper["model_2"].atomic_model.fitting,
        )
        self.assertIs(
            wrapper["model_1"].atomic_model.fitting.nets,
            wrapper["model_2"].atomic_model.fitting.nets,
        )
        self.assertIs(
            wrapper["model_1"].atomic_model.fitting.fparam_avg,
            wrapper["model_2"].atomic_model.fitting.fparam_avg,
        )
        self.assertIs(
            wrapper["model_1"].atomic_model.fitting.fparam_inv_std,
            wrapper["model_2"].atomic_model.fitting.fparam_inv_std,
        )
        self.assertIsNot(
            wrapper["model_1"].atomic_model.fitting.bias_atom_e,
            wrapper["model_2"].atomic_model.fitting.bias_atom_e,
        )
        self.assertEqual(
            serialized["model_def_script"]["shared_links"]["my_fitting"]["links"][0][
                "shared_type"
            ],
            "fitting_net",
        )

        wrapper.set_case_embd("model_1")
        case_embd = wrapper["model_1"].atomic_model.fitting.serialize()["@variables"][
            "case_embd"
        ]
        np.testing.assert_array_equal(case_embd, np.array([1.0, 0.0]))
        wrapper.set_case_embd("model_2")
        case_embd = wrapper["model_2"].atomic_model.fitting.serialize()["@variables"][
            "case_embd"
        ]
        np.testing.assert_array_equal(case_embd, np.array([0.0, 1.0]))
        case_embd_model_1 = wrapper["model_1"].atomic_model.fitting.serialize()[
            "@variables"
        ]["case_embd"]
        np.testing.assert_array_equal(case_embd_model_1, np.array([1.0, 0.0]))

    def test_entrypoint_multitask_train_dpa3(self) -> None:
        config = self._base_multitask_config(
            descriptor=deepcopy(model_dpa3["descriptor"])
        )
        for model_key in config["model"]["model_dict"]:
            config["model"]["model_dict"][model_key]["fitting_net"] = deepcopy(
                model_dpa3["fitting_net"]
            )
        input_path = self._write_config("multitask_dpa3.json", config)

        self._run_entrypoint(input_path)
        serialized, wrapper = self._load_wrapper(self.tmpdir / "model.ckpt-1.jax")
        self.assertEqual(serialized["@variables"]["current_step"], 1)
        self.assertIs(
            wrapper["model_1"].atomic_model.descriptor,
            wrapper["model_2"].atomic_model.descriptor,
        )

    def test_multitask_checkpoint_rejects_single_task_restart(self) -> None:
        input_path = self._write_config(
            "multitask_reject.json",
            self._base_multitask_config(
                descriptor=deepcopy(model_se_e2_a["descriptor"])
            ),
        )
        self._run_entrypoint(input_path)
        ckpt_path = self.tmpdir / "model.ckpt-1.jax"

        single_task_config = {
            "model": deepcopy(model_se_e2_a),
            "learning_rate": {
                "type": "exp",
                "start_lr": 1e-3,
                "decay_steps": 1,
                "stop_lr": 1e-8,
            },
            "loss": {"type": "ener"},
            "training": {"numb_steps": 1},
        }
        single_task_config["model"]["fitting_net"].setdefault("type", "ener")
        from deepmd.jax.train.trainer import (
            DPTrainer,
        )

        with self.assertRaisesRegex(
            ValueError, "single-task JAX target does not accept a multitask checkpoint"
        ):
            DPTrainer(single_task_config, restart=str(ckpt_path))


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXFreezeMultiTask(unittest.TestCase):
    def test_freeze_jax_selects_branch(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {
                        "type": "standard",
                        "model_branch_alias": ["A"],
                    },
                    "branch_b": {
                        "type": "standard",
                    },
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                    "branch_b": {"value": "b"},
                }
            },
        }
        with (
            patch.object(
                jax_freeze_entrypoint,
                "serialize_from_file",
                return_value=deepcopy(data),
            ),
            patch.object(
                jax_freeze_entrypoint,
                "deserialize_to_file",
            ) as mock_deserialize,
            patch(
                "deepmd.jax.entrypoints.freeze.Path.is_dir",
                return_value=True,
            ),
        ):
            jax_freeze_entrypoint.freeze(
                checkpoint_folder="ckpt_dir",
                output="frozen.jax",
                head="A",
            )
        selected_data = mock_deserialize.call_args.args[1]
        self.assertEqual(selected_data["model"]["value"], "a")
        self.assertNotIn("model_dict", selected_data["model_def_script"])

    def test_freeze_jax_hessian_selects_branch(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {
                        "type": "standard",
                        "model_branch_alias": ["A"],
                    },
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                }
            },
        }
        with (
            patch.object(
                jax_freeze_entrypoint,
                "serialize_from_file",
                return_value=deepcopy(data),
            ),
            patch.object(
                jax_freeze_entrypoint,
                "deserialize_to_file",
            ) as mock_deserialize,
            patch(
                "deepmd.jax.entrypoints.freeze.Path.is_dir",
                return_value=True,
            ),
        ):
            jax_freeze_entrypoint.freeze(
                checkpoint_folder="ckpt_dir",
                output="frozen.jax",
                head="A",
                hessian=True,
            )
        selected_data = mock_deserialize.call_args.args[1]
        self.assertEqual(selected_data["model"]["value"], "a")
        self.assertTrue(mock_deserialize.call_args.kwargs["hessian"])

    def test_freeze_hlo_selects_branch(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {
                        "type": "standard",
                        "model_branch_alias": ["A"],
                    },
                    "branch_b": {
                        "type": "standard",
                    },
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                    "branch_b": {"value": "b"},
                }
            },
        }
        with (
            patch.object(
                jax_freeze_entrypoint,
                "serialize_from_file",
                return_value=deepcopy(data),
            ),
            patch.object(
                jax_freeze_entrypoint,
                "deserialize_to_file",
            ) as mock_deserialize,
            patch(
                "deepmd.jax.entrypoints.freeze.Path.is_dir",
                return_value=True,
            ),
        ):
            jax_freeze_entrypoint.freeze(
                checkpoint_folder="ckpt_dir",
                output="frozen.hlo",
                head="A",
            )
        selected_data = mock_deserialize.call_args.args[1]
        self.assertEqual(selected_data["model"]["value"], "a")
        self.assertNotIn("model_dict", selected_data["model_def_script"])

    def test_freeze_hlo_rejects_multitask_without_head(self) -> None:
        data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": {"type": "standard"},
                }
            },
            "model": {
                "model_dict": {
                    "branch_a": {"value": "a"},
                }
            },
        }
        with (
            patch.object(
                jax_freeze_entrypoint,
                "serialize_from_file",
                return_value=deepcopy(data),
            ),
            patch(
                "deepmd.jax.entrypoints.freeze.Path.is_dir",
                return_value=True,
            ),
        ):
            with self.assertRaisesRegex(ValueError, "--head/--model-branch"):
                jax_freeze_entrypoint.freeze(
                    checkpoint_folder="ckpt_dir",
                    output="frozen.hlo",
                )
