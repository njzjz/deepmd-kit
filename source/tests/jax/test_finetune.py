# SPDX-License-Identifier: LGPL-3.0-or-later
import sys
import unittest
from copy import (
    deepcopy,
)
from unittest.mock import (
    Mock,
    patch,
)

import numpy as np

from deepmd.dpmodel.common import (
    to_numpy_array,
)
from deepmd.jax.common import (
    to_jax_array,
)
from deepmd.jax.descriptor.se_e2_a import (
    DescrptSeA,
)
from deepmd.jax.descriptor.dpa3 import (
    DescrptDPA3,
)
from deepmd.jax.entrypoints import train as jax_train_entrypoint
from deepmd.jax.env import (
    jnp,
)
from deepmd.jax.fitting.fitting import (
    EnergyFittingNet,
)
from deepmd.jax.model import (
    EnergyModel,
)
from deepmd.jax.train.trainer import (
    DPTrainer,
)
from deepmd.jax.utils.finetune import (
    get_finetune_rule_single,
    get_finetune_rules,
    merge_finetune_model_data,
)
from deepmd.utils.finetune import (
    FinetuneRuleItem,
)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXFinetuneRules(unittest.TestCase):
    def setUp(self) -> None:
        self.target_model_config = {
            "type": "standard",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3",
                "trainable": False,
                "trainable_ln": False,
                "rcut": 6.0,
            },
            "fitting_net": {
                "type": "ener",
                "trainable": [False, True],
                "neuron": [16],
            },
        }
        self.pretrained_model_config = {
            "type": "standard",
            "type_map": ["O", "H"],
            "descriptor": {
                "type": "dpa3",
                "trainable": True,
                "trainable_ln": True,
                "rcut": 5.0,
            },
            "fitting_net": {
                "type": "ener",
                "trainable": [True, True],
                "neuron": [32],
            },
        }

    def test_single_task_preserves_trainable_flags(self) -> None:
        updated_config, finetune_rule = get_finetune_rule_single(
            self.target_model_config,
            self.pretrained_model_config,
            change_model_params=True,
        )
        self.assertFalse(updated_config["descriptor"]["trainable"])
        self.assertFalse(updated_config["descriptor"]["trainable_ln"])
        self.assertEqual(updated_config["descriptor"]["rcut"], 5.0)
        self.assertEqual(updated_config["fitting_net"]["trainable"], [False, True])
        self.assertEqual(updated_config["fitting_net"]["neuron"], [32])
        self.assertFalse(finetune_rule.get_random_fitting())

    def test_single_task_random_fitting_keeps_target_fitting(self) -> None:
        updated_config, finetune_rule = get_finetune_rule_single(
            self.target_model_config,
            self.pretrained_model_config,
            model_branch_from="RANDOM",
            change_model_params=True,
        )
        self.assertTrue(finetune_rule.get_random_fitting())
        self.assertEqual(updated_config["descriptor"]["rcut"], 5.0)
        self.assertEqual(updated_config["fitting_net"]["neuron"], [16])

    def test_get_finetune_rules_accepts_jax(self) -> None:
        finetune_data = {
            "model_def_script": self.pretrained_model_config,
            "model": {"unused": True},
        }
        with patch(
            "deepmd.jax.utils.finetune.serialize_from_file",
            return_value=finetune_data,
        ):
            updated_config, finetune_links, returned_data = get_finetune_rules(
                "pretrained.jax",
                deepcopy(self.target_model_config),
                change_model_params=True,
            )
        self.assertEqual(updated_config["descriptor"]["rcut"], 5.0)
        self.assertEqual(finetune_links["Default"].get_model_branch(), "Default")
        self.assertIs(returned_data, finetune_data)

    def test_get_finetune_rules_accepts_hlo(self) -> None:
        finetune_data = {
            "model_def_script": self.pretrained_model_config,
            "model": {"unused": True},
        }
        with patch(
            "deepmd.jax.utils.finetune.serialize_from_file",
            return_value=finetune_data,
        ):
            _, finetune_links, _ = get_finetune_rules(
                "pretrained.hlo",
                deepcopy(self.target_model_config),
                change_model_params=False,
            )
        self.assertEqual(finetune_links["Default"].get_model_branch(), "Default")

    def test_get_finetune_rules_rejects_savedmodel(self) -> None:
        with self.assertRaisesRegex(ValueError, ".savedmodel"):
            get_finetune_rules(
                "pretrained.savedmodel",
                deepcopy(self.target_model_config),
            )

    def test_single_task_target_rejects_multitask_target(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "multitask targets"):
            get_finetune_rules(
                "pretrained.jax",
                {"model_dict": {"Default": deepcopy(self.target_model_config)}},
            )

    def test_multitask_source_random_fitting_uses_first_branch(self) -> None:
        updated_config, finetune_rule = get_finetune_rule_single(
            deepcopy(self.target_model_config),
            {"model_dict": {"branch_a": deepcopy(self.pretrained_model_config)}},
            from_multitask=True,
            change_model_params=True,
        )
        self.assertTrue(finetune_rule.get_random_fitting())
        self.assertEqual(finetune_rule.get_model_branch(), "branch_a")
        self.assertEqual(updated_config["descriptor"]["rcut"], 5.0)
        self.assertEqual(updated_config["fitting_net"]["neuron"], [16])

    def test_multitask_source_explicit_random_keyword_uses_first_branch(self) -> None:
        _, finetune_rule = get_finetune_rule_single(
            deepcopy(self.target_model_config),
            {"model_dict": {"branch_a": deepcopy(self.pretrained_model_config)}},
            from_multitask=True,
            model_branch_from="RANDOM",
            change_model_params=True,
        )
        self.assertTrue(finetune_rule.get_random_fitting())
        self.assertEqual(finetune_rule.get_model_branch(), "branch_a")

    def test_get_finetune_rules_from_multitask_source_branch(self) -> None:
        finetune_data = {
            "model_def_script": {
                "model_dict": {
                    "branch_a": deepcopy(self.pretrained_model_config),
                    "branch_b": {
                        **deepcopy(self.pretrained_model_config),
                        "descriptor": {
                            **deepcopy(self.pretrained_model_config["descriptor"]),
                            "rcut": 7.0,
                        },
                    },
                }
            },
            "model": {"unused": True},
        }
        with patch(
            "deepmd.jax.utils.finetune._validate_finetune_source",
            return_value=finetune_data,
        ):
            updated_config, finetune_links, returned_data = get_finetune_rules(
                "pretrained.hlo",
                {
                    **deepcopy(self.target_model_config),
                    "finetune_head": "branch_b",
                },
                change_model_params=True,
            )
        self.assertEqual(updated_config["descriptor"]["rcut"], 7.0)
        self.assertEqual(finetune_links["Default"].get_model_branch(), "branch_b")
        self.assertIs(returned_data, finetune_data)


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXFinetuneMerge(unittest.TestCase):
    def test_merge_replaces_array_leaves_only(self) -> None:
        target = {
            "descriptor": {
                "weights": np.zeros((2, 2)),
                "trainable": False,
            },
            "fitting": {
                "weights": np.ones((2, 2)),
                "trainable": True,
            },
        }
        source = {
            "descriptor": {
                "weights": np.full((2, 2), 2.0),
                "trainable": True,
            },
            "fitting": {
                "weights": np.full((2, 2), 3.0),
                "trainable": False,
            },
        }
        merged = merge_finetune_model_data(
            target,
            source,
            FinetuneRuleItem(["O", "H"], ["O", "H"]),
        )
        np.testing.assert_array_equal(merged["descriptor"]["weights"], source["descriptor"]["weights"])
        np.testing.assert_array_equal(merged["fitting"]["weights"], source["fitting"]["weights"])
        self.assertFalse(merged["descriptor"]["trainable"])
        self.assertTrue(merged["fitting"]["trainable"])

    def test_merge_reshapes_same_size_leaves(self) -> None:
        target = {"descriptor": {"weights": np.zeros((2, 3))}}
        source = {"descriptor": {"weights": np.arange(6).reshape(3, 2)}}
        merged = merge_finetune_model_data(
            target,
            source,
            FinetuneRuleItem(["O", "H"], ["O", "H"]),
        )
        np.testing.assert_array_equal(
            merged["descriptor"]["weights"],
            np.arange(6).reshape(2, 3),
        )

    def test_random_fitting_only_inherits_descriptor(self) -> None:
        target = {
            "descriptor": {"weights": np.zeros((2, 2))},
            "fitting": {"weights": np.ones((2, 2))},
        }
        source = {
            "descriptor": {"weights": np.full((2, 2), 2.0)},
            "fitting": {"weights": np.full((2, 2), 3.0)},
        }
        merged = merge_finetune_model_data(
            target,
            source,
            FinetuneRuleItem(["O", "H"], ["O", "H"], random_fitting=True),
        )
        np.testing.assert_array_equal(merged["descriptor"]["weights"], source["descriptor"]["weights"])
        np.testing.assert_array_equal(merged["fitting"]["weights"], target["fitting"]["weights"])


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXFinetuneTypeMapConsistency(unittest.TestCase):
    def test_change_type_map_consistency(self) -> None:
        descriptor_kwargs = {
            "repflow": {
                "n_dim": 8,
                "e_dim": 4,
                "a_dim": 4,
                "nlayers": 2,
                "e_rcut": 4.0,
                "e_rcut_smth": 2.0,
                "e_sel": 8,
                "a_rcut": 3.0,
                "a_rcut_smth": 1.5,
                "a_sel": 4,
                "axis_neuron": 2,
                "a_compress_rate": 1,
                "a_compress_e_rate": 1,
                "a_compress_use_split": True,
                "update_angle": True,
                "update_style": "res_residual",
                "update_residual": 0.1,
                "update_residual_init": "const",
                "smooth_edge_update": True,
            },
            "activation_function": "tanh",
            "use_tebd_bias": False,
            "precision": "float64",
            "concat_output_tebd": False,
        }
        fitting_kwargs = {
            "neuron": [8, 8],
            "resnet_dt": True,
            "precision": "float64",
            "activation_function": "tanh",
            "seed": 1,
        }
        pretrained_ds = DescrptDPA3(
            ntypes=3,
            type_map=["H", "O", "B"],
            **deepcopy(descriptor_kwargs),
        )
        pretrained_ft = EnergyFittingNet(
            3,
            pretrained_ds.get_dim_out(),
            mixed_types=pretrained_ds.mixed_types(),
            type_map=["H", "O", "B"],
            **deepcopy(fitting_kwargs),
        )
        pretrained_model = EnergyModel(
            pretrained_ds,
            pretrained_ft,
            type_map=["H", "O", "B"],
        )
        target_ds = DescrptDPA3(
            ntypes=3,
            type_map=["O", "H", "B"],
            **deepcopy(descriptor_kwargs),
        )
        target_ft = EnergyFittingNet(
            3,
            target_ds.get_dim_out(),
            mixed_types=target_ds.mixed_types(),
            type_map=["O", "H", "B"],
            **deepcopy(fitting_kwargs),
        )
        target_model = EnergyModel(
            target_ds,
            target_ft,
            type_map=["O", "H", "B"],
        )
        finetune_rule = FinetuneRuleItem(["H", "O", "B"], ["O", "H", "B"])

        pretrained_model.change_type_map(
            target_model.get_type_map(),
            model_with_new_type_stat=target_model.atomic_model,
        )
        changed_pretrained_model = EnergyModel.deserialize(pretrained_model.serialize())
        merged = merge_finetune_model_data(
            target_model.serialize(),
            pretrained_model.serialize(),
            finetune_rule,
        )
        finetuned_model = EnergyModel.deserialize(merged)

        coord = np.array(
            [[[0.0, 0.0, 0.0], [0.0, 1.1, 0.0], [0.9, 0.0, 0.0]]],
            dtype=np.float64,
        ).reshape(1, 9)
        box = (5.0 * np.eye(3)).reshape(1, 9)
        atype_new = np.array([[1, 0, 1]], dtype=np.int64)

        old_ret = changed_pretrained_model.call(
            to_jax_array(coord),
            to_jax_array(atype_new),
            box=to_jax_array(box),
        )
        new_ret = finetuned_model.call(
            to_jax_array(coord),
            to_jax_array(atype_new),
            box=to_jax_array(box),
        )
        np.testing.assert_allclose(
            to_numpy_array(old_ret["energy"]),
            to_numpy_array(new_ret["energy"]),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            to_numpy_array(old_ret["force"]),
            to_numpy_array(new_ret["force"]),
            atol=1e-10,
        )


@unittest.skipIf(
    sys.version_info < (3, 10),
    "JAX requires Python 3.10 or later",
)
class TestJAXFinetuneWiring(unittest.TestCase):
    def _trainer_jdata(self) -> dict:
        return {
            "model": {"type": "standard"},
            "learning_rate": {"start_lr": 1e-3, "decay_steps": 1, "stop_lr": 1e-8},
            "training": {"numb_steps": 1},
        }

    def test_trainer_finetune_does_not_restore_step(self) -> None:
        dummy_model = Mock()
        dummy_model.get_dim_fparam.return_value = 0
        with patch("deepmd.jax.train.trainer.get_model_for_wrapper", return_value=dummy_model), patch(
            "deepmd.jax.train.trainer.EnergyLoss.get_loss",
            return_value=Mock(label_requirement=[]),
        ), patch("deepmd.jax.train.trainer.serialize_from_file") as mock_serialize:
            trainer = DPTrainer(
                self._trainer_jdata(),
                finetune_model="pretrained.jax",
                finetune_links={"Default": FinetuneRuleItem(["H"], ["H"])},
                finetune_model_data={"model": {"dummy": True}},
            )
        self.assertEqual(trainer.start_step, 0)
        mock_serialize.assert_not_called()

    def test_trainer_finetune_selects_multitask_source_branch(self) -> None:
        dummy_model = Mock()
        dummy_model.get_dim_fparam.return_value = 0
        with patch("deepmd.jax.train.trainer.get_model_for_wrapper", return_value=dummy_model), patch(
            "deepmd.jax.train.trainer.EnergyLoss.get_loss",
            return_value=Mock(label_requirement=[]),
        ), patch.object(
            DPTrainer,
            "_apply_single_finetune",
            return_value=dummy_model,
        ) as mock_apply, patch(
            "deepmd.jax.train.trainer.select_model_branch",
            return_value={
                "model_def_script": {"type": "standard"},
                "model": {"branch_value": "selected"},
            },
        ) as mock_select, patch(
            "deepmd.jax.train.trainer._pack_data_for_bias_adjust",
            return_value={"coord": np.zeros((1, 1, 3))},
        ), patch(
            "deepmd.jax.train.trainer.model_change_out_bias",
            return_value=dummy_model,
        ):
            trainer = DPTrainer(
                self._trainer_jdata(),
                finetune_model="pretrained.jax",
                finetune_links={"Default": FinetuneRuleItem(["H"], ["H"], model_branch="branch_b")},
                finetune_model_data={
                    "model_def_script": {
                        "model_dict": {
                            "branch_a": {"type": "standard"},
                            "branch_b": {"type": "standard"},
                        }
                    },
                    "model": {
                        "model_dict": {
                            "branch_a": {"branch_value": "a"},
                            "branch_b": {"branch_value": "b"},
                        }
                    },
                },
            )
            train_data = Mock()
            train_data.get_nsystems.return_value = 1
            trainer._finetune_single(train_data)
        mock_select.assert_called_once()
        self.assertEqual(mock_select.call_args.args[1], "branch_b")
        self.assertEqual(mock_apply.call_args.args[1], {"branch_value": "selected"})

    def test_trainer_finetune_with_new_type_computes_stats(self) -> None:
        dummy_model = Mock()
        dummy_model.get_dim_fparam.return_value = 0
        dummy_model.atomic_model = Mock()
        dummy_model.serialize.return_value = {"descriptor": np.zeros((1,))}
        loss = Mock(label_requirement=[])

        with patch("deepmd.jax.train.trainer.get_model_for_wrapper", return_value=dummy_model), patch(
            "deepmd.jax.train.trainer.EnergyLoss.get_loss",
            return_value=loss,
        ), patch(
            "deepmd.jax.train.trainer.make_stat_input",
            return_value={"type": [[np.array([[0]], dtype=np.int32)]], "coord": [[np.zeros((1, 3))]]},
        ), patch(
            "deepmd.jax.train.trainer.jnp.asarray",
            side_effect=lambda x: x,
        ), patch.object(
            DPTrainer,
            "_apply_single_finetune",
            return_value=dummy_model,
        ), patch(
            "deepmd.jax.train.trainer.jax.make_mesh",
            side_effect=RuntimeError("stop_after_stats"),
        ):
            trainer = DPTrainer(
                self._trainer_jdata(),
                finetune_model="pretrained.jax",
                finetune_links={"Default": FinetuneRuleItem(["H"], ["H", "He"])},
                finetune_model_data={"model": {"dummy": True}},
            )
            train_data = Mock()
            train_data.get_nsystems.return_value = 1
            train_data.mixed_type = False
            train_data.data_systems = [Mock(pbc=False)]
            with self.assertRaisesRegex(RuntimeError, "stop_after_stats"):
                trainer.train(train_data)
        dummy_model.atomic_model.descriptor.compute_input_stats.assert_called_once()
        dummy_model.atomic_model.fitting.compute_output_stats.assert_called_once()

    def test_train_entrypoint_wires_finetune_rule(self) -> None:
        fake_jdata = {
            "model": {"type": "standard", "type_map": ["H"]},
            "learning_rate": {"start_lr": 1e-3, "decay_steps": 1, "stop_lr": 1e-8},
            "training": {
                "numb_steps": 1,
                "training_data": {"systems": []},
            },
        }
        train_data = Mock()
        train_data.type_map = ["H"]
        trainer_instance = Mock()
        trainer_instance.model.get_rcut.return_value = 6.0
        trainer_instance.model.get_type_map.return_value = ["H"]
        trainer_instance.data_requirements = []

        with patch.object(jax_train_entrypoint, "j_loader", return_value=deepcopy(fake_jdata)), patch.object(
            jax_train_entrypoint,
            "get_finetune_rules",
            return_value=(
                deepcopy(fake_jdata["model"]),
                {"Default": FinetuneRuleItem(["H"], ["H"])},
                {"model": {"dummy": True}, "model_def_script": deepcopy(fake_jdata["model"])},
            ),
        ) as mock_rules, patch.object(
            jax_train_entrypoint, "update_deepmd_input", side_effect=lambda x, **kwargs: x
        ), patch.object(
            jax_train_entrypoint, "normalize", side_effect=lambda x, **kwargs: x
        ), patch.object(
            jax_train_entrypoint, "update_sel", side_effect=lambda x, **kwargs: x
        ), patch.object(
            jax_train_entrypoint, "SummaryPrinter", return_value=Mock(__call__=Mock())
        ), patch.object(
            jax_train_entrypoint, "DPTrainer", return_value=trainer_instance
        ) as mock_trainer, patch.object(
            jax_train_entrypoint, "get_data", return_value=train_data
        ), patch.object(
            jax_train_entrypoint.dp_random, "seed"
        ), patch(
            "builtins.open",
            unittest.mock.mock_open(),
        ), patch.object(
            jax_train_entrypoint.json, "dump"
        ):
            jax_train_entrypoint.train(
                INPUT="input.json",
                init_model=None,
                restart=None,
                output="out.json",
                init_frz_model="",
                mpi_log="master",
                log_level=2,
                log_path=None,
                finetune="pretrained.jax",
                use_pretrain_script=False,
            )
        mock_rules.assert_called_once()
        _, rule_kwargs = mock_rules.call_args
        self.assertFalse(rule_kwargs["change_model_params"])
        _, kwargs = mock_trainer.call_args
        self.assertEqual(kwargs["finetune_model"], "pretrained.jax")
        self.assertEqual(kwargs["finetune_links"]["Default"].get_model_branch(), "Default")
        self.assertIn("model", kwargs["finetune_model_data"])
