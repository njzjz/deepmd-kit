"""Unittests for argument parser."""

import re
import unittest
from argparse import (
    Namespace,
)
from contextlib import (
    redirect_stderr,
)
from io import (
    StringIO,
)
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Tuple,
    Union,
)

from deepmd.entrypoints.main import (
    get_ll,
    parse_args,
)

if TYPE_CHECKING:
    try:
        from typing import TypedDict  # python==3.8
    except ImportError:
        from typing_extensions import TypedDict  # python<=3.7

    DATA = TypedDict("DATA", {"type": Union[type, Tuple[type]], "value": Any})
    TEST_DICT = Dict[str, DATA]


def build_args(args: "TEST_DICT", command: str) -> List[str]:
    """Build list of arguments similar to one generated by `sys.argv` used by argparse.

    Parameters
    ----------
    args : TEST_DICT
        from dictionary with specifications how to build each argument
    command : str
        first argument that chooses subparser

    Returns
    -------
    List[str]
        arguments with options as list of strings, goal is to emulate `sys.argv`
    """
    args_list = [command]

    for argument, test_data in args.items():
        # arguments without dash are positional, their name should not appear in
        # arguments list
        if argument.startswith("-"):
            args_list.append(argument)
        # arguments without value are passed as such, typically these are where action
        # is 'count' or 'store_true'
        if "value" in test_data:
            args_list += str(test_data["value"]).split()

    return args_list


class TestParserOutput(unittest.TestCase):
    """Test if parser correctly parses supplied arguments."""

    def attr_and_type_check(
        self, namespace: Namespace, mapping: "TEST_DICT", command: str, test_value: bool
    ):
        """Check attributes of `argparse.Manespace` types and values are as expected.

        First check for attribute existence, if it exists check its type and if type is
        as expected check value

        Parameters
        ----------
        namespace : Namespace
            `argparse.Manespace` object aoutput from parser
        mapping : TEST_DICT
            mapping of argument names and their types and values
        command : str
            first argument that sets subparser
        test_value : bool
            whether to test for value match
        """
        mapping = {**{"command": dict(type=str, value=command)}, **mapping}

        for argument, test_data in mapping.items():

            # get expected type
            expected_type = test_data["type"]

            # if data has different destination attribute, use it
            if "dest" in test_data:
                argument = test_data["dest"]

            # remove first one/two hyphens from argument name
            argument = re.sub(r"^-{1,2}", "", argument)

            # remove any hyphens from string as these are replaced to
            # underscores by argparse
            attribute = re.sub("-", "_", argument)

            # first check if namespace object hat the expected attribute
            self.assertTrue(
                hasattr(namespace, attribute),
                msg=f"Namespace object does not have expected attribute: {attribute}",
            )
            # than check if the attribute is of expected type
            self.assertIsInstance(
                getattr(namespace, attribute),
                expected_type,
                msg=f"Namespace attribute '{attribute}' is of wrong type, expected: "
                f"{expected_type}, got: {type(getattr(namespace, attribute))}",
            )
            # if argument has associated value check if it is same as expected
            if "value" in test_data and test_value:
                # use expected value if supplied
                if "expected" in test_data:
                    expected = test_data["expected"]
                else:
                    expected = test_data["value"]
                self.assertEqual(
                    expected,
                    getattr(namespace, attribute),
                    msg=f"Got wrong parsed value, expected: {test_data['value']}, got "
                    f"{getattr(namespace, attribute)}",
                )

    def run_test(self, *, command: str, mapping: "TEST_DICT"):
        """Run test first for specified arguments and then for default.

        Parameters
        ----------
        command : str
            first argument that sets subparser
        mapping : TEST_DICT
            mapping of argument names and their types and values

        Raises
        ------
        SystemExit
            If parser for some reason fails
        NotImplementedError
            [description]
        """
        # test passed in arguments
        cmd_args = build_args(mapping, command)
        buffer = StringIO()
        try:
            with redirect_stderr(buffer):
                namespace = parse_args(cmd_args)
        except SystemExit:
            raise SystemExit(
                f"Encountered expection when parsing arguments ->\n\n"
                f"{buffer.getvalue()}\n"
                f"passed in arguments were: {cmd_args}\n"
                f"built from dict {mapping}"
            )
        self.attr_and_type_check(namespace, mapping, command, test_value=True)

        # check for required arguments
        required = []
        for argument, data in mapping.items():
            if not argument.startswith("-"):
                if isinstance(data["type"], tuple):
                    t = data["type"][0]
                else:
                    t = data["type"]
                if t == str:
                    required.append("STRING")
                elif t in (int, float):
                    required.append("11111")
                else:
                    raise NotImplementedError(
                        f"Option for type: {t} not implemented, please do so!"
                    )

        # test default values
        cmd_args = [command] + required
        buffer = StringIO()
        try:
            with redirect_stderr(buffer):
                namespace = parse_args(cmd_args)
        except SystemExit:
            raise SystemExit(
                f"Encountered expection when parsing DEFAULT arguments ->\n\n"
                f"{buffer.getvalue()}\n"
                f"passed in arguments were: {cmd_args}\n"
                f"built from dict {mapping}"
            )
        self.attr_and_type_check(namespace, mapping, command, test_value=False)

    def test_no_command(self):
        """Test that parser outputs nothing when no command is input and does not fail."""
        self.assertIsNone(parse_args([]).command)

    def test_wrong_command(self):
        """Test that parser fails if no command is passed in."""
        with self.assertRaises(SystemExit):
            parse_args(["RANDOM_WRONG_COMMAND"])

    def test_parser_log(self):
        """Check if logging associated attributes are present in specified parsers."""
        ARGS = {
            "--log-level": dict(type=int, value="INFO", expected=20),
            "--log-path": dict(type=(str, type(None)), value="LOGFILE"),
        }

        for parser in ("config", "transfer", "train", "freeze", "test", "compress"):
            if parser in ("train"):
                args = {**{"INPUT": dict(type=str, value="INFILE")}, **ARGS}
            else:
                args = ARGS

            self.run_test(command=parser, mapping=args)

    def test_parser_mpi(self):
        """Check if mpi-log attribute is present in specified parsers."""
        ARGS = {"--mpi-log": dict(type=str, value="master")}

        for parser in ("train", "compress"):
            if parser in ("train"):
                args = {**{"INPUT": dict(type=str, value="INFILE")}, **ARGS}
            else:
                args = ARGS
            self.run_test(command=parser, mapping=args)

    def test_parser_config(self):
        """Test config subparser."""
        ARGS = {
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="config", mapping=ARGS)

    def test_parser_transfer(self):
        """Test transfer subparser."""
        ARGS = {
            "--raw-model": dict(type=str, value="INFILE.PB"),
            "--old-model": dict(type=str, value="OUTFILE.PB"),
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="transfer", mapping=ARGS)

    def test_parser_train_init_model(self):
        """Test train init-model subparser."""
        ARGS = {
            "INPUT": dict(type=str, value="INFILE"),
            "--init-model": dict(type=(str, type(None)), value="SYSTEM_DIR"),
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="train", mapping=ARGS)

    def test_parser_train_restart(self):
        """Test train restart subparser."""
        ARGS = {
            "INPUT": dict(type=str, value="INFILE"),
            "--restart": dict(type=(str, type(None)), value="RESTART"),
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="train", mapping=ARGS)

    def test_parser_train_init_frz_model(self):
        """Test train init-frz-model subparser."""
        ARGS = {
            "INPUT": dict(type=str, value="INFILE"),
            "--init-frz-model": dict(type=(str, type(None)), value="INIT_FRZ_MODEL"),
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="train", mapping=ARGS)

    def test_parser_train_finetune(self):
        """Test train finetune subparser."""
        ARGS = {
            "INPUT": dict(type=str, value="INFILE"),
            "--finetune": dict(type=(str, type(None)), value="FINETUNE"),
            "--output": dict(type=str, value="OUTPUT"),
        }

        self.run_test(command="train", mapping=ARGS)

    def test_parser_train_wrong_subcommand(self):
        """Test train with multiple subparsers."""
        ARGS = {
            "INPUT": dict(type=str, value="INFILE"),
            "--init-model": dict(type=(str, type(None)), value="SYSTEM_DIR"),
            "--restart": dict(type=(str, type(None)), value="RESTART"),
            "--output": dict(type=str, value="OUTPUT"),
        }
        with self.assertRaises(SystemExit):
            self.run_test(command="train", mapping=ARGS)

    def test_parser_freeze(self):
        """Test freeze subparser."""
        ARGS = {
            "--checkpoint-folder": dict(type=str, value="FOLDER"),
            "--output": dict(type=str, value="FROZEN.PB"),
            "--node-names": dict(type=(str, type(None)), value="NODES"),
        }

        self.run_test(command="freeze", mapping=ARGS)

    def test_parser_test(self):
        """Test test subparser."""
        ARGS = {
            "--model": dict(type=str, value="MODEL.PB"),
            "--system": dict(type=str, value="SYSTEM_DIR"),
            "--set-prefix": dict(type=str, value="SET_PREFIX"),
            "--numb-test": dict(type=int, value=1),
            "--rand-seed": dict(type=(int, type(None)), value=12321),
            "--detail-file": dict(type=(str, type(None)), value="TARGET.FILE"),
            "--atomic": dict(type=bool),
        }

        self.run_test(command="test", mapping=ARGS)

    def test_parser_compress(self):
        """Test compress subparser."""
        ARGS = {
            "--output": dict(type=str, value="OUTFILE"),
            "--extrapolate": dict(type=int, value=5),
            "--step": dict(type=float, value=0.1),
            "--frequency": dict(type=int, value=-1),
            "--checkpoint-folder": dict(type=str, value="."),
        }

        self.run_test(command="compress", mapping=ARGS)

    def test_parser_doc(self):
        """Test doc subparser."""
        ARGS = {
            "--out-type": dict(type=str, value="rst"),
        }

        self.run_test(command="doc-train-input", mapping=ARGS)

    def test_parser_model_devi(self):
        """Test model-devi subparser."""
        ARGS = {
            "--models": dict(
                type=list,
                value="GRAPH.000.pb GRAPH.001.pb",
                expected=["GRAPH.000.pb", "GRAPH.001.pb"],
            ),
            "--system": dict(type=str, value="SYSTEM_DIR"),
            "--set-prefix": dict(type=str, value="SET_PREFIX"),
            "--output": dict(type=str, value="OUTFILE"),
            "--frequency": dict(type=int, value=1),
        }

        self.run_test(command="model-devi", mapping=ARGS)

    def test_get_log_level(self):
        MAPPING = {
            "DEBUG": 10,
            "INFO": 20,
            "WARNING": 30,
            "ERROR": 40,
            "3": 10,
            "2": 20,
            "1": 30,
            "0": 40,
        }

        for input_val, expected_result in MAPPING.items():
            self.assertEqual(
                get_ll(input_val),
                expected_result,
                msg=f"Expected: {expected_result} result for input value: {input_val} "
                f"but got {get_ll(input_val)}",
            )
