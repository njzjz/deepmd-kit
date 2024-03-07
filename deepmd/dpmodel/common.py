# SPDX-License-Identifier: LGPL-3.0-or-later
from abc import (
    ABC,
    abstractmethod,
)
from typing import (
    Any,
)

import array_api_compat
import numpy as np

from deepmd.env import (
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_NP_FLOAT_PRECISION,
)

PRECISION_DICT = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
    "half": np.float16,
    "single": np.float32,
    "double": np.float64,
    "int32": np.int32,
    "int64": np.int64,
    "default": GLOBAL_NP_FLOAT_PRECISION,
}
RESERVED_PRECISON_DICT = {
    np.float16: "float16",
    np.float32: "float32",
    np.float64: "float64",
    np.int32: "int32",
    np.int64: "int64",
}
DEFAULT_PRECISION = "float64"

Array = Any


class NativeOP(ABC):
    """The unit operation of a native model."""

    @abstractmethod
    def call(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        pass

    def __call__(self, *args, **kwargs):
        """Forward pass in NumPy implementation."""
        return self.call(*args, **kwargs)

    def to(self, sampled_array: Array):
        """Convert the implementation and the device of the
        current model to sampled array like.

        Loop over all attributes of this class. If the attribute
        is a `NativeOP` instance, call its `to` method. If the
        attribute is a NumPy array, convert it to the same
        implementation and device as the sampled array.

        Parameters
        ----------
        sampled_array : Array
            The array to be sampled.
        """
        xp = array_api_compat.array_namespace(sampled_array)
        device = array_api_compat.device(sampled_array)
        for key, value in self.__dict__.items():
            if isinstance(value, NativeOP):
                value.to(sampled_array)
            elif isinstance(value, np.ndarray):
                self.__dict__[key] = array_api_compat.to_device(
                    xp.from_dlpack(value), device
                )


__all__ = [
    "GLOBAL_NP_FLOAT_PRECISION",
    "GLOBAL_ENER_FLOAT_PRECISION",
    "PRECISION_DICT",
    "RESERVED_PRECISON_DICT",
    "DEFAULT_PRECISION",
    "NativeOP",
    "Array",
]
