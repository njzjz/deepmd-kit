# SPDX-License-Identifier: LGPL-3.0-or-later
import pickle
from typing import (
    Any,
)


def deepcopy(obj: Any) -> Any:
    """Deep copy an object using pickle.

    Deep copy is a performance killer. Use this function only when necessary.

    Parameters
    ----------
    obj
        The object to be copied.

    Returns
    -------
    object
        The copied object.
    """
    # pickle is faster than copy.deepcopy
    # https://stackoverflow.com/a/29385667/9567349
    return pickle.loads(pickle.dumps(obj))
