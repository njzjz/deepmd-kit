# SPDX-License-Identifier: LGPL-3.0-or-later
import functools

from deepmd.utils.copy import deepcopy as copy_lib


def lru_cache(maxsize=16, typed=False, copy=False, deepcopy=False):
    if deepcopy:

        def decorator(f):
            cached_func = functools.lru_cache(maxsize, typed)(f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return copy_lib.deepcopy(cached_func(*args, **kwargs))

            return wrapper

    elif copy:

        def decorator(f):
            cached_func = functools.lru_cache(maxsize, typed)(f)

            @functools.wraps(f)
            def wrapper(*args, **kwargs):
                return copy_lib.copy(cached_func(*args, **kwargs))

            return wrapper

    else:
        decorator = functools.lru_cache(maxsize, typed)
    return decorator
