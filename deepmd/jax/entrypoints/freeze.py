# SPDX-License-Identifier: LGPL-3.0-or-later
from pathlib import (
    Path,
)
from typing import (
    Any,
)

from deepmd.jax.utils.serialization import (
    deserialize_to_file,
    select_model_branch,
    serialize_from_file,
)


def freeze(
    *,
    checkpoint_folder: str,
    output: str,
    head: str | None = None,
    model_branch: str | None = None,
    hessian: bool = False,
    **kwargs: Any,
) -> None:
    """Freeze the graph in supplied folder.

    Parameters
    ----------
    checkpoint_folder : str
        location of either the folder with checkpoint or the checkpoint prefix
    output : str
        output file name
    hessian : bool, optional
        whether to freeze the hessian, by default False
    **kwargs
        other arguments
    """
    if (Path(checkpoint_folder) / "checkpoint").is_file():
        checkpoint_meta = Path(checkpoint_folder) / "checkpoint"
        checkpoint_folder = checkpoint_meta.read_text().strip()
    if Path(checkpoint_folder).is_dir():
        data = serialize_from_file(checkpoint_folder)
        selected_branch = model_branch or head
        if selected_branch and "model_dict" in data["model_def_script"]:
            data = select_model_branch(data, selected_branch)
        deserialize_to_file(output, data, hessian=hessian)
    else:
        raise FileNotFoundError(f"Checkpoint {checkpoint_folder} does not exist.")
