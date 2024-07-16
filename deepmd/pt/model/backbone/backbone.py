# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    NoReturn,
)

import torch


class BackBone(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        """BackBone base method."""
        super().__init__()

    def forward(self, **kwargs) -> NoReturn:
        """Calculate backBone."""
        raise NotImplementedError
