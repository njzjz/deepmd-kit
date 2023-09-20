# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess
import sys


def test_on_kaggle():
    subprocess.check_output(
        ["git", "clone", "https://github.com/njzjz/deepmd-kit", "-b", "test_on_kaggle"]
    )
    os.chdir("deepmd-kit")
    subprocess.check_output([sys.executable, "-m", "pip", "install", "-U", "cmake"])
    subprocess.check_output(
        ["bash", "./source/install/test_cc_local.sh"],
        env={
            "DP_VARIANT": "cuda",
            **os.environ,
        },
    )


if __name__ == "__main__":
    test_on_kaggle()
