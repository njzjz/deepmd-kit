# SPDX-License-Identifier: LGPL-3.0-or-later
import os
import subprocess


def test_on_kaggle():
    subprocess.check_output(
        ["git", "clone", "https://github.com/njzjz/deepmd-kit", "-b", "test_on_kaggle"]
    )
    os.chdir("deepmd-kit")
    subprocess.check_output(["./source/install/test_cc_local.sh"])


if __name__ == "__main__":
    test_on_kaggle()
