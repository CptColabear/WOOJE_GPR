#!/usr/bin/env python

from pathlib import Path
import runpy


TARGET_SCRIPT = Path(__file__).resolve().parent / "wo_cluster" / "groupembedding_wo_cluster.py"


if __name__ == "__main__":
    runpy.run_path(str(TARGET_SCRIPT), run_name="__main__")
