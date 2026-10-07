import argparse
import subprocess
import sys
from fashion_atlas.common import ROOT, ARTIFACTS

MODULES = {stage: f"fashion_atlas.training.{stage}" for stage in (
    "prepare", "train_shape", "train_pattern", "train_text", "evaluate"
)}
MODULES["catalog"] = "fashion_atlas.catalog"

STAGES = (
    "prepare",
    "train_shape",
    "train_pattern",
    "catalog",
    "train_text",
    "evaluate",
)


def main(start="prepare"):
    ARTIFACTS.mkdir(exist_ok=True)
    for stage in STAGES[STAGES.index(start) :]:
        print(f"\nRunning {stage}...", flush=True)
        with (ARTIFACTS / f"{stage}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(
                [sys.executable, "-u", "-m", MODULES[stage]],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"{stage} failed. See artifacts/{stage}.log")
    print("Done. Run python -m fashion_atlas to open the search demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=STAGES, default="prepare")
    main(**vars(parser.parse_args()))
