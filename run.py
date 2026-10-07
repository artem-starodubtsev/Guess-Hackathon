import argparse
import subprocess
import sys
from common import ROOT, ARTIFACTS

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
                [sys.executable, "-u", str(ROOT / f"{stage}.py")],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise SystemExit(f"{stage} failed. See artifacts/{stage}.log")
    print("Done. Run python app.py to open the search demo.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", choices=STAGES, default="prepare")
    main(**vars(parser.parse_args()))
