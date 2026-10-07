"""Build one reusable cache from the original RGB photos."""

from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import time

import cv2
import numpy as np

from fashion_atlas.common import DATA, CACHE, read_json, record_metrics
from fashion_atlas.preprocessing.segmentation import _garment_mask
from fashion_atlas.preprocessing.shape import extract_shape
from fashion_atlas.preprocessing.pattern import extract_pattern, three_crops
from fashion_atlas.preprocessing.palette import extract_palette


def preprocess_item(item):
    cv2.setNumThreads(1)
    result = {"id": item["id"], "errors": {}}
    image = cv2.imread(str(DATA / item["source"]))
    if image is None:
        result["errors"]["image"] = "Cannot decode image"
        return result
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result["pixel_hash"] = hashlib.sha256(rgb.tobytes()).hexdigest()
    try:
        result["shape"] = cv2.resize(
            extract_shape(rgb), (128, 128), interpolation=cv2.INTER_NEAREST
        )
    except (ValueError, RuntimeError, cv2.error) as error:
        result["errors"]["shape"] = str(error)
    try:
        result["crops"], _, _ = three_crops(rgb, _garment_mask(image))
    except (ValueError, RuntimeError, cv2.error) as error:
        result["errors"]["pattern"] = str(error)
    try:
        result["palette"] = extract_palette(extract_pattern(rgb))["vector"]
    except (ValueError, RuntimeError, cv2.error) as error:
        result["errors"]["palette"] = str(error)
    return result


def main(workers=4, force=False):
    CACHE.mkdir(exist_ok=True)
    target = CACHE / "images.npz"
    if target.exists() and not force:
        print("Image cache already exists; pass --force to rebuild.")
        return
    items = read_json(DATA / "items.json")
    assert [item["id"] for item in items] == list(range(len(items)))
    split_styles = {
        split: {x["style"] for x in items if x["split"] == split}
        for split in ("train", "evaluation", "test")
    }
    if split_styles["train"] & split_styles["evaluation"]:
        raise ValueError("A STYLE occurs in both train and evaluation")
    count = len(items)
    masks = np.zeros((count, 128, 128), np.uint8)
    crops = np.zeros((count, 3, 128, 128), np.uint8)
    palettes = np.zeros((count, 16), np.float32)
    valid = np.zeros((count, 3), bool)
    hashes = np.full(count, "", dtype="U64")
    failures = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for number, result in enumerate(
            pool.map(preprocess_item, items, chunksize=4), 1
        ):
            index = result["id"]
            for column, (name, array) in enumerate(
                (("shape", masks), ("crops", crops), ("palette", palettes))
            ):
                if name in result:
                    array[index] = result[name]
                    valid[index, column] = True
            hashes[index] = result.get("pixel_hash", "")
            if result["errors"]:
                failures.append({"name": items[index]["name"], **result["errors"]})
            if number % 200 == 0:
                print(
                    f"Prepared {number}/{count} in {time.perf_counter()-start:.0f}s",
                    flush=True,
                )
    # Exact duplicate pixels must not cross the training/validation boundary.
    train_hashes = {
        hashes[i]
        for i, item in enumerate(items)
        if item["split"] == "train" and hashes[i]
    }
    evaluation_safe = np.array(
        [
            item["split"] == "train" or hashes[i] not in train_hashes
            for i, item in enumerate(items)
        ]
    )
    np.savez_compressed(
        target,
        masks=masks,
        crops=crops,
        palettes=palettes,
        valid=valid,
        evaluation_safe=evaluation_safe,
    )
    record_metrics(
        "preparation",
        {
            "seconds": time.perf_counter() - start,
            "images": count,
            "shape_valid": int(valid[:, 0].sum()),
            "pattern_valid": int(valid[:, 1].sum()),
            "palette_valid": int(valid[:, 2].sum()),
            "complete": int(valid.all(1).sum()),
            "excluded_duplicate_evaluation": sum(
                not evaluation_safe[i] and item["split"] == "evaluation"
                for i, item in enumerate(items)
            ),
            "failures": failures,
        },
    )
    print("Preprocessing complete", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    main(**vars(parser.parse_args()))
