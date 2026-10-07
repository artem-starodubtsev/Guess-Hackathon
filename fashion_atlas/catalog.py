import cv2
import numpy as np

from fashion_atlas.common import DATA, CACHE, ARTIFACTS, read_json, write_json, record_metrics


def palette_signature(vector):
    colors = np.asarray(vector, dtype=np.float32).reshape(4, 4)
    lab = cv2.cvtColor(np.clip(colors[:, :3], 0, 1)[None], cv2.COLOR_RGB2LAB)[0]
    mass = np.maximum(colors[:, 3], 0)
    mass /= max(mass.sum(), 1e-8)
    return np.column_stack([mass, lab]).astype(np.float32)


def palette_distance(a, b):
    return cv2.EMD(a, b, cv2.DIST_L2)[0]


def distances(descriptor, items, scales):
    vectors = np.asarray([item["descriptor"] for item in items], np.float64)
    descriptor = np.asarray(descriptor, np.float64)
    signature = palette_signature(descriptor[:16])
    palette = np.array(
        [
            palette_distance(signature, np.array(item["palette"], np.float32))
            for item in items
        ]
    )
    pattern = cosine_distance(vectors[:, 16:80], descriptor[16:80])
    shape = cosine_distance(vectors[:, 80:], descriptor[80:])
    return np.column_stack([palette, pattern, shape]) / np.asarray(scales)


def cosine_distance(vectors, query):
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query)
    return np.clip(1 - vectors @ query / np.maximum(norms, 1e-12), 0, 2)


def main():
    metadata = read_json(DATA / "items.json")
    cache = np.load(CACHE / "images.npz")
    shape = np.load(CACHE / "shape_embeddings.npz")["embeddings"]
    pattern = np.load(CACHE / "pattern_embeddings.npz")["embeddings"]
    valid = cache["valid"].all(1)
    palette = cache["palettes"]
    items = []
    for source_id in np.flatnonzero(valid):
        source = metadata[source_id]
        if source["split"] == "test":
            continue
        vector = np.concatenate(
            [palette[source_id], pattern[source_id], shape[source_id]]
        )
        items.append(
            {
                **source,
                "id": len(items),
                "source_id": int(source_id),
                "descriptor": vector.tolist(),
                "palette": palette_signature(vector[:16]).tolist(),
            }
        )
    train = [item for item in items if item["split"] == "train"]
    rng = np.random.default_rng(42)
    pairs = rng.integers(len(train), size=(32768, 2))
    pair_distances = []
    for a, b in pairs:
        left, right = train[a], train[b]
        if (left["style"], left["color"]) == (right["style"], right["color"]):
            continue
        x, y = np.array(left["descriptor"]), np.array(right["descriptor"])
        pair_distances.append(
            [
                palette_distance(
                    np.array(left["palette"], np.float32),
                    np.array(right["palette"], np.float32),
                ),
                max(0, 1 - x[16:80] @ y[16:80]),
                max(0, 1 - x[80:] @ y[80:]),
            ]
        )
    scales = np.maximum(np.median(pair_distances, axis=0), 1e-6).tolist()
    config = {
        "scales": scales,
        "dimensions": [16, 64, 64],
        "attributes": ["palette", "pattern", "shape"],
        "tie_epsilon": 1e-6,
    }
    write_json(ARTIFACTS / "catalog.json", {"items": items, "config": config})
    record_metrics(
        "catalog",
        {
            "images": len(items),
            "products": len({(i["style"], i["color"]) for i in items}),
            "scales": scales,
        },
    )
    print("Catalog:", len(items), "images", flush=True)


if __name__ == "__main__":
    main()
