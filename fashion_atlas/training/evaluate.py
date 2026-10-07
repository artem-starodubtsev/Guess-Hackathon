import base64
import html
import time

import cv2
import numpy as np
import torch

from fashion_atlas.common import DATA, CACHE, ARTIFACTS, read_json, write_json, record_metrics
from fashion_atlas.catalog import distances
from fashion_atlas.search import TextEncoder, rank
from fashion_atlas.models import ShapeAutoencoder


def image_url(item):
    image = cv2.imread(str(DATA / item["source"]))
    scale = 170 / max(image.shape[:2])
    image = cv2.resize(
        image,
        (round(image.shape[1] * scale), round(image.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode()


@torch.inference_mode()
def shape_preview(items):
    candidates = [item for item in items if item["split"] == "evaluation"]
    selected = np.random.default_rng(42).choice(len(candidates), 6, replace=False)
    chosen = [candidates[i] for i in selected]
    masks = np.load(CACHE / "images.npz")["masks"]
    targets = np.stack([masks[item["source_id"]] for item in chosen])
    model = ShapeAutoencoder().eval()
    model.load_state_dict(
        torch.load(
            ARTIFACTS / "shape_autoencoder.pt", map_location="cpu", weights_only=True
        )
    )
    logits, _ = model(torch.tensor(targets[:, None], dtype=torch.float32))
    predictions = (logits[:, 0].sigmoid().numpy() * 255).astype(np.uint8)
    cards = []
    for item, target, prediction in zip(chosen, targets, predictions):
        combined = np.concatenate([target * 255, prediction], axis=1)
        _, buffer = cv2.imencode(".png", combined)
        url = "data:image/png;base64," + base64.b64encode(buffer).decode()
        cards.append(
            f"<div class='card' style='width:256px'><img style='width:256px;height:128px' src='{url}'><p>{html.escape(item['style'])} · target / reconstruction</p></div>"
        )
    return "<div class='grid'>" + "".join(cards) + "</div>"


def main():
    torch.set_num_threads(4)
    start = time.perf_counter()
    catalog = read_json(ARTIFACTS / "catalog.json")
    items, scales = catalog["items"], catalog["config"]["scales"]
    by_name = {item["name"]: item for item in items}
    benchmark = read_json(DATA / "benchmark.json")
    safe = np.load(CACHE / "images.npz")["evaluation_safe"]
    examples = [
        row
        for row in benchmark["validation"]
        if row["source_name"] in by_name
        and by_name[row["source_name"]]["split"] == "evaluation"
        and safe[by_name[row["source_name"]]["source_id"]]
    ]
    texts = [row["text"] for row in examples] + benchmark["manual"]
    result = {
        "validation_queries": len(examples),
        "excluded_benchmark_queries": len(benchmark["validation"]) - len(examples),
    }
    manual = []
    for tuned in (False, True):
        name = "tuned" if tuned else "frozen"
        encoder = TextEncoder(tuned=tuned)
        vectors, weights = encoder.encode(texts)
        ranks_style, ranks_product = [], []
        for i, row in enumerate(examples):
            ranked = rank(items, distances(vectors[i], items, scales), weights[i])
            source = by_name[row["source_name"]]
            style_rank = next(
                (
                    j + 1
                    for j, r in enumerate(ranked)
                    if items[r["id"]]["style"] == source["style"]
                ),
                len(ranked) + 1,
            )
            product_rank = next(
                (
                    j + 1
                    for j, r in enumerate(ranked)
                    if (items[r["id"]]["style"], items[r["id"]]["color"])
                    == (source["style"], source["color"])
                ),
                len(ranked) + 1,
            )
            ranks_style.append(style_rank)
            ranks_product.append(product_rank)
        ranks_style, ranks_product = np.array(ranks_style), np.array(ranks_product)
        result[name] = {
            "style_r10": float((ranks_style <= 10).mean()),
            "product_r10": float((ranks_product <= 10).mean()),
            "style_mrr": float((1 / ranks_style).mean()),
            "median_style_rank": float(np.median(ranks_style)),
        }
        for offset, text in enumerate(benchmark["manual"]):
            index = len(examples) + offset
            ranked = rank(
                items, distances(vectors[index], items, scales), weights[index]
            )
            manual.append(
                {
                    "model": name,
                    "query": text,
                    "weights": weights[index].tolist(),
                    "top": ranked[:5],
                }
            )
        del encoder
        torch.cuda.empty_cache()
        print("Retrieval", name, result[name], flush=True)
    record_metrics("retrieval", {**result, "seconds": time.perf_counter() - start})
    write_json(ARTIFACTS / "examples.json", manual)
    build_report(items, manual)


def build_report(items, examples):
    metrics = read_json(ARTIFACTS / "metrics.json")
    escape = html.escape
    sections = [
        "<!doctype html><meta charset='utf-8'><title>Fashion Atlas — training report</title><style>body{font:16px system-ui;background:#f5f4ef;color:#202927;max-width:1150px;margin:40px auto;padding:24px}h1{font-size:40px}h2{margin-top:36px}.grid{display:flex;gap:12px;flex-wrap:wrap}.card{background:white;padding:12px;border-radius:12px;width:170px;font-size:12px}.card img{height:170px;width:170px;object-fit:contain}table{border-collapse:collapse;background:white;width:100%}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;background:white;padding:20px;border-radius:12px}.muted{color:#65706b}</style><h1>Fashion Atlas</h1><p>Clean rebuild · RGB image preprocessing · 144D attribute descriptor · local text and image search.</p>"
    ]
    preparation = metrics["preparation"]
    sections.append(
        f"<p>{preparation['images']} input photographs; {metrics['catalog']['images']} searchable images; {metrics['catalog']['products']} products. {preparation['excluded_duplicate_evaluation']} exact duplicates excluded from validation. Data split by STYLE. The original test folder was not used for model selection.</p>"
    )
    sections.append(
        "<h2>Shape · 128 × 128 → 64</h2><p>Reconstruction uses MSE only. A separate metadata-trained linear projection adjusts retrieval distances.</p>"
    )
    shape = metrics["shape"]
    sections.append(
        f"<p>Validation MSE: {shape['reconstruction']['mse']:.5f}; silhouette IoU: {shape['reconstruction']['iou']:.3f}.</p><table><tr><th>Precision@5</th><th>Autoencoder</th><th>Metadata projection</th></tr>"
    )
    for key in shape["baseline_p5"]:
        sections.append(
            f"<tr><td>{key}</td><td>{shape['baseline_p5'][key]:.3f}</td><td>{shape['projected_p5'][key]:.3f}</td></tr>"
        )
    sections.append("</table>" + shape_preview(items))
    sections.append(
        "<h2>Pattern · three fabric regions → DINOv2 Small → 64</h2><p>Cross-crop retrieval checks whether the first and third regions retrieve the same product. It is a consistency proxy, not human-rated pattern relevance. DINO stays frozen; the projection is retrained.</p><table><tr><th>Representation</th><th>Recall@1</th><th>Recall@5</th></tr>"
    )
    for key in ("pca_cross_crop", "learned_cross_crop"):
        values = metrics["pattern"][key]
        sections.append(
            f"<tr><td>{key}</td><td>{values['r1']:.3f}</td><td>{values['r5']:.3f}</td></tr>"
        )
    retrieval = metrics["retrieval"]
    sections.append(
        f"</table><h2>Text retrieval</h2><p>{metrics['text']['queries']:,} accepted training/validation queries. Same {retrieval['validation_queries']} held-out queries for both models. Products are grouped by STYLE + color. Recall measures recovery of the annotated source, not every relevant alternative. These are validation comparisons, not an untouched final test.</p><table><tr><th>Text encoder</th><th>Style Recall@10</th><th>Product Recall@10</th><th>Style MRR</th></tr>"
    )
    for name in ("frozen", "tuned"):
        values = retrieval[name]
        sections.append(
            f"<tr><td>{name}</td><td>{values['style_r10']:.3f}</td><td>{values['product_r10']:.3f}</td><td>{values['style_mrr']:.3f}</td></tr>"
        )
    sections.append(
        "</table><h2>Runtime</h2><table><tr><th>Stage</th><th>Minutes</th></tr>"
    )
    for key in ("preparation", "shape", "pattern", "text", "retrieval"):
        sections.append(
            f"<tr><td>{key}</td><td>{metrics[key]['seconds']/60:.1f}</td></tr>"
        )
    sections.append(
        "</table><h2>Ten manual queries</h2><p>These examples were not added to the training set as manual labels. Weights express relative attribute emphasis, not match confidence. Abstract occasion queries remain a known limitation.</p>"
    )
    thumbnails = {}
    for query in dict.fromkeys(row["query"] for row in examples):
        sections.append(f"<h3>{escape(query)}</h3>")
        for example in (row for row in examples if row["query"] == query):
            weights = " · ".join(
                f"{name} {value:.2f}"
                for name, value in zip(
                    ("Palette", "Pattern", "Shape"), example["weights"]
                )
            )
            sections.append(
                f"<p><b>{example['model']}</b> · {weights}</p><div class='grid'>"
            )
            for match in example["top"]:
                item = items[match["id"]]
                if item["id"] not in thumbnails:
                    thumbnails[item["id"]] = image_url(item)
                sections.append(
                    f"<div class='card'><img src='{thumbnails[item['id']]}'><p>{escape(item['style'])} · {escape(item['color'])}</p><p>{escape(item['category'])} / {escape(item['subcategory'])}</p><p>Distance {match['distance']:.3f}</p></div>"
                )
            sections.append("</div>")
    sections.append(
        "<h2>Limits</h2><p>White product backgrounds are required. Segmentation and fabric cropping can reject photos; failures are listed in metrics.json. Four palette colors can miss small accents. Shape metadata is coarse, pattern positives use product identity, and synthetic text is noisy. A zero slider completely ignores that attribute; all-zero sliders intentionally return no ranking. Original pretrained backbones and accepted annotations are reused; all project-specific trainable models were retrained.</p>"
    )
    (ARTIFACTS / "report.html").write_text("\n".join(sections), encoding="utf-8")


if __name__ == "__main__":
    main()
