# Fashion Atlas

A local fashion search demo with three independent attributes: palette, pattern and shape. Search with an English sentence or pick a catalog photo, then change how much each attribute matters.

The descriptor has 144 values: **16 palette + 64 pattern + 64 shape**. The website keeps the blocks separate when measuring similarity. A zero slider ignores that attribute; the sliders are relative weights, not confidence scores. Results group different photo views of the same product.

## Upload and find a matching piece

Upload one garment photo (JPEG, PNG or WebP, up to 12 MB / 20 megapixels). The local backend corrects EXIF orientation, places transparency on white, and resizes large images to at most 1024 pixels per side before applying the existing RGB encoders. Photos and descriptors are held in memory for the request; uploads are not saved or added to the catalog. A plain white background and a fully visible garment work best. Complex scenes or insufficient fabric regions return a readable error and preserve the previous results.

Choose **Find a matching piece** and select a target category, for example PANTS for a reference T-shirt. Category is a strict catalog metadata filter. Palette and pattern start at 1 each; shape stays disabled at 0 because a T-shirt silhouette should not penalize trousers. Change the two active sliders or the category without re-encoding the photo. Result cards open a preview without replacing your reference. Catalog photos can also be used as references in this mode.

**Find similar clothes** retains the existing image/text search with all three adjustable attributes. Matching means similar colors and prints; it is not a trained outfit-compatibility score. `POST /api/image-search` accepts raw image bytes and returns the descriptor, all three per-item distances, image/crop/mask previews and processing time. Category filtering and weight changes run in the browser.

## Run

The existing `.venv` and local weights are ready on this machine:

```powershell
.\.venv\Scripts\python.exe app.py
```

Open http://127.0.0.1:8767. The page lets you compare frozen FashionCLIP with the version fine-tuned on its last two transformer blocks. Both have separately trained attribute heads.

For a new environment, use Python 3.12 and install `requirements.txt`. This setup uses PyTorch with CUDA 12.8 on an RTX 5090; training requires CUDA. Inference can also run on CPU. The photographs, annotations and pretrained weights must be supplied separately; `.gitignore` keeps them out of the code repository.

## Docker

See [DOCKER.md](DOCKER.md) for a self-contained image and automatic GPU/CPU launcher. The container includes the current models and catalog. Inference defaults to `FASHION_DEVICE=auto`; use `python app.py --device cpu` to force CPU outside Docker. `/api/health` reports the active device.

## Retrain

```powershell
.\.venv\Scripts\python.exe run.py
```

Stages run sequentially and write one log each in `artifacts/`. Resume at a stage with `python run.py --start train_text`. A full run initializes the project-specific models from scratch. Cached image preprocessing and frozen DINO features are reused when present. To regenerate those too, remove `cache/images.npz` and `cache/pattern_features.npz` before running. Keep `data/` and `pretrained/`.

`evaluate.py` builds `artifacts/report.html`, with validation metrics and ten visual search examples. `artifacts/metrics.json` contains the exact results, elapsed times and rejected-image reasons. The website links to the report.

## What runs where

| File | Role |
| --- | --- |
| `segmentation.py`, `shape.py` | Foreground mask and centered silhouette |
| `pattern.py`, `palette.py` | Fabric crops and adaptive 1–4 color palette |
| `prepare.py` | Build the image cache from original photos |
| `models.py`, `training.py` | Networks, feature extraction and training helpers |
| `train_shape.py` | CNN autoencoder and metadata-based shape projection |
| `train_pattern.py` | DINO features and contrastive pattern projection |
| `catalog.py` | Build descriptors and calibrate attribute distances |
| `train_text.py` | Frozen text heads, then last-two-block fine-tuning |
| `search.py` | Image/text encoders and grouped ranking |
| `app.py`, `web/` | Local HTTP server and plain HTML/CSS/JavaScript UI |
| `evaluate.py`, `run.py`, `common.py` | Report, stage runner and paths |

There is no frontend build step, service account, separate configuration layer or required API connection.

## Data and models

`data/items.json` is the metadata manifest. Each row contains a contiguous `id`, image `name`, relative `source`, `style`, `color`, `category`, `subcategory`, `construction`, and `split` (`train`, `evaluation` or `test`). Photos live under `data/trainval/` and `data/test/`. Existing Excel and CSV source metadata are retained in their original folders.

`data/queries.jsonl` contains accepted synthetic annotations: one image name and its query list per line. Queries include `text`, `kind`, per-attribute `weights`, `attribute_mask`, `evidence` and `warnings`. Queries with warnings, absent evidence or no supervised attribute are excluded. API responses and credentials are not required. This rebuild reuses the accepted annotations rather than buying them again.

`data/benchmark.json` fixes 300 source-associated validation queries and ten manual diagnostic queries. Unavailable or duplicate validation sources are excluded explicitly. Train/evaluation separation is by STYLE; exact pixel duplicates of training photos are also excluded from validation. Checkpoints are selected on validation, so these numbers are development metrics. The original test folder is not used for model selection.

`pretrained/` holds the original DINOv2 Small weights, its upstream runtime and FashionCLIP's text-only backbone/tokenizer. DINO's upstream license stays with its runtime. These external pretrained models are reused; they are not trained from zero. The project-specific checkpoints in `artifacts/` are trained afresh.

## Pipeline

- **Palette:** find a square with at least 90% fabric, fill the small background remainder by template matching, then choose K=1…4 with a Lab reconstruction threshold. Each palette entry is `[R, G, B, proportion]`, with values in 0…1. Compare palettes by Earth Mover's Distance in Lab, independently of color ordering.
- **Pattern:** take three separated, almost fully covered fabric regions, resize to 128×128 and convert to grayscale. Frozen DINOv2 Small supplies 384D mean patch features. A trained projection maps them to 64D; normalize and average across crops. Product identity supplies contrastive positives. Shared color codes are excluded as ambiguous negatives, not assumed to be proven pattern labels.
- **Shape:** fit a binary garment mask into a padded square at 128×128. A CNN autoencoder learns a 64D bottleneck using MSE reconstruction only. A separate linear projection uses category/subcategory labels to improve the shape neighborhood. The decoder is only needed for training and reconstruction inspection.
- **Text:** FashionCLIP outputs 512D features. Small heads predict the three blocks and attribute emphasis. First train the heads with a frozen backbone; then unfreeze the last two blocks, final layer norm and text projection. Palette supervision is permutation-invariant; the pattern and shape heads use cosine alignment.

Distances are divided by fixed median distances estimated from training-only pairs, then averaged using the selected weights. Product grouping retains the best matching photo view.

## Use from Python

```python
from search import ImageEncoder, TextEncoder

image_encoder = ImageEncoder()
descriptor = image_encoder.encode(rgb_image)  # uint8 H×W×3 -> float32 (144,)

text_encoder = TextEncoder(tuned=True)
descriptors, weights = text_encoder.encode(["blue and white striped shirt"])
```

`extract_shape`, `extract_pattern` and `extract_palette` also accept RGB NumPy arrays directly and perform no file writes. `extract_pattern(image, output_size=(256, 256))` uses `(width, height)` ordering. The three grayscale DINO crops are separate from this color-preserving palette preprocessing.

## Limits

The demo expects clothing on a mostly white background. Cropping can fail on narrow or occluded items; rejected photos are recorded rather than silently replaced. Four colors can miss small accents. Metadata is coarse and the text labels are synthetic. Specific visual descriptions work better than broad occasions such as business meetings or nights out. Retrieval of an annotated source is a useful diagnostic, not a complete human relevance score.

## Cluster maps

Open the **Clusters** tab in the header. Seven precomputed UMAP maps cover every non-empty combination of palette, pattern and shape. Hover over a point to inspect its photo and eight nearest products, or click to pin it. Scroll to zoom and drag to pan. Category colors are metadata labels, not computed cluster assignments.

Neighbors are ranked by the original normalized attribute distances with equal weights for the enabled attributes, never by 2D coordinates. Other views of the query product are excluded, and neighbors are grouped by style + color. Switching maps preserves the selected item and changes both the projection and its neighbor list.

The static map data ships in the Docker image; UMAP is not required for serving. After rebuilding the catalog, regenerate the maps with `python -m pip install umap-learn==0.5.9.post2` and `python build_clusters.py`, then rebuild Docker. The map includes every catalog photo view.
