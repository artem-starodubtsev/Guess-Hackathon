# GUESS Hackathon

A visual fashion search prototype built around a simple idea: people often know what they like when they see it, but catalog categories and keywords do not fully describe that preference. Let users search with a photo or a few words, then decide whether color, pattern or shape matters most.

## How it works

The pipeline turns product images and user requests into comparable **visual signatures**, then searches the GUESS catalog using adjustable priorities.

1. **Extract three attributes.** Each garment gets a palette vector (16 values), a pattern vector (64) and a shape vector (64). Palette extraction captures dominant colors and their proportions. A frozen DINOv2 Small model with a trained projection represents fabric patterns. A CNN silhouette encoder with a metadata-trained projection represents shape and garment type.
2. **Represent the catalog and the request.** The three blocks form a 144-value descriptor stored with each product. An uploaded photo goes through the same image pipeline. A fine-tuned FashionCLIP text encoder maps written requests into these attribute spaces and predicts initial search weights.
3. **Let users control similarity.** The search engine combines attribute distances using the Color, Pattern and Shape sliders. Set a weight to zero to ignore that attribute, or increase it to give it more importance relative to the others. For example, keep the floral pattern while exploring other colors, or prioritize a similar silhouette. Multiple photo views are grouped into product results.
4. **Explore the collection.** An interactive catalog map shows visual neighborhoods for individual attributes or their combinations. It offers another way to discover products and inspect visual families, with potential uses for analytics, merchandising and fashion design teams. Neighbors are ranked in the original attribute spaces, not by their positions on the 2D map.

The prototype also supports finding a matching piece within a chosen category, such as trousers for an uploaded blouse, using palette and pattern similarity. This is visual matching, not a learned outfit-compatibility score. Specific visual descriptions work best; broad occasion-based requests are still limited.

**Future extension:** use prompts, clicks and feedback to learn preferred attribute weights over time. This preference-learning loop is not implemented in the current prototype.

![Architecture overview: palette, pattern and shape extraction, text alignment, and weighted search](docs/images/architecture.png)

## Run

Start Docker Desktop with Linux containers, then launch the prepared image:

```sh
docker run --rm --init -p 127.0.0.1:8767:8767 fashion-atlas
```

Open **http://localhost:8767**. Press Ctrl+C to stop. This command runs on CPU; add `--gpus all` to use an NVIDIA GPU supported by Docker.

Build the image once from the project folder (and rebuild after changes):

```sh
docker build -t fashion-atlas .
```

**Data required:** the GitHub repository contains code, not catalog photos or model weights. Before building, copy the prepared `data/`, `pretrained/`, `artifacts/` and `cache/images.npz` from the local project. No API key is needed. [Docker details](docs/docker.md).

## Files

- `fashion_atlas/` — server, models and search.
- `fashion_atlas/preprocessing/` — image, palette and silhouette processing.
- `fashion_atlas/training/` — preparation, training and evaluation.
- `web/` — interface and cluster maps.
- `docs/` — [development and training](docs/development.md).

Local launch: `python -m fashion_atlas` (after installing `requirements.txt`).

## Disclaimer

The original idea and product concept were developed by our team. Generative AI tools assisted with rapid coding, testing, prototyping and subsequent refinement, including documentation and visuals.
