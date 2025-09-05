# PixelBliss

Stateless CLI that generates a single high‑quality wallpaper per run, ranks candidates, saves variants, updates a manifest, and optionally posts to X/Twitter with alt text.

## Quick start

1. Install deps: `pip install -r requirements.txt`
2. Copy env and fill keys: `cp .env.example .env`
3. Optionally edit `config.yaml`
4. Run once: `python main.py post-once` (use `dry-run` to skip posting)

## Commands

- `python main.py post-once` — Full pipeline and post
- `python main.py dry-run` — Full pipeline without posting
- `python main.py repair-manifest` — Placeholder (not implemented)

## How it works (high level)

- Prompt generation: OpenAI (model `gpt-5`) by default; pluggable (`openai` or `dummy`)
- Image generation: Tries FAL first, Replicate as fallback (configurable models, retries)
- Scoring & selection: brightness, entropy, local quality floors; aesthetic score; weighted ranking; duplicate filtering via pHash
- Upscale & variants: Optional upscale via FAL/Replicate (or local dummy); creates wallpaper sizes from `config.yaml`
- Posting: Uploads image, sets alt text, tweets with empty text
- Storage & manifest: Saves under `outputs/YYYY-MM-DD/<slug>/`; writes `meta.json`; updates `manifest/index.json`
- Alerts: Optional webhook on success/failure

## Configuration

See `config.yaml`. Key sections: `prompt_generation`, `image_generation`, `ranking`, `aesthetic_scoring`, `local_quality`, `upscale`, `wallpaper_variants`, `alerts`, `timezone`, `categories`, `art_styles`, `category_selection_method` (`random` or `time`) and `rotation_minutes`.

## Environment variables

```
FAL_API_KEY
REPLICATE_API_TOKEN
OPENAI_API_KEY
X_API_KEY
X_API_SECRET
X_ACCESS_TOKEN
X_ACCESS_TOKEN_SECRET
ALERT_WEBHOOK_URL
```

## Outputs

- Images: `outputs/YYYY-MM-DD/<slug>/*.png`
- Metadata: `outputs/YYYY-MM-DD/<slug>/meta.json`
- Manifest: `manifest/index.json`

## Testing

```
pytest tests/ --cov=pixelbliss --cov-report=term-missing
```

## Cron

```
0 10-22/3 * * * cd /path/to/pixelbliss && python main.py post-once >> log.txt 2>&1
