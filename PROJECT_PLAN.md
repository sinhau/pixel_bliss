# PROJECT_PLAN.md — PixelBliss v4.1 (single post per run, website-ready, configurable prompt gen)

> **Goal:** Automate a stateless pipeline that posts **one** highly aesthetic image per run to X/Twitter (with **alt text only**), while saving an **upscaled wallpaper set** to a simple VPS for a future website.  
> **Key design:** 1 base prompt → 3 prompt variants → **9 images** → rank (brightness, entropy, lightweight aesthetic model) → **pick 1 winner** → **upscale** → generate **configurable** smartphone/PC resolutions → save → tweet (alt-only) → alert.  
> **New:** **Configurable prompt generation provider** (default: **OpenAI GPT‑5**) for both prompt creation and alt-text generation.

---

## 0) Scope & Non-Goals
**In-scope**
- End-to-end CLI that runs once per cron invocation and posts exactly one image.
- Stateless category rotation (time-based), no DB.
- **Prompt generation is configurable**; default provider is **OpenAI GPT‑5**.
- FAL AI as preferred image provider; Replicate as fallback.
- 1 base prompt → 3 variants → 3 images each (9 total).
- Ranking by brightness + entropy + lightweight aesthetic model.
- Upscale winner only, then generate multiple wallpaper sizes (configurable).
- Save outputs to static web root on a VPS; maintain a `manifest/index.json` for a future site.
- Minimal alerts: one webhook message on success/failure.
- Alt text only; no caption/hashtags.

**Out-of-scope (for now)**
- Web UI, gallery site, auth, analytics.
- Advanced moderation or style memory.
- Budgets, queues, or background workers.
- Multi-post scheduling within a single run.

---

## 1) Tech Stack
- **Language:** Python 3.11
- **Runtime:** CLI, cron-friendly (no server).
- **Deps:** `requests`, `python-dotenv`, `pydantic`, `Pillow`, `imagehash`, `tenacity`
- **Prompt Gen:** Configurable; default **OpenAI GPT‑5** (pluggable interface)
- **Providers:** FAL AI (preferred), Replicate (fallback)
- **Hosting (files):** VPS (e.g., Nginx) serving `/var/www/pixelbliss` as static files
- **Posting:** X/Twitter API (media upload, set alt text, create tweet)
- **Alerts:** single webhook (Discord/Slack/Telegram)

---

## 2) Directory Layout
```
pixelbliss/
  main.py
  pixelbliss/
    config.py
    prompt_engine/              # configurable prompt provider (GPT‑5 by default)
      __init__.py
      base.py
      openai_gpt5.py
      dummy_local.py            # optional template-only fallback
    prompts.py                  # orchestration: base prompt + 3 variants
    providers/
      base.py
      fal.py
      replicate.py
      upscale.py                # wrapper for ESRGAN or similar
    imaging/
      metrics.py                # brightness, entropy
      sanity.py                 # floors/thresholds
      phash.py                  # dedupe helpers
      variants.py               # crop/pad -> wallpaper sizes
    scoring/
      aesthetic.py              # lightweight hosted aesthetic scorer
    twitter/
      client.py                 # upload_media, set_alt_text, create_tweet
    storage/
      fs.py                     # save winners, write meta.json
      manifest.py               # append/update manifest/index.json
      paths.py                  # slugs, paths, filenames
    alerts/
      webhook.py
    run_once.py                 # main pipeline
  config.yaml
  .env.example
  README.md
```

---

## 3) Environment Variables (`.env`)
```
FAL_API_KEY=
REPLICATE_API_TOKEN=
X_API_KEY=
X_API_SECRET=
X_ACCESS_TOKEN=
X_ACCESS_TOKEN_SECRET=
ALERT_WEBHOOK_URL=      # Discord/Slack/Telegram
OPENAI_API_KEY=         # used when prompt provider is openai:gpt-5
```

---

## 4) Configuration (`config.yaml`)
### Schema
```yaml
timezone: "America/Los_Angeles"

categories: [ "sci-fi", "tech", "mystic", "geometry", "nature", "neo-noir", "watercolor", "cosmic-minimal" ]

rotation_minutes: 180   # category index = floor(now_min / rotation_minutes) % len(categories)

prompt_generation:
  provider: "openai"        # openai | dummy
  model: "gpt-5"            # default model for base prompt, variants, and alt text
  temperature: 0.8
  max_tokens: 400
  # system/style rules enforced at runtime (no real people/logos/NSFW, negative prompts, style hints)

generation:
  num_prompt_variants: 3        # 1 base -> 3 variants
  images_per_variant: 3         # 3 images per variant -> total 9
  provider_order: ["fal", "replicate"]
  model_fal: "black-forest-labs/flux-1.1"     # example id
  model_replicate: "black-forest-labs/flux"   # example id
  retries_per_image: 2

ranking:
  # final = w_brightness * B_norm + w_entropy * H_norm + w_aesthetic * Aesthetic
  w_brightness: 0.25
  w_entropy: 0.25
  w_aesthetic: 0.50

  # sanity floors (discard if violated)
  entropy_min: 3.5
  brightness_min: 10
  brightness_max: 245

  # dedupe against recent manifest pHashes
  phash_distance_min: 6

upscale:
  enabled: true
  provider: "replicate"          # or "fal"
  model: "real-esrgan-4x"        # pick your exact model
  factor: 2

# output sizes — edit this list to change wallpaper variants without code changes
wallpaper_variants:
  - {name: "square_1x1_2k",        w: 2048, h: 2048}
  - {name: "phone_9x16_2k",        w: 1125, h: 2000}
  - {name: "phone_20x9_3.2k",      w: 1440, h: 3200}
  - {name: "desktop_16x9_1080p",   w: 1920, h: 1080}
  - {name: "desktop_16x9_1440p",   w: 2560, h: 1440}
  - {name: "desktop_16x9_4k",      w: 3840, h: 2160}
  - {name: "desktop_16x10_1600p",  w: 2560, h: 1600}
  - {name: "ultrawide_21x9",       w: 3440, h: 1440}

alerts:
  enabled: true
  webhook_url_env: "ALERT_WEBHOOK_URL"
```

---

## 5) Static Hosting on VPS
- Web root: `/var/www/pixelbliss`
- Public structure:
```
/var/www/pixelbliss/
  outputs/YYYY-MM-DD/<slug>/
    winner_upscaled.png (optional)
    <variant>.png (for each wallpaper size)
    meta.json
  manifest/index.json   # append-only gallery manifest
```
- Expose the root at `https://<your-domain>/` (Nginx example below).

### Nginx (example)
```
server {
  listen 80;
  server_name pixelbliss.example.com;
  root /var/www/pixelbliss;

  location / {
    autoindex off;
    try_files $uri $uri/ =404;
  }
}
```

---

## 6) Data Schemas
### meta.json (per winner)
```json
{
  "category": "sci-fi",
  "base_prompt": "…",
  "variant_prompt": "…",
  "variant_index": 1,
  "provider": "fal",
  "model": "flux-1.1",
  "seed": 12345678,
  "created_at": "2025-09-04T10:02:45-07:00",
  "alt_text": "A glowing crystal city floating in space with soft auroras weaving among glassy spires.",
  "phash": "a1b2c3d4",
  "scores": { "aesthetic": 0.72, "brightness": 128.4, "entropy": 5.1, "final": 0.78 },
  "files": {
    "square_1x1_2k": "/outputs/2025-09-04/crystal-archipelago/square_1x1_2k.png",
    "phone_9x16_2k": "/outputs/2025-09-04/crystal-archipelago/phone_9x16_2k.png",
    "phone_20x9_3.2k": "/outputs/2025-09-04/crystal-archipelago/phone_20x9_3.2k.png",
    "desktop_16x9_1080p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_1080p.png",
    "desktop_16x9_1440p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_1440p.png",
    "desktop_16x9_4k": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_4k.png",
    "desktop_16x10_1600p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x10_1600p.png",
    "ultrawide_21x9": "/outputs/2025-09-04/crystal-archipelago/ultrawide_21x9.png"
  },
  "tweet_id": "0000000000000000000"
}
```

### manifest/index.json (append one object per posted winner)
```json
{
  "id": "2025-09-04_sci-fi_crystal-archipelago",
  "date": "2025-09-04",
  "category": "sci-fi",
  "files": {
    "square_1x1_2k": "/outputs/2025-09-04/crystal-archipelago/square_1x1_2k.png",
    "phone_9x16_2k": "/outputs/2025-09-04/crystal-archipelago/phone_9x16_2k.png",
    "phone_20x9_3.2k": "/outputs/2025-09-04/crystal-archipelago/phone_20x9_3.2k.png",
    "desktop_16x9_1080p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_1080p.png",
    "desktop_16x9_1440p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_1440p.png",
    "desktop_16x9_4k": "/outputs/2025-09-04/crystal-archipelago/desktop_16x9_4k.png",
    "desktop_16x10_1600p": "/outputs/2025-09-04/crystal-archipelago/desktop_16x10_1600p.png",
    "ultrawide_21x9": "/outputs/2025-09-04/crystal-archipelago/ultrawide_21x9.png"
  },
  "tweet_id": "0000000000000000000",
  "phash": "a1b2c3d4"
}
```

---

## 7) Module Contracts (for AI agent)

### CLI
- `pixelbliss post-once` → executes the full pipeline once
- `pixelbliss dry-run` → runs everything except the X post; still saves files
- `pixelbliss repair-manifest` → re-scan outputs and rebuild manifest

### prompt_engine/base.py
```python
from typing import Protocol, List

class PromptProvider(Protocol):
    def make_base(self, category: str) -> str: ...
    def make_variants_from_base(self, base_prompt: str, k: int) -> List[str]: ...
    def make_alt_text(self, base_prompt: str, variant_prompt: str) -> str: ...
```

### prompt_engine/openai_gpt5.py
- Implement `PromptProvider` using OpenAI GPT‑5 (model from config).

### prompts.py
```python
def make_base(category: str) -> str: ...
def make_variants_from_base(base_prompt: str, k: int) -> list[str]: ...
def make_alt_text(base_prompt: str, variant_prompt: str) -> str: ...
# These delegate to prompt_engine based on config.prompt_generation.provider
```

### providers/base.py
```python
# ImageResult is a dict with: image (PIL.Image), provider (str), model (str), seed (int)
def generate_image(prompt: str, provider: str, model: str, retries: int): ...
```

### providers/fal.py and providers/replicate.py
- Implement `generate_image(...)` calling respective APIs (respect retries).

### providers/upscale.py
```python
def upscale(image, provider: str, model: str, factor: int):
    # return PIL.Image
    pass
```

### imaging/metrics.py
```python
def brightness(image) -> float: ...
def entropy(image) -> float: ...
```

### imaging/sanity.py
```python
def passes_floors(brightness_value: float, entropy_value: float, cfg) -> bool: ...
```

### imaging/phash.py
```python
def phash_hex(image) -> str: ...
def is_duplicate(phash_hex_str: str, recent_hashes: list, distance_min: int) -> bool: ...
```

### imaging/variants.py
```python
def crop_pad(image, w: int, h: int):
    # return resized/cropped/padded PIL.Image
    pass

def make_wallpaper_variants(image, variants_cfg: list) -> dict:
    # returns mapping {variant_name: PIL.Image}
    pass
```

### scoring/aesthetic.py
```python
def aesthetic(image) -> float:
    # Return [0,1]. Use lightweight hosted model; fallback to 0.5 on error.
    pass
```

### Ranking utilities
```python
def final_score(bright, entr, aesth, weights) -> float: ...
def normalize_and_rescore(items: list, weights) -> list:
    # Min-max normalize brightness/entropy over items, set item["final"]
    pass
```

### twitter/client.py
```python
def upload_media(paths: list) -> list: ...       # returns media_ids
def set_alt_text(media_id: str, alt: str) -> None: ...
def create_tweet(text: str, media_ids: list) -> str:  # returns tweet_id
```

### storage/paths.py
```python
def make_slug(category: str, base_prompt: str) -> str: ...
def output_dir(date_str: str, slug: str) -> str: ...
```

### storage/fs.py
```python
def save_images(dir_path: str, images: dict) -> dict:
    # Save to disk; return {variant_name: public_path}
    pass

def save_meta(dir_path: str, meta: dict) -> str:
    # Write meta.json; return path
    pass
```

### storage/manifest.py
```python
def append(entry: dict) -> None: ...
def update_tweet_id(item_id: str, tweet_id: str) -> None: ...
def load_recent_hashes(limit: int = 200) -> list: ...
```

### alerts/webhook.py
```python
def send_success(category: str, model: str, tweet_url: str, image_url: str) -> None: ...
def send_failure(reason: str, details: str = "") -> None: ...
```

---

## 8) Main Pipeline (pseudocode)

```python
def category_by_time(categories, rotation_minutes, now=None):
    # Stateless round-robin by current time
    now = now or datetime.now(local_tz)
    idx = ((now.hour*60 + now.minute) // rotation_minutes) % len(categories)
    return categories[idx]

def post_once():
    cfg = load_config()

    # A) Stateless category selection
    category = category_by_time(cfg.categories, cfg.rotation_minutes)

    # B) Base prompt -> 3 prompt variants (via configurable prompt provider; default GPT‑5)
    base_prompt = prompts.make_base(category)
    variant_prompts = prompts.make_variants_from_base(base_prompt, cfg.generation.num_prompt_variants)

    # C) Generate 3 images per variant (FAL first, then Replicate)
    candidates = []
    for vp in variant_prompts:
        for _ in range(cfg.generation.images_per_variant):
            imgres = try_in_order(
                vp,
                providers=cfg.generation.provider_order,
                models=[cfg.generation.model_fal, cfg.generation.model_replicate],
                retries=cfg.generation.retries_per_image
            )
            if imgres:
                candidates.append({**imgres, "prompt": vp})

    if not candidates:
        alerts.send_failure("no images produced"); return 1

    # D) Rank 9: sanity -> score -> normalize -> choose winner; dedupe vs manifest
    scored = []
    for c in candidates:
        b = metrics.brightness(c["image"])
        e = metrics.entropy(c["image"])
        if not sanity.passes_floors(b, e, cfg): 
            continue
        a = scoring.aesthetic(c["image"])  # fallback 0.5 on error
        c["brightness"] = b
        c["entropy"] = e
        c["aesthetic"] = a
        scored.append(c)

    if not scored:
        alerts.send_failure("all candidates failed sanity/scoring"); return 1

    scored = normalize_and_rescore(scored, cfg.ranking)
    recent_hashes = manifest.load_recent_hashes(limit=200)

    winner = None
    for c in sorted(scored, key=lambda x: x["final"], reverse=True):
        if not phash.is_duplicate(phash.phash_hex(c["image"]), recent_hashes, cfg.ranking.phash_distance_min):
            winner = c
            break
    if winner is None:
        alerts.send_failure("near-duplicate with manifest history"); return 0  # not fatal

    # E) Upscale winner -> wallpaper variants
    base_img = winner["image"]
    if cfg.upscale.enabled:
        base_img = upscale.upscale(base_img, cfg.upscale.provider, cfg.upscale.model, cfg.upscale.factor)

    wallpapers = imaging.variants.make_wallpaper_variants(base_img, cfg.wallpaper_variants)

    # Alt text only (via configurable prompt provider; default GPT‑5)
    alt = prompts.make_alt_text(base_prompt, winner["prompt"])

    # Save to web root & manifest
    date_str = today_local()
    slug = storage.paths.make_slug(category, base_prompt)
    out_dir = storage.paths.output_dir(date_str, slug)
    public_paths = storage.fs.save_images(out_dir, wallpapers)

    ph = phash.phash_hex(wallpapers[next(iter(wallpapers))])  # hash on first variant

    meta = {
      "category": category,
      "base_prompt": base_prompt,
      "variant_prompt": winner["prompt"],
      "provider": winner["provider"],
      "model": winner["model"],
      "seed": winner.get("seed"),
      "created_at": now_iso(),
      "alt_text": alt,
      "phash": ph,
      "scores": {
        "aesthetic": round(winner["aesthetic"], 4),
        "brightness": round(winner["brightness"], 2),
        "entropy": round(winner["entropy"], 3),
        "final": round(winner["final"], 4)
      },
      "files": public_paths,
      "tweet_id": None
    }
    storage.fs.save_meta(out_dir, meta)

    manifest.append({
      "id": f"{date_str}_{category}_{slug}",
      "date": date_str,
      "category": category,
      "files": public_paths,
      "tweet_id": None,
      "phash": ph
    })

    # Post to X
    first_key = "phone_9x16_2k" if "phone_9x16_2k" in public_paths else next(iter(public_paths))
    media_ids = twitter.client.upload_media([fs_abs(public_paths[first_key])])
    twitter.client.set_alt_text(media_ids[0], alt)
    tweet_id = twitter.client.create_tweet(text="", media_ids=media_ids)

    # Update manifest/meta with tweet id
    manifest.update_tweet_id(f"{date_str}_{category}_{slug}", tweet_id)
    meta["tweet_id"] = tweet_id
    storage.fs.save_meta(out_dir, meta)  # overwrite with tweet id

    alerts.webhook.send_success(category, meta["model"], tweet_url(tweet_id), public_paths[first_key])
    return 0
```

---

## 9) Alert Payloads (examples)
**Success (text content)**
```
[PixelBliss] Posted sci-fi via flux-1.1 → https://x.com/.../status/1234567890
Image: https://pixelbliss.example.com/outputs/2025-09-04/crystal-archipelago/phone_9x16_2k.png
```

**Failure (text content)**
```
[PixelBliss] FAIL: no images produced (FAL + Replicate both errored)
```

---

## 10) Cron Examples
Every 3 hours, 10am–10pm PT:
```
0 10-22/3 * * * cd /var/app/pixelbliss && /usr/bin/python3 -m pixelbliss post-once >> /var/app/pixelbliss/cron.log 2>&1
```

---

## 11) Acceptance Criteria
1. Each `post-once` run produces **exactly one** tweet (or cleanly aborts on duplicate) with **alt text only**.
2. Pipeline: **1 base prompt → 3 variants → 9 images → rank (brightness+entropy+aesthetic) → 1 winner → upscale → wallpaper variants**.
3. All winner files are written under `/var/www/pixelbliss/outputs/YYYY-MM-DD/<slug>/` and listed in `manifest/index.json`.
4. `meta.json` and `manifest/index.json` include `tweet_id` after posting.
5. One compact webhook message fires on success/failure.
6. Stateless category rotation works (time-based).
7. Prompt provider is **configurable**; default is **OpenAI GPT‑5** for prompts and alt text.

---

## 12) Test Plan (minimal)
- Unit-ish: mock providers to return generated noise and verify ranking & floors.
- Dry run: run `dry-run` to ensure files & manifest are created without tweeting.
- Happy path: providers succeed; confirm 9 candidates → winner → saved → posted.
- Provider failure: force FAL to fail; ensure Replicate fallback works.
- Scorer down: make `aesthetic()` return default; ensure ranking still works.
- Duplicate guard: copy a previous winner to candidates; ensure skip/abort behavior.
- Alt text: ensure non-empty, 1–2 sentences, no hashtags/emojis.
- Prompt provider swap: switch `prompt_generation.provider` to `dummy` and confirm pipeline still runs.

---

## 13) Notes for the Agent
- Prefer stdlib + tiny deps. Keep functions short and pure where possible.
- All external calls must be retried with backoff (`tenacity`); fail fast after limit.
- Never write secrets to logs. Keep logs concise.
- PNG outputs. Use high-quality resampling for resize/crop (e.g., Lanczos).
- Prompt safety: no real people/logos/NSFW; include a negative prompt field.
- Ensure consistent time zone handling (config `timezone`).

---

End of plan. Ship it.
