import datetime
from typing import List, Dict, Any, Optional
from . import config, prompts, providers, imaging, scoring, twitter, storage, alerts
from .providers.base import ImageResult
from .imaging import metrics, sanity, phash
from .scoring import aesthetic
from .storage import manifest
from .config import Config
import pytz

def category_by_time(categories: List[str], rotation_minutes: int, now=None):
    if now is None:
        tz = pytz.timezone("America/Los_Angeles")  # From config
        now = datetime.datetime.now(tz)
    idx = ((now.hour * 60 + now.minute) // rotation_minutes) % len(categories)
    return categories[idx]

def try_in_order(prompt: str, providers: List[str], models: List[str], retries: int) -> Optional[ImageResult]:
    for prov, mod in zip(providers, models):
        result = providers.base.generate_image(prompt, prov, mod, retries)
        if result:
            return result
    return None

def normalize_and_rescore(items: List[Dict], cfg: Config) -> List[Dict]:
    if not items:
        return items
    brights = [i['brightness'] for i in items]
    ents = [i['entropy'] for i in items]
    aesths = [i['aesthetic'] for i in items]

    b_min, b_max = min(brights), max(brights)
    e_min, e_max = min(ents), max(ents)

    for i in items:
        b_norm = (i['brightness'] - b_min) / (b_max - b_min) if b_max > b_min else 0.5
        e_norm = (i['entropy'] - e_min) / (e_max - e_min) if e_max > e_min else 0.5
        i['final'] = (
            cfg.ranking.w_brightness * b_norm +
            cfg.ranking.w_entropy * e_norm +
            cfg.ranking.w_aesthetic * i['aesthetic']
        )
    return items

def today_local() -> str:
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")

def now_iso() -> str:
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.datetime.now(tz).isoformat()

def tweet_url(tweet_id: str) -> str:
    return f"https://x.com/user/status/{tweet_id}"  # Replace user with actual

def fs_abs(path: str) -> str:
    return path  # Assuming relative to web root

def post_once():
    cfg = config.load_config()

    # A) Stateless category selection
    category = category_by_time(cfg.categories, cfg.rotation_minutes)

    # B) Base prompt -> 3 prompt variants
    base_prompt = prompts.make_base(category, cfg)
    variant_prompts = prompts.make_variants_from_base(base_prompt, cfg.prompt_generation.num_prompt_variants, cfg)

    # C) Generate images - try FAL models first, then Replicate as fallback
    candidates = []
    for vp in variant_prompts:
        # Try FAL models first
        imgres = None
        for model in cfg.image_generation.model_fal:
            imgres = providers.base.generate_image(vp, "fal", model, cfg.image_generation.retries_per_image)
            if imgres:
                candidates.append({**imgres, "prompt": vp})
                break

        # If no FAL model worked, try Replicate models
        if not imgres:
            for model in cfg.image_generation.model_replicate:
                imgres = providers.base.generate_image(vp, "replicate", model, cfg.image_generation.retries_per_image)
                if imgres:
                    candidates.append({**imgres, "prompt": vp})
                    break

    if not candidates:
        alerts.webhook.send_failure("no images produced")
        return 1

    # D) Rank candidates
    scored = []
    for c in candidates:
        b = metrics.brightness(c["image"])
        e = metrics.entropy(c["image"])
        if not sanity.passes_floors(b, e, cfg):
            continue
        a = aesthetic.aesthetic(c["image"])
        c["brightness"] = b
        c["entropy"] = e
        c["aesthetic"] = a
        scored.append(c)

    if not scored:
        alerts.webhook.send_failure("all candidates failed sanity/scoring")
        return 1

    scored = normalize_and_rescore(scored, cfg)
    recent_hashes = manifest.load_recent_hashes(limit=200)

    winner = None
    for c in sorted(scored, key=lambda x: x["final"], reverse=True):
        if not phash.is_duplicate(phash.phash_hex(c["image"]), recent_hashes, cfg.ranking.phash_distance_min):
            winner = c
            break
    if winner is None:
        alerts.webhook.send_failure("near-duplicate with manifest history")
        return 0  # not fatal

    # E) Upscale winner -> wallpaper variants
    base_img = winner["image"]
    if cfg.upscale.enabled:
        from .providers import upscale
        base_img = upscale.upscale(base_img, cfg.upscale.provider, cfg.upscale.model, cfg.upscale.factor)
        if base_img is None:
            base_img = winner["image"]  # fallback

    from .imaging import variants
    wallpapers = variants.make_wallpaper_variants(base_img, cfg.wallpaper_variants)

    # Alt text
    alt = prompts.make_alt_text(base_prompt, winner["prompt"], cfg)

    # Save
    date_str = today_local()
    slug = storage.paths.make_slug(category, base_prompt)
    out_dir = storage.paths.output_dir(date_str, slug)
    public_paths = storage.fs.save_images(out_dir, wallpapers)

    ph = phash.phash_hex(wallpapers[next(iter(wallpapers))])

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

    # Update
    manifest.update_tweet_id(f"{date_str}_{category}_{slug}", tweet_id)
    meta["tweet_id"] = tweet_id
    storage.fs.save_meta(out_dir, meta)

    alerts.webhook.send_success(category, meta["model"], tweet_url(tweet_id), public_paths[first_key])
    return 0
