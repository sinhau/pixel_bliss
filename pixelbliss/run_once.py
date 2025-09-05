import datetime
import random
from typing import List, Dict, Any, Optional
from . import config, prompts, providers, imaging, scoring, twitter, storage, alerts
from .providers.base import ImageResult
from .imaging import metrics, sanity, phash
from .scoring import aesthetic
from .storage import manifest
from .config import Config
import pytz

def category_by_time(categories: List[str], rotation_minutes: int, now=None):
    """
    Select a category based on the current time using rotation.
    
    Args:
        categories: List of available categories to rotate through.
        rotation_minutes: Number of minutes each category should be active.
        now: Optional datetime to use instead of current time. Defaults to None.
        
    Returns:
        str: The category that should be active at the given time.
    """
    if now is None:
        tz = pytz.timezone("America/Los_Angeles")  # From config
        now = datetime.datetime.now(tz)
    idx = ((now.hour * 60 + now.minute) // rotation_minutes) % len(categories)
    return categories[idx]

def category_by_random(categories: List[str]):
    """
    Select a category randomly from the available categories.
    
    Args:
        categories: List of available categories to choose from.
        
    Returns:
        str: A randomly selected category.
    """
    return random.choice(categories)

def select_category(cfg: Config):
    """
    Select a category based on the configured selection method.
    
    Args:
        cfg: Configuration object containing selection method and categories.
        
    Returns:
        str: The selected category.
    """
    if cfg.category_selection_method == "random":
        return category_by_random(cfg.categories)
    else:
        # Default to time-based selection
        return category_by_time(cfg.categories, cfg.rotation_minutes)

def try_in_order(prompt: str, providers: List[str], models: List[str], retries: int) -> Optional[ImageResult]:
    """
    Try to generate an image using providers and models in order until one succeeds.
    
    Args:
        prompt: The text prompt for image generation.
        providers: List of provider names to try in order.
        models: List of model names corresponding to each provider.
        retries: Number of retries per provider/model combination.
        
    Returns:
        Optional[ImageResult]: The first successful image result, or None if all fail.
    """
    for prov, mod in zip(providers, models):
        result = providers.base.generate_image(prompt, prov, mod, retries)
        if result:
            return result
    return None

def normalize_and_rescore(items: List[Dict], cfg: Config) -> List[Dict]:
    """
    Normalize brightness and entropy scores and calculate final ranking scores.
    
    Args:
        items: List of image candidates with brightness, entropy, and aesthetic scores.
        cfg: Configuration object containing ranking weights.
        
    Returns:
        List[Dict]: Items with added normalized scores and final ranking score.
    """
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
    """
    Get today's date in local timezone as a string.
    
    Returns:
        str: Today's date in YYYY-MM-DD format.
    """
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.datetime.now(tz).strftime("%Y-%m-%d")

def now_iso() -> str:
    """
    Get current timestamp in local timezone as ISO format string.
    
    Returns:
        str: Current timestamp in ISO format.
    """
    tz = pytz.timezone("America/Los_Angeles")
    return datetime.datetime.now(tz).isoformat()

def tweet_url(tweet_id: str) -> str:
    """
    Generate a Twitter/X URL from a tweet ID.
    
    Args:
        tweet_id: The ID of the tweet.
        
    Returns:
        str: Full URL to the tweet.
    """
    return f"https://x.com/user/status/{tweet_id}"  # Replace user with actual

def fs_abs(path: str) -> str:
    """
    Convert a relative path to an absolute filesystem path.
    
    Args:
        path: Relative path to convert.
        
    Returns:
        str: Absolute path (currently just returns the input path).
    """
    return path  # Assuming relative to web root

def post_once(dry_run: bool = False):
    """
    Execute the complete pipeline to generate, rank, and post a wallpaper image.
    
    This function orchestrates the entire process:
    1. Select category based on time rotation
    2. Generate base prompt and variants
    3. Generate images using configured providers
    4. Score and rank candidates
    5. Select winner avoiding duplicates
    6. Upscale and create wallpaper variants
    7. Save files and metadata
    8. Post to social media (unless dry_run=True)
    
    Args:
        dry_run: If True, skip posting to social media. Defaults to False.
        
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    cfg = config.load_config()

    # A) Category selection based on configured method
    category = select_category(cfg)

    # B) Base prompt -> 3 prompt variants
    base_prompt = prompts.make_base(category, cfg)
    variant_prompts = prompts.make_variants_from_base(base_prompt, cfg.prompt_generation.num_prompt_variants, cfg)

    # C) Generate images - try FAL models with Replicate fallback by index
    candidates = []
    for vp in variant_prompts:
        # Loop through models by index, trying FAL first then Replicate fallback for same index
        for i in range(len(cfg.image_generation.model_fal)):
            # Try FAL model at index i
            fal_model = cfg.image_generation.model_fal[i]
            imgres = providers.base.generate_image(vp, cfg.image_generation.provider_order[0], fal_model)

            # If FAL failed and we have a corresponding Replicate model, try it
            if not imgres:
                replicate_model = cfg.image_generation.model_replicate[i]
                imgres = providers.base.generate_image(vp, cfg.image_generation.provider_order[1], replicate_model)

            # Add image candidate if available
            if imgres:
                candidates.append({**imgres, "prompt": vp})

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
        # Pass image URL and config to aesthetic scoring
        image_url = c.get("image_url")
        if image_url:
            a = aesthetic.aesthetic(image_url, cfg)
        else:
            # No URL available, use default score
            a = 0.5
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

    if dry_run:
        return 0

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
