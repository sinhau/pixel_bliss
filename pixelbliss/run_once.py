import datetime
import random
import asyncio
import logging
from typing import List, Dict, Any, Optional
from . import config, prompts, providers, imaging, scoring, twitter, storage, alerts
from .providers.base import ImageResult
from .imaging import metrics, sanity, phash, collage, quality
from .scoring import aesthetic
from .storage import manifest
from .config import Config
from .logging_config import get_logger, ProgressLogger
import pytz


async def generate_theme_hint_async(cfg: Config, progress_logger=None) -> str:
    """
    Generate a theme hint using trending topics analysis asynchronously.
    
    Args:
        cfg: Configuration object containing trending themes settings.
        progress_logger: Optional progress logger for tracking.
    
    Returns:
        str: A theme hint for the knobs system.
    """
    if cfg.trending_themes.enabled:
        try:
            from .prompt_engine.trending_topics import TrendingTopicsProvider
            
            provider = TrendingTopicsProvider(
                model=cfg.trending_themes.model
            )
            
            return await provider.get_trending_theme_async(progress_logger)
                
        except Exception as e:
            logger = get_logger('theme_generation')
            logger.error(f"Trending topics failed: {e}")
            if progress_logger:
                progress_logger.warning("Trending topics failed, using fallback themes")
            
            if not cfg.trending_themes.fallback_enabled:
                raise e
    
    # Fallback to curated theme hints
    themes = [
        # keep the broad, content-agnostic originals
        "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",

        # concept & idea themes (not style/mood)
        "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
        "cycles", "growth", "renewal", "emergence", "evolution",
        "interconnection", "networks", "continuum", "wholeness", "infinity",
        "order and randomness", "pattern", "repetition", "rhythm",

        # math & structure (conceptual domains, not aesthetics)
        "fractal", "spirals", "tessellation", "lattice", "grid",
        "waveforms", "fields", "orbits", "constellations", "topography", "cartography",

        # natural domains (generic, non-specific)
        "elemental", "terrestrial", "celestial", "aquatic", "mineral",
        "botanical", "aerial", "seasonal", "weather",

        # metaphor & abstract idea spaces
        "journey", "thresholds", "liminality", "sanctuary", "play",
        "curiosity", "wonder", "stillness", "openness", "simplicity",
        "order and flow", "cause and effect", "microcosm and macrocosm"
    ]
    return random.choice(themes)


def try_in_order(prompt: str, provider_names: List[str], models: List[str], retries: int) -> Optional[ImageResult]:
    """
    Try to generate an image using providers and models in order until one succeeds.
    
    Args:
        prompt: The text prompt for image generation.
        provider_names: List of provider names to try in order.
        models: List of model names corresponding to each provider.
        retries: Number of retries per provider/model combination.
        
    Returns:
        Optional[ImageResult]: The first successful image result, or None if all fail.
    """
    for prov, mod in zip(provider_names, models):
        result = providers.base.generate_image(prompt, prov, mod, retries)
        if result:
            return result
    return None

def normalize_and_rescore(items: List[Dict], cfg: Config) -> List[Dict]:
    """
    Normalize brightness and entropy scores and calculate final ranking scores.
    
    Args:
        items: List of image candidates with brightness, entropy, aesthetic, and local quality scores.
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
        # Handle backward compatibility - local_quality might not be present in older tests
        local_q = i.get('local_quality', 0.5)  # Default to 0.5 if not present
        i['final'] = (
            cfg.ranking.w_brightness * b_norm +
            cfg.ranking.w_entropy * e_norm +
            cfg.ranking.w_aesthetic * i['aesthetic'] +
            cfg.ranking.w_local_quality * local_q
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
        str: Absolute path with leading slash removed if present.
    """
    # Remove leading slash if present to make it a proper relative path
    if path.startswith('/'):
        return path[1:]
    return path

async def generate_for_variant(variant_prompt: str, cfg: Config, semaphore: Optional[asyncio.Semaphore] = None, progress_logger=None, variant_index: int = 0) -> List[Dict[str, Any]]:
    """
    Generate images for a single prompt variant, trying FAL first then Replicate fallback per model index.
    
    Args:
        variant_prompt: The prompt variant to generate images for.
        cfg: Configuration object containing provider and model settings.
        semaphore: Optional semaphore to limit concurrent API calls.
        progress_logger: Optional progress logger for tracking generation progress.
        variant_index: Index of this variant for progress tracking.
        
    Returns:
        List[Dict[str, Any]]: List of successful image candidates for this variant.
    """
    logger = get_logger('generation')
    candidates = []
    
    try:
        # Loop through models by index, trying FAL first then Replicate fallback for same index
        for i in range(len(cfg.image_generation.model_fal)):
            try:
                # Try FAL model at index i
                fal_model = cfg.image_generation.model_fal[i]
                fal_provider = cfg.image_generation.provider_order[0]
                
                logger.debug(f"Trying {fal_provider}/{fal_model} for variant: {variant_prompt[:50]}...")
                
                if semaphore:
                    async with semaphore:
                        imgres = await asyncio.to_thread(
                            providers.base.generate_image, 
                            variant_prompt, 
                            fal_provider, 
                            fal_model
                        )
                else:
                    imgres = await asyncio.to_thread(
                        providers.base.generate_image, 
                        variant_prompt, 
                        fal_provider, 
                        fal_model
                    )
                
                # If FAL failed and we have a corresponding Replicate model, try it
                if not imgres and i < len(cfg.image_generation.model_replicate):
                    replicate_model = cfg.image_generation.model_replicate[i]
                    replicate_provider = cfg.image_generation.provider_order[1]
                    
                    logger.debug(f"FAL failed, trying {replicate_provider}/{replicate_model}")
                    
                    if semaphore:
                        async with semaphore:
                            imgres = await asyncio.to_thread(
                                providers.base.generate_image, 
                                variant_prompt, 
                                replicate_provider, 
                                replicate_model
                            )
                    else:
                        imgres = await asyncio.to_thread(
                            providers.base.generate_image, 
                            variant_prompt, 
                            replicate_provider, 
                            replicate_model
                        )
                
                # Add image candidate if available
                if imgres:
                    logger.debug(f"Successfully generated image with {imgres['provider']}/{imgres['model']}")
                    candidates.append({**imgres, "prompt": variant_prompt})
                    
            except Exception as e:
                # Log the error but continue with other model indices
                # This ensures one failing model doesn't stop the entire variant
                logger.warning(f"Error generating image for variant '{variant_prompt[:50]}...' with model index {i}: {e}")
                continue
        
        logger.debug(f"Generated {len(candidates)} candidates for variant")
        
        # Update progress
        if progress_logger:
            progress_logger.update_operation_progress("image_generation")
        
        return candidates
        
    except Exception as e:
        # Update progress even on failure
        if progress_logger:
            progress_logger.update_operation_progress("image_generation")
        raise e

async def run_all_variants(variant_prompts: List[str], cfg: Config, progress_logger=None) -> List[Dict[str, Any]]:
    """
    Generate images for all prompt variants in parallel.
    
    Args:
        variant_prompts: List of prompt variants to generate images for.
        cfg: Configuration object containing provider and model settings.
        progress_logger: Optional progress logger for tracking generation progress.
        
    Returns:
        List[Dict[str, Any]]: Flattened list of all successful image candidates.
    """
    logger = get_logger('generation')
    
    # Set up concurrency control
    max_concurrency = cfg.image_generation.max_concurrency
    if max_concurrency is None:
        max_concurrency = len(variant_prompts)
    
    logger.info(f"Starting parallel generation for {len(variant_prompts)} variants (max_concurrency: {max_concurrency})")
    
    # Start progress tracking
    if progress_logger:
        progress_logger.start_operation("image_generation", len(variant_prompts), "parallel image generation")
    
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency > 0 else None
    
    # Create tasks for all variants
    tasks = [
        asyncio.create_task(generate_for_variant(vp, cfg, semaphore, progress_logger, i)) 
        for i, vp in enumerate(variant_prompts)
    ]
    
    # Execute all tasks in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Flatten successful results and handle exceptions
    candidates = []
    failed_variants = 0
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            failed_variants += 1
            logger.error(f"Error processing variant '{variant_prompts[i][:50]}...': {result}")
            # Optionally send alert for failed variants
            try:
                alerts.webhook.send_failure(f"variant failed: {variant_prompts[i]} - {str(result)}", cfg)
            except Exception as alert_error:
                logger.debug(f"Failed to send alert for variant failure: {alert_error}")
        else:
            # result is a list of candidates for this variant
            candidates.extend(result)
            logger.debug(f"Variant {i+1} produced {len(result)} candidates")
    
    # Finish progress tracking
    if progress_logger:
        success = failed_variants == 0
        progress_logger.finish_operation("image_generation", success)
        if failed_variants > 0:
            progress_logger.warning(f"{failed_variants} image generation variants failed")
    
    logger.info(f"Parallel generation completed: {len(candidates)} total candidates, {failed_variants} failed variants")
    return candidates


async def post_once(dry_run: bool = False, logger: Optional[logging.Logger] = None, progress_logger: Optional[ProgressLogger] = None):
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
        logger: Optional logger instance for detailed logging.
        progress_logger: Optional progress logger for pipeline tracking.
        
    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    # Set up logging if not provided
    if logger is None:
        logger = get_logger('pipeline')
    if progress_logger is None:
        from .logging_config import setup_logging
        _, progress_logger = setup_logging()
    
    try:
        # Start pipeline tracking
        total_steps = 10 if not dry_run else 9  # Skip posting step in dry run
        progress_logger.start_pipeline(total_steps)
        
        # Step 1: Load configuration
        progress_logger.step("Loading configuration")
        cfg = config.load_config()
        logger.debug("Configuration loaded")
        progress_logger.success("Configuration loaded successfully")

        # Step 2: Generate theme hint
        progress_logger.step("Generating theme hint")
        if cfg.trending_themes.enabled:
            theme_hint = await generate_theme_hint_async(cfg, progress_logger)
        else:
            # Fallback to curated theme hints
            themes = [
                # keep the broad, content-agnostic originals
                "abstract", "nature", "cosmic", "geometric", "organic", "crystalline", "flow",

                # concept & idea themes (not style/mood)
                "balance", "harmony", "unity", "duality", "symmetry", "asymmetry",
                "cycles", "growth", "renewal", "emergence", "evolution",
                "interconnection", "networks", "continuum", "wholeness", "infinity",
                "order and randomness", "pattern", "repetition", "rhythm",

                # math & structure (conceptual domains, not aesthetics)
                "fractal", "spirals", "tessellation", "lattice", "grid",
                "waveforms", "fields", "orbits", "constellations", "topography", "cartography",

                # natural domains (generic, non-specific)
                "elemental", "terrestrial", "celestial", "aquatic", "mineral",
                "botanical", "aerial", "seasonal", "weather",

                # metaphor & abstract idea spaces
                "journey", "thresholds", "liminality", "sanctuary", "play",
                "curiosity", "wonder", "stillness", "openness", "simplicity",
                "order and flow", "cause and effect", "microcosm and macrocosm"
            ]
            theme_hint = random.choice(themes)
        logger.info(f"Generated theme hint: {theme_hint}")
        progress_logger.success(f"Theme hint generated", f"{theme_hint}")

        # Step 3: Generate prompts
        progress_logger.step("Generating prompts")
        
        # Generate base prompt with detailed logging
        base_prompt = prompts.make_base(theme_hint, cfg, progress_logger)
        logger.info(f"Base prompt generated: {base_prompt[:100]}...")
        
        # Generate variant prompts with detailed logging
        logger.info("Starting prompt variant generation (async enabled)")
        if cfg.prompt_generation.async_enabled:
            variant_prompts = await prompts.make_variants_from_base_async(base_prompt, cfg.prompt_generation.num_prompt_variants, cfg, progress_logger)
        else:
            variant_prompts = prompts.make_variants_from_base(base_prompt, cfg.prompt_generation.num_prompt_variants, cfg, progress_logger)
        
        logger.info(f"Generated {len(variant_prompts)} prompt variants")
        for i, vp in enumerate(variant_prompts, 1):
            logger.debug(f"Variant {i}: {vp[:80]}...")
        progress_logger.success(f"Generated {len(variant_prompts)} prompt variants")

        # Step 4: Generate images
        progress_logger.step("Generating images")
        logger.info("Starting image generation (async enabled)")
        
        progress_logger.substep("Using parallel generation")
        candidates = await run_all_variants(variant_prompts, cfg, progress_logger)

        logger.info(f"Generated {len(candidates)} image candidates")
        if not candidates:
            progress_logger.error("No images were generated")
            alerts.webhook.send_failure("no images produced", cfg)
            progress_logger.finish_pipeline(success=False)
            return 1
        
        progress_logger.success(f"Generated {len(candidates)} image candidates")

        # Check for human-in-the-loop selection
        if cfg.discord.enabled:
            progress_logger.step("Human selection via Discord")
            from .alerts import discord_select
            
            selected = await discord_select.ask_user_to_select_raw(candidates, cfg, logger)
            if selected == "none":
                # User rejected all candidates - end pipeline without posting or saving
                logger.info("User rejected all candidates via Discord selection - ending pipeline")
                progress_logger.success("All candidates rejected by user - pipeline ended")
                progress_logger.finish_pipeline(success=True)
                return 0
            elif selected is None:
                # No response received (timeout) - end pipeline without posting or saving
                logger.info("No Discord selection received within timeout - ending pipeline")
                progress_logger.success("No user selection received - pipeline ended")
                progress_logger.finish_pipeline(success=True)
                return 0
            else:
                # Your pick is authoritative
                winner = selected
                timeout_fallback = False
                user_rank = candidates.index(selected) + 1
                logger.info(f"User selected candidate #{user_rank} via Discord")
            
            progress_logger.success(f"Winner selected via human choice (rank #{user_rank})")
            
            # Skip all scoring/collage/duplicate checks - proceed directly to upscaling
            # Set up output directory for saving files
            date_str = today_local()
            slug = storage.paths.make_slug(theme_hint, base_prompt)
            out_dir = storage.paths.output_dir(date_str, slug)
            logger.debug(f"Output directory: {out_dir}")
            
            # Jump to Step 7: Upscale winner
            base_img = winner["image"]
            if cfg.upscale.enabled:
                logger.info(f"Upscaling with {cfg.upscale.provider}/{cfg.upscale.model} (factor: {cfg.upscale.factor}x)")
                from .providers import upscale
                upscaled_img = upscale.upscale(base_img, cfg.upscale.provider, cfg.upscale.model, cfg.upscale.factor)
                if upscaled_img is not None:
                    base_img = upscaled_img
                    progress_logger.success("Image upscaled successfully")
                else:
                    logger.warning("Upscaling failed, using original image")
                    progress_logger.warning("Upscaling failed, using original")
            else:
                logger.info("Upscaling disabled, using original image")
                progress_logger.substep("Upscaling disabled")

            # Step 8: Create wallpaper variants
            progress_logger.step("Creating wallpaper variants")
            from .imaging import variants
            wallpapers = variants.make_wallpaper_variants(base_img, cfg.wallpaper_variants)
            logger.info(f"Created {len(wallpapers)} wallpaper variants: {list(wallpapers.keys())}")
            progress_logger.success(f"Created {len(wallpapers)} wallpaper variants")

            # Step 9: Generate alt text and save files
            progress_logger.step("Saving files and metadata")
            alt = prompts.make_alt_text(base_prompt, winner["prompt"], cfg)
            logger.debug(f"Alt text generated: {alt[:100]}...")
            
            public_paths = storage.fs.save_images(out_dir, wallpapers, base_img)
            logger.info(f"Images saved to: {out_dir}")
            
            # Save all original candidate images in candidates subfolder
            candidate_paths = storage.fs.save_candidate_images(out_dir, candidates)
            logger.info(f"Saved {len(candidate_paths)} candidate images to candidates subfolder")
            
            ph = phash.phash_hex(wallpapers[next(iter(wallpapers))])
            
            meta = {
                "theme_hint": theme_hint,
                "base_prompt": base_prompt,
                "variant_prompt": winner["prompt"],
                "provider": winner["provider"],
                "model": winner["model"],
                "seed": winner.get("seed"),
                "created_at": now_iso(),
                "alt_text": alt,
                "phash": ph,
                "human_selection": {
                    "enabled": True,
                    "method": "discord_dm_raw_batches",
                    "selected_rank_raw": user_rank,
                    "timeout_fallback": timeout_fallback
                },
                "files": public_paths,
                "tweet_id": None
            }
            storage.fs.save_meta(out_dir, meta)
            
            manifest.append({
                "id": f"{date_str}_{theme_hint}_{slug}",
                "date": date_str,
                "theme_hint": theme_hint,
                "files": public_paths,
                "tweet_id": None,
                "phash": ph
            })
            
            logger.info("Metadata saved and manifest updated")
            progress_logger.success("Files and metadata saved")

            if dry_run:
                logger.info("Dry run mode - skipping social media posting")
                progress_logger.finish_pipeline(success=True)
                return 0

            # Step 10: Post to social media
            progress_logger.step("Posting to social media")
            # Use base image instead of wallpaper variants for social media posting
            base_img_key = "base_img"
            logger.info(f"Posting base image: {base_img_key}")
            
            # Generate Twitter blurb for human selection path using the actual image
            twitter_blurb = prompts.make_twitter_blurb(theme_hint, fs_abs(public_paths["base_img"]), cfg)
            if twitter_blurb:
                logger.info(f"Twitter blurb generated: {twitter_blurb}")
                # Append blurb to alt text for enhanced accessibility
                alt_with_blurb = f"{alt} {twitter_blurb}"
            else:
                logger.info("No Twitter blurb generated, using alt text only")
                alt_with_blurb = alt
                twitter_blurb = ""  # Ensure empty string for tweet
            
            try:
                media_ids = twitter.client.upload_media([fs_abs(public_paths[base_img_key])])
                logger.debug(f"Media uploaded, ID: {media_ids[0]}")
                
                twitter.client.set_alt_text(media_ids[0], alt_with_blurb)
                logger.debug("Alt text with blurb set for media")
                
                tweet_id = twitter.client.create_tweet(text=twitter_blurb, media_ids=media_ids)
                logger.info(f"Tweet posted successfully with blurb, ID: {tweet_id}")
                
                # Update records with tweet ID
                manifest.update_tweet_id(f"{date_str}_{theme_hint}_{slug}", tweet_id)
                meta["tweet_id"] = tweet_id
                storage.fs.save_meta(out_dir, meta)
                
                tweet_link = tweet_url(tweet_id)
                alerts.webhook.send_success(theme_hint, meta["model"], tweet_link, public_paths[base_img_key], cfg)
                
                progress_logger.success("Posted to social media", f"Tweet ID: {tweet_id}")
                
            except Exception as e:
                logger.error(f"Failed to post to social media: {e}")
                progress_logger.error("Social media posting failed", str(e))
                progress_logger.finish_pipeline(success=False)
                return 1

            progress_logger.finish_pipeline(success=True)
            logger.info("Pipeline completed successfully with human selection")
            return 0

        # Step 5: Score and filter candidates (original path when HITL disabled)
        progress_logger.step("Scoring and filtering candidates")
        scored = []
        filtered_sanity = 0
        filtered_quality = 0
        
        for i, c in enumerate(candidates):
            logger.debug(f"Processing candidate {i+1}/{len(candidates)}")
            
            # Basic metrics
            b = metrics.brightness(c["image"])
            e = metrics.entropy(c["image"])
            
            # Sanity checks
            if not sanity.passes_floors(b, e, cfg):
                filtered_sanity += 1
                logger.debug(f"Candidate {i+1} failed sanity check (brightness: {b:.2f}, entropy: {e:.3f})")
                continue
            
            # Local quality assessment
            passes_local, local_q = quality.evaluate_local(c["image"], cfg)
            if not passes_local:
                filtered_quality += 1
                logger.debug(f"Candidate {i+1} failed quality check (score: {local_q:.4f})")
                continue
            
            c["brightness"] = b
            c["entropy"] = e
            c["local_quality"] = local_q
            scored.append(c)
            logger.debug(f"Candidate {i+1} passed initial scoring (B:{b:.2f}, E:{e:.3f}, Q:{local_q:.4f})")
        
        logger.info(f"Filtering results: {len(scored)} passed, {filtered_sanity} failed sanity, {filtered_quality} failed quality")
        progress_logger.substep(f"Filtered candidates", f"{len(scored)} passed initial scoring")
        
        # Add aesthetic scores
        if scored:
            progress_logger.substep("Computing aesthetic scores")
            logger.debug("Using parallel aesthetic scoring")
            scored = await aesthetic.score_candidates_parallel(scored, cfg, progress_logger)
            
            # Log aesthetic scores
            for i, c in enumerate(scored):
                logger.debug(f"Candidate {i+1} aesthetic score: {c['aesthetic']:.4f}")

        if not scored:
            progress_logger.error("All candidates failed scoring")
            alerts.webhook.send_failure("all candidates failed sanity/scoring", cfg)
            progress_logger.finish_pipeline(success=False)
            return 1

        # Normalize and calculate final scores
        scored = normalize_and_rescore(scored, cfg)
        logger.info(f"Final scoring completed for {len(scored)} candidates")
        
        # Log top candidates
        top_candidates = sorted(scored, key=lambda x: x["final"], reverse=True)[:3]
        for i, c in enumerate(top_candidates):
            logger.info(f"Top candidate {i+1}: final={c['final']:.4f}, aesthetic={c['aesthetic']:.4f}, provider={c['provider']}")
        
        progress_logger.success(f"Scored {len(scored)} candidates")

        # Step 6: Create collage and select winner
        progress_logger.step("Creating collage and selecting winner")
        date_str = today_local()
        slug = storage.paths.make_slug(theme_hint, base_prompt)
        out_dir = storage.paths.output_dir(date_str, slug)
        logger.debug(f"Output directory: {out_dir}")
        
        collage_path = collage.save_collage(scored, out_dir, "candidates_collage.jpg")
        logger.info(f"Collage saved to: {collage_path}")
        progress_logger.substep("Collage created")
        
        # Load recent hashes for duplicate detection
        recent_hashes = manifest.load_recent_hashes(limit=200)
        logger.debug(f"Loaded {len(recent_hashes)} recent hashes for duplicate detection")
        
        # Select winner (avoiding duplicates)
        winner = None
        for i, c in enumerate(sorted(scored, key=lambda x: x["final"], reverse=True)):
            candidate_hash = phash.phash_hex(c["image"])
            if not phash.is_duplicate(candidate_hash, recent_hashes, cfg.ranking.phash_distance_min):
                winner = c
                logger.info(f"Winner selected: rank {i+1}, final_score={c['final']:.4f}, provider={c['provider']}")
                break
            else:
                logger.debug(f"Candidate {i+1} rejected as duplicate (hash: {candidate_hash})")
        
        if winner is None:
            progress_logger.warning("All candidates are near-duplicates")
            alerts.webhook.send_failure("near-duplicate with manifest history", cfg)
            progress_logger.finish_pipeline(success=True)  # Not fatal
            return 0
        
        progress_logger.success("Winner selected", f"score: {winner['final']:.4f}")

        # Step 7: Upscale winner
        progress_logger.step("Upscaling winner")
        base_img = winner["image"]
        if cfg.upscale.enabled:
            logger.info(f"Upscaling with {cfg.upscale.provider}/{cfg.upscale.model} (factor: {cfg.upscale.factor}x)")
            from .providers import upscale
            upscaled_img = upscale.upscale(base_img, cfg.upscale.provider, cfg.upscale.model, cfg.upscale.factor)
            if upscaled_img is not None:
                base_img = upscaled_img
                progress_logger.success("Image upscaled successfully")
            else:
                logger.warning("Upscaling failed, using original image")
                progress_logger.warning("Upscaling failed, using original")
        else:
            logger.info("Upscaling disabled, using original image")
            progress_logger.substep("Upscaling disabled")

        # Step 8: Create wallpaper variants
        progress_logger.step("Creating wallpaper variants")
        from .imaging import variants
        wallpapers = variants.make_wallpaper_variants(base_img, cfg.wallpaper_variants)
        logger.info(f"Created {len(wallpapers)} wallpaper variants: {list(wallpapers.keys())}")
        progress_logger.success(f"Created {len(wallpapers)} wallpaper variants")

        # Step 9: Generate alt text, Twitter blurb, and save files
        progress_logger.step("Generating content and saving files")
        alt = prompts.make_alt_text(base_prompt, winner["prompt"], cfg)
        logger.debug(f"Alt text generated: {alt[:100]}...")
        
        # Generate Twitter blurb using the actual image
        twitter_blurb = prompts.make_twitter_blurb(theme_hint, base_img, cfg)
        if twitter_blurb:
            logger.info(f"Twitter blurb generated: {twitter_blurb}")
            # Append blurb to alt text for enhanced accessibility
            alt_with_blurb = f"{alt} {twitter_blurb}"
        else:
            logger.info("No Twitter blurb generated, using alt text only")
            alt_with_blurb = alt
            twitter_blurb = ""  # Ensure empty string for tweet
        
        public_paths = storage.fs.save_images(out_dir, wallpapers, base_img)
        logger.info(f"Images saved to: {out_dir}")
        
        # Save all original candidate images in candidates subfolder
        candidate_paths = storage.fs.save_candidate_images(out_dir, scored)
        logger.info(f"Saved {len(candidate_paths)} candidate images to candidates subfolder")
        
        ph = phash.phash_hex(wallpapers[next(iter(wallpapers))])
        
        meta = {
            "theme_hint": theme_hint,
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
                "local_quality": round(winner["local_quality"], 4),
                "final": round(winner["final"], 4)
            },
            "files": public_paths,
            "tweet_id": None
        }
        storage.fs.save_meta(out_dir, meta)
        
        manifest.append({
            "id": f"{date_str}_{theme_hint}_{slug}",
            "date": date_str,
            "theme_hint": theme_hint,
            "files": public_paths,
            "tweet_id": None,
            "phash": ph
        })
        
        logger.info("Metadata saved and manifest updated")
        progress_logger.success("Files and metadata saved")

        if dry_run:
            logger.info("Dry run mode - skipping social media posting")
            progress_logger.finish_pipeline(success=True)
            return 0

        # Step 10: Post to social media
        progress_logger.step("Posting to social media")
        # Use base image instead of wallpaper variants for social media posting
        base_img_key = "base_img"
        logger.info(f"Posting base image: {base_img_key}")
        
        try:
            media_ids = twitter.client.upload_media([fs_abs(public_paths[base_img_key])])
            logger.debug(f"Media uploaded, ID: {media_ids[0]}")
            
            twitter.client.set_alt_text(media_ids[0], alt_with_blurb)
            logger.debug("Alt text with blurb set for media")
            
            tweet_id = twitter.client.create_tweet(text=twitter_blurb, media_ids=media_ids)
            logger.info(f"Tweet posted successfully with blurb, ID: {tweet_id}")
            
            # Update records with tweet ID
            manifest.update_tweet_id(f"{date_str}_{theme_hint}_{slug}", tweet_id)
            meta["tweet_id"] = tweet_id
            storage.fs.save_meta(out_dir, meta)
            
            tweet_link = tweet_url(tweet_id)
            alerts.webhook.send_success(theme_hint, meta["model"], tweet_link, public_paths[base_img_key], cfg)
            
            progress_logger.success("Posted to social media", f"Tweet ID: {tweet_id}")
            
        except Exception as e:
            logger.error(f"Failed to post to social media: {e}")
            progress_logger.error("Social media posting failed", str(e))
            progress_logger.finish_pipeline(success=False)
            return 1

        progress_logger.finish_pipeline(success=True)
        logger.info("Pipeline completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}", exc_info=True)
        progress_logger.error("Pipeline failed", str(e))
        progress_logger.finish_pipeline(success=False)
        return 1
