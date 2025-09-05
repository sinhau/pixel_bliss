# PixelBliss

PixelBliss is a stateless CLI tool that automates posting highly aesthetic images to X/Twitter. It generates one image per run, saves wallpaper variants to a VPS, and maintains a manifest for future website use.

## Features

- **Configurable Prompt Generation**: Default OpenAI GPT-5, with pluggable providers.
- **Image Generation**: FAL AI (preferred) or Replicate (fallback).
- **Ranking**: Brightness, entropy, and aesthetic scoring.
- **Upscaling**: Optional upscaling before generating wallpaper variants.
- **Twitter Posting**: Posts with alt text only.
- **Storage**: Saves to static web root with manifest.
- **Alerts**: Webhook notifications on success/failure.

## Setup

1. Clone the repo.
2. Install dependencies: `pip install -r requirements.txt` (create this with listed deps).
3. Copy `.env.example` to `.env` and fill in API keys.
4. Configure `config.yaml` as needed.
5. Run `python main.py post-once` to test.

## Dependencies

- requests
- python-dotenv
- pydantic
- Pillow
- imagehash
- tenacity
- openai
- tweepy
- replicate
- fal_client (hypothetical)
- pytz

## Usage

- `python main.py post-once`: Run the full pipeline.
- `python main.py dry-run`: Run without posting to Twitter.
- `python main.py repair-manifest`: Rebuild manifest from outputs.

## Cron Example

```
0 10-22/3 * * * cd /path/to/pixelbliss && python main.py post-once >> log.txt 2>&1
```

## Configuration

See `config.yaml` for all options.

## Project Structure

As per `PROJECT_PLAN.md`.
