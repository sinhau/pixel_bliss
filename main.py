#!/usr/bin/env python3
import sys
import asyncio
import argparse
from dotenv import load_dotenv
from pixelbliss.run_once import post_once
from pixelbliss.logging_config import setup_logging

# Load environment variables from .env file
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="PixelBliss CLI")
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Set logging level')
    parser.add_argument('--log-file', help='Optional log file path')
    parser.add_argument('--no-colors', action='store_true', help='Disable colored output')
    
    subparsers = parser.add_subparsers(dest='command')
    subparsers.add_parser('post-once', help='Execute the full pipeline once')
    subparsers.add_parser('dry-run', help='Run everything except the X post')
    subparsers.add_parser('repair-manifest', help='Re-scan outputs and rebuild manifest')

    args = parser.parse_args()

    # Setup logging
    logger, progress_logger = setup_logging(
        level=args.log_level,
        log_file=args.log_file,
        enable_colors=not args.no_colors
    )

    if args.command == 'post-once':
        logger.info("Starting PixelBliss post-once command")
        sys.exit(asyncio.run(post_once(dry_run=False, logger=logger, progress_logger=progress_logger)))
    elif args.command == 'dry-run':
        logger.info("Starting PixelBliss dry-run command")
        sys.exit(asyncio.run(post_once(dry_run=True, logger=logger, progress_logger=progress_logger)))
    elif args.command == 'repair-manifest':
        logger.warning("Repair manifest not implemented yet")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
