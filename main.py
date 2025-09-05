#!/usr/bin/env python3
import sys
import argparse
from pixelbliss.run_once import post_once

def main():
    parser = argparse.ArgumentParser(description="PixelBliss CLI")
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('post-once', help='Execute the full pipeline once')
    subparsers.add_parser('dry-run', help='Run everything except the X post')
    subparsers.add_parser('repair-manifest', help='Re-scan outputs and rebuild manifest')

    args = parser.parse_args()

    if args.command == 'post-once':
        sys.exit(post_once())
    elif args.command == 'dry-run':
        # Implement dry run, similar but skip posting
        print("Dry run not implemented yet")
        sys.exit(0)
    elif args.command == 'repair-manifest':
        # Implement repair
        print("Repair manifest not implemented yet")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()
