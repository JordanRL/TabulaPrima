#!/usr/bin/env python
"""
Dataset Cache Manager - A utility to manage tokenized dataset caches
"""

import os
import sys
import pickle
import argparse
import hashlib
from datetime import datetime
import humanize
from tabulate import tabulate


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    @staticmethod
    def header(text):
        return f"{Colors.HEADER}{Colors.BOLD}{text}{Colors.ENDC}"

    @staticmethod
    def info(text):
        return f"{Colors.BLUE}{text}{Colors.ENDC}"

    @staticmethod
    def success(text):
        return f"{Colors.GREEN}{text}{Colors.ENDC}"

    @staticmethod
    def warning(text):
        return f"{Colors.YELLOW}{text}{Colors.ENDC}"

    @staticmethod
    def error(text):
        return f"{Colors.RED}{text}{Colors.ENDC}"

    @staticmethod
    def highlight(text):
        return f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.ENDC}"


def parse_args():
    parser = argparse.ArgumentParser(description="Manage tokenized dataset caches")
    parser.add_argument("--cache-dir", type=str, default="dataset_cache",
                        help="Directory containing cached datasets")

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List command
    list_parser = subparsers.add_parser("list", help="List all cached datasets")
    list_parser.add_argument("--detail", action="store_true", help="Show detailed information")

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete specific cache files")
    delete_parser.add_argument("--hash", type=str, help="MD5 hash of cache file to delete")
    delete_parser.add_argument("--all", action="store_true", help="Delete all cache files")
    delete_parser.add_argument("--older-than", type=int, help="Delete cache files older than N days")

    # Info command
    info_parser = subparsers.add_parser("info", help="Show detailed information about a specific cache file")
    info_parser.add_argument("hash", type=str, help="MD5 hash of cache file to inspect")

    return parser.parse_args()


def list_caches(cache_dir, detailed=False):
    """List all cached datasets"""
    if not os.path.exists(cache_dir):
        print(Colors.error(f"Cache directory does not exist: {cache_dir}"))
        return

    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]

    if not cache_files:
        print(Colors.warning("No cache files found."))
        return

    cache_info = []
    total_size = 0

    for file in cache_files:
        file_path = os.path.join(cache_dir, file)
        file_size = os.path.getsize(file_path)
        total_size += file_size

        # Get file creation/modification time
        mtime = os.path.getmtime(file_path)
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        # Extract hash from filename
        file_hash = file.split('.')[0]

        # Try to extract dataset info by loading cache
        dataset_info = "Unknown"
        examples_count = 0

        if detailed:
            try:
                with open(file_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    examples_count = len(cache_data['examples'])

                    # Try to infer dataset from the first example
                    if examples_count > 0:
                        # This is a very crude way to guess the dataset
                        # A better approach would be to store metadata in the cache
                        sample_len = len(cache_data['examples'][0])
                        tokens_str = str(cache_data['examples'][0][:10].tolist())
                        if "wikitext" in file_path.lower():
                            dataset_info = "WikiText"
                        elif sample_len > 1000:
                            dataset_info = "Long-context dataset"
                        else:
                            dataset_info = "Unknown dataset"

            except Exception as e:
                dataset_info = f"Error: {str(e)}"

        readable_size = humanize.naturalsize(file_size)

        cache_info.append({
            "Hash": file_hash[:8] + "...",
            "Size": readable_size,
            "Date": mtime_str,
            "Dataset": dataset_info if detailed else "N/A",
            "Examples": examples_count if detailed else "N/A"
        })

    # Sort by modification time (newest first)
    cache_info.sort(key=lambda x: x["Date"], reverse=True)

    # Print table
    headers = ["Hash", "Size", "Date"]
    if detailed:
        headers.extend(["Dataset", "Examples"])

    print(Colors.header(f"\n{'=' * 50}"))
    print(Colors.header(f" Dataset Cache Files"))
    print(Colors.header(f"{'=' * 50}"))

    print(tabulate(
        [list(info.values()) for info in cache_info],
        headers=headers,
        tablefmt="grid"
    ))

    # Print summary
    print(f"\nTotal cache size: {Colors.highlight(humanize.naturalsize(total_size))}")
    print(f"Cache files: {Colors.highlight(str(len(cache_files)))}")
    print(f"Cache directory: {Colors.highlight(os.path.abspath(cache_dir))}")


def delete_caches(cache_dir, file_hash=None, delete_all=False, older_than=None):
    """Delete specific cache files"""
    if not os.path.exists(cache_dir):
        print(Colors.error(f"Cache directory does not exist: {cache_dir}"))
        return

    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl")]

    if not cache_files:
        print(Colors.warning("No cache files found."))
        return

    files_to_delete = []

    # Filter files to delete based on criteria
    if delete_all:
        files_to_delete = cache_files
        print(Colors.warning(f"Preparing to delete ALL cache files ({len(files_to_delete)})"))
    elif file_hash:
        # Find files that match the hash prefix
        matching_files = [f for f in cache_files if f.startswith(file_hash)]
        if not matching_files:
            print(Colors.error(f"No cache files found with hash: {file_hash}"))
            return
        files_to_delete = matching_files
        print(Colors.warning(f"Preparing to delete {len(files_to_delete)} cache files matching hash: {file_hash}"))
    elif older_than:
        # Find files older than N days
        cutoff_time = datetime.now().timestamp() - (older_than * 24 * 60 * 60)
        older_files = [f for f in cache_files if os.path.getmtime(os.path.join(cache_dir, f)) < cutoff_time]
        if not older_files:
            print(Colors.warning(f"No cache files found older than {older_than} days."))
            return
        files_to_delete = older_files
        print(Colors.warning(f"Preparing to delete {len(files_to_delete)} cache files older than {older_than} days"))

    # Request confirmation
    if files_to_delete:
        print("\nFiles to be deleted:")
        for i, file in enumerate(files_to_delete[:5], 1):
            file_path = os.path.join(cache_dir, file)
            file_size = humanize.naturalsize(os.path.getsize(file_path))
            mtime = datetime.fromtimestamp(os.path.getmtime(file_path)).strftime("%Y-%m-%d %H:%M:%S")
            print(f"  {i}. {file} ({file_size}, {mtime})")

        if len(files_to_delete) > 5:
            print(f"  ... and {len(files_to_delete) - 5} more files")

        confirm = input("\nAre you sure you want to delete these files? (yes/no): ")
        if confirm.lower() != "yes":
            print(Colors.info("Deletion cancelled."))
            return

        # Delete the files
        deleted_count = 0
        deleted_size = 0
        for file in files_to_delete:
            file_path = os.path.join(cache_dir, file)
            file_size = os.path.getsize(file_path)
            try:
                os.remove(file_path)
                deleted_count += 1
                deleted_size += file_size
                print(f"  Deleted: {file}")
            except Exception as e:
                print(Colors.error(f"  Error deleting {file}: {str(e)}"))

        print(Colors.success(f"\n✓ Successfully deleted {deleted_count} files ({humanize.naturalsize(deleted_size)})"))


def show_cache_info(cache_dir, file_hash):
    """Show detailed information about a specific cache file"""
    if not os.path.exists(cache_dir):
        print(Colors.error(f"Cache directory does not exist: {cache_dir}"))
        return

    # Find the cache file that matches the hash
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".pkl") and f.startswith(file_hash)]

    if not cache_files:
        print(Colors.error(f"No cache file found with hash: {file_hash}"))
        return

    if len(cache_files) > 1:
        print(Colors.warning(f"Multiple cache files found with hash prefix: {file_hash}"))
        print("Please be more specific:")
        for file in cache_files:
            print(f"  - {file}")
        return

    # Load and analyze the cache file
    file_path = os.path.join(cache_dir, cache_files[0])
    file_size = os.path.getsize(file_path)
    mtime = os.path.getmtime(file_path)
    mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

    print(Colors.header(f"\n{'=' * 50}"))
    print(Colors.header(f" Cache File Information"))
    print(Colors.header(f"{'=' * 50}"))

    print(f"File: {Colors.highlight(cache_files[0])}")
    print(f"Path: {Colors.highlight(os.path.abspath(file_path))}")
    print(f"Size: {Colors.highlight(humanize.naturalsize(file_size))}")
    print(f"Last modified: {Colors.highlight(mtime_str)}")

    # Try to load and analyze the cache contents
    try:
        with open(file_path, 'rb') as f:
            cache_data = pickle.load(f)

        examples = cache_data.get('examples', [])
        total_tokens = cache_data.get('total_tokens', 0)

        print(f"\nContents:")
        print(f"  • Number of examples: {Colors.highlight(f'{len(examples):,}')}")
        print(f"  • Total tokens: {Colors.highlight(f'{total_tokens:,}')}")

        if examples:
            # Analyze token length distribution
            lengths = [len(ex) for ex in examples]
            avg_len = sum(lengths) / len(lengths)
            min_len = min(lengths)
            max_len = max(lengths)
            print(f"  • Average tokens per example: {Colors.highlight(f'{avg_len:.1f}')}")
            print(f"  • Min tokens per example: {Colors.highlight(str(min_len))}")
            print(f"  • Max tokens per example: {Colors.highlight(str(max_len))}")

            # Display some info about the first example
            print(f"\nSample data (first example):")
            sample = examples[0]
            print(f"  • Length: {Colors.highlight(str(len(sample)))}")
            print(f"  • First 10 tokens: {Colors.highlight(str(sample[:10].tolist()))}")
            print(f"  • Last 10 tokens: {Colors.highlight(str(sample[-10:].tolist()))}")

    except Exception as e:
        print(Colors.error(f"\nError analyzing cache file: {str(e)}"))


def main():
    args = parse_args()

    if args.command == "list":
        list_caches(args.cache_dir, args.detail)
    elif args.command == "delete":
        delete_caches(args.cache_dir, args.hash, args.all, args.older_than)
    elif args.command == "info":
        show_cache_info(args.cache_dir, args.hash)
    else:
        # If no command specified, show the list by default
        list_caches(args.cache_dir, False)


if __name__ == "__main__":
    main()