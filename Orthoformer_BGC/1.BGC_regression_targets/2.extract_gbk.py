#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract GBK files from antiSMASH output directories.
"""

import os
import shutil
import argparse
from pathlib import Path


def extract_gbk_files(source_dir: str, output_dir: str) -> None:
    """
    Extract .gbk files from antiSMASH output subdirectories.

    Args:
        source_dir: Path to antiSMASH output directory containing sample subdirectories.
        output_dir: Path to output directory for extracted GBK files.
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)

    if not source_path.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return

    output_path.mkdir(parents=True, exist_ok=True)

    copied_count = 0
    skipped_count = 0

    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue

        basename = folder.name
        gbk_file = folder / f"{basename}.gbk"

        if gbk_file.exists():
            dest_file = output_path / f"{basename}.gbk"
            shutil.copy2(gbk_file, dest_file)
            print(f"Copied: {gbk_file.name}")
            copied_count += 1
        else:
            print(f"Warning: {gbk_file.name} not found")
            skipped_count += 1

    print(f"\nExtraction complete.")
    print(f"  Copied: {copied_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract GBK files from antiSMASH output directories."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to antiSMASH output directory.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to output directory for GBK files.")
    args = parser.parse_args()

    extract_gbk_files(args.input, args.output)


if __name__ == "__main__":
    main()
