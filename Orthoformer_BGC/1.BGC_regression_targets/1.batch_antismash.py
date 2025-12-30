#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch run antiSMASH using Docker with checkpointing support.
"""

import os
import argparse
import subprocess
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def run_antismash_for_file(fna_file, input_dir, output_dir, cpus):
    """
    Run antiSMASH for a single .fna file.

    Args:
        fna_file: Name of the .fna file.
        input_dir: Directory containing the input file.
        output_dir: Output directory for antiSMASH results.
        cpus: Number of CPUs to allocate for antiSMASH.

    Returns:
        Status string indicating success, skip, or failure.
    """
    full_input_path = os.path.join(input_dir, fna_file)
    sample_name = os.path.splitext(fna_file)[0]
    specific_output_dir = os.path.join(output_dir, sample_name)

    success_flag = os.path.join(specific_output_dir, 'index.html')
    if os.path.exists(success_flag):
        logging.info(f"Skipping '{fna_file}' as results already exist in '{specific_output_dir}'.")
        return f"Skipped: {fna_file}"

    logging.info(f"Processing '{fna_file}'...")
    os.makedirs(specific_output_dir, exist_ok=True)

    command = [
        "docker", "run",
        "--rm",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--volume", f"{os.path.abspath(input_dir)}:/input:ro",
        "--volume", f"{os.path.abspath(specific_output_dir)}:/output:rw",
        "docker.io/antismash/standalone:8.0.2",
        fna_file,
        "--output-dir", "/output",
        "--cpus", str(cpus),
        "--genefinding-tool", "prodigal",
        "--taxon", "bacteria"
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info(f"Successfully processed '{fna_file}'.")
        logging.debug(f"STDOUT for {fna_file}:\n{result.stdout}")
        return f"Success: {fna_file}"
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to process '{fna_file}'.")
        logging.error(f"Command: {' '.join(command)}")
        logging.error(f"Stderr: {e.stderr}")
        return f"Failed: {fna_file}"
    except Exception as e:
        logging.error(f"An unexpected error occurred while processing '{fna_file}': {e}")
        return f"Error: {fna_file}"


def main():
    """Main function for argument parsing and batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch run antiSMASH using Docker with checkpointing.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Input directory containing .fna files.")
    parser.add_argument("-o", "--output", required=True,
                        help="Main output directory for antiSMASH results.")
    parser.add_argument("--cpus", type=int, default=1,
                        help="Number of CPUs to allocate for each antiSMASH job.")
    parser.add_argument("--jobs", type=int, default=cpu_count(),
                        help="Number of parallel jobs to run.")

    args = parser.parse_args()

    if not os.path.isdir(args.input):
        logging.error(f"Input directory not found: {args.input}")
        return

    os.makedirs(args.output, exist_ok=True)

    try:
        fna_files = [f for f in os.listdir(args.input) if f.endswith('.fna')]
        if not fna_files:
            logging.warning(f"No .fna files found in {args.input}")
            return
        logging.info(f"Found {len(fna_files)} .fna files to process.")
    except OSError as e:
        logging.error(f"Could not read input directory {args.input}: {e}")
        return

    task_func = partial(
        run_antismash_for_file,
        input_dir=args.input,
        output_dir=args.output,
        cpus=args.cpus
    )

    with Pool(processes=args.jobs) as pool:
        results = pool.map(task_func, fna_files)

    logging.info("Batch processing complete.")
    logging.info("Summary:")
    success_count = sum(1 for r in results if r.startswith("Success"))
    skipped_count = sum(1 for r in results if r.startswith("Skipped"))
    failed_count = sum(1 for r in results if r.startswith("Failed") or r.startswith("Error"))

    logging.info(f"  - Successfully processed: {success_count}")
    logging.info(f"  - Skipped (already done): {skipped_count}")
    logging.info(f"  - Failed/Errors: {failed_count}")


if __name__ == "__main__":
    main()
