"""
Generate embeddings using Bacformer with Dual GPU Support
This script splits the workload across two GPUs for parallel processing.

Environment: bacformer (conda activate bacformer)
"""

import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import json
import multiprocessing as mp
from multiprocessing import Process, Queue
import time

def load_sampled_genome_ids(sample_file):
    """Load list of sampled genome IDs"""
    with open(sample_file, 'r') as f:
        genome_ids = [line.strip() for line in f if line.strip()]
    return genome_ids

def read_faa(faa_file, max_proteins=6000):
    """Read faa file and extract protein sequences"""
    proteins = []
    for record in SeqIO.parse(faa_file, "fasta"):
        seq = str(record.seq)
        if seq.endswith('*'):
            seq = seq[:-1]
        proteins.append(seq)
    
    if len(proteins) > max_proteins:
        proteins = proteins[:max_proteins]
    
    return proteins

def get_bacformer_embedding(protein_sequences, model, device, batch_size=128, 
                            max_n_proteins=6000, pooling='mean'):
    """Generate genome embedding using Bacformer"""
    from bacformer.pp import protein_seqs_to_bacformer_inputs
    
    try:
        inputs = protein_seqs_to_bacformer_inputs(
            protein_sequences,
            device=device,
            batch_size=batch_size,
            max_n_proteins=max_n_proteins,
        )
        
        with torch.no_grad():
            outputs = model(**inputs, return_dict=True)
        
        last_hidden_state = outputs["last_hidden_state"]
        
        if pooling == 'mean':
            genome_embedding = last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
        elif pooling == 'max':
            genome_embedding = last_hidden_state.max(dim=1)[0].squeeze(0).cpu().numpy()
        elif pooling == 'cls':
            genome_embedding = last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        return genome_embedding
    
    except Exception as e:
        print(f"  Error generating embedding: {e}")
        return None

def worker_process(gpu_id, genome_ids, faa_dir, output_dir, model_name, 
                   max_n_proteins, batch_size, genome_pooling, result_queue):
    """Worker process for a single GPU"""
    try:
        device = torch.device(f"cuda:{gpu_id}")
        
        # Load model on this GPU
        from transformers import AutoModel
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        model = model.to(device).eval()
        
        processed = 0
        failed = []
        protein_counts = []
        
        for genome_id in tqdm(genome_ids, desc=f"GPU {gpu_id}", position=gpu_id):
            output_file = f"{genome_id}.npy"
            output_path = os.path.join(output_dir, output_file)
            
            # Skip if already exists
            if os.path.exists(output_path):
                continue
            
            faa_file = os.path.join(faa_dir, f"{genome_id}.faa")
            
            if not os.path.exists(faa_file):
                failed.append(genome_id)
                continue
            
            try:
                protein_sequences = read_faa(faa_file, max_proteins=max_n_proteins)
                
                if len(protein_sequences) == 0:
                    failed.append(genome_id)
                    continue
                
                protein_counts.append(len(protein_sequences))
                
                embedding = get_bacformer_embedding(
                    protein_sequences, model, device,
                    batch_size=batch_size,
                    max_n_proteins=max_n_proteins,
                    pooling=genome_pooling
                )
                
                if embedding is None:
                    failed.append(genome_id)
                    continue
                
                np.save(output_path, embedding)
                processed += 1
                
            except Exception as e:
                failed.append(genome_id)
                continue
        
        result_queue.put({
            'gpu_id': gpu_id,
            'processed': processed,
            'failed': failed,
            'protein_counts': protein_counts
        })
        
    except Exception as e:
        result_queue.put({
            'gpu_id': gpu_id,
            'error': str(e),
            'processed': 0,
            'failed': genome_ids,
            'protein_counts': []
        })

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate embeddings using Bacformer with Dual GPU Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--faa_dir",
        type=str,
        default="../CNGBdb_faa",
        help="Directory containing ORF sequence (*.faa files)"
    )
    parser.add_argument(
        "--sample_file",
        type=str,
        default="../sampled_genomes.all.txt",
        help="Path to file containing sampled genome IDs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../embeddings/bacformer",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="macwiatrak/bacformer-masked-complete-genomes",
        help="Bacformer model name or path"
    )
    parser.add_argument(
        "--max_n_proteins",
        type=int,
        default=6000,
        help="Maximum number of proteins per genome"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--genome_pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="Pooling method for genome-level embedding"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1' or '0')"
    )
    
    args = parser.parse_args()
    
    # Parse GPU list
    try:
        args.use_gpus = [int(gpu.strip()) for gpu in args.gpus.split(',')]
    except ValueError:
        raise ValueError(f"Invalid GPU format: {args.gpus}. Expected comma-separated integers (e.g., '0,1')")
    
    return args

def main():
    args = parse_args()
    
    print("="*80)
    print("GENERATING EMBEDDINGS WITH BACFORMER (DUAL GPU)")
    print("="*80)
    
    # Check GPUs
    print(f"\nUsing GPUs: {args.use_gpus}")
    for gpu_id in args.use_gpus:
        if torch.cuda.is_available():
            print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nOutput directory: {args.output_dir}")
    
    # Load sampled genome IDs
    print(f"\nLoading sampled genome IDs from {args.sample_file}...")
    genome_ids = load_sampled_genome_ids(args.sample_file)
    print(f"Number of genomes to process: {len(genome_ids)}")
    
    # Check existing files
    existing_files = set(os.listdir(args.output_dir)) if os.path.exists(args.output_dir) else set()
    existing_genomes = set([f.replace('.npy', '') for f in existing_files if f.endswith('.npy')])
    genome_ids_to_process = [gid for gid in genome_ids if gid not in existing_genomes]
    
    print(f"Existing embedding files: {len(existing_genomes)}")
    print(f"Remaining to process: {len(genome_ids_to_process)}")
    
    if len(genome_ids_to_process) == 0:
        print("\n✓ All embeddings already generated!")
        return
    
    # Split genome IDs between GPUs
    n_gpus = len(args.use_gpus)
    genome_splits = [genome_ids_to_process[i::n_gpus] for i in range(n_gpus)]
    
    print(f"\nSplitting workload:")
    for i, split in enumerate(genome_splits):
        print(f"  GPU {args.use_gpus[i]}: {len(split)} genomes")
    
    print(f"\nStarting parallel processing...")
    print("NOTE: First time may take 5-10 minutes to download Bacformer model (~2.5 GB)")
    
    start_time = time.time()
    
    # Create result queue
    result_queue = Queue()
    
    # Start worker processes
    processes = []
    for i, gpu_id in enumerate(args.use_gpus):
        p = Process(target=worker_process, args=(
            gpu_id, genome_splits[i], args.faa_dir, args.output_dir, args.model_name,
            args.max_n_proteins, args.batch_size, args.genome_pooling, result_queue
        ))
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join()
    
    # Collect results
    results = []
    while not result_queue.empty():
        results.append(result_queue.get())
    
    elapsed_time = time.time() - start_time
    
    # Aggregate statistics
    total_processed = sum(r['processed'] for r in results)
    total_failed = []
    all_protein_counts = []
    
    for r in results:
        if 'error' in r:
            print(f"\n✗ GPU {r['gpu_id']} encountered error: {r['error']}")
        total_failed.extend(r.get('failed', []))
        all_protein_counts.extend(r.get('protein_counts', []))
    
    print(f"\n{'='*80}")
    print("✓ EMBEDDING GENERATION COMPLETED")
    print(f"{'='*80}")
    print(f"Total genomes: {len(genome_ids)}")
    print(f"Processed: {total_processed}")
    print(f"Already existed: {len(existing_genomes)}")
    print(f"Failed: {len(total_failed)}")
    print(f"Time elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    if total_processed > 0:
        print(f"Average time per sample: {elapsed_time/total_processed:.2f} seconds")
        print(f"Average proteins per genome: {np.mean(all_protein_counts):.0f}")
    
    # Verify output
    output_files = [f for f in os.listdir(args.output_dir) if f.endswith('.npy')]
    print(f"Total output files: {len(output_files)}")
    
    if output_files:
        sample_emb = np.load(os.path.join(args.output_dir, output_files[0]))
        print(f"Embedding shape: {sample_emb.shape}")
    
    # Save statistics
    stats = {
        'total_samples': len(genome_ids),
        'processed': total_processed,
        'already_existed': len(existing_genomes),
        'failed': len(total_failed),
        'elapsed_time_hours': elapsed_time/3600,
        'avg_proteins_per_genome': float(np.mean(all_protein_counts)) if all_protein_counts else 0,
        'gpus_used': args.use_gpus,
    }
    
    stats_file = os.path.join(args.output_dir, 'generation_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")
    
    if total_failed:
        failed_file = os.path.join(args.output_dir, 'failed_samples.txt')
        with open(failed_file, 'w') as f:
            for name in total_failed:
                f.write(f"{name}\n")
        print(f"Failed samples saved to: {failed_file}")
    
    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()

""" Run Example:
python s2_generate_bacformer_embeddings_dual_gpu.py \
    --faa_dir /path/to/faa \
    --output_dir /path/to/output \
    --batch_size 256 \
    --gpus 0,1
"""
