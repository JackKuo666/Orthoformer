"""
Generate Evo embeddings for multi-level samples (2232 genomes)
Uses GPU 1 and evo-1-8k-base model
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# Memory optimization
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def load_evo_model(model_name, device):
    """Load Evo model"""
    print(f"Loading Evo model: {model_name}")
    print(f"Target device: {device}")
    
    from evo import Evo
    
    # Load model
    evo_model = Evo(model_name)
    model, tokenizer = evo_model.model, evo_model.tokenizer
    
    # Move to device
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully on {device}")
    print(f"Model dtype: {next(model.parameters()).dtype}")
    
    return model, tokenizer

def read_fna_file(fna_path, max_length=None):
    """Read DNA sequence from .fna file"""
    sequences = []
    current_seq = []
    
    with open(fna_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_seq:
                    sequences.append(''.join(current_seq))
                    current_seq = []
            else:
                current_seq.append(line.upper())
        
        if current_seq:
            sequences.append(''.join(current_seq))
    
    # Concatenate all sequences
    full_sequence = ''.join(sequences)
    
    # Truncate if necessary
    if max_length and len(full_sequence) > max_length:
        # Random sampling from the genome
        start_pos = np.random.randint(0, len(full_sequence) - max_length + 1)
        full_sequence = full_sequence[start_pos:start_pos + max_length]
    
    return full_sequence

def generate_embedding(model, tokenizer, sequence, device, max_length=8192, genome_pooling='mean'):
    """Generate embedding for a DNA sequence"""
    # Tokenize using Evo's tokenizer
    input_ids = torch.tensor(
        tokenizer.tokenize(sequence),
        dtype=torch.int,
    ).to(device).unsqueeze(0)
    
    # Truncate if needed
    if input_ids.size(1) > max_length:
        input_ids = input_ids[:, :max_length]
    
    # Inference
    with torch.no_grad():
        try:
            logits, _ = model(input_ids)
            
            # Convert to float32 if bfloat16 (NumPy doesn't support bfloat16)
            if logits.dtype == torch.bfloat16:
                logits = logits.float()
            
            # Genome-level pooling
            if genome_pooling == 'mean':
                embedding = logits.mean(dim=1).squeeze(0)
            elif genome_pooling == 'max':
                embedding = logits.max(dim=1)[0].squeeze(0)
            elif genome_pooling == 'cls':
                embedding = logits[:, 0, :].squeeze(0)
            elif genome_pooling == 'last':
                embedding = logits[:, -1, :].squeeze(0)
            else:
                embedding = logits.mean(dim=1).squeeze(0)
            
            # Convert to numpy
            embedding = embedding.cpu().numpy()
            
            return embedding
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  WARNING: OOM error, clearing cache and retrying...")
                torch.cuda.empty_cache()
                # Try again with shorter sequence
                if len(sequence) > 4096:
                    sequence = sequence[:4096]
                    return generate_embedding(model, tokenizer, sequence, device, max_length, genome_pooling)
                else:
                    raise e
            else:
                raise e

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Generate Evo embeddings for multi-level samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--fna_dir",
        type=str,
        default="../CNGBdb_fna",
        help="Directory containing genome sequences (*.fna files)"
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
        default="../embeddings/evo",
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="evo-1-131k-base",
        help="Evo model name"
    )
    parser.add_argument(
        "--device_id",
        type=int,
        default=1,
        help="GPU device ID to use (use -1 for CPU)"
    )
    parser.add_argument(
        "--max_sequence_length",
        type=int,
        default=8192,
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--genome_pooling",
        type=str,
        default="mean",
        choices=["mean", "max", "cls", "last"],
        help="Pooling method for genome-level embedding"
    )
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("="*80)
    print("GENERATING EVO EMBEDDINGS FOR MULTI-LEVEL SAMPLES")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Device: GPU {args.device_id}")
    print(f"Max sequence length: {args.max_sequence_length}")
    print(f"Sample file: {args.sample_file}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)
    
    # Setup device
    if args.device_id < 0 or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device_id}")
    
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available() and args.device_id >= 0:
        print(f"GPU {args.device_id}: {torch.cuda.get_device_name(args.device_id)}")
        print(f"Available memory: {torch.cuda.get_device_properties(args.device_id).total_memory / 1024**3:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print("\n" + "="*80)
    print("LOADING MODEL")
    print("="*80)
    model, tokenizer = load_evo_model(args.model_name, device)
    
    # Clear cache after loading
    torch.cuda.empty_cache()
    
    # Load sample list
    print("\n" + "="*80)
    print("LOADING SAMPLE LIST")
    print("="*80)
    with open(args.sample_file, 'r') as f:
        genome_ids = [line.strip() for line in f if line.strip()]
    
    print(f"Total samples: {len(genome_ids)}")
    
    # Check existing embeddings
    existing_embeddings = set()
    for fname in os.listdir(args.output_dir):
        if fname.endswith('.npy'):
            existing_embeddings.add(fname.replace('.npy', ''))
    
    print(f"Existing embeddings: {len(existing_embeddings)}")
    
    # Filter to samples that need processing
    genomes_to_process = [gid for gid in genome_ids if gid not in existing_embeddings]
    print(f"Genomes to process: {len(genomes_to_process)}")
    
    if len(genomes_to_process) == 0:
        print("\n✓ All embeddings already exist!")
        return
    
    # Generate embeddings
    print("\n" + "="*80)
    print("GENERATING EMBEDDINGS")
    print("="*80)
    
    successful = 0
    failed = 0
    failed_genomes = []
    
    for i, genome_id in enumerate(tqdm(genomes_to_process, desc="Processing genomes")):
        try:
            # Find .fna file
            fna_file = os.path.join(args.fna_dir, f"{genome_id}.fna")
            
            if not os.path.exists(fna_file):
                print(f"\n  WARNING: {genome_id}.fna not found, skipping...")
                failed += 1
                failed_genomes.append((genome_id, "File not found"))
                continue
            
            # Read sequence
            sequence = read_fna_file(fna_file, max_length=args.max_sequence_length)
            
            if len(sequence) == 0:
                print(f"\n  WARNING: {genome_id} has empty sequence, skipping...")
                failed += 1
                failed_genomes.append((genome_id, "Empty sequence"))
                continue
            
            # Generate embedding
            embedding = generate_embedding(model, tokenizer, sequence, device, args.max_sequence_length, args.genome_pooling)
            
            # Save embedding
            output_file = os.path.join(args.output_dir, f"{genome_id}.npy")
            np.save(output_file, embedding)
            
            successful += 1
            
            # Progress update every 100 genomes
            if (i + 1) % 100 == 0:
                torch.cuda.empty_cache()
                if torch.cuda.is_available() and args.device_id >= 0:
                    mem_allocated = torch.cuda.memory_allocated(args.device_id) / 1024**3
                    mem_reserved = torch.cuda.memory_reserved(args.device_id) / 1024**3
                    print(f"\n  Progress: {i+1}/{len(genomes_to_process)}")
                    print(f"  GPU memory: {mem_allocated:.2f} GB allocated, {mem_reserved:.2f} GB reserved")
                print(f"  Success: {successful}, Failed: {failed}")
            
        except Exception as e:
            print(f"\n  ERROR processing {genome_id}: {e}")
            failed += 1
            failed_genomes.append((genome_id, str(e)))
            torch.cuda.empty_cache()
            continue
    
    # Summary
    print("\n" + "="*80)
    print("EMBEDDING GENERATION COMPLETED")
    print("="*80)
    print(f"Total processed: {len(genomes_to_process)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total embeddings now: {len(existing_embeddings) + successful}")
    
    if failed_genomes:
        print(f"\nFailed genomes ({len(failed_genomes)}):")
        for genome_id, reason in failed_genomes[:10]:
            print(f"  - {genome_id}: {reason}")
        if len(failed_genomes) > 10:
            print(f"  ... and {len(failed_genomes) - 10} more")
    
    print(f"\nEmbeddings saved to: {args.output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

""" Run Example:
python s3_generate_evo_embeddings_multi_level.py \
    --fna_dir /path/to/fna \
    --output_dir /path/to/output \
    --model_name evo-1-131k-base \
    --device_id 0 \
    --max_sequence_length 8192 \
    --genome_pooling mean

"""
