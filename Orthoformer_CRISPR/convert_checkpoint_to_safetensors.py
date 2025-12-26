#!/usr/bin/env python3
"""
Convert existing PyTorch checkpoints to safetensors format
This allows resuming training without torch >= 2.6 requirement
"""
import os
import sys
import argparse
from pathlib import Path
import torch

try:
    from safetensors.torch import save_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False
    print("Warning: safetensors not installed. Install with: pip install safetensors")

def convert_checkpoint(checkpoint_dir, output_dir=None):
    """
    Convert pytorch_model.bin to model.safetensors in a checkpoint directory
    
    Args:
        checkpoint_dir: Path to checkpoint directory containing pytorch_model.bin
        output_dir: Optional output directory (default: same as checkpoint_dir)
    """
    checkpoint_path = Path(checkpoint_dir)
    pytorch_model_path = checkpoint_path / "pytorch_model.bin"
    
    if not pytorch_model_path.exists():
        print(f"Warning: {pytorch_model_path} not found, skipping...")
        return False
    
    if output_dir is None:
        output_dir = checkpoint_path
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    safetensors_path = output_dir / "model.safetensors"
    
    print(f"Converting {pytorch_model_path} -> {safetensors_path}")
    
    try:
        # Load PyTorch model
        # Try with weights_only=True first (safer), fallback to False if needed
        try:
            state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=True)
        except (ValueError, TypeError):
            # Fallback for older PyTorch versions or if weights_only not supported
            state_dict = torch.load(pytorch_model_path, map_location='cpu', weights_only=False)
        
        if HAS_SAFETENSORS:
            # Save as safetensors
            save_file(state_dict, safetensors_path)
            print(f"✓ Successfully converted to {safetensors_path}")
        else:
            # Alternative: Use transformers to save (if available)
            try:
                from transformers import PreTrainedModel
                # This is a workaround - we'll need the model class
                # For now, just inform user to install safetensors
                print("✗ safetensors not available. Please install: pip install safetensors")
                print("  Or use transformers>=4.21.0 which supports safetensors natively")
                return False
            except ImportError:
                print("✗ Cannot convert without safetensors. Install: pip install safetensors")
                return False
        
        # Optionally remove pytorch_model.bin (commented out for safety)
        # pytorch_model_path.unlink()
        # print(f"✓ Removed {pytorch_model_path}")
        
        return True
    except Exception as e:
        print(f"✗ Error converting {checkpoint_dir}: {e}")
        import traceback
        traceback.print_exc()
        return False

def convert_all_checkpoints(base_dir):
    """Convert all checkpoints in output directory"""
    base_path = Path(base_dir)
    
    if not base_path.exists():
        print(f"Error: {base_path} does not exist")
        return
    
    # Find all checkpoint directories
    checkpoint_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('checkpoint-')]
    
    if not checkpoint_dirs:
        print(f"No checkpoint directories found in {base_path}")
        return
    
    print(f"Found {len(checkpoint_dirs)} checkpoint directories")
    
    success_count = 0
    for checkpoint_dir in sorted(checkpoint_dirs):
        if convert_checkpoint(checkpoint_dir):
            success_count += 1
    
    print(f"\n✓ Converted {success_count}/{len(checkpoint_dirs)} checkpoints")

def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch checkpoints to safetensors format")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory or output directory containing checkpoints")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: same as checkpoint_dir)")
    parser.add_argument("--all", action="store_true", help="Convert all checkpoints in the directory")
    
    args = parser.parse_args()
    
    if args.all:
        convert_all_checkpoints(args.checkpoint_dir)
    else:
        convert_checkpoint(args.checkpoint_dir, args.output_dir)

if __name__ == "__main__":
    main()

