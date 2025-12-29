#!/usr/bin/env python
"""Calculate RF and nRF metrics for the generated trees using embedding_3M_v10."""

import os
import sys
from ete3 import Tree as EteTree

def calculate_rf_metrics(pred_tree_path, ref_tree_path):
    """Calculate RF and nRF metrics between predicted and reference trees."""
    print(f"\nComparing:")
    print(f"  Predicted: {pred_tree_path}")
    print(f"  Reference: {ref_tree_path}")
    
    if not os.path.exists(pred_tree_path):
        print(f"  ERROR: Predicted tree not found!")
        return None
    
    if not os.path.exists(ref_tree_path):
        print(f"  ERROR: Reference tree not found!")
        return None
    
    try:
        pred_tree = EteTree(pred_tree_path, format=1)
        ref_tree = EteTree(ref_tree_path, format=1)
        
        # Get common leaves
        pred_leaves = set(pred_tree.get_leaf_names())
        ref_leaves = set(ref_tree.get_leaf_names())
        common_leaves = pred_leaves & ref_leaves
        
        print(f"  Common leaves: {len(common_leaves)}")
        
        if len(common_leaves) < 4:
            print(f"  ERROR: Not enough common leaves for RF calculation!")
            return None
        
        # Prune trees to common leaves
        pred_tree.prune(common_leaves, preserve_branch_length=True)
        ref_tree.prune(common_leaves, preserve_branch_length=True)
        
        # Calculate RF distance
        rf_result = pred_tree.robinson_foulds(ref_tree, unrooted_trees=True)
        rf = rf_result[0]
        max_rf = rf_result[1]
        nrf = rf / max_rf if max_rf > 0 else 0.0
        
        print(f"  RF = {rf}, maxRF = {max_rf}, nRF = {nrf:.6f}")
        
        return {
            'rf': rf,
            'max_rf': max_rf,
            'nrf': nrf,
            'common_leaves': len(common_leaves)
        }
    
    except Exception as e:
        print(f"  ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Reference tree
    ref_tree = "gtdb_ref_trees/ar53_r226.nwk"
    
    # Trees to compare - using the updated trees in nj_sele_tree/ and upgma_sele_tree/
    trees = [
        ("NJ (selected samples)", "nj_tree/ar53_r226.nj_skbio.nwk"),
        ("UPGMA (selected samples)", "upgma_tree/ar53_r226.upgma_scipy.nwk"),
    ]
    
    print("=" * 60)
    print("RF and nRF Metrics for 3M_v10 Trees (Selected Samples)")
    print("=" * 60)
    
    results = {}
    for name, tree_path in trees:
        print(f"\n{name}:")
        result = calculate_rf_metrics(tree_path, ref_tree)
        if result:
            results[name] = result
    
    # Summary table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Tree':<30} {'RF':<10} {'maxRF':<10} {'nRF':<10} {'Leaves':<10}")
    print("-" * 60)
    for name, result in results.items():
        print(f"{name:<30} {result['rf']:<10} {result['max_rf']:<10} {result['nrf']:<10.6f} {result['common_leaves']:<10}")
    
    # Save results to file
    output_file = "rf_metrics.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("RF and nRF Metrics for 3M_v10 Trees (Selected Samples)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Tree':<30} {'RF':<10} {'maxRF':<10} {'nRF':<10} {'Leaves':<10}\n")
        f.write("-" * 60 + "\n")
        for name, result in results.items():
            f.write(f"{name:<30} {result['rf']:<10} {result['max_rf']:<10} {result['nrf']:<10.6f} {result['common_leaves']:<10}\n")
    
    print(f"\nResults saved to: {output_file}")
    return results

if __name__ == "__main__":
    main()

