#!/usr/bin/env python3
"""
Analyze class imbalance issues and model degradation causes
"""
import csv
import sys

def analyze_metrics(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No data found")
        return
    
    print("=" * 80)
    print("Class Imbalance Analysis")
    print("=" * 80)
    
    # 1. Class distribution
    first_row = rows[0]
    print("\n1. Validation set class distribution (support):")
    for i in range(8):
        support = float(first_row.get(f'eval_support_c{i}', 0))
        print(f"   Class {i}: {support:,.0f} samples")
    
    total = sum(float(first_row.get(f'eval_support_c{i}', 0)) for i in range(8))
    print(f"   Total: {total:,.0f} samples")
    
    # 2. Early vs late performance
    print("\n2. Training process performance changes:")
    print("\n   Early performance (first 3 evaluations):")
    for i, row in enumerate(rows[:3]):
        epoch = row['epoch']
        f1_c1 = float(row.get('eval_f1_c1', 0))
        f1_c2 = float(row.get('eval_f1_c2', 0))
        f1_c3 = float(row.get('eval_f1_c3', 0))
        print(f"     Epoch {epoch}: Class1 F1={f1_c1:.4f}, Class2 F1={f1_c2:.4f}, Class3 F1={f1_c3:.4f}")
    
    print("\n   Late performance (last 3 evaluations):")
    for i, row in enumerate(rows[-3:]):
        epoch = row['epoch']
        f1_c1 = float(row.get('eval_f1_c1', 0))
        f1_c2 = float(row.get('eval_f1_c2', 0))
        f1_c3 = float(row.get('eval_f1_c3', 0))
        print(f"     Epoch {epoch}: Class1 F1={f1_c1:.4f}, Class2 F1={f1_c2:.4f}, Class3 F1={f1_c3:.4f}")
    
    # 3. Find best performance
    print("\n3. Best F1 scores for each class:")
    best_f1 = {}
    best_epoch = {}
    for i in range(8):
        best_f1[i] = 0.0
        best_epoch[i] = None
        for row in rows:
            f1 = float(row.get(f'eval_f1_c{i}', 0))
            if f1 > best_f1[i]:
                best_f1[i] = f1
                best_epoch[i] = row['epoch']
    
    for i in range(8):
        print(f"   Class {i}: F1={best_f1[i]:.4f} (epoch {best_epoch[i]})")
    
    # 4. Problem diagnosis
    print("\n4. Problem diagnosis:")
    print("\n   Problem 1: Extreme class imbalance")
    c0_support = float(first_row.get('eval_support_c0', 0))
    c1_support = float(first_row.get('eval_support_c1', 0))
    imbalance_ratio = c0_support / c1_support if c1_support > 0 else float('inf')
    print(f"     - Class 0 to Class 1 ratio: {imbalance_ratio:.1f}:1")
    print(f"     - Class 0 accounts for {c0_support/total*100:.2f}% of total samples")
    
    print("\n   Problem 2: Model degradation (Catastrophic Forgetting)")
    early_avg_f1 = sum(float(row.get('eval_f1_c1', 0)) for row in rows[:3]) / 3
    late_avg_f1 = sum(float(row.get('eval_f1_c1', 0)) for row in rows[-3:]) / 3
    print(f"     - Class 1 early average F1: {early_avg_f1:.4f}")
    print(f"     - Class 1 late average F1: {late_avg_f1:.4f}")
    print(f"     - Decrease: {(early_avg_f1 - late_avg_f1)*100:.2f} percentage points")
    
    print("\n   Problem 3: Class weights may be insufficient")
    print("     - Although class weights are set, they may be too small after normalization")
    print("     - Suggest increasing class weight strength")
    
    # 5. Suggestions
    print("\n5. Solution suggestions:")
    print("\n   A. Adjust class weight strategy:")
    print("      - Use more aggressive class weights (no normalization, or use square root normalization)")
    print("      - Consider using focal loss alpha parameter")
    print("      - Increase focal loss gamma value (currently 2.0)")
    
    print("\n   B. Use better loss function:")
    print("      - Ensure focal loss correctly applies class weights")
    print("      - Consider using Dice Loss or Tversky Loss")
    print("      - Use label smoothing to prevent overfitting")
    
    print("\n   C. Training strategy adjustments:")
    print("      - Use smaller learning rate")
    print("      - Increase warmup steps")
    print("      - Use early stopping (based on eval_f1_macro)")
    print("      - Resume from best checkpoint (around epoch 1.86)")
    
    print("\n   D. Data level:")
    print("      - Use class-balanced sampling")
    print("      - Perform data augmentation for minority classes")
    print("      - Consider using class-balanced variant of focal loss")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "eval_metrics.csv"
    analyze_metrics(csv_file)

