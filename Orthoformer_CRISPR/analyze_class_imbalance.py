#!/usr/bin/env python3
"""
分析类别不平衡问题和模型退化原因
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
    print("类别不平衡问题分析")
    print("=" * 80)
    
    # 1. 类别分布
    first_row = rows[0]
    print("\n1. 验证集类别分布 (support):")
    for i in range(8):
        support = float(first_row.get(f'eval_support_c{i}', 0))
        print(f"   Class {i}: {support:,.0f} 样本")
    
    total = sum(float(first_row.get(f'eval_support_c{i}', 0)) for i in range(8))
    print(f"   总计: {total:,.0f} 样本")
    
    # 2. 早期 vs 后期表现
    print("\n2. 训练过程表现变化:")
    print("\n   早期表现 (前3次评估):")
    for i, row in enumerate(rows[:3]):
        epoch = row['epoch']
        f1_c1 = float(row.get('eval_f1_c1', 0))
        f1_c2 = float(row.get('eval_f1_c2', 0))
        f1_c3 = float(row.get('eval_f1_c3', 0))
        print(f"     Epoch {epoch}: Class1 F1={f1_c1:.4f}, Class2 F1={f1_c2:.4f}, Class3 F1={f1_c3:.4f}")
    
    print("\n   后期表现 (最后3次评估):")
    for i, row in enumerate(rows[-3:]):
        epoch = row['epoch']
        f1_c1 = float(row.get('eval_f1_c1', 0))
        f1_c2 = float(row.get('eval_f1_c2', 0))
        f1_c3 = float(row.get('eval_f1_c3', 0))
        print(f"     Epoch {epoch}: Class1 F1={f1_c1:.4f}, Class2 F1={f1_c2:.4f}, Class3 F1={f1_c3:.4f}")
    
    # 3. 找出最佳表现
    print("\n3. 各类别最佳 F1 分数:")
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
    
    # 4. 问题诊断
    print("\n4. 问题诊断:")
    print("\n   问题1: 类别极度不平衡")
    c0_support = float(first_row.get('eval_support_c0', 0))
    c1_support = float(first_row.get('eval_support_c1', 0))
    imbalance_ratio = c0_support / c1_support if c1_support > 0 else float('inf')
    print(f"     - Class 0 与 Class 1 的比例: {imbalance_ratio:.1f}:1")
    print(f"     - Class 0 占总样本的 {c0_support/total*100:.2f}%")
    
    print("\n   问题2: 模型退化 (Catastrophic Forgetting)")
    early_avg_f1 = sum(float(row.get('eval_f1_c1', 0)) for row in rows[:3]) / 3
    late_avg_f1 = sum(float(row.get('eval_f1_c1', 0)) for row in rows[-3:]) / 3
    print(f"     - Class 1 早期平均 F1: {early_avg_f1:.4f}")
    print(f"     - Class 1 后期平均 F1: {late_avg_f1:.4f}")
    print(f"     - 下降: {(early_avg_f1 - late_avg_f1)*100:.2f} 个百分点")
    
    print("\n   问题3: 类别权重可能不足")
    print("     - 虽然设置了类别权重，但归一化后可能太小")
    print("     - 建议增加类别权重的强度")
    
    # 5. 建议
    print("\n5. 解决方案建议:")
    print("\n   A. 调整类别权重策略:")
    print("      - 使用更激进的类别权重（不归一化，或使用平方根归一化）")
    print("      - 考虑使用 focal loss 的 alpha 参数")
    print("      - 增加 focal loss 的 gamma 值（当前是 2.0）")
    
    print("\n   B. 使用更好的损失函数:")
    print("      - 确保 focal loss 正确应用类别权重")
    print("      - 考虑使用 Dice Loss 或 Tversky Loss")
    print("      - 使用 label smoothing 防止过拟合")
    
    print("\n   C. 训练策略调整:")
    print("      - 使用更小的学习率")
    print("      - 增加 warmup steps")
    print("      - 使用 early stopping（基于 eval_f1_macro）")
    print("      - 从最佳 checkpoint 恢复（epoch 1.86 附近）")
    
    print("\n   D. 数据层面:")
    print("      - 使用 class-balanced sampling")
    print("      - 对少数类进行数据增强")
    print("      - 考虑使用 focal loss 的 class-balanced 变体")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "eval_metrics.csv"
    analyze_metrics(csv_file)

