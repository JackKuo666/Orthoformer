#!/usr/bin/env python3
"""
View eval metrics in a nicely formatted table
"""
import csv
import sys

def print_table(data, headers):
    """Print a simple text table"""
    if not data:
        return
    
    # Calculate column widths
    col_widths = [len(str(h)) for h in headers]
    for row in data:
        for i, val in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(str(val)))
    
    # Print header
    header_line = " | ".join(str(h).ljust(col_widths[i]) for i, h in enumerate(headers))
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        row_line = " | ".join(str(val).ljust(col_widths[i]) for i, val in enumerate(row))
        print(row_line)

def main():
    if len(sys.argv) < 2:
        print("Usage: python view_metrics_table.py <metrics_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print("No data found in CSV file")
        return
    
    # Key metrics to display
    key_metrics = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_f1_micro', 'eval_f1_macro']
    available_keys = [k for k in key_metrics if k in rows[0].keys()]
    
    # Prepare table data
    table_data = []
    for row in rows:
        table_data.append([row.get(k, 'N/A') for k in available_keys])
    
    print("=== Evaluation Metrics Summary ===\n")
    print_table(table_data, available_keys)
    
    # Per-class summary (latest)
    print("\n=== Per-Class Metrics (Latest Evaluation) ===\n")
    latest = rows[-1]
    
    class_metrics = []
    for i in range(8):  # Assuming 8 classes
        f1_key = f'eval_f1_c{i}'
        prec_key = f'eval_precision_c{i}'
        rec_key = f'eval_recall_c{i}'
        sup_key = f'eval_support_c{i}'
        
        if f1_key in latest:
            class_metrics.append([
                f'Class {i}',
                latest.get(f1_key, '0.0'),
                latest.get(prec_key, '0.0'),
                latest.get(rec_key, '0.0'),
                latest.get(sup_key, '0.0'),
            ])
    
    if class_metrics:
        print_table(class_metrics, ['Class', 'F1', 'Precision', 'Recall', 'Support'])
    
    print(f"\nTotal evaluations: {len(rows)}")

if __name__ == "__main__":
    main()

