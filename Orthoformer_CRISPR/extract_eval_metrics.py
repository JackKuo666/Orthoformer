#!/usr/bin/env python3
"""
Extract evaluation metrics from training log and format as table
"""
import re
import json
import sys
import csv
from pathlib import Path

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

def parse_eval_line(line):
    """Parse a single eval metrics line from log"""
    try:
        # Extract the dict from the line
        # Format: {'eval_loss': 0.0015, 'eval_accuracy': 0.9991, ...}
        match = re.search(r"\{'eval_loss'.*?\}", line)
        if not match:
            return None
        
        # Parse as Python dict (using eval, but we validate it's a dict first)
        eval_str = match.group(0)
        # Replace single quotes with double quotes for JSON parsing
        eval_str = eval_str.replace("'", '"')
        # Handle Python True/False/None
        eval_str = eval_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
        
        try:
            metrics = json.loads(eval_str)
        except:
            # Fallback: use eval (less safe but works for this use case)
            metrics = eval(match.group(0))
        
        return metrics
    except Exception as e:
        print(f"Warning: Failed to parse line: {e}", file=sys.stderr)
        return None

def extract_metrics_from_log(log_file):
    """Extract all eval metrics from log file"""
    all_metrics = []
    
    with open(log_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if "'eval_loss'" in line:
                metrics = parse_eval_line(line)
                if metrics:
                    metrics['line_number'] = line_num
                    all_metrics.append(metrics)
    
    return all_metrics

def create_metrics_table(metrics_list):
    """Create a formatted table from metrics"""
    if not metrics_list:
        return None
    
    # Extract epoch info if available
    for m in metrics_list:
        if 'epoch' not in m:
            # Try to infer from line number or step
            m['epoch'] = m.get('line_number', 0)
    
    if HAS_PANDAS:
        # Create DataFrame
        df = pd.DataFrame(metrics_list)
        
        # Reorder columns for better readability
        priority_cols = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_f1_micro', 'eval_f1_macro', 
                         'eval_precision_macro', 'eval_recall_macro']
        
        # Get all columns
        all_cols = list(df.columns)
        # Remove priority cols from all_cols
        remaining_cols = [c for c in all_cols if c not in priority_cols]
        # Reorder: priority first, then others
        ordered_cols = [c for c in priority_cols if c in df.columns] + sorted(remaining_cols)
        
        df = df[ordered_cols]
        return df
    else:
        # Fallback: return as list of dicts
        return metrics_list

def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_eval_metrics.py <log_file> [output_file]")
        print("  log_file: Path to training log file")
        print("  output_file: Optional output CSV file (default: eval_metrics.csv)")
        sys.exit(1)
    
    log_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "eval_metrics.csv"
    
    print(f"Extracting metrics from {log_file}...")
    metrics_list = extract_metrics_from_log(log_file)
    
    if not metrics_list:
        print("No eval metrics found in log file!")
        sys.exit(1)
    
    print(f"Found {len(metrics_list)} evaluation records")
    
    # Create table
    table = create_metrics_table(metrics_list)
    
    # Save to CSV
    if HAS_PANDAS and isinstance(table, pd.DataFrame):
        table.to_csv(output_file, index=False, float_format='%.6f')
        print(f"\nMetrics saved to {output_file}")
        
        # Print summary table
        print("\n=== Summary Table ===")
        # Select key metrics for display
        display_cols = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_f1_micro', 'eval_f1_macro']
        display_cols = [c for c in display_cols if c in table.columns]
        
        print(table[display_cols].to_string(index=False))
        
        # Print per-class metrics summary (if available)
        class_cols = [c for c in table.columns if c.startswith('eval_f1_c')]
        if class_cols:
            print("\n=== Per-Class F1 Scores (Latest) ===")
            latest = table.iloc[-1]
            class_data = []
            for col in sorted(class_cols):
                class_id = col.replace('eval_f1_c', '')
                class_data.append({
                    'Class': class_id,
                    'F1': latest[col],
                    'Precision': latest.get(f'eval_precision_c{class_id}', 'N/A'),
                    'Recall': latest.get(f'eval_recall_c{class_id}', 'N/A'),
                    'Support': latest.get(f'eval_support_c{class_id}', 'N/A'),
                })
            class_df = pd.DataFrame(class_data)
            print(class_df.to_string(index=False))
    else:
        # Fallback: save as CSV without pandas
        if not metrics_list:
            print("No metrics to save")
            return 1
        
        # Get all unique keys
        all_keys = set()
        for m in metrics_list:
            all_keys.update(m.keys())
        
        # Order keys
        priority_keys = ['epoch', 'eval_loss', 'eval_accuracy', 'eval_f1_micro', 'eval_f1_macro']
        ordered_keys = [k for k in priority_keys if k in all_keys]
        ordered_keys += sorted([k for k in all_keys if k not in priority_keys])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=ordered_keys)
            writer.writeheader()
            for m in metrics_list:
                row = {k: m.get(k, '') for k in ordered_keys}
                writer.writerow(row)
        
        print(f"\nMetrics saved to {output_file}")
        print(f"Total records: {len(metrics_list)}")
        print("\nNote: Install pandas for better table formatting")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

