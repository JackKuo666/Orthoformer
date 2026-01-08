import sys
import pandas as pd

### python3 analysis_metrics.py embeddings_4000_v1_a2_binary_classification/gideon_Lactose/bacformer_ft/xgboost_classification_report.csv


def get_simple_metrics(file_path):
    """
    Extract simplest key metrics
    """
    df = pd.read_csv(file_path)
    
    # Use iloc for positional indexing directly to avoid column name issues
    return {
        'class_0_f1': df.iloc[0, 3],     # Row 0, column 3 is f1-score
        'class_1_f1': df.iloc[1, 3],     # Row 1, column 3 is f1-score
        'accuracy': df.iloc[2, 1],       # Row 2, column 1 is accuracy value
        'macro_f1': df.iloc[3, 3],       # Row 3, column 3 is macro avg f1
        'weighted_f1': df.iloc[4, 3]     # Row 4, column 3 is weighted avg f1
    }

#metrics = get_simple_metrics(sys.argv[1])
#for key, value in metrics.items():
#    print(f"{key}: {value:.4f}")

from pathlib import Path

def get_folders_pathlib(path):
    """
    Get all folders using pathlib (recommended approach)
    """
    path_obj = Path(path)
    folders = [item.name for item in path_obj.iterdir() if item.is_dir()]
    return folders
# Usage example
#folders = get_folders_pathlib(".")


import argparse

def parse_argv():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Example program that processes three parameters')
    # Add three required positional parameters
    parser.add_argument('-i','--input_file', default="embeddings_4000_v1_a2_binary_classification",help='Input file path')
    parser.add_argument('-o','--output_dir', default="binary",help='Output directory path')
    parser.add_argument('-e','--embedding_dir', default="embeddings_4000_v1_a2",help='Run mode')
    parser.add_argument('--force', '-f', action='store_true', default=False, help='Force overwrite existing files')
    # Parse arguments
    args = parser.parse_args()
    # Use arguments
    return args

#    return {
#        'class_0_f1': df.iloc[0, 3],     # Row 0, column 3 is f1-score
#        'class_1_f1': df.iloc[1, 3],     # Row 1, column 3 is f1-score
#        'accuracy': df.iloc[2, 1],       # Row 2, column 1 is accuracy value
#        'macro_f1': df.iloc[3, 3],       # Row 3, column 3 is macro avg f1
#        'weighted_f1': df.iloc[4, 3]     # Row 4, column 3 is weighted avg f1
#    }

def get_all_phenotype_binary(folder):
    phenotypes = get_folders_pathlib(folder)
    prefix = folder
    our_accuracy = []; bac_accuracy = []
    our_macro_f1 = []; bac_macro_f1 = []
    our_weighted_f1 = [] ; bac_weighted_f1 = []
    for pt in phenotypes:
        our_result = prefix + "/" + pt + "/our_ft/xgboost_classification_report.csv"
        bacformer_result = prefix + "/" + pt + "/bacformer_ft/xgboost_classification_report.csv"
        our_metrics = get_simple_metrics(our_result)
        bacformer_metrics = get_simple_metrics(bacformer_result)
        #print(our_metrics)
        #print(bacformer_metrics)
        our_accuracy.append(our_metrics["accuracy"])
        bac_accuracy.append(bacformer_metrics["accuracy"])
        our_macro_f1.append(our_metrics["macro_f1"])
        bac_macro_f1.append(bacformer_metrics["macro_f1"])
        our_weighted_f1.append(our_metrics["weighted_f1"])
        bac_weighted_f1.append(bacformer_metrics["weighted_f1"])
    
    df = pd.DataFrame({
    'orthoformer accuracy': our_accuracy,
    'bacformer accuracy': bac_accuracy, 
    'orthoformer macro_f1': our_macro_f1,
    'bacformer macro_f1': bac_macro_f1,
    'orthoformer weighted_f1': our_weighted_f1,
    'bacformer weighted_f1': bac_weighted_f1
    }, index=phenotypes)
    df_rounded = df.round(3)
    df_rounded.to_csv("binary_classification.csv")

args = parse_argv()
get_all_phenotype_binary(args.input_file)
### python3 analysis_metrics.py 
