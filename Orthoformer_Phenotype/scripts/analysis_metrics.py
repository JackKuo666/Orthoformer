import sys
import pandas as pd

### python3 analysis_metrics.py embeddings_4000_v1_a2_binary_classification/gideon_Lactose/bacformer_ft/xgboost_classification_report.csv


def get_simple_metrics(file_path):
    """
    最简单的关键指标提取
    """
    df = pd.read_csv(file_path)
    
    # 直接使用iloc按位置索引，避免列名问题
    return {
        'class_0_f1': df.iloc[0, 3],     # 第0行第3列是f1-score
        'class_1_f1': df.iloc[1, 3],     # 第1行第3列是f1-score
        'accuracy': df.iloc[2, 1],       # 第2行第1列是accuracy值
        'macro_f1': df.iloc[3, 3],       # 第3行第3列是macro avg f1
        'weighted_f1': df.iloc[4, 3]     # 第4行第3列是weighted avg f1
    }

#metrics = get_simple_metrics(sys.argv[1])
#for key, value in metrics.items():
#    print(f"{key}: {value:.4f}")

from pathlib import Path

def get_folders_pathlib(path):
    """
    使用pathlib获取所有文件夹（推荐方式）
    """
    path_obj = Path(path)
    folders = [item.name for item in path_obj.iterdir() if item.is_dir()]
    return folders
# 使用示例
#folders = get_folders_pathlib(".")


import argparse

def parse_argv():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='这是一个处理三个参数的示例程序')
    # 添加三个必需的位置参数
    parser.add_argument('-i','--input_file', default="embeddings_4000_v1_a2_binary_classification",help='输入文件路径')
    parser.add_argument('-o','--output_dir', default="binary",help='输出目录路径')
    parser.add_argument('-e','--embedding_dir', default="embeddings_4000_v1_a2",help='运行模式')
    parser.add_argument('--force', '-f', action='store_true', default=False, help='强制覆盖已存在文件')
    # 解析参数
    args = parser.parse_args()
    # 使用参数
    return args

#    return {
#        'class_0_f1': df.iloc[0, 3],     # 第0行第3列是f1-score
#        'class_1_f1': df.iloc[1, 3],     # 第1行第3列是f1-score
#        'accuracy': df.iloc[2, 1],       # 第2行第1列是accuracy值
#        'macro_f1': df.iloc[3, 3],       # 第3行第3列是macro avg f1
#        'weighted_f1': df.iloc[4, 3]     # 第4行第3列是weighted avg f1
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
