import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, auc
)

import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
plt.switch_backend('Agg')

# ========== 工具函数 ==========
def get_file_stem(path: str) -> str:
    """
    给定一个文件路径，返回不带扩展名的文件名
    例如: /a/b/c/file.txt -> file
    """
    return Path(path).stem


# ========== 主类 ========== #
class BinaryClassificationModel:
    def __init__(self, csv_path, bacformer, data_folder,
                 test_size=0.2, random_state=42,
                 outdir="/mnt/disk_c/user_data/liuping/bacformer/results"):
        """
        初始化二分类模型
        """
        self.csv_path = csv_path
        self.data_folder = data_folder
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.bacformer = bacformer
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def _to_numpy_2d(self, x_tensor: torch.Tensor) -> np.ndarray:
        """
        把 torch.Tensor(N, D) 转为 numpy.ndarray(N, D)
        """
        if isinstance(x_tensor, torch.Tensor):
            return x_tensor.detach().cpu().numpy()
        return np.asarray(x_tensor)

    def load_and_preprocess_data(self):
        """
        加载CSV数据并从文件中提取特征
        返回:
            X (torch.Tensor[N, D]),
            y (np.ndarray[N]),
            sample_names (list[str]),
            missing_files (list[str]),
            data_info (dict)
        """
        # 加载CSV数据
        df = pd.read_csv(self.csv_path)

        # 基本信息
        total_samples = len(df)
        label_counts = df['Label'].value_counts().to_dict()
        positive_ratio = float(df['Label'].mean())

        print("数据基本信息:")
        print(f"总样本数: {total_samples}")
        print(f"类别分布: {label_counts}")
        print(f"阳性比例: {positive_ratio:.3f}")

        if not os.path.exists(self.data_folder):
            raise ValueError(f"数据文件夹 '{self.data_folder}' 不存在")

        X, y = [], []
        sample_names = []
        missing_files = []

        print("\n正在从文件中提取特征...")
        for _, row in df.iterrows():
            orig_name = row['genome_name']
            genome_name = orig_name
            found_file = None
            base_path = os.path.join(self.data_folder, genome_name)

            # 常见扩展名尝试
            for ext in ['', '.fasta', '.fna', '.fa', '.txt', '.gbff', '.gbk', '.pt']:
                test_path = base_path + ext
                if os.path.exists(test_path):
                    found_file = test_path
                    break

            # 尝试 RS_/GB_ 前缀
            if found_file is None:
                for prefix in ['RS', 'GB']:
                    genome_name = f"{prefix}_{orig_name}"
                    base_path = os.path.join(self.data_folder, genome_name)
                    for ext in ['', '.fasta', '.fna', '.fa', '.txt', '.gbff', '.gbk', '.pt']:
                        test_path = base_path + ext
                        if os.path.exists(test_path):
                            found_file = test_path
                            break
                    if found_file:
                        break

            if found_file:
                try:
                    # 提取特征
                    if self.bacformer:
                        # CSV 中保存的字符串向量 -> tensor
                        features_str = row['bacformer_genome_embedding']
                        vec = [float(x) for x in features_str.replace('\n', ' ')
                                                       .replace('[', '')
                                                       .replace(']', '')
                                                       .split()]
                        features = torch.tensor(vec, dtype=torch.float32)
                    else:
                        # 从 .pt 读你的 embedding（或兼容 list/ndarray）
                        features = torch.load(found_file)
                        if isinstance(features, np.ndarray):
                            features = torch.tensor(features, dtype=torch.float32)
                        elif isinstance(features, list):
                            features = torch.tensor(features, dtype=torch.float32)
                        else:
                            features = features.to(dtype=torch.float32)

                    # 确保是一维
                    features = features.flatten()

                    X.append(features)
                    y.append(int(row['Label']))
                    sample_names.append(genome_name)
                except Exception as e:
                    print(f"处理文件 {found_file} 时出错: {e}")
                    missing_files.append(genome_name)
            else:
                missing_files.append(genome_name)

        print(f"\n成功处理: {len(sample_names)} 个文件")
        print(f"缺失文件: {len(missing_files)} 个")
        if missing_files:
            print("示例缺失文件:", missing_files[:5])

        y = np.array(y)
        if len(X) == 0:
            X_tensor = torch.empty((0, 0), dtype=torch.float32)
        else:
            # 对齐维度（如果各样本向量长度差异，右侧零填充）
            max_dim = max(x.numel() for x in X)
            X_padded = []
            for x in X:
                if x.numel() < max_dim:
                    pad = torch.zeros(max_dim - x.numel(), dtype=torch.float32)
                    X_padded.append(torch.cat([x, pad], dim=0))
                else:
                    X_padded.append(x)
            X_tensor = torch.stack(X_padded, dim=0)

        print(f"\n特征矩阵形状: {tuple(X_tensor.shape)}")
        if X_tensor.numel() > 0:
            print(f"特征数量: {X_tensor.shape[1]}")

        #data_info = {
        #    "total samples": total_samples,
        #    "processed samples": len(sample_names),
        #    "miss": len(missing_files),
        #    "positive ratio": positive_ratio,
        #    "distribution": label_counts
        #}
        data_info = {                                                                                         
                "总样本数": total_samples,
                "成功处理文件数": len(sample_names),
                "缺失文件数": len(missing_files),
                "阳性比例": positive_ratio,
                "类别分布": label_counts
        }   



        return X_tensor, y, sample_names, missing_files, data_info

    def train_test_split_data(self, X, y, sample_names):
        """
        分割训练集和测试集（保持名称对齐）
        """
        X_np = self._to_numpy_2d(X)
        names = np.array(sample_names)

        X_tr, X_te, y_tr, y_te, n_tr, n_te = train_test_split(
            X_np, y, names,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        print("\n数据分割结果:")
        print(f"训练集: {X_tr.shape[0]} 个样本")
        print(f"测试集: {X_te.shape[0]} 个样本")
        print(f"训练集类别分布: {np.bincount(y_tr)}")
        print(f"测试集类别分布: {np.bincount(y_te)}")

        return X_tr, X_te, y_tr, y_te, n_tr.tolist(), n_te.tolist()

    # ===== 三种模型：按需选择 =====
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        print("\n训练逻辑回归模型...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000,
            class_weight='balanced'
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        cm = confusion_matrix(y_test, y_pred)

        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'feature_importance': model.coef_[0] if hasattr(model, 'coef_') else None,
            'model_name': 'logistic_regression'
        }

        print("逻辑回归模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        print("\n训练随机森林模型...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'feature_importance': getattr(model, 'feature_importances_', None),
            'model_name': 'random_forest'
        }

        print("随机森林模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    def train_xgboost_classifier(self, X_train, X_test, y_train, y_test):
        print("\n训练 XGBoost 分类模型...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        scale_pos_weight = float(neg) / float(pos) if pos > 0 else 1.0

        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            n_estimators=300,
            learning_rate=0.08,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            verbosity=1
        )

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': cm,
            'feature_importance': getattr(model, 'feature_importances_', None),
            'model_name': 'xgboost'
        }

        print("XGBoost 分类模型性能:")
        print(f"  准确率: {accuracy:.4f}")
        print(f"  精确率: {precision:.4f}")
        print(f"  召回率: {recall:.4f}")
        print(f"  F1分数: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    # ===== 可视化并保存 =====
    def plot_results(self, y_test, results, prefix="model"):
        """
        绘制并保存结果可视化（混淆矩阵/ROC/指标条形图），并可选保存特征重要性图
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 混淆矩阵
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Prediction')
        axes[0, 0].set_ylabel('True')

        # 2. ROC 曲线
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 性能指标条形图
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metrics_label = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
        metric_names = ['准确率', '精确率', '召回率', 'F1分数', 'ROC AUC']
        scores = [results[m] for m in metrics]

        #bars = axes[1, 0].bar(metric_names, scores, alpha=0.85)
        bars = axes[1, 0].bar(metrics_label, scores, alpha=0.85)
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].tick_params(axis='x', rotation=45)

        for bar, score in zip(bars, scores):
            h = bar.get_height()
            axes[1, 0].text(bar.get_x()+bar.get_width()/2., h + 0.01,
                            f'{score:.3f}', ha='center', va='bottom')

        # 4. 预留子图位（可显示特征重要性缩略图）
        axes[1, 1].axis('off')

        plt.tight_layout()

        # 保存主结果图
        fig_path = os.path.join(self.outdir, f"{prefix}_results.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight',format='pdf')
        print(f"结果图已保存: {fig_path}")
        plt.close(fig)

        # 可选：保存特征重要性（若有）
        fi = results.get('feature_importance', None)
        if fi is not None:
            try:
                plt.figure(figsize=(10, 6))
                # 若特征太多，只画前 30 个
                idx = np.argsort(fi)[::-1][:30]
                plt.bar(range(len(idx)), np.array(fi)[idx])
                plt.xticks(range(len(idx)), [f"f{int(i)}" for i in idx], rotation=90)
                plt.title(f"{prefix} - Top Feature Importances")
                plt.ylabel("Importance")
                fi_path = os.path.join(self.outdir, f"{prefix}_feature_importance.pdf")
                plt.tight_layout()
                plt.savefig(fi_path, dpi=300, bbox_inches='tight',format='pdf')
                print(f"特征重要性图已保存: {fi_path}")
                plt.close()
            except Exception as _:
                # 一些模型的 coef_ 形状可能不匹配，或者数量太大
                pass

    # ===== 保存模型 =====
    def save_model(self, model, model_name):
        """
        保存训练好的模型和标准化器
        """
        model_path = os.path.join(self.outdir, f'{model_name}_model.pkl')
        scaler_path = os.path.join(self.outdir, 'scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"模型和标准化器已保存：\n  {model_path}\n  {scaler_path}")

    # ===== CSV 保存：数据与评估 =====
    def save_results_to_csv(self, results, data_info, y_test, y_pred, y_pred_proba, test_names,
                            prefix: str):
        """
        保存以下表格：
        1) 训练评估摘要  -> {prefix}_summary.csv
        2) 逐类指标      -> {prefix}_classification_report.csv
        3) 混淆矩阵      -> {prefix}_confusion_matrix.csv
        4) 每样本预测    -> {prefix}_per_sample_predictions.csv
        """
        # 1) 训练评估摘要
        summary_dict = {
            "total samples": data_info["总样本数"],
            "processed samples": data_info["成功处理文件数"],
            "missing": data_info["缺失文件数"],
            "positive ratio": data_info["阳性比例"],
            "label distribution": str(data_info["类别分布"]),
            "accuracy": results['accuracy'],
            "precision": results['precision'],
            "recall": results['recall'],
            "F1-score": results['f1'],
            "ROC_AUC": results['roc_auc'],
            "model type": results.get('model_name', 'unknown')
        }
        pd.DataFrame([summary_dict]).to_csv(
            os.path.join(self.outdir, f"{prefix}_summary.csv"),
            index=False, encoding="utf-8"
        )

        # 2) 逐类指标
        cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(cls_report).transpose().reset_index().rename(columns={"index": "class"})
        rep_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_classification_report.csv"),
            index=False, encoding="utf-8"
        )

        # 3) 混淆矩阵
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, columns=['pred_0', 'pred_1'])
        cm_df.insert(0, 'true', ['true_0', 'true_1'])
        cm_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_confusion_matrix.csv"),
            index=False, encoding="utf-8"
        )

        # 4) 每样本预测
        per_sample = pd.DataFrame({
            "sample_name": test_names,
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba
        })
        per_sample.to_csv(
            os.path.join(self.outdir, f"{prefix}_per_sample_predictions.csv"),
            index=False, encoding="utf-8"
        )

        print(f"结果 CSV 已保存到目录：{self.outdir}")

    def save_missing_list(self, missing_files, prefix: str):
        """
        保存缺失文件列表
        """
        if len(missing_files) == 0:
            return
        miss_df = pd.DataFrame({"missing_file": missing_files})
        miss_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_missing_files.csv"),
            index=False, encoding="utf-8"
        )
        print(f"缺失文件列表已保存：{prefix}_missing_files.csv")

    def run(self, which_model="xgboost"):
        """
        运行完整的训练流程
        which_model: "logistic" | "rf" | "xgboost"
        """
        print("开始二分类模型训练...")
        try:
            # 1) 加载数据 + 特征
            X, y, names, missing_files, data_info = self.load_and_preprocess_data()
            if X.shape[0] == 0:
                print("没有成功提取特征的文件，无法训练模型")
                return None

            # 2) 划分数据（含名称）
            X_train, X_test, y_train, y_test, names_train, names_test = self.train_test_split_data(X, y, names)

            # 3) 训练模型（任选其一）
            if which_model == "logistic":
                results = self.train_logistic_regression(X_train, X_test, y_train, y_test)
            elif which_model == "rf":
                results = self.train_random_forest(X_train, X_test, y_train, y_test)
            else:
                results = self.train_xgboost_classifier(X_train, X_test, y_train, y_test)

            # 4) 可视化并保存图像
            self.plot_results(y_test, results, prefix=results.get('model_name', 'model'))

            # 5) 保存模型
            self.save_model(results['model'], results.get('model_name', 'model'))

            # 6) 保存 CSV（摘要、报告、混淆矩阵、逐样本）
            prefix = f"{results.get('model_name','model')}"
            self.save_results_to_csv(
                results=results,
                data_info=data_info,
                y_test=y_test,
                y_pred=results['y_pred'],
                y_pred_proba=results['y_pred_proba'],
                test_names=names_test,
                prefix=prefix
            )

            # 7) 缺失文件列表
            self.save_missing_list(missing_files, prefix=prefix)

            # 8) 控制台输出详细分类报告
            print("\n详细分类报告:")
            print(classification_report(y_test, results['y_pred'], zero_division=0))

            return results['model']

        except Exception as e:
            print(f"训练过程中出错: {e}")
            import traceback
            traceback.print_exc()
            return None

import argparse

def parse_argv():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='这是一个处理三个参数的示例程序')
    
    # 添加三个必需的位置参数
    parser.add_argument('-i','--input_file', default="classified_output/binary/madin_carbsubs_caprate.csv",help='输入文件路径')
    parser.add_argument('-o','--output_dir', default="binary",help='输出目录路径')
    parser.add_argument('-e','--embedding_dir', default="embeddings_4000_v1_a2",help='运行模式')
    parser.add_argument('--force', '-f', action='store_true', default=False, help='强制覆盖已存在文件')
    
    # 解析参数
    args = parser.parse_args()
    
    # 使用参数
    print(f"输入文件: {args.input_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"运行模式: {args.embedding_dir}")
    
    return args

# ========== 使用示例 ========== #
if __name__ == "__main__":
    args = parse_argv()
    csv_path = args.input_file ###, csv

    # 按文件名自动创建输出子目录
    
    feature_marker = "our_ft"
    
    if args.force:
        feature_marker = "bacformer_ft"

    output_dir = os.path.join(
        args.output_dir,
        get_file_stem(csv_path), feature_marker
    )
    data_folder = args.embedding_dir

    classifier = BinaryClassificationModel(
        csv_path=csv_path,
        data_folder=data_folder ,  # 替换为你的数据文件夹
        bacformer=args.force,
        test_size=0.2,   # 4:1 分割
        random_state=42,
        outdir=output_dir # 输出目录
    )

    model = classifier.run(which_model="xgboost")  # "logistic" | "rf" | "xgboost"

    if model:
        print("\n训练完成并已保存所有 CSV/图像结果与模型。")
    else:
        print("\n训练失败，请检查数据和代码。")
