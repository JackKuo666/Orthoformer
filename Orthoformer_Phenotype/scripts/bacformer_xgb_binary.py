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

# ========== Utility Functions ==========
def get_file_stem(path: str) -> str:
    """
    Given a file path, return the filename without extension
    Example: /a/b/c/file.txt -> file
    """
    return Path(path).stem


# ========== Main Class ========== #
class BinaryClassificationModel:
    def __init__(self, csv_path, bacformer, data_folder,
                 test_size=0.2, random_state=42,
                 outdir="/mnt/disk_c/user_data/liuping/bacformer/results"):
        """
        Initialize binary classification model
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
        Convert torch.Tensor(N, D) to numpy.ndarray(N, D)
        """
        if isinstance(x_tensor, torch.Tensor):
            return x_tensor.detach().cpu().numpy()
        return np.asarray(x_tensor)

    def load_and_preprocess_data(self):
        """
        Load CSV data and extract features from files
        Returns:
            X (torch.Tensor[N, D]),
            y (np.ndarray[N]),
            sample_names (list[str]),
            missing_files (list[str]),
            data_info (dict)
        """
        # Load CSV data
        df = pd.read_csv(self.csv_path)

        # Basic information
        total_samples = len(df)
        label_counts = df['Label'].value_counts().to_dict()
        positive_ratio = float(df['Label'].mean())

        print("Basic data information:")
        print(f"Total samples: {total_samples}")
        print(f"Class distribution: {label_counts}")
        print(f"Positive ratio: {positive_ratio:.3f}")

        if not os.path.exists(self.data_folder):
            raise ValueError(f"Data folder '{self.data_folder}' does not exist")

        X, y = [], []
        sample_names = []
        missing_files = []

        print("\nExtracting features from files...")
        for _, row in df.iterrows():
            orig_name = row['genome_name']
            genome_name = orig_name
            found_file = None
            base_path = os.path.join(self.data_folder, genome_name)

            # Try common extensions
            for ext in ['', '.fasta', '.fna', '.fa', '.txt', '.gbff', '.gbk', '.pt']:
                test_path = base_path + ext
                if os.path.exists(test_path):
                    found_file = test_path
                    break

            # Try RS_/GB_ prefix
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
                    # Extract features
                    if self.bacformer:
                        # String vector saved in CSV -> tensor
                        features_str = row['bacformer_genome_embedding']
                        vec = [float(x) for x in features_str.replace('\n', ' ')
                                                       .replace('[', '')
                                                       .replace(']', '')
                                                       .split()]
                        features = torch.tensor(vec, dtype=torch.float32)
                    else:
                        # Read embedding from .pt (or compatible with list/ndarray)
                        features = torch.load(found_file)
                        if isinstance(features, np.ndarray):
                            features = torch.tensor(features, dtype=torch.float32)
                        elif isinstance(features, list):
                            features = torch.tensor(features, dtype=torch.float32)
                        else:
                            features = features.to(dtype=torch.float32)

                    # Ensure 1D
                    features = features.flatten()

                    X.append(features)
                    y.append(int(row['Label']))
                    sample_names.append(genome_name)
                except Exception as e:
                    print(f"Error processing file {found_file}: {e}")
                    missing_files.append(genome_name)
            else:
                missing_files.append(genome_name)

        print(f"\nSuccessfully processed: {len(sample_names)} files")
        print(f"Missing files: {len(missing_files)}")
        if missing_files:
            print("Example missing files:", missing_files[:5])

        y = np.array(y)
        if len(X) == 0:
            X_tensor = torch.empty((0, 0), dtype=torch.float32)
        else:
            # Align dimensions (if vector lengths differ, zero-pad on the right)
            max_dim = max(x.numel() for x in X)
            X_padded = []
            for x in X:
                if x.numel() < max_dim:
                    pad = torch.zeros(max_dim - x.numel(), dtype=torch.float32)
                    X_padded.append(torch.cat([x, pad], dim=0))
                else:
                    X_padded.append(x)
            X_tensor = torch.stack(X_padded, dim=0)

        print(f"\nFeature matrix shape: {tuple(X_tensor.shape)}")
        if X_tensor.numel() > 0:
            print(f"Number of features: {X_tensor.shape[1]}")

        #data_info = {
        #    "total samples": total_samples,
        #    "processed samples": len(sample_names),
        #    "miss": len(missing_files),
        #    "positive ratio": positive_ratio,
        #    "distribution": label_counts
        #}
        data_info = {                                                                                         
                "total_samples": total_samples,
                "processed_samples": len(sample_names),
                "missing_files": len(missing_files),
                "positive_ratio": positive_ratio,
                "label_distribution": label_counts
        }   



        return X_tensor, y, sample_names, missing_files, data_info

    def train_test_split_data(self, X, y, sample_names):
        """
        Split training and test sets (maintain name alignment)
        """
        X_np = self._to_numpy_2d(X)
        names = np.array(sample_names)

        X_tr, X_te, y_tr, y_te, n_tr, n_te = train_test_split(
            X_np, y, names,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )

        print("\nData split results:")
        print(f"Training set: {X_tr.shape[0]} samples")
        print(f"Test set: {X_te.shape[0]} samples")
        print(f"Training set class distribution: {np.bincount(y_tr)}")
        print(f"Test set class distribution: {np.bincount(y_te)}")

        return X_tr, X_te, y_tr, y_te, n_tr.tolist(), n_te.tolist()

    # ===== Three models: choose as needed =====
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):
        print("\nTraining logistic regression model...")
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

        print("Logistic regression model performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    def train_random_forest(self, X_train, X_test, y_train, y_test):
        print("\nTraining random forest model...")
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

        print("Random forest model performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    def train_xgboost_classifier(self, X_train, X_test, y_train, y_test):
        print("\nTraining XGBoost classifier model...")
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

        print("XGBoost classifier model performance:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-score: {f1:.4f}")
        print(f"  ROC AUC: {roc_auc:.4f}")

        return results

    # ===== Visualization and Saving =====
    def plot_results(self, y_test, results, prefix="model"):
        """
        Plot and save result visualizations (confusion matrix/ROC/metric bar chart), optionally save feature importance plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Confusion matrix
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Prediction')
        axes[0, 0].set_ylabel('True')

        # 2. ROC curve
        fpr, tpr, _ = roc_curve(y_test, results['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        axes[0, 1].plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})', linewidth=2)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Performance metrics bar chart
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        metrics_label = ['Accuracy', 'Precision', 'Recall', 'F1-score', 'AUC']
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

        # 4. Reserved subplot position (can display feature importance thumbnail)
        axes[1, 1].axis('off')

        plt.tight_layout()

        # Save main result figure
        fig_path = os.path.join(self.outdir, f"{prefix}_results.pdf")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight',format='pdf')
        print(f"Result figure saved: {fig_path}")
        plt.close(fig)

        # Optional: save feature importance (if available)
        fi = results.get('feature_importance', None)
        if fi is not None:
            try:
                plt.figure(figsize=(10, 6))
                # If too many features, only plot top 30
                idx = np.argsort(fi)[::-1][:30]
                plt.bar(range(len(idx)), np.array(fi)[idx])
                plt.xticks(range(len(idx)), [f"f{int(i)}" for i in idx], rotation=90)
                plt.title(f"{prefix} - Top Feature Importances")
                plt.ylabel("Importance")
                fi_path = os.path.join(self.outdir, f"{prefix}_feature_importance.pdf")
                plt.tight_layout()
                plt.savefig(fi_path, dpi=300, bbox_inches='tight',format='pdf')
                print(f"Feature importance figure saved: {fi_path}")
                plt.close()
            except Exception as _:
                # Some models' coef_ shape may not match, or quantity too large
                pass

    # ===== Save Model =====
    def save_model(self, model, model_name):
        """
        Save trained model and scaler
        """
        model_path = os.path.join(self.outdir, f'{model_name}_model.pkl')
        scaler_path = os.path.join(self.outdir, 'scaler.pkl')
        joblib.dump(model, model_path)
        joblib.dump(self.scaler, scaler_path)
        print(f"Model and scaler saved:\n  {model_path}\n  {scaler_path}")

    # ===== CSV Saving: Data and Evaluation =====
    def save_results_to_csv(self, results, data_info, y_test, y_pred, y_pred_proba, test_names,
                            prefix: str):
        """
        Save the following tables:
        1) Training evaluation summary  -> {prefix}_summary.csv
        2) Per-class metrics            -> {prefix}_classification_report.csv
        3) Confusion matrix             -> {prefix}_confusion_matrix.csv
        4) Per-sample predictions       -> {prefix}_per_sample_predictions.csv
        """
        # 1) Training evaluation summary
        summary_dict = {
            "total samples": data_info["total_samples"],
            "processed samples": data_info["processed_samples"],
            "missing": data_info["missing_files"],
            "positive ratio": data_info["positive_ratio"],
            "label distribution": str(data_info["label_distribution"]),
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

        # 2) Per-class metrics
        cls_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(cls_report).transpose().reset_index().rename(columns={"index": "class"})
        rep_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_classification_report.csv"),
            index=False, encoding="utf-8"
        )

        # 3) Confusion matrix
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, columns=['pred_0', 'pred_1'])
        cm_df.insert(0, 'true', ['true_0', 'true_1'])
        cm_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_confusion_matrix.csv"),
            index=False, encoding="utf-8"
        )

        # 4) Per-sample predictions
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

        print(f"Result CSVs saved to directory: {self.outdir}")

    def save_missing_list(self, missing_files, prefix: str):
        """
        Save missing file list
        """
        if len(missing_files) == 0:
            return
        miss_df = pd.DataFrame({"missing_file": missing_files})
        miss_df.to_csv(
            os.path.join(self.outdir, f"{prefix}_missing_files.csv"),
            index=False, encoding="utf-8"
        )
        print(f"Missing file list saved: {prefix}_missing_files.csv")

    def run(self, which_model="xgboost"):
        """
        Run complete training pipeline
        which_model: "logistic" | "rf" | "xgboost"
        """
        print("Starting binary classification model training...")
        try:
            # 1) Load data + features
            X, y, names, missing_files, data_info = self.load_and_preprocess_data()
            if X.shape[0] == 0:
                print("No files successfully extracted features, cannot train model")
                return None

            # 2) Split data (including names)
            X_train, X_test, y_train, y_test, names_train, names_test = self.train_test_split_data(X, y, names)

            # 3) Train model (choose one)
            if which_model == "logistic":
                results = self.train_logistic_regression(X_train, X_test, y_train, y_test)
            elif which_model == "rf":
                results = self.train_random_forest(X_train, X_test, y_train, y_test)
            else:
                results = self.train_xgboost_classifier(X_train, X_test, y_train, y_test)

            # 4) Visualize and save images
            self.plot_results(y_test, results, prefix=results.get('model_name', 'model'))

            # 5) Save model
            self.save_model(results['model'], results.get('model_name', 'model'))

            # 6) Save CSV (summary, report, confusion matrix, per-sample)
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

            # 7) Missing file list
            self.save_missing_list(missing_files, prefix=prefix)

            # 8) Console output detailed classification report
            print("\nDetailed classification report:")
            print(classification_report(y_test, results['y_pred'], zero_division=0))

            return results['model']

        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return None

import argparse

def parse_argv():
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(description='Example program that processes three parameters')
    
    # Add three required positional parameters
    parser.add_argument('-i','--input_file', default="classified_output/binary/madin_carbsubs_caprate.csv",help='Input file path')
    parser.add_argument('-o','--output_dir', default="binary",help='Output directory path')
    parser.add_argument('-e','--embedding_dir', default="embeddings_4000_v1_a2",help='Run mode')
    parser.add_argument('--force', '-f', action='store_true', default=False, help='Force overwrite existing files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Use arguments
    print(f"Input file: {args.input_file}")
    print(f"Output directory: {args.output_dir}")
    print(f"Run mode: {args.embedding_dir}")
    
    return args

# ========== Usage Example ========== #
if __name__ == "__main__":
    args = parse_argv()
    csv_path = args.input_file ###, csv

    # Automatically create output subdirectory based on filename
    
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
        data_folder=data_folder ,  # Replace with your data folder
        bacformer=args.force,
        test_size=0.2,   # 4:1 split
        random_state=42,
        outdir=output_dir # Output directory
    )

    model = classifier.run(which_model="xgboost")  # "logistic" | "rf" | "xgboost"

    if model:
        print("\nTraining completed and all CSV/image results and model saved.")
    else:
        print("\nTraining failed, please check data and code.")
