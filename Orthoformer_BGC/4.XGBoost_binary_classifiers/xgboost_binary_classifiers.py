#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XGBoost Binary Classifier for BGC abundance analysis.

This script performs binary classification using XGBoost with:
- Adaptive hyperparameter tuning based on dataset size
- Class imbalance handling
- Two-phase grid search (coarse-to-fine)
- Cross-validation (Leave-One-Out for small datasets, Stratified K-Fold otherwise)
- Visualization (t-SNE, ROC curve, feature importance, boxplots)
"""

import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import (
    train_test_split, GridSearchCV, LeaveOneOut,
    StratifiedKFold, cross_validate, cross_val_predict
)
from sklearn.metrics import (
    roc_curve, auc, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, roc_auc_score
)
from sklearn.manifold import TSNE
import xgboost as xgb

warnings.filterwarnings('ignore')


def remove_low_frequency_features(X, feature_cols, threshold=0.05):
    """
    Remove features that appear in fewer than threshold proportion of samples.

    Args:
        X: Feature matrix (numpy array).
        feature_cols: List of feature column names.
        threshold: Minimum proportion of samples with non-zero values.

    Returns:
        Filtered feature matrix and list of remaining feature names.
    """
    non_zero_ratio = (X > 0).mean(axis=0)
    keep_features = non_zero_ratio >= threshold

    removed_features = [f for f, keep in zip(feature_cols, keep_features) if not keep]
    kept_features = [f for f, keep in zip(feature_cols, keep_features) if keep]

    if removed_features:
        print(f"Removed low-frequency features (<{threshold*100}% samples): {removed_features}")

    return X[:, keep_features], kept_features


def get_class_weights(y, imbalance_ratio):
    """
    Calculate class weights based on imbalance ratio.

    Args:
        y: Target labels.
        imbalance_ratio: Ratio of majority to minority class.

    Returns:
        Class weight dictionary or 'balanced' string.
    """
    classes = np.unique(y)
    class_counts = pd.Series(y).value_counts()

    if imbalance_ratio <= 1.5:
        return 'balanced'
    elif imbalance_ratio <= 2.0:
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        return {minority_class: 1.5, majority_class: 1.0}
    else:
        minority_class = class_counts.idxmin()
        majority_class = class_counts.idxmax()
        return {minority_class: 2.0, majority_class: 1.0}


def get_xgboost_params(n_samples):
    """
    Get XGBoost parameter grid based on dataset size.

    Args:
        n_samples: Number of samples in the dataset.

    Returns:
        Parameter grid dictionary for GridSearchCV.
    """
    if n_samples < 200:
        param_grid = {
            'max_depth': [3, 4, 5],
            'learning_rate': [0.05, 0.1, 0.2],
            'n_estimators': [50, 100],
            'subsample': [0.7, 0.8],
            'colsample_bytree': [0.7, 0.8],
            'min_child_weight': [3, 5],
            'reg_alpha': [0.1, 0.5],
            'reg_lambda': [1, 2]
        }
    elif n_samples < 1000:
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.05, 0.1, 0.15],
            'n_estimators': [100, 150, 200],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 3],
            'reg_alpha': [0, 0.1],
            'reg_lambda': [1, 1.5]
        }
    else:
        param_grid = {
            'max_depth': [6, 8, 10],
            'learning_rate': [0.03, 0.05, 0.1],
            'n_estimators': [150, 200, 300],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9],
            'min_child_weight': [1, 2],
            'reg_alpha': [0, 0.01],
            'reg_lambda': [0.5, 1]
        }

    return param_grid


def coarse_to_fine_search(X, y, param_grid, cv, scoring='f1_weighted', n_jobs=-1):
    """
    Perform two-phase grid search: coarse then fine.

    Args:
        X: Feature matrix.
        y: Target labels.
        param_grid: Full parameter grid.
        cv: Cross-validation strategy.
        scoring: Scoring metric.
        n_jobs: Number of parallel jobs.

    Returns:
        Best parameters and best CV score.
    """
    coarse_params = {
        'max_depth': param_grid['max_depth'][::2],
        'learning_rate': param_grid['learning_rate'][::2],
        'n_estimators': param_grid['n_estimators'][::2],
        'subsample': [param_grid['subsample'][0]],
        'colsample_bytree': [param_grid['colsample_bytree'][0]],
        'min_child_weight': [param_grid['min_child_weight'][0]],
        'reg_alpha': [param_grid['reg_alpha'][0]],
        'reg_lambda': [param_grid['reg_lambda'][0]]
    }

    print("Phase 1: Coarse grid search...")
    xgb_model = xgb.XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        base_score=0.5
    )
    coarse_search = GridSearchCV(
        xgb_model, coarse_params, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
    )
    coarse_search.fit(X, y)

    best_coarse = coarse_search.best_params_
    print(f"Best coarse params: {best_coarse}")

    fine_params = {}

    best_depth = best_coarse['max_depth']
    fine_params['max_depth'] = [max(3, best_depth-1), best_depth, min(10, best_depth+1)]

    best_lr = best_coarse['learning_rate']
    fine_params['learning_rate'] = [max(0.01, best_lr*0.7), best_lr, min(0.3, best_lr*1.3)]

    best_n = best_coarse['n_estimators']
    fine_params['n_estimators'] = [max(50, best_n-50), best_n, min(300, best_n+50)]

    fine_params.update({
        'subsample': param_grid['subsample'],
        'colsample_bytree': param_grid['colsample_bytree'],
        'min_child_weight': param_grid['min_child_weight'],
        'reg_alpha': param_grid['reg_alpha'],
        'reg_lambda': param_grid['reg_lambda']
    })

    print("\nPhase 2: Fine grid search...")
    fine_search = GridSearchCV(
        xgb_model, fine_params, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
    )
    fine_search.fit(X, y)

    return fine_search.best_params_, fine_search.best_score_


def calculate_significance(data1, data2):
    """
    Calculate statistical significance using Mann-Whitney U test.

    Args:
        data1: First group data.
        data2: Second group data.

    Returns:
        Significance symbol and p-value.
    """
    stat, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')

    if p_value < 0.001:
        return '***', p_value
    elif p_value < 0.01:
        return '**', p_value
    elif p_value < 0.05:
        return '*', p_value
    else:
        return 'ns', p_value


def main(input_file, output_prefix, n_jobs=-1, random_state=42):
    """
    Main function for XGBoost binary classification analysis.

    Args:
        input_file: Path to input CSV file with features and Group_Label column.
        output_prefix: Prefix for output files.
        n_jobs: Number of parallel jobs for grid search.
        random_state: Random seed for reproducibility.
    """
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.2,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10
    })

    df = pd.read_csv(input_file)

    original_feature_cols = [
        'mixed', 'nrps', 'nrps related peptides', 'other',
        'pks', 'ripps', 'saccharides & derivatives', 'terpenes'
    ]
    feature_cols = original_feature_cols.copy()

    X = df[feature_cols].values
    y = df['Group_Label']

    n_samples = len(X)
    classes = sorted(y.unique())
    class_counts = y.value_counts()
    n_minority = min(class_counts.values)
    imbalance_ratio = max(class_counts.values) / n_minority

    print(f"\n{'='*60}")
    print(f"Dataset Analysis:")
    print(f"{'='*60}")
    print(f"Total samples: {n_samples}")
    print(f"Class distribution: {dict(class_counts)}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    X, feature_cols = remove_low_frequency_features(X, feature_cols, threshold=0.05)
    print(f"Remaining features: {len(feature_cols)}")

    if len(feature_cols) == 0:
        print("ERROR: All features were removed! Adjusting threshold...")
        X = df[original_feature_cols].values
        X, feature_cols = remove_low_frequency_features(X, original_feature_cols, threshold=0.01)

    print(f"Classes after filtering: {classes}")
    print(f"Samples per class: {y.value_counts().to_dict()}")

    if len(classes) < 2:
        print("ERROR: Less than 2 classes remaining. Cannot perform binary classification.")
        return

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\nRunning t-SNE...")
    perplexity = min(30, len(X) / 4)
    tsne = TSNE(
        n_components=2, random_state=random_state, perplexity=perplexity,
        init='random', learning_rate='auto'
    )
    X_tsne = tsne.fit_transform(X_scaled)

    if n_samples < 150:
        print("\nUsing Leave-One-Out CV (small dataset)")
        cv = LeaveOneOut()
        grid_cv = 3
    else:
        print("\nUsing 5-Fold Stratified CV")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        grid_cv = cv

    param_grid = get_xgboost_params(n_samples)

    class_weight = get_class_weights(y, imbalance_ratio)
    if isinstance(class_weight, dict):
        minority_class = list(class_weight.keys())[0]
        scale_pos_weight = class_weight[minority_class]
    else:
        scale_pos_weight = n_samples / (2 * n_minority)

    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    best_params, best_cv_score = coarse_to_fine_search(
        X, y_encoded, param_grid, grid_cv, scoring='f1_weighted', n_jobs=n_jobs
    )

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV F1-score: {best_cv_score:.3f}")

    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        use_label_encoder=False,
        eval_metric='logloss',
        base_score=0.5
    )

    if n_samples < 150:
        y_proba = cross_val_predict(
            final_model, X, y_encoded, cv=cv, method='predict_proba', n_jobs=n_jobs
        )

        fpr, tpr, thresholds = roc_curve(y_encoded, y_proba[:, 1])
        roc_auc = auc(fpr, tpr)

        cv_results = cross_validate(
            final_model, X, y_encoded, cv=cv,
            scoring=['accuracy', 'f1_weighted'],
            n_jobs=n_jobs
        )

        print(f"\nLOOCV Results:")
        print(f"Accuracy: {cv_results['test_accuracy'].mean():.3f} "
              f"(+/- {cv_results['test_accuracy'].std()*2:.3f})")
        print(f"F1-score: {cv_results['test_f1_weighted'].mean():.3f} "
              f"(+/- {cv_results['test_f1_weighted'].std()*2:.3f})")
        print(f"ROC AUC: {roc_auc:.3f}")

        if cv_results['test_accuracy'].mean() > 0.95:
            print("\nWARNING: Very high accuracy detected. Possible overfitting!")
            print("Consider reducing model complexity or using more regularization.")

        final_model.fit(X, y_encoded)
        feature_importance = final_model.feature_importances_

    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded
        )

        final_model.fit(X_train, y_train)
        feature_importance = final_model.feature_importances_

        y_pred = final_model.predict(X_test)
        y_pred_proba = final_model.predict_proba(X_test)[:, 1]

        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        mcc = matthews_corrcoef(y_test, y_pred)

        y_test_original = le.inverse_transform(y_test)
        y_pred_original = le.inverse_transform(y_pred)

        print(f"\nTest Set Results:")
        print(classification_report(y_test_original, y_pred_original))
        print(f"MCC: {mcc:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")

    # Visualization
    colors_rdylbu = ['#d73027', '#4575b4']

    fig = plt.figure(figsize=(14, 12))

    # t-SNE plot
    ax1 = plt.subplot(2, 2, 1)
    for idx, class_name in enumerate(classes):
        mask = y == class_name
        ax1.scatter(
            X_tsne[mask, 0], X_tsne[mask, 1],
            label=f'{class_name} (n={sum(mask)})',
            alpha=0.7, s=50,
            color=colors_rdylbu[idx],
            edgecolors='white', linewidth=0.8
        )
    ax1.set_xlabel('tSNE1', fontweight='bold')
    ax1.set_ylabel('tSNE2', fontweight='bold')
    ax1.set_title('t-SNE Visualization', fontweight='bold', pad=10)
    ax1.legend(frameon=True, fancybox=True, shadow=True, framealpha=0.5, facecolor='white')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # ROC curve
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(fpr, tpr, color='#d73027', lw=2.5,
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax2.plot([0, 1], [0, 1], color='#999999', lw=2, linestyle='--',
             label='Random classifier')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate', fontweight='bold')
    ax2.set_ylabel('True Positive Rate', fontweight='bold')
    ax2.set_title('ROC Curve', fontweight='bold', pad=10)
    ax2.legend(loc="lower right", frameon=True, fancybox=True,
               shadow=True, framealpha=0.5, facecolor='white')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Feature importance
    ax3 = plt.subplot(2, 2, 3)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    colors_importance = plt.cm.RdYlBu_r(np.linspace(0.3, 0.7, len(importance_df)))
    bars = ax3.barh(
        range(len(importance_df)), importance_df['importance'],
        color=colors_importance, edgecolor='white', linewidth=1.5
    )
    ax3.set_yticks(range(len(importance_df)))
    ax3.set_yticklabels(importance_df['feature'])
    ax3.set_xlabel('Feature Importance', fontweight='bold')
    ax3.set_title('XGBoost Feature Importance', fontweight='bold', pad=10)
    ax3.invert_yaxis()
    for i, v in enumerate(importance_df['importance']):
        ax3.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9, fontweight='bold')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    ax3.grid(True, alpha=0.2, axis='x', linestyle='--')

    # Boxplot for top features
    ax4 = plt.subplot(2, 2, 4)
    top_features = importance_df.head(4)['feature'].tolist()
    data_for_box = []
    for feature in top_features:
        for class_name in classes:
            mask = y == class_name
            values = df.loc[mask, feature].values
            for v in values:
                data_for_box.append({
                    'Feature': feature,
                    'Class': class_name,
                    'Value': v
                })

    box_df = pd.DataFrame(data_for_box)

    positions = []
    box_colors = []
    for i, feature in enumerate(top_features):
        for j, class_name in enumerate(classes):
            positions.append(i * (len(classes) + 0.5) + j)
            box_colors.append(colors_rdylbu[j])

    feature_data = []
    for feature in top_features:
        for class_name in classes:
            subset = box_df[(box_df['Feature'] == feature) & (box_df['Class'] == class_name)]
            feature_data.append(subset['Value'].values)

    bp = ax4.boxplot(
        feature_data, positions=positions, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor='white', linewidth=1.5),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(linewidth=1.5),
        capprops=dict(linewidth=1.5),
        flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.3)
    )

    for patch, color in zip(bp['boxes'], box_colors * len(top_features)):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    xtick_positions = [i * (len(classes) + 0.5) + 0.5 for i in range(len(top_features))]
    ax4.set_xticks(xtick_positions)
    ax4.set_xticklabels(top_features, rotation=45, ha='right')
    ax4.set_ylabel('Value', fontweight='bold')
    ax4.set_title('Top 4 Features Distribution', fontweight='bold', pad=10)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    ax4.grid(True, alpha=0.2, axis='y', linestyle='--')

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_rdylbu[i], alpha=0.7, label=classes[i])
        for i in range(len(classes))
    ]
    ax4.legend(
        handles=legend_elements, loc='upper right', frameon=True,
        fancybox=True, shadow=True, framealpha=0.5, facecolor='white'
    )

    y_max = box_df['Value'].max()
    y_min = box_df['Value'].min()
    y_range = y_max - y_min

    for idx, feature in enumerate(top_features):
        class1_data = df.loc[y == classes[0], feature].values
        class2_data = df.loc[y == classes[1], feature].values

        sig_symbol, p_val = calculate_significance(class1_data, class2_data)

        if sig_symbol != 'ns':
            x_center = idx * (len(classes) + 0.5) + 0.5
            y_pos = y_max + y_range * 0.1

            line_x = [x_center - 0.4, x_center - 0.4, x_center + 0.4, x_center + 0.4]
            line_y = [y_pos - y_range * 0.02, y_pos, y_pos, y_pos - y_range * 0.02]
            ax4.plot(line_x, line_y, 'k-', lw=1.5)

            ax4.text(
                x_center, y_pos + y_range * 0.03, sig_symbol,
                ha='center', va='bottom', fontsize=12, fontweight='bold'
            )

        print(f"{feature}: {sig_symbol} (p = {p_val:.4f})")

    ax4.set_ylim(y_min - y_range * 0.05, y_max + y_range * 0.25)

    plt.tight_layout()
    plt.savefig(f'{output_prefix}_xgboost_analysis.png', bbox_inches='tight', dpi=300)
    plt.savefig(f'{output_prefix}_xgboost_analysis.pdf', bbox_inches='tight')
    plt.close()

    # Save results
    importance_df.to_csv(f'{output_prefix}_feature_importance.csv', index=False)

    params_df = pd.DataFrame([best_params])
    params_df.to_csv(f'{output_prefix}_best_params.csv', index=False)

    coords_df = pd.DataFrame({
        'sample': df.index,
        'Group_Label': y,
        'tsne1': X_tsne[:, 0],
        'tsne2': X_tsne[:, 1]
    })
    coords_df.to_csv(f'{output_prefix}_coordinates.csv', index=False)

    roc_df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'threshold': thresholds
    })
    roc_df.to_csv(f'{output_prefix}_roc_data.csv', index=False)

    significance_results = []
    for feature in top_features:
        class1_data = df.loc[y == classes[0], feature].values
        class2_data = df.loc[y == classes[1], feature].values
        sig_symbol, p_val = calculate_significance(class1_data, class2_data)
        significance_results.append({
            'feature': feature,
            'significance': sig_symbol,
            'p_value': p_val
        })

    sig_df = pd.DataFrame(significance_results)
    sig_df.to_csv(f'{output_prefix}_significance_test.csv', index=False)

    print(f"\n{'='*50}")
    print("Analysis complete! Files saved:")
    print(f"- {output_prefix}_xgboost_analysis.png/pdf")
    print(f"- {output_prefix}_feature_importance.csv")
    print(f"- {output_prefix}_best_params.csv")
    print(f"- {output_prefix}_coordinates.csv")
    print(f"- {output_prefix}_roc_data.csv")
    print(f"- {output_prefix}_significance_test.csv")
    print(f"{'='*50}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python xgboost_binary_classifiers.py <input_file> <output_prefix> [n_jobs]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_prefix = sys.argv[2]
    n_jobs = int(sys.argv[3]) if len(sys.argv) > 3 else -1

    main(input_file, output_prefix, n_jobs)
