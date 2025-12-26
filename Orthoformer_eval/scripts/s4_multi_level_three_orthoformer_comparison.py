"""
Multi-Level Three Orthoformer Models Comparison
Compare model_2048_v18 vs model_3M_2048_v8 vs model_3M_2048_v10 across different taxonomic levels
(phylum, class, order, family, genus)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

def parse_lineage(lineage_str):
    """Parse GTDB lineage string"""
    parts = lineage_str.split(';')
    taxonomy = {}
    
    for part in parts:
        part = part.strip()
        if part.startswith('d__'):
            taxonomy['domain'] = part[3:]
        elif part.startswith('p__'):
            taxonomy['phylum'] = part[3:]
        elif part.startswith('c__'):
            taxonomy['class'] = part[3:]
        elif part.startswith('o__'):
            taxonomy['order'] = part[3:]
        elif part.startswith('f__'):
            taxonomy['family'] = part[3:]
        elif part.startswith('g__'):
            taxonomy['genus'] = part[3:]
        elif part.startswith('s__'):
            taxonomy['species'] = part[3:]
    
    return taxonomy

def load_embeddings(embedding_dir, genome_ids):
    """Load embeddings for specified genome IDs"""
    embeddings = []
    valid_ids = []
    
    for genome_id in genome_ids:
        emb_file = os.path.join(embedding_dir, f"{genome_id}.npy")
        if os.path.exists(emb_file):
            emb = np.load(emb_file)
            embeddings.append(emb)
            valid_ids.append(genome_id)
    
    if len(embeddings) == 0:
        return None, []
    
    embeddings = np.array(embeddings)
    return embeddings, valid_ids

def prepare_labels(genome_info_df, genome_ids, label_level='phylum'):
    """Prepare labels for evaluation"""
    labels = []
    label_names = []
    
    for genome_id in genome_ids:
        row = genome_info_df[genome_info_df['user_genome'] == genome_id]
        if len(row) > 0:
            label = row[label_level].values[0]
            labels.append(label)
            label_names.append(label)
        else:
            labels.append('Unknown')
            label_names.append('Unknown')
    
    unique_labels = list(set(label_names))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    numeric_labels = np.array([label_to_id[label] for label in label_names])
    
    return numeric_labels, label_names, unique_labels

def calculate_metrics(embeddings, numeric_labels, n_clusters):
    """Calculate all evaluation metrics"""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(embeddings_scaled)
    
    # Calculate metrics
    metrics = {
        'ARI': adjusted_rand_score(numeric_labels, pred_labels),
        'NMI': normalized_mutual_info_score(numeric_labels, pred_labels),
        'AMI': adjusted_mutual_info_score(numeric_labels, pred_labels),
        'Silhouette': silhouette_score(embeddings_scaled, pred_labels),
        'Davies_Bouldin': davies_bouldin_score(embeddings_scaled, pred_labels),
        'Calinski_Harabasz': calinski_harabasz_score(embeddings_scaled, pred_labels)
    }
    
    return metrics

def plot_pca_comparison(embeddings_dict, label_names, unique_labels, level, output_dir, genome_ids=None):
    """Plot PCA comparison for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    n_colors = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
    if n_colors > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_colors))
    
    # Store PCA data for all models
    pca_data_all = []
    
    for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embeddings_scaled)
        explained_var = pca.explained_variance_ratio_
        
        # Save PCA coordinates to DataFrame
        if genome_ids is not None:
            pca_df = pd.DataFrame({
                'genome_id': genome_ids,
                'label': label_names,
                'PC1': reduced[:, 0],
                'PC2': reduced[:, 1],
                'PC1_explained_var': explained_var[0],
                'PC2_explained_var': explained_var[1]
            })
            pca_file = os.path.join(output_dir, f'{level}_{model_name}_PCA.csv')
            pca_df.to_csv(pca_file, index=False)
            print(f"  Saved PCA data: {pca_file}")
            pca_data_all.append((model_name, pca_df))
        
        ax = axes[idx]
        for i, label in enumerate(unique_labels):
            mask = np.array(label_names) == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                      c=[colors[i]], label=label if len(unique_labels) <= 20 else '', 
                      alpha=0.6, s=30)
        
        ax.set_xlabel(f'PC1 ({explained_var[0]:.1%})', fontsize=11)
        ax.set_ylabel(f'PC2 ({explained_var[1]:.1%})', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        
        if len(unique_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    plt.suptitle(f'PCA Comparison - {level.capitalize()} Level', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{level}_PCA_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def plot_tsne_comparison(embeddings_dict, label_names, unique_labels, level, output_dir, genome_ids=None):
    """Plot t-SNE comparison for all three models"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    
    n_colors = len(unique_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, min(n_colors, 20)))
    if n_colors > 20:
        colors = plt.cm.hsv(np.linspace(0, 1, n_colors))
    
    for idx, (model_name, embeddings) in enumerate(embeddings_dict.items()):
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(embeddings_scaled)
        
        # Save TSNE coordinates to DataFrame
        if genome_ids is not None:
            tsne_df = pd.DataFrame({
                'genome_id': genome_ids,
                'label': label_names,
                'TSNE1': reduced[:, 0],
                'TSNE2': reduced[:, 1]
            })
            tsne_file = os.path.join(output_dir, f'{level}_{model_name}_TSNE.csv')
            tsne_df.to_csv(tsne_file, index=False)
            print(f"  Saved TSNE data: {tsne_file}")
        
        ax = axes[idx]
        for i, label in enumerate(unique_labels):
            mask = np.array(label_names) == label
            ax.scatter(reduced[mask, 0], reduced[mask, 1], 
                      c=[colors[i]], label=label if len(unique_labels) <= 20 else '', 
                      alpha=0.6, s=30)
        
        ax.set_xlabel('t-SNE 1', fontsize=11)
        ax.set_ylabel('t-SNE 2', fontsize=11)
        ax.set_title(f'{model_name}', fontsize=13, fontweight='bold')
        
        if len(unique_labels) <= 20:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
    
    plt.suptitle(f't-SNE Comparison - {level.capitalize()} Level', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{level}_TSNE_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def plot_metrics_comparison(metrics_df, level, output_dir):
    """Plot metrics comparison bar chart for three models"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metric_names = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
    model_names = metrics_df.index.tolist()
    colors_models = ['#3498db', '#9b59b6', '#e74c3c']  # Blue for v18, Purple for v8, Red for v10
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        values = metrics_df[metric].values
        bars = ax.bar(model_names, values, color=colors_models[:len(model_names)], 
                     alpha=0.8, edgecolor='black')
        
        # Highlight best performer
        if metric == 'Davies_Bouldin':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)
        
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle(f'Metrics Comparison - {level.capitalize()} Level', 
                 fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, f'{level}_metrics_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def plot_heatmap(embeddings, label_names, unique_labels, model_name, level, output_dir, genome_ids=None):
    """Plot distance heatmap for a single model"""
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    
    # Calculate full distance matrix for saving
    distances_full = squareform(pdist(embeddings_scaled, metric='euclidean'))
    
    # Save full distance matrix
    if genome_ids is not None:
        # Create DataFrame with genome IDs as index and columns
        distance_df = pd.DataFrame(
            distances_full,
            index=genome_ids,
            columns=genome_ids
        )
        heatmap_file = os.path.join(output_dir, f'{level}_{model_name}_heatmap_matrix.csv')
        distance_df.to_csv(heatmap_file)
        print(f"  Saved heatmap matrix: {heatmap_file}")
        
        # Also save label mapping for reference
        label_df = pd.DataFrame({
            'genome_id': genome_ids,
            'label': label_names
        })
        label_file = os.path.join(output_dir, f'{level}_{model_name}_labels.csv')
        label_df.to_csv(label_file, index=False)
        print(f"  Saved label mapping: {label_file}")
    
    # Sample if too large for visualization
    if len(embeddings) > 200:
        indices = np.random.choice(len(embeddings), 200, replace=False)
        embeddings_scaled = embeddings_scaled[indices]
        label_names_subset = [label_names[i] for i in indices]
        distances = distances_full[np.ix_(indices, indices)]
    else:
        label_names_subset = label_names
        distances = distances_full
    
    sorted_indices = np.argsort(label_names_subset)
    distances_sorted = distances[sorted_indices][:, sorted_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(distances_sorted, cmap='RdBu_r', aspect='auto')
    ax.set_title(f'{model_name} - Distance Heatmap ({level.capitalize()})', 
                 fontsize=13, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Euclidean Distance', rotation=270, labelpad=20)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{level}_{model_name}_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def evaluate_level(genome_info_df, embedding_dirs, level, output_dir):
    """Evaluate all three models at a specific taxonomic level"""
    print(f"\n{'='*80}")
    print(f"EVALUATING AT {level.upper()} LEVEL")
    print(f"{'='*80}")
    
    # Get genome IDs
    genome_ids = genome_info_df['user_genome'].tolist()
    
    # Load embeddings for all models
    print(f"\nLoading embeddings...")
    embeddings_dict = {}
    valid_ids_dict = {}
    
    for model_name, emb_dir in embedding_dirs.items():
        embeddings, valid_ids = load_embeddings(emb_dir, genome_ids)
        if embeddings is not None:
            embeddings_dict[model_name] = embeddings
            valid_ids_dict[model_name] = valid_ids
            print(f"  {model_name}: {len(embeddings)} embeddings")
    
    # Find common IDs across all three models
    common_ids = set(valid_ids_dict['model_2048_v18']) & \
                 set(valid_ids_dict['model_3M_2048_v8']) & \
                 set(valid_ids_dict['model_3M_2048_v10'])
    common_ids = list(common_ids)
    print(f"\nCommon genomes across all models: {len(common_ids)}")
    
    # Filter to common IDs
    for model_name in embeddings_dict.keys():
        indices = [valid_ids_dict[model_name].index(gid) for gid in common_ids]
        embeddings_dict[model_name] = embeddings_dict[model_name][indices]
    
    # Prepare labels
    numeric_labels, label_names, unique_labels = prepare_labels(
        genome_info_df, common_ids, label_level=level
    )
    
    # Remove 'Unknown' if present
    if 'Unknown' in unique_labels:
        mask = np.array(label_names) != 'Unknown'
        numeric_labels = numeric_labels[mask]
        label_names = [l for l, m in zip(label_names, mask) if m]
        unique_labels = [l for l in unique_labels if l != 'Unknown']
        for model_name in embeddings_dict.keys():
            embeddings_dict[model_name] = embeddings_dict[model_name][mask]
        common_ids = [cid for cid, m in zip(common_ids, mask) if m]
        # Recompute numeric labels
        label_to_id = {label: i for i, label in enumerate(unique_labels)}
        numeric_labels = np.array([label_to_id[label] for label in label_names])
    
    n_samples = len(common_ids)
    n_clusters = len(unique_labels)
    print(f"Samples after filtering: {n_samples}")
    print(f"Unique {level}: {n_clusters}")
    
    if n_samples < 10 or n_clusters < 2:
        print(f"  WARNING: Too few samples or clusters, skipping {level} level")
        return None
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    metrics_results = {}
    for model_name, embeddings in embeddings_dict.items():
        print(f"  {model_name}...")
        metrics = calculate_metrics(embeddings, numeric_labels, n_clusters)
        metrics_results[model_name] = metrics
    
    metrics_df = pd.DataFrame(metrics_results).T
    print(f"\nMetrics Summary:")
    print(metrics_df.to_string())
    
    # Save metrics
    metrics_file = os.path.join(output_dir, f'{level}_metrics.csv')
    metrics_df.to_csv(metrics_file)
    print(f"\nSaved metrics to: {metrics_file}")
    
    # Generate visualizations
    print(f"\nGenerating visualizations...")
    
    # PCA comparison
    print("  1. PCA comparison...")
    plot_pca_comparison(embeddings_dict, label_names, unique_labels, level, output_dir, genome_ids=common_ids)
    
    # t-SNE comparison
    print("  2. t-SNE comparison...")
    plot_tsne_comparison(embeddings_dict, label_names, unique_labels, level, output_dir, genome_ids=common_ids)
    
    # Metrics comparison
    print("  3. Metrics comparison...")
    plot_metrics_comparison(metrics_df, level, output_dir)
    
    # Heatmaps for each model
    print("  4. Heatmaps...")
    for model_name, embeddings in embeddings_dict.items():
        plot_heatmap(embeddings, label_names, unique_labels, model_name, level, output_dir, genome_ids=common_ids)
    
    return metrics_df

def create_summary_report(all_metrics, output_dir):
    """Create comprehensive summary report"""
    print(f"\n{'='*80}")
    print("CREATING SUMMARY REPORT")
    print(f"{'='*80}")
    
    # Combine all metrics
    summary_data = []
    for level, metrics_df in all_metrics.items():
        for model_name in metrics_df.index:
            row = {'Level': level, 'Model': model_name}
            row.update(metrics_df.loc[model_name].to_dict())
            summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary
    summary_file = os.path.join(output_dir, 'three_orthoformer_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Saved summary to: {summary_file}")
    
    # Create summary visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    metric_names = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
    levels = summary_df['Level'].unique()
    models = summary_df['Model'].unique()
    
    colors_models = {
        'model_2048_v18': '#3498db', 
        'model_3M_2048_v8': '#9b59b6',
        'model_3M_2048_v10': '#e74c3c'
    }
    
    for idx, metric in enumerate(metric_names):
        ax = axes[idx]
        
        x = np.arange(len(levels))
        width = 0.25  # Adjusted width for three bars
        
        for i, model in enumerate(models):
            values = []
            for level in levels:
                val = summary_df[(summary_df['Level'] == level) & 
                                (summary_df['Model'] == model)][metric].values
                values.append(val[0] if len(val) > 0 else 0)
            
            ax.bar(x + i*width, values, width, label=model, 
                  color=colors_models.get(model, '#95a5a6'), alpha=0.8)
        
        ax.set_xlabel('Taxonomic Level', fontsize=11)
        ax.set_ylabel(metric, fontsize=11, fontweight='bold')
        ax.set_title(metric, fontsize=12)
        ax.set_xticks(x + width)
        ax.set_xticklabels(levels, rotation=15)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.suptitle('Multi-Level Performance Comparison (v18 vs v8 vs v10)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_plot = os.path.join(output_dir, 'three_orthoformer_summary.png')
    plt.savefig(summary_plot, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot to: {summary_plot}")
    
    return summary_df

def main():
    base_dir = "../embeddings"
    genome_info_file = os.path.join(base_dir, 'sampled_genomes_multi_level_info.csv')
    output_base_dir = os.path.join(base_dir, 'three_orthoformer_comparison_results')
    
    embedding_dirs = {
        'model_2048_v18': os.path.join(base_dir, 'model_2048_v18'),
        'model_3M_2048_v8': os.path.join(base_dir, 'model_3M_2048_v8'),
        'model_3M_2048_v10': os.path.join(base_dir, 'model_3M_2048_v10')
    }
    
    levels = ['phylum', 'class', 'order', 'family', 'genus']
    
    print("="*80)
    print("THREE ORTHOFORMER MODELS COMPARISON")
    print("="*80)
    print(f"Levels to evaluate: {', '.join(levels)}")
    print(f"Models: {', '.join(embedding_dirs.keys())}")
    
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Load genome info
    print(f"\nLoading genome information from {genome_info_file}...")
    genome_info_df = pd.read_csv(genome_info_file)
    print(f"Total genomes in info file: {len(genome_info_df)}")
    
    # Evaluate each level
    all_metrics = {}
    for level in levels:
        output_dir = os.path.join(output_base_dir, level)
        os.makedirs(output_dir, exist_ok=True)
        
        metrics_df = evaluate_level(genome_info_df, embedding_dirs, level, output_dir)
        if metrics_df is not None:
            all_metrics[level] = metrics_df
    
    # Create summary report
    if all_metrics:
        summary_df = create_summary_report(all_metrics, output_base_dir)
        
        print(f"\n{'='*80}")
        print("MULTI-LEVEL EVALUATION COMPLETED")
        print(f"{'='*80}")
        print(f"\nResults saved to: {output_base_dir}")
        print(f"\nEvaluated levels: {', '.join(all_metrics.keys())}")
        print(f"\nSummary:")
        print(summary_df.to_string())
    else:
        print("\nNo valid results obtained!")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()



