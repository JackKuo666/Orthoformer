"""
Generate Additional Visualizations for Six-Model Comparison
Creates radar charts, heatmaps, and ranking visualizations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
import warnings
warnings.filterwarnings('ignore')

def create_radar_chart(summary_df, level, output_dir):
    """Create radar chart for model comparison at a specific level"""
    level_data = summary_df[summary_df['Level'] == level]
    
    # Select metrics for radar chart (normalize Davies-Bouldin by inverting)
    metrics = ['ARI', 'NMI', 'AMI', 'Silhouette']
    
    # Save radar chart data
    radar_data = level_data[['Model'] + metrics].copy()
    radar_file = os.path.join(output_dir, f'{level}_radar_chart.csv')
    radar_data.to_csv(radar_file, index=False)
    print(f"  Saved radar chart data: {radar_file}")
    
    # Number of variables
    num_vars = len(metrics)
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]
    
    # Initialize plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Colors for models
    colors = {
        'model_2048_v18': '#3498db',
        'model_3M_2048_v8': '#9b59b6',
        'model_3M_2048_v10': '#e74c3c',
        'Bacformer': '#f39c12',
        'Evo': '#2ecc71',
        'Evo_131k': '#1abc9c'
    }
    
    # Plot each model
    for idx, model in enumerate(level_data['Model'].values):
        values = level_data[level_data['Model'] == model][metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, 
               color=colors.get(model, '#95a5a6'))
        ax.fill(angles, values, alpha=0.15, color=colors.get(model, '#95a5a6'))
    
    # Fix axis to go in the right order
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=12)
    ax.set_ylim(0, 1)
    ax.set_title(f'Model Comparison - {level.capitalize()} Level', 
                size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f'{level}_radar_chart.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_performance_heatmap(summary_df, output_dir):
    """Create heatmap showing model performance across all levels"""
    # Prepare data for ARI metric
    pivot_data = summary_df.pivot(index='Model', columns='Level', values='ARI')
    
    # Reorder levels
    level_order = ['phylum', 'class', 'order', 'family', 'genus']
    pivot_data = pivot_data[level_order]
    
    # Save ARI heatmap data
    ari_file = os.path.join(output_dir, 'ari_heatmap_all_levels.csv')
    pivot_data.to_csv(ari_file)
    print(f"  Saved ARI heatmap data: {ari_file}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdBu_r', 
               cbar_kws={'label': 'ARI Score'}, ax=ax, vmin=0, vmax=0.6)
    ax.set_title('ARI Performance Across Taxonomic Levels', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Taxonomic Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ari_heatmap_all_levels.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")
    
    # Create heatmap for NMI
    pivot_data_nmi = summary_df.pivot(index='Model', columns='Level', values='NMI')
    pivot_data_nmi = pivot_data_nmi[level_order]
    
    # Save NMI heatmap data
    nmi_file = os.path.join(output_dir, 'nmi_heatmap_all_levels.csv')
    pivot_data_nmi.to_csv(nmi_file)
    print(f"  Saved NMI heatmap data: {nmi_file}")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_data_nmi, annot=True, fmt='.3f', cmap='RdBu_r', 
               cbar_kws={'label': 'NMI Score'}, ax=ax, vmin=0, vmax=1.0)
    ax.set_title('NMI Performance Across Taxonomic Levels', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Taxonomic Level', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'nmi_heatmap_all_levels.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_ranking_plot(summary_df, output_dir):
    """Create ranking visualization for each metric"""
    metrics = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
    levels = ['phylum', 'class', 'order', 'family', 'genus']
    
    # Collect all ranking data
    all_ranking_data = []
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        # Calculate rankings for each level
        ranking_data = []
        for level in levels:
            level_data = summary_df[summary_df['Level'] == level]
            if metric == 'Davies_Bouldin':
                level_data = level_data.sort_values(metric, ascending=True)
            else:
                level_data = level_data.sort_values(metric, ascending=False)
            
            for rank, (_, row) in enumerate(level_data.iterrows(), 1):
                ranking_data.append({
                    'Level': level,
                    'Model': row['Model'],
                    'Rank': rank,
                    'Value': row[metric]
                })
        
        ranking_df = pd.DataFrame(ranking_data)
        all_ranking_data.append(ranking_df)
        
        # Create line plot
        colors = {
            'model_2048_v18': '#3498db',
            'model_3M_2048_v8': '#9b59b6',
            'model_3M_2048_v10': '#e74c3c',
            'Bacformer': '#f39c12',
            'Evo': '#2ecc71',
            'Evo_131k': '#1abc9c'
        }
        
        for model in ranking_df['Model'].unique():
            model_data = ranking_df[ranking_df['Model'] == model]
            ax.plot(model_data['Level'], model_data['Rank'], 
                   marker='o', linewidth=2, markersize=8, 
                   label=model, color=colors.get(model, '#95a5a6'))
        
        ax.set_xlabel('Taxonomic Level', fontsize=11)
        ax.set_ylabel('Rank (1=Best)', fontsize=11)
        ax.set_title(f'{metric} Rankings', fontsize=12, fontweight='bold')
        ax.set_ylim(6.5, 0.5)
        ax.set_yticks(range(1, 7))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    
    # Save combined ranking data
    combined_ranking = pd.concat(all_ranking_data, keys=metrics, names=['Metric', 'Index'])
    combined_ranking = combined_ranking.reset_index(level='Metric').reset_index(drop=True)
    ranking_file = os.path.join(output_dir, 'ranking_trends.csv')
    combined_ranking.to_csv(ranking_file, index=False)
    print(f"  Saved ranking trends data: {ranking_file}")
    
    plt.suptitle('Model Rankings Across Taxonomic Levels', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = os.path.join(output_dir, 'ranking_trends.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_metric_correlation_heatmap(summary_df, output_dir):
    """Create correlation heatmap between different metrics"""
    metrics = ['ARI', 'NMI', 'AMI', 'Silhouette', 'Davies_Bouldin', 'Calinski_Harabasz']
    
    # Calculate correlation matrix
    corr_matrix = summary_df[metrics].corr()
    
    # Save correlation matrix data
    corr_file = os.path.join(output_dir, 'metric_correlation.csv')
    corr_matrix.to_csv(corr_file)
    print(f"  Saved metric correlation data: {corr_file}")
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', 
               center=0, square=True, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Metric Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'metric_correlation.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")

def create_model_comparison_table(summary_df, output_dir):
    """Create a comprehensive comparison table"""
    levels = ['phylum', 'class', 'order', 'family', 'genus']
    models = summary_df['Model'].unique()
    
    # Calculate average ranks
    metrics = ['ARI', 'NMI', 'AMI', 'Silhouette']
    
    avg_ranks = []
    for model in models:
        ranks = []
        for level in levels:
            level_data = summary_df[summary_df['Level'] == level]
            for metric in metrics:
                if metric == 'Davies_Bouldin':
                    rank = level_data.sort_values(metric, ascending=True)['Model'].tolist().index(model) + 1
                else:
                    rank = level_data.sort_values(metric, ascending=False)['Model'].tolist().index(model) + 1
                ranks.append(rank)
        avg_ranks.append({
            'Model': model,
            'Average_Rank': np.mean(ranks),
            'Best_Ranks': sum(1 for r in ranks if r == 1),
            'Top3_Ranks': sum(1 for r in ranks if r <= 3)
        })
    
    rank_df = pd.DataFrame(avg_ranks).sort_values('Average_Rank')
    
    # Save to CSV
    output_file = os.path.join(output_dir, 'model_ranking_summary.csv')
    rank_df.to_csv(output_file, index=False)
    print(f"  Saved: {output_file}")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(rank_df))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, rank_df['Average_Rank'], width, 
                   label='Average Rank', alpha=0.8, color='#3498db')
    bars2 = ax.bar(x + width/2, rank_df['Best_Ranks'], width, 
                   label='# of Best Ranks', alpha=0.8, color='#2ecc71')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rank_df['Model'], rotation=15, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, 'model_ranking_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file}")
    
    return rank_df

def main():
    base_dir = ".."
    results_dir = os.path.join(base_dir, 'six_model_comparison_results')
    summary_file = os.path.join(results_dir, 'six_model_summary.csv')
    
    print("="*80)
    print("GENERATING ADDITIONAL VISUALIZATIONS")
    print("="*80)
    
    # Load summary data
    print("\nLoading summary data...")
    summary_df = pd.read_csv(summary_file)
    print(f"Loaded data for {len(summary_df)} model-level combinations")
    
    # Create visualizations
    print("\n1. Creating radar charts for each level...")
    levels = ['phylum', 'class', 'order', 'family', 'genus']
    for level in levels:
        create_radar_chart(summary_df, level, results_dir)
    
    print("\n2. Creating performance heatmaps...")
    create_performance_heatmap(summary_df, results_dir)
    
    print("\n3. Creating ranking trends plot...")
    create_ranking_plot(summary_df, results_dir)
    
    print("\n4. Creating metric correlation heatmap...")
    create_metric_correlation_heatmap(summary_df, results_dir)
    
    print("\n5. Creating model comparison table...")
    rank_df = create_model_comparison_table(summary_df, results_dir)
    
    print("\n" + "="*80)
    print("ADDITIONAL VISUALIZATIONS COMPLETED")
    print("="*80)
    print(f"\nResults saved to: {results_dir}")
    
    print("\n" + "="*80)
    print("MODEL RANKING SUMMARY")
    print("="*80)
    print(rank_df.to_string(index=False))
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()

