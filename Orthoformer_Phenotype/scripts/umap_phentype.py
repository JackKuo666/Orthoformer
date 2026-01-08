import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
from matplotlib import rcParams
plt.switch_backend('Agg')

from sklearn.decomposition import PCA
# Set Nature journal style

def set_nature_style():
    """Set Nature journal figure style"""
    plt.rcParams['font.size'] = 8
    plt.rcParams['axes.linewidth'] = 0.5
    plt.rcParams['lines.linewidth'] = 0.5
    plt.rcParams['xtick.major.width'] = 0.5
    plt.rcParams['ytick.major.width'] = 0.5
    plt.rcParams['xtick.major.size'] = 2
    plt.rcParams['ytick.major.size'] = 2
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.1
    plt.rcParams['pdf.fonttype'] = 42  # Ensure text is editable
    plt.rcParams['ps.fonttype'] = 42
    
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.color': '0.8'
    })

set_nature_style()

class GenomeDimensionalityReduction:
    def __init__(self, data_dir=None):
        """Initialize genome dimensionality reduction analyzer"""
        self.data = None
        self.labels = None
        self.genome_names = None
        self.scaler = StandardScaler()
    
    def load_phenotype_genome(self,filepath):
        df = pd.read_csv(filepath)
        self.genomes = df["genome_name"].tolist()
        self.phenotype = df[df.columns[3]].tolist() ###
        self.filepath = filepath

    def load_pt_files(self, data_dir):
        """ Load all .pt files from directory """
        print("Loading genome data...")
        
        all_embeddings = []
        all_labels = []
        all_names = []
        
        # Iterate through all .pt files in directory
        
        for i,genome in enumerate(self.genomes):
            file_path = data_dir + "/" + genome + ".pt"
            if os.path.exists(file_path):
                data = torch.load(file_path)
                all_embeddings.append(data)
                all_labels.append(self.phenotype[i])
                all_names.append(genome)
            else:
                print("Not found embedding for genome "+genome)
        
        self.data = np.array(all_embeddings)
        self.labels = np.array(all_labels)
        self.genome_names = np.array(all_names)
        
        print(f"Successfully loaded {len(self.data)} genomes")
        print(f"Data shape: {self.data.shape}")
        print(f"Phenotype distribution: {pd.Series(self.labels).value_counts().to_dict()}")
        
        return self.data, self.labels, self.genome_names
    
    def load_from_arrays(self, embeddings, labels, names=None):
        """ Load data directly from arrays """
        self.data = np.array(embeddings)
        self.labels = np.array(labels)
        self.genome_names = np.array(names) if names is not None else np.array([f"genome_{i}" for i in range(len(embeddings))])
        
        print(f"Loaded {len(self.data)} samples")
        print(f"Data shape: {self.data.shape}")
        
        return self.data, self.labels, self.genome_names
    
    def preprocess_data(self):
        """ Data preprocessing """
        print("Preprocessing data...")
        
        # Standardize data
        self.data_scaled = self.scaler.fit_transform(self.data)
        
        # Ensure labels are string type (for legend)
        self.labels_str = np.array([str(label) for label in self.labels])
        
        # Get unique phenotypes
        self.unique_phenotypes = np.unique(self.labels_str)
        self.n_phenotypes = len(self.unique_phenotypes)
        
        print(f"Phenotype types: {list(self.unique_phenotypes)}")
        print(f"Data shape after standardization: {self.data_scaled.shape}")
        
        return self.data_scaled
    
    def perform_umap(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """ Perform UMAP dimensionality reduction """
        print("Performing UMAP dimensionality reduction...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='euclidean'
        )
        
        self.umap_result = reducer.fit_transform(self.data_scaled)
        print("UMAP completed!")
        
        return self.umap_result
    
    def perform_tsne(self, n_components=2, perplexity=30, random_state=42):
        """ Perform t-SNE dimensionality reduction """
        print("Performing t-SNE dimensionality reduction...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            learning_rate='auto',
            init='random'
        )
        
        self.tsne_result = tsne.fit_transform(self.data_scaled)
        print("t-SNE completed!")
        
        return self.tsne_result
    
    def optimized_tsne_genomic(self, n_components=2, random_state=42):
        """Optimized t-SNE parameters for genomic data"""
        tsne = TSNE(
            n_components=n_components,
            perplexity=30,                    # Adjust according to data size
            random_state=random_state,
            learning_rate=200,                # Fixed learning rate, avoid auto selection
            init='pca',                       # Use PCA initialization, more stable
            max_iter=1000,                      # Increase iterations
            n_iter_without_progress=300,      # Early stopping condition
            min_grad_norm=1e-7,               # Gradient threshold
            metric='euclidean',               # Genomic data suitable for Euclidean distance
            early_exaggeration=12.0,          # Initial exaggeration factor
            verbose=1                         # Show progress
        )
        self.tsne_result = tsne.fit_transform(self.data_scaled)
        return self.tsne_result

    def pca_pretrained_tsne(self, n_components=2, pca_components=50, random_state=42):
        """Perform PCA pre-reduction before t-SNE"""
        # 2. PCA reduce to 50 dimensions (remove noise)
        print(f"Original dimensions: {self.data_scaled.shape[1]}")
        
        pca = PCA(n_components=min(pca_components, self.data_scaled.shape[1]), 
                  random_state=random_state)
        
        data_pca = pca.fit_transform(self.data_scaled)
        
        print(f"Dimensions after PCA: {data_pca.shape[1]}")
        print(f"Variance retained: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 3. t-SNE dimensionality reduction
        
        #tsne = TSNE(
        #    n_components=n_components,
        #    perplexity=30,
        #    random_state=random_state,
        #    learning_rate=200,
        #    init='pca',
        #    max_iter=1000,
        #    verbose=1
        #)
        tsne = TSNE(
            n_components=n_components,
            perplexity=30,                    # Adjust according to data size
            random_state=random_state,
            learning_rate=200,                # Fixed learning rate, avoid auto selection
            init='pca',                       # Use PCA initialization, more stable
            max_iter=1000,                      # Increase iterations
            n_iter_without_progress=300,      # Early stopping condition
            min_grad_norm=1e-7,               # Gradient threshold
            metric='euclidean',               # Genomic data suitable for Euclidean distance
            early_exaggeration=12.0,          # Initial exaggeration factor
            verbose=1                         # Show progress
        )
        self.tsne_result = tsne.fit_transform(data_pca)
        return self.tsne_result
    
    def create_nature_style_plot(self, results, method='UMAP', save_path=None,output_dir="results"):
        """ Create Nature-style dimensionality reduction plot """
        
        # Nature journal color scheme
        nature_colors = ['#2E86AB', '#A23B72']  # Blue and magenta
        if self.n_phenotypes > 2:
            # If more colors needed
            nature_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23','#6A0572','#1A936F']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        
        ### also save coordinate to file

        # Figure 1: Scatter plot
        for i, phenotype in enumerate(self.unique_phenotypes):
            mask = self.labels_str == phenotype
            axes[0].scatter(
                results[mask, 0], results[mask, 1],
                c=[nature_colors[i]], label=phenotype,
                alpha=0.7, s=6, edgecolors='white', linewidth=0.3
            )
        
        axes[0].set_xlabel(f'{method} 1')
        axes[0].set_ylabel(f'{method} 2')
        axes[0].set_title(f'{method} Visualization\nGenome Embeddings', fontsize=10, fontweight='bold')
        axes[0].legend(frameon=True, fancybox=False, framealpha=0.8, edgecolor='black')
        axes[0].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(output_dir+"/"+save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        #plt.show()
        
        return fig
    
    def create_combined_plot(self, umap_result, tsne_result, save_path=None):
        """ Create comparison plot of UMAP and t-SNE """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Nature color scheme
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23','#6A0572','#1A936F']
        
        # UMAP plot
        for i, phenotype in enumerate(self.unique_phenotypes):
            mask = self.labels_str == phenotype
            axes[0].scatter(
                umap_result[mask, 0], umap_result[mask, 1],
                c=[colors[i]], label=phenotype,
                alpha=0.7, s=40, edgecolors='white', linewidth=0.3
            )
        
        axes[0].set_xlabel('UMAP 1')
        axes[0].set_ylabel('UMAP 2')
        axes[0].set_title('UMAP Projection', fontsize=10, fontweight='bold')
        axes[0].legend(frameon=True, fancybox=False, framealpha=0.8)
        axes[0].grid(True, alpha=0.3)
        
        # t-SNE plot
        for i, phenotype in enumerate(self.unique_phenotypes):
            mask = self.labels_str == phenotype
            axes[1].scatter(
                tsne_result[mask, 0], tsne_result[mask, 1],
                c=[colors[i]], label=phenotype,
                alpha=0.7, s=6, edgecolors='white', linewidth=0.3
            )
        
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        axes[1].set_title('t-SNE Projection', fontsize=10, fontweight='bold')
        axes[1].legend(frameon=True, fancybox=False, framealpha=0.8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to: {save_path}")
        
        #plt.show()
        
        return fig
    
    def save_results(self, tsne_result, output_dir='results'):
        """ Save dimensionality reduction results """
        os.makedirs(output_dir, exist_ok=True)
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'genome_name': self.genome_names,
            'phenotype': self.labels_str,
            'tsne_1': tsne_result[:, 0],
            'tsne_2': tsne_result[:, 1]
        })
        
        # Save as CSV
        csv_path = os.path.join(output_dir, self.filepath+'.dimensionality_reduction_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        return results_df


import sys

def main():
    """ Main function """
    # Create analyzer
    analyzer = GenomeDimensionalityReduction()
    
    # Method 1: Load .pt files from directory
    data_dir = "/mnt/MAG/jinfang/phenotype/model_3M_2048_v10_phenotype_embedding"  # Modify to your data directory
    #analyzer.load_phenotype_genome("gideon_Raffinose.csv") ### 
    analyzer.load_phenotype_genome(sys.argv[1]) ### 
    analyzer.load_pt_files(data_dir)
    # Data preprocessing
    analyzer.preprocess_data()
    
    # Perform UMAP dimensionality reduction
    umap_result = analyzer.perform_umap(n_neighbors=15, min_dist=0.1)
    
    # Perform t-SNE dimensionality reduction
    tsne_result = analyzer.pca_pretrained_tsne()
    
    # Create UMAP plot
    print("Generating UMAP plot...")
    # Create t-SNE plot
    print("Generating t-SNE plot...")
    analyzer.create_nature_style_plot(tsne_result, method='t-SNE',
                                   save_path=sys.argv[1]+'.tsne_visualization.pdf')
    
    # Create comparison plot
    #print("Generating comparison plot...")
    #analyzer.create_combined_plot(umap_result, tsne_result,
    #                            save_path=sys.argv[1]+'comparison_plot.pdf')
    
    # Save results
    results_df = analyzer.save_results(tsne_result)
    
    print("\n" + "="*50)
    print("Analysis complete!")
    print("="*50)
    print(f"Number of genomes processed: {len(analyzer.data)}")
    print(f"Phenotype distribution: {pd.Series(analyzer.labels_str).value_counts().to_dict()}")
    print("Generated files:")
    print("  - results/dimensionality_reduction_results.csv")

if __name__ == "__main__":
    main()
