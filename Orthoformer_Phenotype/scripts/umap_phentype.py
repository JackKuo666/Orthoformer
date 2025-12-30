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
# 设置Nature期刊风格

def set_nature_style():
    """设置Nature期刊的图形风格"""
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
    plt.rcParams['pdf.fonttype'] = 42  # 确保文字可编辑
    plt.rcParams['ps.fonttype'] = 42
    
    sns.set_style("whitegrid", {
        'grid.linestyle': ':',
        'grid.linewidth': 0.5,
        'grid.color': '0.8'
    })

set_nature_style()

class GenomeDimensionalityReduction:
    def __init__(self, data_dir=None):
        """初始化基因组降维分析器 """
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
        """ 从目录加载所有.pt文件 """
        print("正在加载基因组数据...")
        
        all_embeddings = []
        all_labels = []
        all_names = []
        
        # 遍历目录中的所有.pt文件
        
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
        
        print(f"成功加载 {len(self.data)} 个基因组")
        print(f"数据形状: {self.data.shape}")
        print(f"表型分布: {pd.Series(self.labels).value_counts().to_dict()}")
        
        return self.data, self.labels, self.genome_names
    
    def load_from_arrays(self, embeddings, labels, names=None):
        """ 直接从数组加载数据 """
        self.data = np.array(embeddings)
        self.labels = np.array(labels)
        self.genome_names = np.array(names) if names is not None else np.array([f"genome_{i}" for i in range(len(embeddings))])
        
        print(f"加载 {len(self.data)} 个样本")
        print(f"数据形状: {self.data.shape}")
        
        return self.data, self.labels, self.genome_names
    
    def preprocess_data(self):
        """ 数据预处理 """
        print("正在预处理数据...")
        
        # 标准化数据
        self.data_scaled = self.scaler.fit_transform(self.data)
        
        # 确保标签是字符串类型（用于图例）
        self.labels_str = np.array([str(label) for label in self.labels])
        
        # 获取唯一的表型
        self.unique_phenotypes = np.unique(self.labels_str)
        self.n_phenotypes = len(self.unique_phenotypes)
        
        print(f"表型类型: {list(self.unique_phenotypes)}")
        print(f"标准化后数据形状: {self.data_scaled.shape}")
        
        return self.data_scaled
    
    def perform_umap(self, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
        """ 执行UMAP降维 """
        print("正在执行UMAP降维...")
        
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=random_state,
            metric='euclidean'
        )
        
        self.umap_result = reducer.fit_transform(self.data_scaled)
        print("UMAP完成!")
        
        return self.umap_result
    
    def perform_tsne(self, n_components=2, perplexity=30, random_state=42):
        """ 执行t-SNE降维 """
        print("正在执行t-SNE降维...")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            random_state=random_state,
            learning_rate='auto',
            init='random'
        )
        
        self.tsne_result = tsne.fit_transform(self.data_scaled)
        print("t-SNE完成!")
        
        return self.tsne_result
    
    def optimized_tsne_genomic(self, n_components=2, random_state=42):
        """针对基因组数据的优化t-SNE参数"""
        tsne = TSNE(
            n_components=n_components,
            perplexity=30,                    # 根据数据量调整
            random_state=random_state,
            learning_rate=200,                # 固定学习率，避免自动选择
            init='pca',                       # 使用PCA初始化，更稳定
            max_iter=1000,                      # 增加迭代次数
            n_iter_without_progress=300,      # 早停条件
            min_grad_norm=1e-7,               # 梯度阈值
            metric='euclidean',               # 基因组数据适合欧氏距离
            early_exaggeration=12.0,          # 初始放大系数
            verbose=1                         # 显示进度
        )
        self.tsne_result = tsne.fit_transform(self.data_scaled)
        return self.tsne_result

    def pca_pretrained_tsne(self, n_components=2, pca_components=50, random_state=42):
        """PCA预降维后再进行t-SNE"""
        # 2. PCA降维到50维（去除噪声）
        print(f"原始维度: {self.data_scaled.shape[1]}")
        
        pca = PCA(n_components=min(pca_components, self.data_scaled.shape[1]), 
                  random_state=random_state)
        
        data_pca = pca.fit_transform(self.data_scaled)
        
        print(f"PCA后维度: {data_pca.shape[1]}")
        print(f"保留方差: {pca.explained_variance_ratio_.sum():.3f}")
        
        # 3. t-SNE降维
        
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
            perplexity=30,                    # 根据数据量调整
            random_state=random_state,
            learning_rate=200,                # 固定学习率，避免自动选择
            init='pca',                       # 使用PCA初始化，更稳定
            max_iter=1000,                      # 增加迭代次数
            n_iter_without_progress=300,      # 早停条件
            min_grad_norm=1e-7,               # 梯度阈值
            metric='euclidean',               # 基因组数据适合欧氏距离
            early_exaggeration=12.0,          # 初始放大系数
            verbose=1                         # 显示进度
        )
        self.tsne_result = tsne.fit_transform(data_pca)
        return self.tsne_result
    
    def create_nature_style_plot(self, results, method='UMAP', save_path=None,output_dir="results"):
        """ 创建Nature风格的降维图 """
        
        # Nature期刊配色方案
        nature_colors = ['#2E86AB', '#A23B72']  # 蓝色和洋红色
        if self.n_phenotypes > 2:
            # 如果需要更多颜色
            nature_colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23','#6A0572','#1A936F']
        
        # 创建图形
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        
        ### also save coordinate to file

        # 图1: 散点图
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
            print(f"图片已保存至: {save_path}")
        
        #plt.show()
        
        return fig
    
    def create_combined_plot(self, umap_result, tsne_result, save_path=None):
        """ 创建UMAP和t-SNE的对比图 """
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Nature配色
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6B8E23','#6A0572','#1A936F']
        
        # UMAP图
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
        
        # t-SNE图
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
            print(f"对比图已保存至: {save_path}")
        
        #plt.show()
        
        return fig
    
    def save_results(self, tsne_result, output_dir='results'):
        """ 保存降维结果 """
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建结果DataFrame
        results_df = pd.DataFrame({
            'genome_name': self.genome_names,
            'phenotype': self.labels_str,
            'tsne_1': tsne_result[:, 0],
            'tsne_2': tsne_result[:, 1]
        })
        
        # 保存为CSV
        csv_path = os.path.join(output_dir, self.filepath+'.dimensionality_reduction_results.csv')
        results_df.to_csv(csv_path, index=False)
        print(f"结果已保存至: {csv_path}")
        
        return results_df


import sys

def main():
    """ 主函数 """
    # 创建分析器
    analyzer = GenomeDimensionalityReduction()
    
    # 方法1: 从目录加载.pt文件
    data_dir = "/mnt/MAG/jinfang/phenotype/model_3M_2048_v10_phenotype_embedding"  # 修改为您的数据目录
    #analyzer.load_phenotype_genome("gideon_Raffinose.csv") ### 
    analyzer.load_phenotype_genome(sys.argv[1]) ### 
    analyzer.load_pt_files(data_dir)
    # 数据预处理
    analyzer.preprocess_data()
    
    # 执行UMAP降维
    umap_result = analyzer.perform_umap(n_neighbors=15, min_dist=0.1)
    
    # 执行t-SNE降维
    tsne_result = analyzer.pca_pretrained_tsne()
    
    # 创建UMAP图
    print("生成UMAP图...")
    # 创建t-SNE图
    print("生成t-SNE图...")
    analyzer.create_nature_style_plot(tsne_result, method='t-SNE',
                                   save_path=sys.argv[1]+'.tsne_visualization.pdf')
    
    # 创建对比图
    #print("生成对比图...")
    #analyzer.create_combined_plot(umap_result, tsne_result,
    #                            save_path=sys.argv[1]+'comparison_plot.pdf')
    
    # 保存结果
    results_df = analyzer.save_results(tsne_result)
    
    print("\n" + "="*50)
    print("分析完成!")
    print("="*50)
    print(f"处理基因组数量: {len(analyzer.data)}")
    print(f"表型分布: {pd.Series(analyzer.labels_str).value_counts().to_dict()}")
    print("生成的文件:")
    print("  - results/dimensionality_reduction_results.csv")

if __name__ == "__main__":
    main()
