import scanpy as sc
import pandas as pd
import numpy as np
import scipy
import os
import shutil
import bbknn
import scanpy as sc
import gseapy as gp
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import sys
# import harmonypy as hm

import matplotlib
matplotlib.use('Agg') # Anti-Grain Geometry 一个高性能图形库，适合生成静态图片，不需要与GUI进行交互

import warnings
warnings.filterwarnings("ignore") # 直接关掉所有WARNING


# import scgen

# ALLOWED_METHODS = {
#     'batch_correction': {'BBKNN', 'Combat', 'Harmony'},
#     'clustering': {'Leiden Clustering', 'Louvain Clustering'},
#     'diff_expr': {'Wilcoxon Rank-Sum Test', 'T-test', 'Logistic Regression'},
#     'enrichment_analysis': {'GO Biological Process', 'GO Molecular Function', 'GO Cellular Component'},
#     'trajectory_inference': {'Monocle', 'PAGA'}
# }

class scRNAseqUtils:
    def __init__(
        self, 
        adata_list, 
        save_path,
        display_path,
        batch_correction='BBKNN',
        clustering='Leiden Clustering',
        enrichment_analysis='GO Enrichment',
        diff_expr='Wilcoxon Rank-Sum Test',
        trajectory_inference='Monocle'
    ):
        """ 初始化类时，传入 AnnData 对象（scanpy 的核心数据结构）和工作目录 """
        self.adata_list = adata_list
        self.save_path = save_path 
        self.display_path = display_path
        # 确保工作目录存在
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        
        # 存储和验证分析方法参数
        # self.batch_correction = self.validate_method(batch_correction, 'batch_correction')
        # self.clustering = self.validate_method(clustering, 'clustering')
        # self.enrichment_analysis = self.validate_method(enrichment_analysis, 'enrichment_analysis')
        # self.diff_expr = self.validate_method(diff_expr, 'diff_expr')
        # self.trajectory_inference = self.validate_method(trajectory_inference, 'trajectory_inference')
        self.batch_correction = batch_correction
        self.clustering = clustering
        self.enrichment_analysis = enrichment_analysis
        self.diff_expr = diff_expr
        self.trajectory_inference = trajectory_inference
        self.results = []
        # 合并数据
        self.merge_data()
        
    # def validate_method(self, method, method_type):
    #     """
    #     验证传入的方法是否在允许的选项中
    #     """
    #     if method not in ALLOWED_METHODS[method_type]:
    #         raise ValueError(f"Invalid method '{method}' for {method_type}. Allowed methods: {ALLOWED_METHODS[method_type]}")
    #     return method

    def merge_data(self):
        '''合并多个AnnData对象'''
        # 如果传入的list只有一个元素，直接返回好吧
        if not self.adata_list:
            raise ValueError("adata_list cannot be empty.")
        elif len(self.adata_list) == 1:
            self.adata = self.adata_list[0]
        else:
            # 需要确保所有数据的基因名称一致
            var_names = set(self.adata_list[0].var_names)
            for adata in self.adata_list[1:]:
                var_names = var_names.intersection(set(adata.var_names))
            var_names = list(var_names)
            # 只保留共有的基因
            for i, adata in enumerate(self.adata_list):
                self.adata_list[i] = adata[:, var_names]
                adata.obs['batch'] = f'batch{i + 1}'  # 添加批次信息
            # 合并数据集
            self.adata = sc.concat(self.adata_list, merge='same')

    def calculate_qc_metrics(self):
        """ 计算质量控制所需的指标，如总 counts、基因数和线粒体基因比例 """
        # 标记线粒体基因
        self.adata.var['mt'] = self.adata.var_names.str.startswith('MT-')
        
        # 计算质量控制指标
        sc.pp.calculate_qc_metrics(
            self.adata, 
            qc_vars=['mt'], 
            percent_top=None, 
            log1p=False, 
            inplace=True
        )
        
    def filter_cells_and_genes(self, min_genes=200, min_cells=3, max_mito=0.1):
        """
        过滤基因和细胞：
            - 过滤在少于min_cells个细胞中表达的基因。
            - 过滤表达低于min_genes或高于max_genes的细胞，以及线粒体基因表达高于max_mito的细胞。
        """
        self.adata.obs[['n_genes_by_counts', 'total_counts', 'pct_counts_mt']].to_csv(
            os.path.join(self.save_path, 'qc_metrics_table.csv')
        )
        self.results.append(
            {
                'name': 'QC Matrics Table',
                'csv_path': os.path.join(self.display_path, 'qc_metrics_table.csv'),
                'description': 'CSV file containing QC metrics for each cell: n_genes_by_counts, total_counts, pct_counts_mt'
            }
        )
        # 过滤之前可视化一遍
        self.violin_plots('violin_before_filtering.png')
        self.plot_highest_expr_genes('highest_expr_genes_before_filtering.png')
        
        sc.pp.filter_genes(self.adata, min_cells=min_cells) # 基于细胞数过滤基因
        sc.pp.filter_cells(self.adata, min_genes=min_genes) # 过滤基于基因表达数的细胞
        
        # 可选，过滤线粒体基因表达比例过高的细胞，暂时注释
        # self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mito, :]
        
        # 更新 QC 指标
        self.calculate_qc_metrics()
        # 过滤之后可视化一遍
        self.violin_plots('violin_after_filtering.png')
        self.plot_highest_expr_genes('highest_expr_genes_after_filtering.png')

    def violin_plots(self, save_name):
        """ 使用小提琴图可视化基因数、UMI counts 和线粒体基因比例，并保存图片。 """
        default_save_path = 'figures/violin.png'
        sc.pl.violin(
            self.adata, 
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            jitter=0.4, 
            multi_panel=True,
            save='.png',
            show=False
        )
        # 移动图像到工作目录
        shutil.move(default_save_path, os.path.join(self.save_path, save_name))
        self.results.append(
            {
                'name': save_name.split('.')[0],
                'img_path': [os.path.join(self.display_path, save_name)],
                'description': 'Violin plot visualizing the distribution of the number of genes per cell (n_genes_by_counts), total UMI counts (total_counts), and the percentage of mitochondrial counts (pct_counts_mt) across all cells. This helps assess the quality of the dataset.'
            }
        )

    def plot_highest_expr_genes(self, save_name, n_top_genes=20):
        """ 可视化表达最高的前 n 个基因，并保存图片。 """
        default_save_path = 'figures/highest_expr_genes.png'
        sc.pl.highest_expr_genes(
            self.adata, 
            n_top=n_top_genes, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, os.path.join(self.save_path, save_name))
        self.results.append(
            {
                'name': save_name.split('.')[0],
                'img_path': [os.path.join(self.display_path, save_name)],
                'description': f'Bar plot showing the top {n_top_genes} most highly expressed genes. This plot highlights the most abundant genes in the dataset, which can provide insights into major biological processes or technical artifacts.'
            }
        )

    def normalize_data(self):
        """ 对数据进行标准化处理，每个细胞归一化到相同的总表达量，并对数转换。 """
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)
        self.results.append(
            {
                'name': 'Normalization and Log Transformation',
                'description': f'The data has been normalized such that the total expression for each cell is scaled to 10,000,<br>'
                            f'and log1p transformation has been applied to reduce the effect of highly-expressed genes. <br>'
                            f'This step ensures comparability between cells and prepares the data for downstream analysis.'
            }
        )

    def find_highly_variable_genes(self, save_name='hvg.png'):
        """ 找出高变异基因，用于后续分析。 """
        default_save_path = 'figures/filter_genes_dispersion.png'
        sc.pp.highly_variable_genes(
            self.adata, 
            min_mean=0.0125, 
            max_mean=3, 
            min_disp=0.5)
        sc.pl.highly_variable_genes(
            self.adata, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, os.path.join(self.save_path, save_name))
        # 将结果添加到 self.results
        self.results.append(
            {
                'name': 'Highly Variable Genes (HVG)',
                'img_path': [os.path.join(self.display_path, save_name)],  # 图像保存路径
                'description': f'A plot showing the highly variable genes detected in the dataset. <br>'
                            f'These genes have high variance across cells, and they are crucial <br>'
                            f'for downstream analysis such as dimensionality reduction and clustering.'
            }
        )

    def filter_hvg(self):
        """
        选择高变异基因并保留原始数据。
        该方法只能执行一次，因为它会修改 adata 对象。
        """
        self.adata.raw = self.adata  # 保留原始数据
        self.adata = self.adata[:, self.adata.var['highly_variable']]  # 选择高变异基因
        print(self.adata.raw.shape, self.adata.shape)
        # 将筛选高变异基因的结果添加到 self.results
        self.results.append(
            {
                'name': 'Filter Highly Variable Genes (HVG)',
                'description': f'Selected highly variable genes from the dataset.<br>'
                            f'Original data shape: {self.adata.raw.shape}.<br>'
                            f'Filtered data shape: {self.adata.shape}.<br>'
                            f'This step retains only the genes with the highest variance across cells.'
            }
        )
    
    def scale_data(self):
        """ 对数据进行标准化，使得每个基因的表达量具有均值为0、方差为1。 """
        # 将数据缩放到单位方差
        sc.pp.regress_out(
            self.adata, 
            ['total_counts', 'pct_counts_mt']# 要回归出去的变量列表
        )
        sc.pp.scale(self.adata, max_value=10)
        self.results.append(
            {
                'name': 'Data Scaling',
                'description': f'The data has been scaled to unit variance, with each gene\'s expression standardized.<br>'
                            f'Total counts and the percentage of mitochondrial counts were regressed out.<br>'
                            f'Expression values are limited to a maximum of 10 to mitigate the influence of outliers.'
            }
        )

    def pca(self, n_comps=50):
        """ 进行主成分分析(PCA)，并将结果保存在adata对象中。 """
        sc.tl.pca(
            self.adata, 
            n_comps=n_comps,
            svd_solver='arpack' # SVD求解器，'arpack' 适用于大型稀疏矩阵
            )
        hvg_genes = self.find_top_genes()
        sc.pl.pca(
            self.adata, 
            color=hvg_genes, # 选择三个高变异基因（这里以HVG基因列表为例）
            save='.png',
            show=False
        )
        sc.pl.pca_variance_ratio(
            self.adata, 
            log=True, 
            show=False,
            save='.png')
        
        shutil.move('figures/pca.png', os.path.join(self.save_path, 'pca.png'))
        shutil.move('figures/pca_variance_ratio.png', os.path.join(self.save_path, 'pca_variance_ratio.png'))
        # 将结果添加到self.results
        self.results.append(
            {
                'name': 'PCA Analysis',
                'img_path': [os.path.join(self.display_path, 'pca.png'), 
                             os.path.join(self.display_path, 'pca_variance_ratio.png')],  # PCA图像保存路径
                'description': f'Performed PCA on the dataset with {n_comps} components.<br>'
                            f'Visualized the first three highly variable genes: {", ".join(hvg_genes)}.<br>'
                            f'Explained variance ratio plot saved as well.'
            }
        )

    def neighbors(self, n_neighbors=10, n_pcs=40):
        """ 计算邻居，用于后续降维（如UMAP和t-SNE）和聚类。 """
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs
        )
        self.results.append(
            {
                'name': 'Neighbor Graph',
                'description': f'Computed a neighborhood graph using {n_neighbors} neighbors and {n_pcs} principal components.<br>'
                            f'This is used for downstream dimensionality reduction and clustering (e.g., UMAP, t-SNE).'
            }
        )
        
    def find_top_genes(self, num_genes=3):
        """ 找到在数据集中高变异基因中表达量差异最大的基因。 """
        # 筛选出高变异基因
        hvg_mask = self.adata.var['highly_variable']
        hvg_data = self.adata.X[:, hvg_mask]
        
        # 获取方差最大的基因的索引
        variances = hvg_data.var(axis=0)
        top_gene_indices = np.argsort(variances)[-num_genes:]
        # 获取名称
        top_genes = self.adata.var_names[hvg_mask][top_gene_indices]
        return top_genes.tolist()


    def tsne(self, suffix=''):
        """ 使用 t-SNE 方法对数据进行降维。 """
        sc.tl.tsne(self.adata)
        sc.pl.tsne(
            self.adata, 
            color=self.find_top_genes(),  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        shutil.move('figures/tsne.png', os.path.join(self.save_path, f'tsne{suffix}.png'))
         # 将结果添加到 self.results
        self.results.append(
            {
                'name': f't-SNE Visualization suffix',
                'img_path': [os.path.join(self.display_path, f'tsne{suffix}.png')],
                'description': 't-SNE plot visualizing the data in lower dimensions. The colors represent the top variable genes.'
            }
        )

    def umap(self, suffix=''):
        """ 使用 UMAP 方法对数据进行降维。 """
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=self.find_top_genes(),  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        shutil.move('figures/umap.png', os.path.join(self.save_path, f'umap{suffix}.png'))
        # 将结果添加到 self.results
        self.results.append(
            {
                'name': f'UMAP Visualization {suffix}',
                'img_path': [os.path.join(self.display_path, f'umap{suffix}.png')],
                'description': 'UMAP plot visualizing the data in lower dimensions. The colors represent the top variable genes.'
            }
        )

    def leiden(self, resolutions=[0.25, 0.5, 0.75, 1.0], suffix=''):
        '''
        使用不同分辨率的Leiden聚类
        '''
        save_path_list = []
        for resolution in resolutions:
            sc.tl.leiden(self.adata, resolution=resolution)
            sc.pl.umap(
                self.adata, 
                color=['leiden'], 
                save='_leiden.png',
                show=False
            )
            shutil.move('figures/umap_leiden.png', os.path.join(self.save_path, f'umap_leiden_res{resolution}{suffix}.png'))
            save_path_list.append(os.path.join(self.display_path, f'umap_leiden_res{resolution}{suffix}.png'))

        self.results.append(
            {
                'name': f'Leiden Clustering {suffix}',
                'img_path': save_path_list,
                'description': f'UMAP visualization of Leiden clustering results with different resolutions: {resolutions}.'
            }
        )
    
    def louvain(self, resolutions=[0.25, 0.5, 0.75, 1.0], suffix=''):
        '''
        使用Louvain聚类
        '''
        save_path_list = []
        for resolution in resolutions:
            sc.tl.louvain(self.adata, resolution=resolution)
            # 可视化结果
            sc.pl.umap(
                self.adata, 
                color=['louvain'], 
                save='_louvain.png',
                show=False
            )
            # 移动保存的文件
            shutil.move('figures/umap_louvain.png', os.path.join(self.save_path, 'umap_louvain_res{resolution}{suffix}.png'))
            save_path_list.append(os.path.join(self.display_path, 'umap_louvain_res{resolution}{suffix}.png'))
        # 将结果添加到 self.results    
        self.results.append(
            {
                'name': f'Louvain Clustering {suffix}',
                'img_path': save_path_list,
                'description': f'UMAP visualization of Louvain clustering results with different resolutions: {resolutions}.'
            }
        )

    ###################下面是几个批次矫正方法###################
    def bbknn_correction(self, neighbors_within_batch=3, n_pcs=50, cluster_method='leiden'):
        """ 使用 BBKNN 方法进行批次效应矫正。 """
        bbknn.bbknn(
            self.adata, 
            batch_key='batch', 
            neighbors_within_batch=neighbors_within_batch, 
            n_pcs=n_pcs
        )
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', cluster_method], 
            save='_bbknn.png',
            show=False
        )
        shutil.move('figures/umap_bbknn.png', os.path.join(self.save_path, 'umap_bbknn.png'))
        
        self.results.append(
            {
                'name': 'BBKNN Batch Correction',
                'img_path': [os.path.join(self.display_path, 'umap_bbknn.png')],
                'description': f'UMAP visualization after BBKNN batch correction using {cluster_method} clustering.'
            }
        )
        print("BBKNN correction finished")

    def combat_correction(self, cluster_method='leiden'):
        """ 使用 Combat 方法进行批次效应矫正。 """
        sc.pp.combat(self.adata, key='batch')
        self.pca()
        self.neighbors()
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', cluster_method], 
            save='_combat.png',
            show=False
        )
        shutil.move('figures/umap_combat.png', os.path.join(self.save_path, 'umap_combat.png'))
        
        self.results.append(
            {
                'name': 'Combat Batch Correction',
                'img_path': [os.path.join(self.display_path, 'umap_combat.png')],
                'description': f'UMAP visualization after Combat batch correction using {cluster_method} clustering.'
            }
        )
        
        print("Combat Correction finished")
    
    def harmony_correction(self, cluster_method='leiden'):
        """ 使用 Scanpy 内置的 Harmony 方法进行批次效应校正。 """
        # 使用 Scanpy 的 Harmony 接口进行批次效应矫正
        sc.external.pp.harmony_integrate(self.adata, key='batch')
        sc.pp.neighbors(self.adata, use_rep='X_pca') # 重新计算邻居
        sc.tl.umap(self.adata) # 计算 UMAP
        filename_umap = 'umap_harmony.png' # 可视化 UMAP
        sc.pl.umap(
            self.adata, 
            color=['batch', cluster_method], 
            save='_harmony.png', 
            show=False
        )
        # 移动保存的图片到工作目录
        shutil.move(f'figures/{filename_umap}', os.path.join(self.save_path, filename_umap))
        
        self.results.append(
            {
                'name': 'Harmony Batch Correction',
                'img_path': [os.path.join(self.display_path, filename_umap)],
                'description': f'UMAP visualization after Harmony batch correction using {cluster_method} clustering.'
            }
        )
        print("Harmony Correction Finished")

    # 添加比较有无批次矫正的可视化方法
    def compare_batch_correction(self):
        """ 比较合并数据在有无批次矫正情况下的结果。 """
        # 先行判断是否需要做批次矫正
        if 'batch' not in self.adata.obs:
            self.results.append(
                {
                    'name': 'Batch Correction',
                    'description': 'No batch information, skip batch correction, please click `NEXT` for more info',
                }
            )
            return
        
        # 下面就是有批次信息，那就做批次矫正
        # 批次矫正之前的 UMAP
        sc.tl.umap(self.adata)
        if self.clustering == 'Leiden Clustering':
            self.leiden(suffix='_before_batch_correction')  # 先进行 Leiden 聚类
            cluster_method = 'leiden'
        elif self.clustering == 'Louvain Clustering':
            self.louvain(suffix='_before_batch_correction')  # 或进行 Louvain 聚类
            cluster_method = 'louvain'
        # 使用self.聚类和self.umap方法均会向self.results中添加内容，无需再写
        self.umap(suffix='_before_batch_correction')
        
        # 开始做批次矫正
        if self.batch_correction == 'BBKNN':
            self.bbknn_correction(cluster_method=cluster_method)
        elif self.batch_correction == "Combat":
            self.combat_correction(cluster_method=cluster_method)
        else:
            self.harmony_correction(cluster_method=cluster_method)
        
        
    ###################添加差异表达分析方法及可视化方法############################
    def calculate_diff_expr(self, n_genes=25):
        """
        计算差异表达基因，支持 t-test、wilcoxon、logreg 等方法。
        
        Parameters:
        - method: 差异表达计算方法 ('t-test', 'wilcoxon', 'logreg')
        - groupby: 按照哪个分组进行差异表达计算 (默认 'leiden')
        - n_genes: 可视化时展示的基因数量
        - save: 是否保存图片 (默认为 False)
        """
        # 判断使用哪种方法
        # 'diff_expr': {'Wilcoxon Rank-Sum Test', 'T-test', 'Logistic Regression'},
        if self.diff_expr == 'Wilcoxon Rank-Sum Test':
            method = 'wilcoxon'
        elif self.diff_expr == 'T-test':
            method = 't-test'
        else:
            method = 'logreg'
        
        if self.clustering == 'Leiden Clustering':
            groupby = 'leiden'
        else:
            groupby = 'louvain'
        
        # 计算差异表达基因
        sc.tl.rank_genes_groups(
            self.adata, 
            groupby, 
            method=method
        )
        # 可视化
        filename = f"_{method}.png"
        sc.pl.rank_genes_groups(
            self.adata, 
            n_genes=n_genes, 
            sharey=False, # 决定不同子图的y轴是否共享
            show=False,
            save=filename
        )
        shutil.move(f'figures/rank_genes_groups_{groupby}'+filename, os.path.join(self.save_path, f'rank_genes_group_{groupby}' + filename))
        
        self.results.append({
            'name': f'Differential Expression ({method})',
            'img_path': [os.path.join(self.display_path, f'rank_genes_group_{groupby}' + filename)],
            'description': f'Top {n_genes} differentially expressed genes using {method} method and grouped by {groupby}.'
        })
        
    def visualize_marker_genes(
        self, 
        marker_genes, 
        groupby='leiden', 
        save=True):
        """
        可视化给定的marker基因，如小提琴图、点图等。
        
        Parameters:
        - marker_genes: 需要可视化的marker基因列表
        - groupby: 按照哪个分组进行可视化 (默认 'leiden')
        - save: 是否保存图片 (默认为True)
        """
        # 小提琴图
        filename_violin = f"violin_marker_genes.png" if save else None
        sc.pl.violin(self.adata, marker_genes, groupby=groupby, save=filename_violin)

        # 点图
        filename_dotplot = f"dotplot_marker_genes.png" if save else None
        sc.pl.dotplot(self.adata, marker_genes, groupby=groupby, save=filename_dotplot)

    def annotate_clusters(self, new_cluster_names, groupby='leiden', save=False):
        """
        重新标注聚类的细胞类型名称，并可视化UMAP。
        
        Parameters:
        - new_cluster_names: 聚类的新名称列表
        - groupby: 按哪个分组进行细胞标注 (默认 'leiden')
        - save: 是否保存图片 (默认为 False)
        """
        # 重新标注聚类
        self.adata.rename_categories(groupby, new_cluster_names)
        
        # 可视化UMAP
        filename_umap = "_pbmc_annotation.png" if save else None
        sc.tl.umap(self.adata)
        sc.pl.umap(self.adata, color=groupby, legend_loc='on data', frameon=False, save=filename_umap)

    def visualize_heatmap(self, marker_genes, groupby='leiden', save=False):
        """
        可视化高差异表达基因的热图。
        
        Parameters:
        - marker_genes: marker基因列表
        - groupby: 按照哪个分组进行可视化 (默认 'leiden')
        - save: 是否保存图片 (默认为 False)
        """
        filename_heatmap = "_annotation_heatmap.png" if save else None
        sc.pl.heatmap(self.adata, marker_genes, groupby=groupby, cmap='viridis', dendrogram=True, save=filename_heatmap)

    def visualize_tracksplot(self, marker_genes, groupby='leiden', save=False):
        """
        可视化基因表达的轨迹图。
        
        Parameters:
        - marker_genes: 需要可视化的marker基因列表
        - groupby: 按照哪个分组进行可视化 (默认 'leiden')
        - save: 是否保存图片 (默认为 False)
        """
        filename_tracksplot = "_annotation_tracksplot.png" if save else None
        sc.pl.tracksplot(self.adata, marker_genes, groupby=groupby, dendrogram=True, save=filename_tracksplot)
    
    def rank_genes_by_clusters(self, method='wilcoxon', n_genes=4, save=False):
        """
        根据聚类对差异基因进行排序，并可视化点图和轨迹图。
        
        Parameters:
        - method: 计算差异表达的统计方法 ('t-test', 'wilcoxon', 'logreg')
        - n_genes: 可视化时展示的前 n 个基因
        - save: 是否保存图片 (默认为 False)
        """
        sc.tl.rank_genes_groups(self.adata, groupby='leiden', method=method)
        
        # 点图
        filename_dotplot = f"_groups_dotplot.png" if save else None
        sc.pl.rank_genes_groups_dotplot(self.adata, n_genes=n_genes, save=filename_dotplot)
        
        
    ######################富集分析，其中已经实现了基因注释######################
    def perform_enrichment_and_annotation(self):
        """富集分析和基因注释，并生成多种可视化结果。"""
        # 获取表达矩阵
        if scipy.sparse.issparse(self.adata.X):
            X = self.adata.X.toarray()
        else:
            X = self.adata.X

        # 计算所有细胞的平均表达
        mean_expression = np.mean(X, axis=0)
        # 获取表达的基因（平均表达量大于0的基因）
        expressed_genes = self.adata.var_names[mean_expression > 0]
        # 获取表达最高的前 5000 个基因
        top_genes = self.adata.var_names[np.argsort(mean_expression)[-5000:]].tolist()

        # 根据用户选择的富集分析类型，设置基因集
        if self.enrichment_analysis == 'GO Biological Process':
            gene_set = ['GO_Biological_Process_2021']
        elif self.enrichment_analysis == 'GO Molecular Function':
            gene_set = ['GO_Molecular_Function_2021']
        else:
            gene_set = ['GO_Cellular_Component_2021']

        # 使用 Enrichr 进行富集分析
        enr = gp.enrichr(gene_list=top_genes,
                        gene_sets=gene_set,
                        organism='Human',
                        outdir=os.path.join(self.save_path, 'enrichr_results'),
                        cutoff=0.05)

        results_df = enr.results # 获取结果

        # 按 Adjusted P-value 排序并获取前 30 个结果
        top_30_results = results_df.sort_values('Adjusted P-value').head(30)

        # 创建交互式条形图，显示富集分析结果
        fig = make_subplots(rows=1, cols=1)
        fig.add_trace(
            go.Bar(
                y=top_30_results['Term'],
                x=-np.log10(top_30_results['Adjusted P-value']),
                orientation='h'
            )
        )

        fig.update_layout(
            title='Top 30 Enriched GO Terms',
            xaxis_title='-log10(Adjusted P-value)',
            yaxis=dict(autorange="reversed"),
            height=800,
            width=1000
        )

        # 保存交互式图表
        html_filename = "top_30_enrichment_interactive.html"
        fig.write_html(os.path.join(self.save_path, html_filename))
        print(f"Interactive plot saved as '{os.path.join(self.display_path, html_filename)}'")

        # 创建并保存基因和富集得分的列表
        gene_enrichment_scores = pd.DataFrame({
            'Gene': results_df['Genes'].str.split(';').explode(),
            'Enrichment Score': -np.log10(
                results_df['Adjusted P-value'].repeat(results_df['Genes'].str.count(';') + 1))
        })
        gene_enrichment_scores = gene_enrichment_scores.groupby('Gene').max().sort_values(
            'Enrichment Score', ascending=False)
        
        # 保存基因富集得分列表
        csv_filename = 'gene_enrichment_scores.csv'
        gene_enrichment_scores.to_csv(os.path.join(self.save_path, csv_filename))
        print(f"Gene enrichment scores saved as '{csv_filename}'")

        self.results.append({
            'name': 'Enrichment Analysis Results',
            'csv_path': os.path.join(self.display_path, csv_filename),  # 基因富集得分文件路径
            'description': f"Top 30 enriched GO terms for {self.enrichment_analysis} with gene enrichment scores.",
        })

        # 第二次向 self.results 添加，使用 img_path
        self.results.append({
            'name': 'Enrichment Analysis Visualization',
            'html_path': os.path.join(self.display_path, html_filename),  # 交互式图表路径
            'description': f"Top 30 enriched GO terms for {self.enrichment_analysis} with interactive visualization.",
        })
        
        # -------------------- 其他可视化部分 --------------------

        # 获取富集的基因列表
        enriched_genes = top_30_results['Genes'].str.split(';').explode().unique().tolist()
        # 确保基因在数据集中存在
        enriched_genes = [gene for gene in enriched_genes if gene in self.adata.var_names]
        
        if enriched_genes:
            # 1. 绘制富集基因在各个聚类中的表达热图
            groupby = 'leiden' if self.clustering == "Leiden Clustering" else 'louvain'
            sc.pl.heatmap(
                self.adata, 
                var_names=enriched_genes, 
                groupby=groupby,
                show=False, 
                save='_enriched_genes.png')
            # 移动并重命名图片
            shutil.move('figures/heatmap_enriched_genes.png', os.path.join(self.save_path, 'enriched_genes_heatmap.png'))
            print("Heatmap saved as enriched_genes_heatmap.png")
            
            self.results.append({
                'name': 'Heatmap of Enriched Genes',
                'img_path': [os.path.join(self.display_path, 'enriched_genes_heatmap.png')],
                'description': 'Heatmap showing expression of enriched genes across different clusters.',
            })

            # 2. 绘制富集基因在各个聚类中的气泡图
            # 获取聚类信息
            clusters = self.adata.obs[groupby]
            # 获取富集基因的表达数据
            expr_data = self.adata[:, enriched_genes].to_df()
            expr_data['cluster'] = clusters.values
            # 计算每个聚类中基因的平均表达
            cluster_expr = expr_data.groupby('cluster').mean(
                ).reset_index().melt(id_vars='cluster', var_name='Gene', value_name='Expression')
            # 合并富集得分
            cluster_expr = cluster_expr.merge(gene_enrichment_scores.reset_index(), on='Gene')
            # 确保表达值是非负的（用于气泡大小）
            cluster_expr['Expression'] = cluster_expr['Expression'].apply(lambda x: max(x, 0))
            
            # 创建气泡图
            fig = go.Figure(go.Scatter(
                x=cluster_expr['cluster'],
                y=cluster_expr['Gene'],
                mode='markers',
                marker=dict(
                    size=cluster_expr['Expression'] * 10,  # 调整大小因子
                    color=cluster_expr['Enrichment Score'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Enrichment Score')
                ),
                text=cluster_expr['Expression'],
                hovertemplate='Cluster: %{x}<br>Gene: %{y}<br>Expression: %{text:.2f}<br>Enrichment Score: %{marker.color:.2f}<extra></extra>',
            ))


            fig.update_layout(
                title='Bubble Plot of Enriched Genes Across Clusters',
                xaxis_title='Cluster',
                yaxis_title='Gene',
                yaxis={'categoryorder': 'total ascending'},
                height=800,
                width=1000
            )

            # 保存气泡图
            fig.write_html(os.path.join(self.save_path, "bubble_plot_enriched_genes.html"))
            print("Bubble plot saved as bubble_plot_enriched_genes.html")

            # 保存气泡图路径
            self.results.append({
                'name': 'Bubble Plot of Enriched Genes',
                'html_path': os.path.join(self.display_path, "bubble_plot_enriched_genes.html"),
                'description': 'Bubble plot showing enriched gene expression across different clusters.',
            })

            # 3. 绘制富集基因在 UMAP 上的表达分布，选择前5个基因进行可视化
            sc.pl.umap(
                self.adata, 
                color=enriched_genes[:5], 
                save='_enriched_genes_expression.png', 
                show=False)
            shutil.move('figures/umap_enriched_genes_expression.png',
                        os.path.join(self.save_path, 'umap_enriched_genes_expression.png'))
            # 打印保存结果
            print(f"UMAP plot saved")

            # 将结果保存到self.results中
            self.results.append({
                'name': 'UMAP Plot of Enriched Genes',
                'img_path': [os.path.join(self.display_path, 'umap_enriched_genes_expression.png')],
                'description': 'UMAP plot showing expression distribution of the top 5 enriched genes.',
            })

            # 4. 绘制富集 GO 术语的网络图
            G = nx.Graph() # 创建图对象

            # 添加节点（GO 术语）
            for idx, row in top_30_results.iterrows():
                term = row['Term']
                genes = row['Genes'].split(';')
                G.add_node(term, genes=genes)

            # 添加边（共享基因数）
            for i, term1 in enumerate(top_30_results['Term']):
                genes1 = set(top_30_results.iloc[i]['Genes'].split(';'))
                for j in range(i + 1, len(top_30_results)):
                    term2 = top_30_results['Term'].iloc[j]
                    genes2 = set(top_30_results.iloc[j]['Genes'].split(';'))
                    shared_genes = genes1.intersection(genes2)
                    if len(shared_genes) > 0:
                        G.add_edge(term1, term2, weight=len(shared_genes))

            # 绘制网络图
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(G, k=0.5)
            node_sizes = [len(G.nodes[node]['genes']) * 100 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')
            nx.draw_networkx_labels(G, pos, font_size=8)
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, width=weights)
            
            plt.title('Network of Enriched GO Terms')
            plt.axis('off')
            plt.savefig(os.path.join(self.save_path, 'enriched_go_terms_network.png'))
            plt.close()
            
            print(f"Network plot saved")

            # 保存网络图路径到self.results
            self.results.append({
                'name': 'Network of Enriched GO Terms',
                'img_path': [os.path.join(self.display_path, 'enriched_go_terms_network.png')],
                'description': 'Network plot showing connections between enriched GO terms based on shared genes.',
            })
            
        else:
            print("No enriched genes found in the dataset for further visualization.")
            self.results.append(
                {
                    'name': 'Enrichment Genes Analysis Vasualization',
                    'description': 'No enriched genes found in the dataset for further visualization.',
                }
            )
        
    # 执行所有方法
    def execute_workflow(self):
        # 首先计算 QC 指标
        print('Calculating QC metrics...')
        self.calculate_qc_metrics()
        
        print('Filtering cells and genes...')
        self.filter_cells_and_genes()
        
        print('Normalizing data...')
        self.normalize_data()
        
        print('Identifying highly variable genes...')
        self.adata.X = np.nan_to_num(self.adata.X)
        self.find_highly_variable_genes('hvg.png')
        
        print('Filtering highly variable genes...')
        self.filter_hvg()
        
        print('Scaling data...')
        self.scale_data()
        
        print('Performing PCA...')
        self.pca()
        
        print('Computing neighbors...')
        self.neighbors()
        
        # bacth correction
        print('Applying batch correction...')
        self.compare_batch_correction()
        
        print('Running UMAP...')
        self.umap()
        
        print(f'clustering with {self.clustering}...')
        if self.clustering == 'Leiden Clustering':
            self.leiden()
        elif self.clustering == 'Louvain Clustering':
            self.louvain()
            
        print(f'Differential Expression Analysis with {self.diff_expr}...')
        self.calculate_diff_expr()
        
        print(f'Enrichment Analysis with {self.enrichment_analysis}...')
        self.perform_enrichment_and_annotation()
        
        print('Workflow Completed...') 



if __name__ == '__main__':
    print('~~~~~~~~~~~~~this is a class for scRNA-seq~~~~~~~~~~~~')