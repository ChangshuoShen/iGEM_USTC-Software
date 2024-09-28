import scanpy as sc
import pandas as pd
import numpy as np
import os
import shutil
import bbknn
import scanpy as sc
import scgen
# from harmony import harmonize

class scRNAseqUtils:
    def __init__(self, adata_list, working_dir, batch_categories=None):
        """
        初始化类时，传入 AnnData 对象（scanpy 的核心数据结构）和工作目录
        """
        self.adata_list = adata_list
        self.working_dir = working_dir
        # 确保工作目录存在
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        # 合并数据
        self.adata = self.merge_data(batch_categories)
        # 计算 QC 指标
        self.calculate_qc_metrics()

    def merge_data(self, batch_categories=None):
        '''合并多个AnnData对象'''
        # 如果传入的list只有一个元素，直接返回好吧
        if len(self.adata_list) == 1:
            self.adata = self.adata_list[0]
            
        # 需要确保所有数据的基因名称一致
        var_names = set(self.adata_list[0].var_names)
        for adata in self.adata_list[1:]:
            var_names = var_names.intersection(set(adata.var_names))
        var_names = list(var_names)
        
        # 只保留共有的基因
        for i in range(len(self.adata_list)):
            self.adata_list[i] = self.adata_list[i][:, var_names]
        
        # 合并数据
        adata = self.adata_list[0].concatenate(
            *self.adata_list[1:], 
            batch_categories=batch_categories
        )
        return adata
    
    
    def calculate_qc_metrics(self):
        """
        计算质量控制所需的指标，如总 counts、基因数和线粒体基因比例
        """
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
        
    def filter_cells_and_genes(
        self, min_genes=200, min_cells=3, max_mito=0.1):
        """
        过滤基因和细胞：
            - 过滤在少于min_cells个细胞中表达的基因。
            - 过滤表达低于min_genes或高于max_genes的细胞，以及线粒体基因表达高于max_mito的细胞。
        """
        # 过滤之前可视化一遍
        self.violin_plots('violin_before.png')
        self.plot_highest_expr_genes('highest_expr_genes_before.png')
        
        sc.pp.filter_genes(self.adata, min_cells=min_cells)# 基于细胞数过滤基因
        sc.pp.filter_cells(self.adata, min_genes=min_genes)# 过滤基于基因表达数的细胞
        
        # 可选，过滤线粒体基因表达比例过高的细胞，暂时注释
        # self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mito, :]
        
        # 更新 QC 指标
        self.calculate_qc_metrics()
        # 过滤之后可视化一遍
        self.violin_plots('violin_after.png')
        self.plot_highest_expr_genes('highest_expr_genes_after.png')

    def violin_plots(self, save_name):
        """
        使用小提琴图可视化基因数、UMI counts 和线粒体基因比例，并保存图片。
        """
        default_save_path = 'figures/violin.png'
        save_path = os.path.join(self.working_dir, save_name)
        sc.pl.violin(
            self.adata, 
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            jitter=0.4, 
            multi_panel=True,
            save='.png',
            show=False
        )
        # 移动图像到工作目录
        shutil.move(default_save_path, save_path)
        return save_path

    def plot_highest_expr_genes(self, save_name, n_top_genes=20):
        """
        可视化表达最高的前 n 个基因，并保存图片。
        """
        default_save_path = 'figures/highest_expr_genes.png'
        save_path = os.path.join(
            self.working_dir, save_name
        )
        sc.pl.highest_expr_genes(
            self.adata, 
            n_top=n_top_genes, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, save_path)
        return save_path

    def normalize_data(self):
        """
        对数据进行标准化处理，每个细胞归一化到相同的总表达量，并对数转换。
        """
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)


    def find_highly_variable_genes(self, save_name):
        """
        找出高变异基因，用于后续分析。
        """
        default_save_path = 'figures/filter_genes_dispersion.png'
        save_path = os.path.join(self.working_dir, save_name )
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
        shutil.move(default_save_path, save_path)
        return save_path

    def filter_hvg(self):
        """
        选择高变异基因并保留原始数据。
        该方法只能执行一次，因为它会修改 adata 对象。
        """
        self.adata.raw = self.adata  # 保留原始数据
        self.adata = self.adata[:, self.adata.var['highly_variable']]  # 选择高变异基因
        print(self.adata.raw.shape, self.adata.shape)
    
    def scale_data(self):
        """
        对数据进行标准化，使得每个基因的表达量具有均值为0、方差为1。
        """
        # 将数据缩放到单位方差
        sc.pp.regress_out(
            self.adata, 
            ['total_counts', 'pct_counts_mt']# 要回归出去的变量列表
        )
        sc.pp.scale(self.adata, max_value=10)

    def pca(self, n_comps=50):
        """
        进行主成分分析(PCA)，并将结果保存在adata对象中。
        """
        default_save_path = 'figures/pca.png'
        save_path = [
            os.path.join(self.working_dir, 'pca.png'),
            os.path.join(self.working_dir, 'pca_variance_ratio.png')
        ]
        sc.tl.pca(
            self.adata, 
            # n_comps=n_comps,
            svd_solver='arpack' # SVD求解器，'arpack' 适用于大型稀疏矩阵
            )
        sc.pl.pca(
            self.adata, 
            color='CST3', 
            save='.png',
            show=False
        )
        sc.pl.pca_variance_ratio(
            self.adata, 
            log=True, 
            show=False,
            save='.png')
        
        shutil.move('figures/pca.png', save_path[0])
        shutil.move('figures/pca_variance_ratio.png', save_path[1])
        return save_path

    def neighbors(self, n_neighbors=10, n_pcs=40):
        """
        计算邻居，用于后续降维（如UMAP和t-SNE）和聚类。
        """
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs
        )
        
        
    def tsne(self):
        """
        使用 t-SNE 方法对数据进行降维。
        """
        sc.tl.tsne(self.adata)
        sc.pl.tsne(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'tsne.png')
        shutil.move('figures/tsne.png', save_path)
        return save_path

    def umap(self, suffix=''):
        """
        使用 UMAP 方法对数据进行降维。
        """
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap.png')
        shutil.move('figures/umap.png', save_path)
        return save_path

    def leiden(self, resolution=1.0):
        '''
        使用Leiden聚类
        '''
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.pl.umap(
            self.adata, 
            color=['leiden'], 
            save='_leiden.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_leiden.png')
        shutil.move('figures/umap_leiden.png', save_path)
        return save_path
    
    def louvain(self, resolution=1.0):
        '''
        使用Louvain聚类
        '''
        # 计算邻近（如果尚未计算）
        # sc.pp.neighbors(self.adata, use_rep='X_pca')

        # 进行Louvain聚类
        sc.tl.louvain(self.adata, resolution=resolution)

        # 可视化结果
        sc.pl.umap(
            self.adata, 
            color=['louvain'], 
            save='_louvain.png',
            show=False
        )
        
        # 移动保存的文件
        save_path = os.path.join(self.working_dir, 'umap_louvain.png')
        shutil.move('figures/umap_louvain.png', save_path)
        
        return save_path
    
    
    ###################下面是几个批次矫正方法###################
    def bbknn_correction(self, neighbors_within_batch=3, n_pcs=50):
        """
        使用 BBKNN 方法进行批次效应矫正。
        """
        bbknn.bbknn(self.adata, batch_key='batch', neighbors_within_batch=neighbors_within_batch, n_pcs=n_pcs)
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_bbknn.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_bbknn.png')
        shutil.move('figures/umap_bbknn.png', save_path)
        print("BBKNN correction finished")
        return save_path


    def combat_correction(self):
        """
        使用 Combat 方法进行批次效应矫正。
        """
        sc.pp.combat(self.adata, key='batch')
        self.pca()
        self.neighbors()
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_combat.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_combat.png')
        shutil.move('figures/umap_combat.png', save_path)
        print("Combat Correction finished")
        return save_path
    
    def harmony_correction(self):
        """
        使用 Scanpy 内置的 Harmony 方法进行批次效应校正。
        """
        # 使用 Scanpy 的 Harmony 接口进行批次效应矫正
        sc.external.pp.harmony_integrate(self.adata, key='batch')

        # 重新计算邻居
        sc.pp.neighbors(self.adata, use_rep='X_pca')
        
        # 计算 UMAP
        sc.tl.umap(self.adata)
        
        # 可视化 UMAP
        filename_umap = 'umap_harmony.png'
        sc.pl.umap(self.adata, color=['batch', 'leiden'], save=filename_umap, show=False)
        
        # 移动保存的图片到工作目录
        save_path = os.path.join(self.working_dir, filename_umap)
        shutil.move(f'figures/{filename_umap}', save_path)
        
        print("Harmony Correction Finished")
        return save_path

    # 添加比较有无批次矫正的可视化方法
    def compare_batch_correction(self):
        """
        比较合并数据在有无批次矫正情况下的结果。
        """
        # 无批次矫正的 UMAP
        self.umap(suffix='_no_batch_correction')
        
        # 应用 BBKNN 批次矫正
        self.bbknn_correction()
        
        # 需要重新计算邻居和聚类
        self.neighbors()
        self.leiden()
        
        # 再次绘制 UMAP
        self.umap(suffix='_bbknn')
        
        
        
    ###################添加差异表达分析方法及可视化方法############################
    def calculate_diff_expr(
        self, 
        method='t-test', 
        groupby='leiden', 
        n_genes=25, 
        save=True):
        """
        计算差异表达基因，支持 t-test、wilcoxon、logreg 等方法。
        
        Parameters:
        - method: 差异表达计算方法 ('t-test', 'wilcoxon', 'logreg')
        - groupby: 按照哪个分组进行差异表达计算 (默认 'leiden')
        - n_genes: 可视化时展示的基因数量
        - save: 是否保存图片 (默认为 False)
        """
        # 计算差异表达基因
        sc.tl.rank_genes_groups(
            self.adata, 
            groupby, 
            method=method
        )
        
        # 可视化
        filename = f"de_{method}.png" if save else None
        sc.pl.rank_genes_groups(
            self.adata, 
            n_genes=n_genes, 
            sharey=False, # 决定不同子图的y轴是否共享
            show=False,
            save=filename
        )
        save_path = os.path.join(
            self.working_dir, f'rank_genes_group_{groupby}' + filename)
        shutil(f'figure/rank_genes_groups_{groupby}'+filename, save_path)
        
    
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
        
class scRNAseqUtils:
    def __init__(self, adata_list, working_dir, batch_categories=None):
        """
        初始化类时，传入 AnnData 对象（scanpy 的核心数据结构）和工作目录
        """
        self.adata_list = adata_list
        self.working_dir = working_dir
        # 确保工作目录存在
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        # 合并数据
        self.adata = self.merge_data(batch_categories)
        # 计算 QC 指标
        self.calculate_qc_metrics()

    def merge_data(self, batch_categories=None):
        '''合并多个AnnData对象'''
        # 如果传入的list只有一个元素，直接返回好吧
        if len(self.adata_list) == 1:
            self.adata = self.adata_list[0]
            
        # 需要确保所有数据的基因名称一致
        var_names = set(self.adata_list[0].var_names)
        for adata in self.adata_list[1:]:
            var_names = var_names.intersection(set(adata.var_names))
        var_names = list(var_names)
        
        # 只保留共有的基因
        for i in range(len(self.adata_list)):
            self.adata_list[i] = self.adata_list[i][:, var_names]
        
        # 合并数据
        adata = self.adata_list[0].concatenate(
            *self.adata_list[1:], 
            batch_categories=batch_categories
        )
        return adata
    
    
    def calculate_qc_metrics(self):
        """
        计算质量控制所需的指标，如总 counts、基因数和线粒体基因比例
        """
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
        
    def filter_cells_and_genes(
        self, min_genes=200, min_cells=3, max_mito=0.1):
        """
        过滤基因和细胞：
            - 过滤在少于min_cells个细胞中表达的基因。
            - 过滤表达低于min_genes或高于max_genes的细胞，以及线粒体基因表达高于max_mito的细胞。
        """
        # 过滤之前可视化一遍
        self.violin_plots('violin_before.png')
        self.plot_highest_expr_genes('highest_expr_genes_before.png')
        
        sc.pp.filter_genes(self.adata, min_cells=min_cells)# 基于细胞数过滤基因
        sc.pp.filter_cells(self.adata, min_genes=min_genes)# 过滤基于基因表达数的细胞
        
        # 可选，过滤线粒体基因表达比例过高的细胞，暂时注释
        # self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mito, :]
        
        # 更新 QC 指标
        self.calculate_qc_metrics()
        # 过滤之后可视化一遍
        self.violin_plots('violin_after.png')
        self.plot_highest_expr_genes('highest_expr_genes_after.png')

    def violin_plots(self, save_name):
        """
        使用小提琴图可视化基因数、UMI counts 和线粒体基因比例，并保存图片。
        """
        default_save_path = 'figures/violin.png'
        save_path = os.path.join(self.working_dir, save_name)
        sc.pl.violin(
            self.adata, 
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            jitter=0.4, 
            multi_panel=True,
            save='.png',
            show=False
        )
        # 移动图像到工作目录
        shutil.move(default_save_path, save_path)
        return save_path

    def plot_highest_expr_genes(self, save_name, n_top_genes=20):
        """
        可视化表达最高的前 n 个基因，并保存图片。
        """
        default_save_path = 'figures/highest_expr_genes.png'
        save_path = os.path.join(
            self.working_dir, save_name
        )
        sc.pl.highest_expr_genes(
            self.adata, 
            n_top=n_top_genes, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, save_path)
        return save_path

    def normalize_data(self):
        """
        对数据进行标准化处理，每个细胞归一化到相同的总表达量，并对数转换。
        """
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)


    def find_highly_variable_genes(self, save_name):
        """
        找出高变异基因，用于后续分析。
        """
        default_save_path = 'figures/filter_genes_dispersion.png'
        save_path = os.path.join(self.working_dir, save_name )
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
        shutil.move(default_save_path, save_path)
        return save_path

    def filter_hvg(self):
        """
        选择高变异基因并保留原始数据。
        该方法只能执行一次，因为它会修改 adata 对象。
        """
        self.adata.raw = self.adata  # 保留原始数据
        self.adata = self.adata[:, self.adata.var['highly_variable']]  # 选择高变异基因
        print(self.adata.raw.shape, self.adata.shape)
    
    def scale_data(self):
        """
        对数据进行标准化，使得每个基因的表达量具有均值为0、方差为1。
        """
        # 将数据缩放到单位方差
        sc.pp.regress_out(
            self.adata, 
            ['total_counts', 'pct_counts_mt']# 要回归出去的变量列表
        )
        sc.pp.scale(self.adata, max_value=10)

    def pca(self, n_comps=50):
        """
        进行主成分分析(PCA)，并将结果保存在adata对象中。
        """
        default_save_path = 'figures/pca.png'
        save_path = [
            os.path.join(self.working_dir, 'pca.png'),
            os.path.join(self.working_dir, 'pca_variance_ratio.png')
        ]
        sc.tl.pca(
            self.adata, 
            # n_comps=n_comps,
            svd_solver='arpack' # SVD求解器，'arpack' 适用于大型稀疏矩阵
            )
        sc.pl.pca(
            self.adata, 
            color='CST3', 
            save='.png',
            show=False
        )
        sc.pl.pca_variance_ratio(
            self.adata, 
            log=True, 
            show=False,
            save='.png')
        
        shutil.move('figures/pca.png', save_path[0])
        shutil.move('figures/pca_variance_ratio.png', save_path[1])
        return save_path

    def neighbors(self, n_neighbors=10, n_pcs=40):
        """
        计算邻居，用于后续降维（如UMAP和t-SNE）和聚类。
        """
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs
        )
        
        
    def tsne(self):
        """
        使用 t-SNE 方法对数据进行降维。
        """
        sc.tl.tsne(self.adata)
        sc.pl.tsne(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'tsne.png')
        shutil.move('figures/tsne.png', save_path)
        return save_path

    def umap(self, suffix=''):
        """
        使用 UMAP 方法对数据进行降维。
        """
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap.png')
        shutil.move('figures/umap.png', save_path)
        return save_path

    def leiden(self, resolution=1.0):
        '''
        使用Leiden聚类
        '''
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.pl.umap(
            self.adata, 
            color=['leiden'], 
            save='_leiden.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_leiden.png')
        shutil.move('figures/umap_leiden.png', save_path)
        return save_path
    
    def louvain(self, resolution=1.0):
        '''
        使用Louvain聚类
        '''
        # 计算邻近（如果尚未计算）
        # sc.pp.neighbors(self.adata, use_rep='X_pca')

        # 进行Louvain聚类
        sc.tl.louvain(self.adata, resolution=resolution)

        # 可视化结果
        sc.pl.umap(
            self.adata, 
            color=['louvain'], 
            save='_louvain.png',
            show=False
        )
        
        # 移动保存的文件
        save_path = os.path.join(self.working_dir, 'umap_louvain.png')
        shutil.move('figures/umap_louvain.png', save_path)
        
        return save_path
    
    
    ###################下面是几个批次矫正方法###################
    def bbknn_correction(self, neighbors_within_batch=3, n_pcs=50):
        """
        使用 BBKNN 方法进行批次效应矫正。
        """
        bbknn.bbknn(self.adata, batch_key='batch', neighbors_within_batch=neighbors_within_batch, n_pcs=n_pcs)
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_bbknn.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_bbknn.png')
        shutil.move('figures/umap_bbknn.png', save_path)
        print("BBKNN correction finished")
        return save_path


    def combat_correction(self):
        """
        使用 Combat 方法进行批次效应矫正。
        """
        sc.pp.combat(self.adata, key='batch')
        self.pca()
        self.neighbors()
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_combat.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_combat.png')
        shutil.move('figures/umap_combat.png', save_path)
        print("Combat Correction finished")
        return save_path
    
    def harmony_correction(self):
        """
        使用 Scanpy 内置的 Harmony 方法进行批次效应校正。
        """
        # 使用 Scanpy 的 Harmony 接口进行批次效应矫正
        sc.external.pp.harmony_integrate(self.adata, key='batch')

        # 重新计算邻居
        sc.pp.neighbors(self.adata, use_rep='X_pca')
        
        # 计算 UMAP
        sc.tl.umap(self.adata)
        
        # 可视化 UMAP
        filename_umap = 'umap_harmony.png'
        sc.pl.umap(self.adata, color=['batch', 'leiden'], save=filename_umap, show=False)
        
        # 移动保存的图片到工作目录
        save_path = os.path.join(self.working_dir, filename_umap)
        shutil.move(f'figures/{filename_umap}', save_path)
        
        print("Harmony Correction Finished")
        return save_path

    # 添加比较有无批次矫正的可视化方法
    def compare_batch_correction(self):
        """
        比较合并数据在有无批次矫正情况下的结果。
        """
        # 无批次矫正的 UMAP
        self.umap(suffix='_no_batch_correction')
        
        # 应用 BBKNN 批次矫正
        self.bbknn_correction()
        
        # 需要重新计算邻居和聚类
        self.neighbors()
        self.leiden()
        
        # 再次绘制 UMAP
        self.umap(suffix='_bbknn')
        
        
        
    ###################添加差异表达分析方法及可视化方法############################
    def calculate_diff_expr(
        self, 
        method='t-test', 
        groupby='leiden', 
        n_genes=25, 
        save=True):
        """
        计算差异表达基因，支持 t-test、wilcoxon、logreg 等方法。
        
        Parameters:
        - method: 差异表达计算方法 ('t-test', 'wilcoxon', 'logreg')
        - groupby: 按照哪个分组进行差异表达计算 (默认 'leiden')
        - n_genes: 可视化时展示的基因数量
        - save: 是否保存图片 (默认为 False)
        """
        # 计算差异表达基因
        sc.tl.rank_genes_groups(
            self.adata, 
            groupby, 
            method=method
        )
        
        # 可视化
        filename = f"de_{method}.png" if save else None
        sc.pl.rank_genes_groups(
            self.adata, 
            n_genes=n_genes, 
            sharey=False, # 决定不同子图的y轴是否共享
            show=False,
            save=filename
        )
        save_path = os.path.join(
            self.working_dir, f'rank_genes_group_{groupby}' + filename)
        shutil(f'figure/rank_genes_groups_{groupby}'+filename, save_path)
        
    
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
        
        # 轨迹图
        filename_tracksplot = f"_groups_tracksplot.png" if save else None
        sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=n_genes, save=filename_tracksplot)
        
class scRNAseqUtils:
    def __init__(self, adata_list, working_dir, batch_categories=None):
        """
        初始化类时，传入 AnnData 对象（scanpy 的核心数据结构）和工作目录
        """
        self.adata_list = adata_list
        self.working_dir = working_dir
        # 确保工作目录存在
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        # 合并数据
        self.adata = self.merge_data(batch_categories)
        # 计算 QC 指标
        self.calculate_qc_metrics()

    def merge_data(self, batch_categories=None):
        '''合并多个AnnData对象'''
        # 如果传入的list只有一个元素，直接返回好吧
        if len(self.adata_list) == 1:
            self.adata = self.adata_list[0]
            
        # 需要确保所有数据的基因名称一致
        var_names = set(self.adata_list[0].var_names)
        for adata in self.adata_list[1:]:
            var_names = var_names.intersection(set(adata.var_names))
        var_names = list(var_names)
        
        # 只保留共有的基因
        for i in range(len(self.adata_list)):
            self.adata_list[i] = self.adata_list[i][:, var_names]
        
        # 合并数据
        adata = self.adata_list[0].concatenate(
            *self.adata_list[1:], 
            batch_categories=batch_categories
        )
        return adata
    
    
    def calculate_qc_metrics(self):
        """
        计算质量控制所需的指标，如总 counts、基因数和线粒体基因比例
        """
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
        
    def filter_cells_and_genes(
        self, min_genes=200, min_cells=3, max_mito=0.1):
        """
        过滤基因和细胞：
            - 过滤在少于min_cells个细胞中表达的基因。
            - 过滤表达低于min_genes或高于max_genes的细胞，以及线粒体基因表达高于max_mito的细胞。
        """
        # 过滤之前可视化一遍
        self.violin_plots('violin_before.png')
        self.plot_highest_expr_genes('highest_expr_genes_before.png')
        
        sc.pp.filter_genes(self.adata, min_cells=min_cells)# 基于细胞数过滤基因
        sc.pp.filter_cells(self.adata, min_genes=min_genes)# 过滤基于基因表达数的细胞
        
        # 可选，过滤线粒体基因表达比例过高的细胞，暂时注释
        # self.adata = self.adata[self.adata.obs['pct_counts_mt'] < max_mito, :]
        
        # 更新 QC 指标
        self.calculate_qc_metrics()
        # 过滤之后可视化一遍
        self.violin_plots('violin_after.png')
        self.plot_highest_expr_genes('highest_expr_genes_after.png')

    def violin_plots(self, save_name):
        """
        使用小提琴图可视化基因数、UMI counts 和线粒体基因比例，并保存图片。
        """
        default_save_path = 'figures/violin.png'
        save_path = os.path.join(self.working_dir, save_name)
        sc.pl.violin(
            self.adata, 
            ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], 
            jitter=0.4, 
            multi_panel=True,
            save='.png',
            show=False
        )
        # 移动图像到工作目录
        shutil.move(default_save_path, save_path)
        return save_path

    def plot_highest_expr_genes(self, save_name, n_top_genes=20):
        """
        可视化表达最高的前 n 个基因，并保存图片。
        """
        default_save_path = 'figures/highest_expr_genes.png'
        save_path = os.path.join(
            self.working_dir, save_name
        )
        sc.pl.highest_expr_genes(
            self.adata, 
            n_top=n_top_genes, 
            save='.png',
            show=False
        )
        shutil.move(default_save_path, save_path)
        return save_path

    def normalize_data(self):
        """
        对数据进行标准化处理，每个细胞归一化到相同的总表达量，并对数转换。
        """
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        sc.pp.log1p(self.adata)


    def find_highly_variable_genes(self, save_name):
        """
        找出高变异基因，用于后续分析。
        """
        default_save_path = 'figures/filter_genes_dispersion.png'
        save_path = os.path.join(self.working_dir, save_name )
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
        shutil.move(default_save_path, save_path)
        return save_path

    def filter_hvg(self):
        """
        选择高变异基因并保留原始数据。
        该方法只能执行一次，因为它会修改 adata 对象。
        """
        self.adata.raw = self.adata  # 保留原始数据
        self.adata = self.adata[:, self.adata.var['highly_variable']]  # 选择高变异基因
        print(self.adata.raw.shape, self.adata.shape)
    
    def scale_data(self):
        """
        对数据进行标准化，使得每个基因的表达量具有均值为0、方差为1。
        """
        # 将数据缩放到单位方差
        sc.pp.regress_out(
            self.adata, 
            ['total_counts', 'pct_counts_mt']# 要回归出去的变量列表
        )
        sc.pp.scale(self.adata, max_value=10)

    def pca(self, n_comps=50):
        """
        进行主成分分析(PCA)，并将结果保存在adata对象中。
        """
        default_save_path = 'figures/pca.png'
        save_path = [
            os.path.join(self.working_dir, 'pca.png'),
            os.path.join(self.working_dir, 'pca_variance_ratio.png')
        ]
        sc.tl.pca(
            self.adata, 
            # n_comps=n_comps,
            svd_solver='arpack' # SVD求解器，'arpack' 适用于大型稀疏矩阵
            )
        sc.pl.pca(
            self.adata, 
            color='CST3', 
            save='.png',
            show=False
        )
        sc.pl.pca_variance_ratio(
            self.adata, 
            log=True, 
            show=False,
            save='.png')
        
        shutil.move('figures/pca.png', save_path[0])
        shutil.move('figures/pca_variance_ratio.png', save_path[1])
        return save_path

    def neighbors(self, n_neighbors=10, n_pcs=40):
        """
        计算邻居，用于后续降维（如UMAP和t-SNE）和聚类。
        """
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=n_neighbors, 
            n_pcs=n_pcs
        )
        
        
    def tsne(self):
        """
        使用 t-SNE 方法对数据进行降维。
        """
        sc.tl.tsne(self.adata)
        sc.pl.tsne(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'tsne.png')
        shutil.move('figures/tsne.png', save_path)
        return save_path

    def umap(self, suffix=''):
        """
        使用 UMAP 方法对数据进行降维。
        """
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['CST3', 'NKG7', 'PPBP'],  # 可根据数据选择颜色参考的基因
            save='.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap.png')
        shutil.move('figures/umap.png', save_path)
        return save_path

    def leiden(self, resolution=1.0):
        '''
        使用Leiden聚类
        '''
        sc.tl.leiden(self.adata, resolution=resolution)
        sc.pl.umap(
            self.adata, 
            color=['leiden'], 
            save='_leiden.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_leiden.png')
        shutil.move('figures/umap_leiden.png', save_path)
        return save_path
    
    def louvain(self, resolution=1.0):
        '''
        使用Louvain聚类
        '''
        # 计算邻近（如果尚未计算）
        # sc.pp.neighbors(self.adata, use_rep='X_pca')

        # 进行Louvain聚类
        sc.tl.louvain(self.adata, resolution=resolution)

        # 可视化结果
        sc.pl.umap(
            self.adata, 
            color=['louvain'], 
            save='_louvain.png',
            show=False
        )
        
        # 移动保存的文件
        save_path = os.path.join(self.working_dir, 'umap_louvain.png')
        shutil.move('figures/umap_louvain.png', save_path)
        
        return save_path
    
    
    ###################下面是几个批次矫正方法###################
    def bbknn_correction(self, neighbors_within_batch=3, n_pcs=50):
        """
        使用 BBKNN 方法进行批次效应矫正。
        """
        bbknn.bbknn(self.adata, batch_key='batch', neighbors_within_batch=neighbors_within_batch, n_pcs=n_pcs)
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_bbknn.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_bbknn.png')
        shutil.move('figures/umap_bbknn.png', save_path)
        print("BBKNN correction finished")
        return save_path


    def combat_correction(self):
        """
        使用 Combat 方法进行批次效应矫正。
        """
        sc.pp.combat(self.adata, key='batch')
        self.pca()
        self.neighbors()
        sc.tl.umap(self.adata)
        sc.pl.umap(
            self.adata, 
            color=['batch', 'leiden'], 
            save='_combat.png',
            show=False
        )
        save_path = os.path.join(self.working_dir, 'umap_combat.png')
        shutil.move('figures/umap_combat.png', save_path)
        print("Combat Correction finished")
        return save_path
    
    def harmony_correction(self):
        """
        使用 Scanpy 内置的 Harmony 方法进行批次效应校正。
        """
        # 使用 Scanpy 的 Harmony 接口进行批次效应矫正
        sc.external.pp.harmony_integrate(self.adata, key='batch')

        # 重新计算邻居
        sc.pp.neighbors(self.adata, use_rep='X_pca')
        
        # 计算 UMAP
        sc.tl.umap(self.adata)
        
        # 可视化 UMAP
        filename_umap = 'umap_harmony.png'
        sc.pl.umap(self.adata, color=['batch', 'leiden'], save=filename_umap, show=False)
        
        # 移动保存的图片到工作目录
        save_path = os.path.join(self.working_dir, filename_umap)
        shutil.move(f'figures/{filename_umap}', save_path)
        
        print("Harmony Correction Finished")
        return save_path

    # 添加比较有无批次矫正的可视化方法
    def compare_batch_correction(self):
        """
        比较合并数据在有无批次矫正情况下的结果。
        """
        # 无批次矫正的 UMAP
        self.umap(suffix='_no_batch_correction')
        
        # 应用 BBKNN 批次矫正
        self.bbknn_correction()
        
        # 需要重新计算邻居和聚类
        self.neighbors()
        self.leiden()
        
        # 再次绘制 UMAP
        self.umap(suffix='_bbknn')
        
        
        
    ###################添加差异表达分析方法及可视化方法############################
    def calculate_diff_expr(
        self, 
        method='t-test', 
        groupby='leiden', 
        n_genes=25, 
        save=True):
        """
        计算差异表达基因，支持 t-test、wilcoxon、logreg 等方法。
        
        Parameters:
        - method: 差异表达计算方法 ('t-test', 'wilcoxon', 'logreg')
        - groupby: 按照哪个分组进行差异表达计算 (默认 'leiden')
        - n_genes: 可视化时展示的基因数量
        - save: 是否保存图片 (默认为 False)
        """
        # 计算差异表达基因
        sc.tl.rank_genes_groups(
            self.adata, 
            groupby, 
            method=method
        )
        
        # 可视化
        filename = f"de_{method}.png" if save else None
        sc.pl.rank_genes_groups(
            self.adata, 
            n_genes=n_genes, 
            sharey=False, # 决定不同子图的y轴是否共享
            show=False,
            save=filename
        )
        save_path = os.path.join(
            self.working_dir, f'rank_genes_group_{groupby}' + filename)
        shutil(f'figure/rank_genes_groups_{groupby}'+filename, save_path)
        
    
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
        
        # 轨迹图
        filename_tracksplot = f"_groups_tracksplot.png" if save else None
        sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=n_genes, save=filename_tracksplot)
        
        # 轨迹图
        filename_tracksplot = f"_groups_tracksplot.png" if save else None
        sc.pl.rank_genes_groups_tracksplot(self.adata, n_genes=n_genes, save=filename_tracksplot)
        
# 测试一下
if __name__ == '__main__':
    
    # 定义 AnnData 对象
    adata_pbmc3k = sc.datasets.pbmc3k()
    adata_pbmc68k = sc.datasets.pbmc68k_reduced()
    adata_pbmc3k.var_names_make_unique()
    adata_pbmc68k.var_names_make_unique()
    # 定义工作目录
    working_dir = './qc_plots'

    # # 初始化 scRNAseqUtils 类
    # utils = scRNAseqUtils(adata, working_dir)

    # # 测试过滤函数
    # utils.filter_cells_and_genes()
    
    # utils.normalize_data()
    # utils.find_highly_variable_genes('HVG.png')
    # utils.filter_hvg()
    # utils.scale_data()
    # utils.neighbors()
    # utils.pca()
    # utils.tsne()
    # utils.umap()
    # utils.leiden()
    # utils.louvain()
    # 初始化 scRNAseqUtils 类
    utils = scRNAseqUtils(
        adata_list=[adata_pbmc3k, adata_pbmc68k], 
        working_dir=working_dir, 
        batch_categories=['PBMC3K', 'PBMC68K']
    )

    # 过滤
    utils.filter_cells_and_genes()

    # 标准化
    utils.normalize_data()

    # 识别高变异基因
    utils.find_highly_variable_genes('HVG.png')
    utils.filter_hvg()

    # 归一化
    utils.scale_data()

    # PCA
    utils.pca()

    # 计算邻居
    utils.neighbors()

    # 聚类
    utils.leiden()

    # 可视化未批次矫正的 UMAP
    utils.umap(suffix='_no_batch_correction')

    # 比较批次矫正前后的结果
    # BBKNN
    utils.bbknn_correction()
    utils.leiden()
    utils.umap(suffix='_bbknn')

    # Combat
    # 需要重新加载原始数据以避免之前的矫正影响
    utils = scRNAseqUtils(
        adata_list=[adata_pbmc3k, adata_pbmc68k], 
        working_dir=working_dir, 
        batch_categories=['PBMC3K', 'PBMC68K']
    )
    utils.filter_cells_and_genes()
    utils.normalize_data()
    utils.find_highly_variable_genes('HVG.png')
    utils.filter_hvg()
    utils.scale_data()
    utils.pca()
    utils.combat_correction()
    utils.neighbors()
    utils.leiden()
    utils.umap(suffix='_combat')

    # Harmony
    # 同样需要重新加载原始数据
    utils = scRNAseqUtils(
        adata_list=[adata_pbmc3k, adata_pbmc68k], 
        working_dir=working_dir, 
        batch_categories=['PBMC3K', 'PBMC68K']
    )
    utils.filter_cells_and_genes()
    utils.normalize_data()
    utils.find_highly_variable_genes('HVG.png')
    utils.filter_hvg()
    utils.scale_data()
    utils.pca()
    utils.harmony_correction()
    utils.leiden()
    utils.umap(suffix='_harmony')
