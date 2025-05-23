#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
热图生成器：从CSV中读取GlobalProductSimilarities和GlobalReactionSimilarities字段，生成热图并保存到data/hitmap目录。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.data.common.paths import project_paths

class HeatmapGenerator:
    """
    从CSV文件中读取相似度数据并生成热图
    """
    
    def __init__(self):
        """初始化热图生成器"""
        self.setup_directories()
    
    def setup_directories(self):
        """设置并确保目录存在"""
        # 创建hitmap目录
        self.heatmap_dir = project_paths.data_heatmap
        if not os.path.exists(self.heatmap_dir):
            os.makedirs(self.heatmap_dir)
            print(f"创建目录: {self.heatmap_dir}")
    
    def read_csv_file(self, file_path):
        """读取CSV文件"""
        print(f"读取CSV文件: {file_path}")
        try:
            df = pd.read_csv(file_path)
            return df
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            return None
    
    def extract_similarity_matrices(self, df):
        """
        从DataFrame中提取产物和反应相似度矩阵
        """
        # 提取全局产物相似度数据
        product_matrix = None
        reaction_matrix = None
        
        if 'GlobalProductSimilarities' in df.columns:
            print("提取产物相似度矩阵...")
            product_data = df['GlobalProductSimilarities'].dropna()
            if not product_data.empty:
                # 将每行的字符串转换为数字列表
                product_lists = [list(map(float, row.split(','))) for row in product_data]
                
                # 确定矩阵大小（取最长列表的长度）
                max_length = max(len(lst) for lst in product_lists)
                
                # 创建相似度矩阵
                product_matrix = np.zeros((len(product_lists), max_length))
                
                # 填充矩阵
                for i, row_list in enumerate(product_lists):
                    for j, value in enumerate(row_list):
                        if j < max_length:  # 确保不超出矩阵边界
                            product_matrix[i, j] = value
                
                print(f"成功创建产物相似度矩阵: 形状 {product_matrix.shape}")
        else:
            print("CSV中未找到'GlobalProductSimilarities'列")
        
        # 提取全局反应相似度数据
        if 'GlobalReactionSimilarities' in df.columns:
            print("提取反应相似度矩阵...")
            reaction_data = df['GlobalReactionSimilarities'].dropna()
            if not reaction_data.empty:
                # 将每行的字符串转换为数字列表
                reaction_lists = [list(map(float, row.split(','))) for row in reaction_data]
                
                # 确定矩阵大小
                max_length = max(len(lst) for lst in reaction_lists)
                
                # 创建相似度矩阵
                reaction_matrix = np.zeros((len(reaction_lists), max_length))
                
                # 填充矩阵
                for i, row_list in enumerate(reaction_lists):
                    for j, value in enumerate(row_list):
                        if j < max_length:
                            reaction_matrix[i, j] = value
                
                print(f"成功创建反应相似度矩阵: 形状 {reaction_matrix.shape}")
        else:
            print("CSV中未找到'GlobalReactionSimilarities'列")
        
        return product_matrix, reaction_matrix
    
    def generate_heatmap(self, matrix, title, output_path):
        """
        生成并保存热图
        
        参数:
        matrix (numpy.ndarray): 相似度矩阵
        title (str): 热图标题
        output_path (str): 输出文件路径
        """
        if matrix is None:
            print(f"无法生成热图: {title} - 矩阵为空")
            return False
        
        try:
            # 设置图像大小和DPI
            plt.figure(figsize=(12, 10), dpi=300)
            
            # 使用seaborn绘制热图
            sns.heatmap(
                matrix, 
                cmap='viridis',  # 使用viridis色图，也可以选择其他如'YlGnBu', 'coolwarm'等
                vmin=0,          # 最小值
                vmax=1,          # 最大值 (相似度的范围是0-1)
                square=True,     # 使单元格为正方形
                xticklabels=False,  # 不显示X轴刻度标签
                yticklabels=False,  # 不显示Y轴刻度标签
                cbar=True,       # 显示颜色条
                cbar_kws={'label': '相似度'}  # 颜色条标签
            )
            
            # 添加标题
            plt.title(title, fontsize=14)
            
            # 紧凑布局
            plt.tight_layout()
            
            # 保存图像
            plt.savefig(output_path)
            print(f"热图已保存到: {output_path}")
            
            # 关闭图像
            plt.close()
            return True
        
        except Exception as e:
            print(f"生成热图时出错: {e}")
            return False
    
    def run(self, csv_path):
        """
        运行热图生成器的主函数
        
        参数:
        csv_path (str): CSV文件路径
        """
        # 读取CSV文件
        df = self.read_csv_file(csv_path)
        if df is None:
            return
        
        # 提取相似度矩阵
        product_matrix, reaction_matrix = self.extract_similarity_matrices(df)
        
        # 生成并保存产物相似度热图
        if product_matrix is not None:
            product_output_path = os.path.join(self.heatmap_dir, 'product_similarity_heatmap.png')
            self.generate_heatmap(product_matrix, '产物相似度热图', product_output_path)
        
        # 生成并保存反应相似度热图
        if reaction_matrix is not None:
            reaction_output_path = os.path.join(self.heatmap_dir, 'reaction_similarity_heatmap.png')
            self.generate_heatmap(reaction_matrix, '反应相似度热图', reaction_output_path)


def main():
    """主函数"""
    # 获取CSV文件路径
    csv_path = os.path.join(project_paths.data_michael, 'labeled_reactions_with_similarity.csv')
    
    # 创建并运行热图生成器
    generator = HeatmapGenerator()
    generator.run(csv_path)


if __name__ == "__main__":
    main()
