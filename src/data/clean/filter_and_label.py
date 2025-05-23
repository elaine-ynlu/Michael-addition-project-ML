from rdkit import Chem
from rdkit import RDLogger # Optional: Suppress RDKit warnings/errors
import os
import pandas as pd
import glob
# Assuming project_paths is defined correctly elsewhere
from src.data.common.paths import project_paths
import re
import traceback # For more detailed error logging if needed
from src.data.common.ring_analysis import RingAnalyzer
from src.data.common.reaction_parser import ReactionParser # 导入 ReactionParser
from src.data.common.canonicalization import SMILESCanonicalizer # 导入 SMILESCanonicalizer
from src.data.common.smiles_utils import Reaction, ReactionComponent
from src.data.common.logger import ProcessLogger
from src.data.molecular_similarity import MolecularSimilarityCalculator  # 导入相似度计算器
from src.data.common.symmetry_analysis import SymmetryAnalyzer # <--- 导入 SymmetryAnalyzer
from src.data.atom_mapping.rxn_mapper_utils import ReactionAtomMapper # <--- 导入 ReactionAtomMapper
from src.data.atom_mapping.local_mapper_utils import LocalReactionAtomMapper # <--- 导入 LocalReactionAtomMapper
import argparse # 导入 argparse
import json # 导入 json 用于序列化
from typing import Optional, List


class MainFilter:
    """主数据过滤器"""
    def __init__(self):
        self.logger = ProcessLogger('RingAndProductSimilarityAnalyzer') # Updated logger name
        # 实例化依赖项
        self.reaction_parser = ReactionParser() # ReactionParser不再依赖Canonicalizer
        self.canonicalizer = SMILESCanonicalizer() # Canonicalizer不再依赖ReactionParser
        self.ring_analyzer = RingAnalyzer() # RingAnalyzer现在没有构造函数参数
        self.symmetry_analyzer = SymmetryAnalyzer() # <--- 初始化 SymmetryAnalyzer
        self.atom_mapper = ReactionAtomMapper() # <--- 初始化 ReactionAtomMapper
        self.local_atom_mapper = LocalReactionAtomMapper() # <--- 初始化 LocalReactionAtomMapper
        # 初始化分子相似度计算器
        self.similarity_calculator = MolecularSimilarityCalculator()

    def add_fingerprints_to_result(self, result: dict, reaction: Reaction):
        """
        将反应产物的指纹添加到结果字典中，以纯字符串形式存储在 'Fingerprints' 键下。
        
        参数:
        result (dict): 分析结果字典
        reaction (Reaction): 反应对象
        """
        # 添加产物指纹
        if not reaction.products or not reaction.products.mols:
            result['Fingerprints'] = None
            return

        try:
            # 使用 get_fingerprint 计算产物指纹 (假设该方法现在只计算产物指纹)
            fp = self.similarity_calculator.get_fingerprint(reaction)
            if fp is not None:
                # 将指纹转换为字符串表示
                fp_str = self.similarity_calculator.fingerprint_to_string(fp)
                result['Fingerprints'] = fp_str
            else:
                result['Fingerprints'] = None
        except Exception as e:
            self.logger.warning(f"为产物生成指纹时发生错误: {e}")
            result['Fingerprints'] = None
            
    def _build_reaction_label(self, ring_analysis_output: dict, reaction_obj: Reaction, reactants_symmetry_flags: Optional[List[bool]], products_symmetry_flags: Optional[List[bool]]) -> str:
        """Helper function to build the reaction label string."""
        product_ring_info = ring_analysis_output.get('product_ring_info', {})
        
        reactants_symmetric_count = sum(1 for x in reactants_symmetry_flags if x) if reactants_symmetry_flags else 0
        products_symmetric_count = sum(1 for x in products_symmetry_flags if x) if products_symmetry_flags else 0
        
        label_parts = [
            f"HasRing={product_ring_info.get('has_ring', False)}",
            f"NumRings={product_ring_info.get('num_rings', 0)}",
            f"IsSingleRing={product_ring_info.get('is_single_ring_molecule', False)}",
            f"IsRingClosing={ring_analysis_output.get('is_ring_closing', False)}",
            f"HasFusedRings={product_ring_info.get('has_fused_rings', False)}",
            f"HasSpiroRings={product_ring_info.get('has_spiro_rings', False)}",
            f"NumReactants={len(reaction_obj.reactants) if reaction_obj.reactants else 0}",
            f"NumProducts={len(reaction_obj.products) if reaction_obj.products else 0}",
            f"ReactantsSymmetricCount={reactants_symmetric_count}",
            f"ProductsSymmetricCount={products_symmetric_count}"
        ]
        return "|".join(label_parts)

    def process_dataframe(self, df, smiles_column='SMILES'):
        """处理DataFrame中的SMILES列，进行环分析并提取反应产物指纹"""
        if smiles_column not in df.columns:
            if not df.empty: # Only log error if df was supposed to have columns
                self.logger.error(f"DataFrame中未找到指定的SMILES列: '{smiles_column}'")
            else:
                self.logger.info(f"输入DataFrame为空或不包含列 '{smiles_column}'，跳过处理。")
            return []

        self.logger.info(f"开始处理DataFrame中的'{smiles_column}'列，共{len(df)}行数据")

        processed_reactions = []
        success_count = 0
        failure_count = 0

        for idx, row in df.iterrows():
            reaction_smiles_original = row[smiles_column]
            source = row.get('Source', 'unknown')
            current_result = {'Source': source}

            if pd.isna(reaction_smiles_original) or not isinstance(reaction_smiles_original, str) or not reaction_smiles_original.strip():
                self.logger.warning(f"跳过索引 {idx}: 无效或空的反应SMILES")
                failure_count += 1
                continue

            # 1. 清理原始SMILES，包括清理立体化学符号
            cleaned_smiles, reason = self.reaction_parser.clean_reaction_smiles(reaction_smiles_original)
            if reason:
                self.logger.warning(f"索引 {idx}: 清理SMILES失败 ('{reaction_smiles_original}') - {reason}")
                failure_count += 1
                continue

            # 2. 分割为组分
            parts, reason = self.reaction_parser.split_reaction_smiles_robust(cleaned_smiles)
            if reason:
                self.logger.warning(f"索引 {idx}: 分割SMILES失败 ('{cleaned_smiles}') - {reason}")
                failure_count += 1
                continue

            # 2.5. 进行配平检测
            is_ok, reason = self.canonicalizer.check_reaction_atom_balance_assumed_1_to_1(parts['reactants'], parts['products'])
            if not is_ok:
                self.logger.warning(f"索引 {idx}: 配平SMILES失败 {reason}")
                failure_count += 1
                continue

            # 3. 规范化每个组分
            final_cano_reactants_no_stereo_str, reason_r_stereo = self.canonicalizer.canonicalize_component_smiles(parts['reactants'])
            final_cano_products_no_stereo_str, reason_p_stereo = self.canonicalizer.canonicalize_component_smiles(parts['products'])

            # 4. 创建 Reaction 对象 (使用已移除立体化学的、规范化的组分SMILES)
            try:
                reaction_obj = Reaction(
                    reactants=ReactionComponent(final_cano_reactants_no_stereo_str.split('.') if final_cano_reactants_no_stereo_str else []),
                    products=ReactionComponent(final_cano_products_no_stereo_str.split('.') if final_cano_products_no_stereo_str else [])
                )
                if not reaction_obj.is_valid_mols(): # 检查 Mol 对象是否都成功解析
                    self.logger.warning(f"索引 {idx}: 从无立体化学SMILES创建Reaction对象时，部分分子解析失败。R: '{final_cano_reactants_no_stereo_str}', P: '{final_cano_products_no_stereo_str}'")
                    failure_count += 1
                    continue
            except Exception as e:
                self.logger.error(f"索引 {idx}: 创建无立体化学Reaction对象时出错 - {str(e)}")
                failure_count += 1
                continue

            # 5. 清理重复反应
            no_duplicate_smiles, reason = self.reaction_parser.standardize_and_check_duplicates(reaction_obj, idx)
            if reason:
                self.logger.warning(f"索引 {idx}: 清理重复SMILES失败 ('{reaction_smiles_original}') - {reason}")
                failure_count += 1
                continue

            current_result['reactants'] = no_duplicate_smiles['reactants'] # 这是用于构成键的反应物部分 (无立体化学，排序)
            current_result['products'] = no_duplicate_smiles['products'] # 这是用于构成键的产物部分 (无立体化学，排序)

            # 5.5 原子映射 (使用无立体化学，排序后的反应物和产物)
            # 初始化映射字段
            current_result['rxn_mapped_SMILES'] = None
            current_result['rxn_confidence'] = None
            current_result['local_mapped_SMILES'] = None
            current_result['local_confidence'] = None
            
            # 构造用于原子映射的SMILES字符串: reactants>>products
            if no_duplicate_smiles['reactants'] and no_duplicate_smiles['products']:
                smiles_for_mapping = f"{no_duplicate_smiles['reactants']}>>{no_duplicate_smiles['products']}"
                
                # RXNMapper 调用
                rxn_mapped_output, rxn_mapping_error = self.atom_mapper.get_mapped_reaction_smiles(smiles_for_mapping)
                if rxn_mapping_error:
                    self.logger.warning(f"索引 {idx}: RXNMapper原子映射失败 for '{smiles_for_mapping}' - {rxn_mapping_error}")
                elif rxn_mapped_output:
                    current_result['rxn_mapped_SMILES'] = rxn_mapped_output.get("mapped_rxn")
                    current_result['rxn_confidence'] = rxn_mapped_output.get("confidence")
                
                # LocalMapper 调用
                # local_mapped_output, local_mapping_error = self.local_atom_mapper.get_mapped_reaction_smiles(smiles_for_mapping)
                # if local_mapping_error:
                #     self.logger.warning(f"索引 {idx}: LocalMapper原子映射失败 for '{smiles_for_mapping}' - {local_mapping_error}")
                # elif local_mapped_output:
                #     current_result['local_mapped_SMILES'] = local_mapped_output.get("mapped_rxn")
                #     current_result['local_confidence'] = local_mapped_output.get("confident")
            else:
                self.logger.warning(f"索引 {idx}: 跳过原子映射，因为反应物或产物为空。R: '{no_duplicate_smiles['reactants']}', P: '{no_duplicate_smiles['products']}'")

            # 6. 环分析 (来自RingAnalyzer, 使用无立体化学的 reaction_obj)
            ring_analysis_output, ring_reason = self.ring_analyzer.analyze_reaction(reaction_obj, idx)
            if ring_reason:
                self.logger.warning(f"索引 {idx}: 环结构分析失败 - {ring_reason}")
                failure_count += 1
                # 构建一个部分标签，即使环分析失败
                current_result['label'] = self._build_reaction_label({}, reaction_obj, None, None)
                current_result['Fingerprints'] = None
                # processed_reactions.append(current_result) # 可选：是否保存部分结果
                # success_count +=1
                continue 
            
            # 6.5 对称性分析 (使用无立体化学的 reaction_obj)
            reactants_symmetry_flags = None
            products_symmetry_flags = None
            try:
                symmetry_labels = self.symmetry_analyzer.label_reaction_symmetry(reaction_obj)
                reactants_symmetry_flags = symmetry_labels.get('reactants_symmetry')
                products_symmetry_flags = symmetry_labels.get('products_symmetry')

            except Exception as e_sym:
                self.logger.warning(f"索引 {idx}: 对称性分析失败 - {str(e_sym)}")
                # 即使对称性分析失败，也继续，标签中将使用默认计数

            # 7. 构建标签 (在MainFilter中，包含对称性信息)
            current_result['label'] = self._build_reaction_label(ring_analysis_output, reaction_obj, reactants_symmetry_flags, products_symmetry_flags)
            
            # 8. 添加产物指纹 (使用无立体化学的 reaction_obj)
            current_result['Fingerprints'] = None
            self.add_fingerprints_to_result(current_result, reaction_obj)
            
            processed_reactions.append(current_result)
            success_count += 1

            if (idx + 1) % 1000 == 0 or (idx + 1) == len(df): 
                current_duplicates = self.reaction_parser.get_duplicate_count()
                self.logger.info(f"已处理{idx + 1}/{len(df)}行 (成功: {success_count}, 失败: {failure_count}, 当前重复: {current_duplicates})")

        final_duplicate_count = self.reaction_parser.get_duplicate_count()
        self.logger.info(f"DataFrame处理完成。成功处理{success_count}个反应，失败/跳过{failure_count}个。检测到总重复反应数: {final_duplicate_count}")
        return processed_reactions

    def filter_and_label(self, df, smiles_column='SMILES'):
        """分析反应中的环结构信息并提取反应产物全局相似度列表"""
        if df.empty:
            self.logger.info("输入DataFrame为空，不进行处理。")
            return pd.DataFrame()
            
        # 第一次处理，获取所有的反应和产物指纹
        processed_reactions = self.process_dataframe(df, smiles_column)
        if not processed_reactions:
            # This can happen if df was not empty but all rows failed processing or were skipped
            self.logger.info("处理后未生成有效反应数据。")
            return pd.DataFrame()

        result_df = pd.DataFrame(processed_reactions)
        
        self.logger.info(f"共处理{len(result_df)}个反应并生成结果DataFrame")
        
        # 第二次处理：收集所有产物信息，创建全局产物字典
        self.logger.info("开始收集所有产物和它们的指纹...")
        all_products = []  # 使用列表而不是字典来保存所有产物和指纹，以便维持顺序
        
        # 收集所有产物的信息
        for idx, row in result_df.iterrows():
            try:
                # 如果有产物指纹数据
                if 'Fingerprints' in row and pd.notna(row['Fingerprints']) and row['Fingerprints']:
                    fp_str = row['Fingerprints']
                    
                    # 获取该反应的第一个产物SMILES (作为代表)
                    if 'products' in row and row['products']:
                        products = row['products'].split('.')
                        if products:  # 确保有至少一个产物
                            first_product = products[0]
                            all_products.append({
                                'idx': idx, # 原始DataFrame索引
                                'smiles': first_product,
                                'fp_str': fp_str
                            })
                # else: # 可选：记录没有指纹的行
                #     self.logger.debug(f"索引 {idx} 没有有效的产物指纹。")

            except Exception as e:
                self.logger.warning(f"处理索引 {idx} 的数据以收集指纹时出错: {e}")
        
        total_products = len(all_products)
        self.logger.info(f"收集到 {total_products} 个产物及其指纹用于相似度计算")
        
        # 第三次处理：计算全局产物相似度列表
        if total_products > 1:  # 至少需要两个产物才能计算相似度
            self.logger.info("开始按索引位置计算所有产物之间的相似度...")
            
            # 先解码所有指纹字符串为指纹对象（预处理以提高效率）
            product_fps = []
            valid_product_indices = []  # 记录 all_products 中有效指纹的索引
            
            for i, product in enumerate(all_products):
                try:
                    fp = self.similarity_calculator.string_to_fingerprint(product['fp_str'])
                    if fp:
                        product_fps.append(fp)
                        valid_product_indices.append(i) # 记录的是在 all_products 列表中的索引
                    else:
                        self.logger.warning(f"原始索引 {product['idx']} 的产物指纹解码失败，将跳过")
                except Exception as e:
                    self.logger.warning(f"解码原始索引 {product['idx']} 的产物指纹时出错: {e}")
            
            total_valid_products = len(product_fps)
            self.logger.info(f"成功解码 {total_valid_products} 个产物的指纹，开始计算相似度列表...")
            
            if total_valid_products > 1:
                # 预先计算所有两两相似度，构建完整相似度矩阵
                # 这样对于每个产物就不需要重复计算
                product_similarity_matrix = [[0.0 for _ in range(total_valid_products)] for _ in range(total_valid_products)]
                
                # 填充产物相似度矩阵
                for i in range(total_valid_products):
                    # 对角线元素（自身相似度）设为1.0
                    product_similarity_matrix[i][i] = 1.0
                    
                    # 计算与其他产物的相似度
                    for j in range(i + 1, total_valid_products):
                        try:
                            similarity = self.similarity_calculator.calculate_similarity(product_fps[i], product_fps[j])
                            rounded_similarity = round(similarity, 4)
                            # 矩阵是对称的
                            product_similarity_matrix[i][j] = rounded_similarity
                            product_similarity_matrix[j][i] = rounded_similarity
                        except Exception as e:
                            # 获取原始索引用于日志记录
                            original_idx_i = all_products[valid_product_indices[i]]['idx']
                            original_idx_j = all_products[valid_product_indices[j]]['idx']
                            self.logger.warning(f"计算原始索引 {original_idx_i} 和 {original_idx_j} 的产物相似度时出错: {e}")
                            # 发生错误时设为0
                            product_similarity_matrix[i][j] = 0.0
                            product_similarity_matrix[j][i] = 0.0
                
                self.logger.info("产物相似度矩阵计算完成，开始为每个产物生成相似度列表...")
                
                # 初始化产物相似度列表数组
                result_df['GlobalProductSimilarities'] = None
                
                # 为每个产物创建一个按索引排列的相似度列表
                # 例如，如果有10个有效产物，每个产物的列表都是长度为10的数组
                # 第k个位置存储与 valid_product_indices[k] 对应产物的相似度
                for product_matrix_idx, all_products_idx in enumerate(valid_product_indices):
                    original_df_idx = all_products[all_products_idx]['idx']  # 获取原始数据框中的索引
                    
                    # 创建相似度列表，按照在 valid_product_indices 中的顺序
                    # 每个位置 k 存储与 valid_product_indices[k] 产物的相似度
                    similarity_list = []
                    
                    for k in range(total_valid_products):
                        similarity_list.append(product_similarity_matrix[product_matrix_idx][k])
                    
                    # 转换为逗号分隔的字符串
                    similarity_str = ",".join(str(sim) for sim in similarity_list)
                    # 使用原始 DataFrame 索引来更新结果
                    result_df.loc[original_df_idx, 'GlobalProductSimilarities'] = similarity_str
                    
                    # 定期打印进度
                    if (product_matrix_idx + 1) % 100 == 0 or (product_matrix_idx + 1) == total_valid_products:
                        self.logger.info(f"已处理 {product_matrix_idx + 1}/{total_valid_products} 个产物的相似度列表")
                
                # 记录统计信息
                num_with_product_similarities = result_df['GlobalProductSimilarities'].notna().sum()
                self.logger.info(f"已添加全局产物相似度列表的反应数量: {num_with_product_similarities}")
            else:
                self.logger.warning("有效产物指纹数量不足2个，无法构建全局产物相似度列表")
        else:
            self.logger.warning("收集到的产物数量不足2个，无法构建全局产物相似度列表")

        # 最终统计信息
        if 'label' in result_df.columns:
            ring_count = len(result_df[result_df['label'].str.contains('HasRing=True', na=False)])
            self.logger.info(f"包含环的反应: {ring_count}")
        
        if 'Fingerprints' in result_df.columns:
            num_with_product_fps = result_df['Fingerprints'].notna().sum()
            self.logger.info(f"包含产物指纹的反应数量: {num_with_product_fps}")
        return result_df


def filter_and_label_run(num_rows_to_process=None):
    """标记反应环结构并分析反应产物分子两两相似度的主函数"""
    print("运行反应环结构标记及反应产物分子两两相似度分析...")
    analyzer = MainFilter()
    
    input_file = os.path.join(project_paths.data_filtered, 'filtered_reactions.csv')
    output_file = os.path.join(project_paths.data_michael, 'labeled_reactions_with_similarity.csv')
    
    try:
        read_nrows = None
        if num_rows_to_process is not None:
            if num_rows_to_process > 0:
                print(f"将读取并处理前 {num_rows_to_process} 行数据。")
                read_nrows = num_rows_to_process
            else:
                print(f"指定的行数 {num_rows_to_process} 无效。将读取并处理所有数据。")
        else:
            print("将读取并处理所有数据。")

        try:
            df = pd.read_csv(input_file, nrows=read_nrows)
        except pd.errors.EmptyDataError:
            print(f"错误: 输入文件 {input_file} 为空或格式不正确。")
            return False
        except FileNotFoundError:
            print(f"错误: 输入文件未找到于 {input_file}")
            return False

        if df.empty:
            if read_nrows is not None and read_nrows > 0:
                 print(f"警告: 从文件 {input_file} 读取的数据为空 (可能文件行数少于指定的 {read_nrows} 行或文件本身为空)。")
            else:
                 print(f"警告: 从文件 {input_file} 读取的数据为空。")
            # Continue with empty df, filter_and_label will handle it

        # 开始解析
        result_df = analyzer.filter_and_label(df)
        
        if not result_df.empty:
            # 在保存前移除指纹字段 (现在只有 'Fingerprints')
            # if 'Fingerprints' in result_df.columns:
            #     result_df = result_df.drop(columns=['Fingerprints'])
            #     print("已移除Fingerprints字段")
            
            result_df.to_csv(output_file, index=False)
            print(f"分析结果已保存到: {output_file}")
        else:
            print("未生成任何结果，输出文件未创建。")
        return True
        
    except Exception as e:
        print(f"处理过程中发生未预料的错误: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        print("处理完成。")


# 如果直接运行此文件
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="标记反应环结构并分析反应产物分子两两相似度。")
    parser.add_argument(
        "--num_rows",
        type=int,
        # default=1000,  # 默认处理1000行
        default=None,  # 全部处理
        help="要处理的数据行数。如果未指定或指定为0或负数，则处理所有行。"
    )
    args = parser.parse_args()
    
    # argparse ensures args.num_rows is int or None.
    # filter_and_label_run handles None or non-positive values by processing all rows.
    filter_and_label_run(num_rows_to_process=args.num_rows)
