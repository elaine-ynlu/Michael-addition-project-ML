# -*- coding: utf-8 -*-
"""
该文件用于计算分子间的Tanimoto相似度。
主要功能包括：
1. 从SMILES字符串生成分子对象。
2. 为分子生成摩根指纹 (Morgan Fingerprints)。
3. 计算两个分子指纹之间的Tanimoto相似度。
4. 处理反应组分和反应的相似度计算。
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import rdChemReactions # 导入ReactionFingerprint
from typing import List, Dict, Tuple, Optional, Any, Union
import pandas as pd
import os
from src.data.common.smiles_utils import Reaction, ReactionComponent
from src.data.common.logger import ProcessLogger
import base64
import numpy as np

class MolecularSimilarityCalculator:
    """分子相似度计算器类"""
    
    def __init__(self, radius: int = 2, n_bits: int = 1024):
        """
        初始化分子相似度计算器
        
        参数:
        radius (int): Morgan指纹的半径，默认为2 (类似于ECFP4)
        n_bits (int): 指纹的位数 (向量长度)，默认为1024
        """
        self.radius = radius
        self.n_bits = n_bits
        self.logger = ProcessLogger('MolecularSimilarity')
    
    def get_morgan_fingerprint(self, mol: Chem.Mol) -> Optional[Any]:
        """
        为分子生成Morgan指纹
        
        参数:
        mol (Chem.Mol): RDKit分子对象
        
        返回:
        Any: 分子的Morgan指纹，如果分子无效则返回None
        """
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, self.radius, self.n_bits)
    def get_drfp_fingerprint(self, reaction: Reaction) -> Optional[DataStructs.ExplicitBitVect]:
        """
        为化学反应生成基于Morgan指纹的差异反应指纹 (DRFP - Difference Reaction Fingerprint)。
        该方法手动计算反应物指纹总和与产物指纹总和之间的差异。

        参数:
        reaction (Reaction): Reaction对象，包含反应物和产物的SMILES列表。

        返回:
        ExplicitBitVect: 反应的差异指纹 (按位异或)，如果无法生成则返回None。
                         返回的指纹长度由类的 n_bits 属性决定。
        """
        if not reaction or not reaction.reactants.smiles_list or not reaction.products.smiles_list:
            self.logger.warning("反应物或产物为空，无法生成DRFP指纹。")
            return None

        # 初始化空的指纹向量
        reactant_fp_sum = DataStructs.ExplicitBitVect(self.n_bits)
        product_fp_sum = DataStructs.ExplicitBitVect(self.n_bits)

        # 计算反应物指纹总和 (按位或)
        for smiles in reaction.reactants.smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = self.get_morgan_fingerprint(mol) # 使用类内方法获取Morgan指纹
                if fp:
                    reactant_fp_sum |= fp
                else:
                    self.logger.warning(f"无法为反应物 {smiles} 生成指纹，已跳过。")
                    # 可以选择更严格的策略，例如如果任何一个分子无效则返回None
                    # return None
            else:
                self.logger.warning(f"无法解析反应物SMILES: {smiles}，已跳过。")
                # return None

        # 计算产物指纹总和 (按位或)
        for smiles in reaction.products.smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                fp = self.get_morgan_fingerprint(mol) # 使用类内方法获取Morgan指纹
                if fp:
                    product_fp_sum |= fp
                else:
                    self.logger.warning(f"无法为产物 {smiles} 生成指纹，已跳过。")
                    # return None
            else:
                self.logger.warning(f"无法解析产物SMILES: {smiles}，已跳过。")
                # return None

        # 计算差异指纹 (按位异或)
        # 这是DRFP的一种常见计算方式，表示反应前后发生变化的特征位
        try:
            drfp = reactant_fp_sum ^ product_fp_sum
            return drfp
        except Exception as e:
            # 理论上 ExplicitBitVect 的异或操作不应失败，但以防万一
            self.logger.error(f"计算差异指纹时发生意外错误: {e}")
            return None

    # 保留旧的 get_fingerprint 作为 get_morgan_fingerprint 的别名，以兼容旧代码
    get_fingerprint = get_drfp_fingerprint
    
    def calculate_similarity(self, fp1: Any, fp2: Any) -> float:
        """
        计算两个指纹之间的Tanimoto相似度
        
        参数:
        fp1 (Any): 第一个分子的指纹
        fp2 (Any): 第二个分子的指纹
        
        返回:
        float: Tanimoto相似度得分 (0到1之间)
        """
        return DataStructs.TanimotoSimilarity(fp1, fp2)
    
    def calculate_tanimoto_similarity(self, smiles1: str, smiles2: str) -> Optional[float]:
        """
        计算两个SMILES字符串代表的分子之间的Tanimoto相似度
        
        参数:
        smiles1 (str): 第一个分子的SMILES字符串
        smiles2 (str): 第二个分子的SMILES字符串
        
        返回:
        float: Tanimoto相似度得分 (0到1之间)，如果任一SMILES无效则返回None
        """
        # 将SMILES字符串转换为RDKit分子对象
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        
        if not mol1:
            self.logger.warning(f"无法解析SMILES字符串: {smiles1}")
            return None
        if not mol2:
            self.logger.warning(f"无法解析SMILES字符串: {smiles2}")
            return None
        
        # 生成Morgan指纹
        fp1 = self.get_fingerprint(mol1)
        fp2 = self.get_fingerprint(mol2)
        
        # 计算Tanimoto相似度
        return self.calculate_similarity(fp1, fp2)
    
    def fingerprint_to_string(self, fp: Any) -> str:
        """
        将RDKit指纹转换为字符串表示，以便于序列化

        参数:
        fp (Any): RDKit分子指纹对象

        返回:
        str: 指纹的字符串表示
        """
        try:
            # 将指纹转换为NumPy数组
            fp_array = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, fp_array)
            
            # 将NumPy数组转换为字节并进行base64编码
            fp_bytes = fp_array.tobytes()
            fp_base64 = base64.b64encode(fp_bytes).decode('utf-8')
            
            return fp_base64
        except Exception as e:
            self.logger.error(f"将指纹转换为字符串时出错: {e}")
            return ""
    
    def string_to_fingerprint(self, fp_str: str) -> Optional[Any]:
        """
        将字符串表示转换回RDKit指纹对象

        参数:
        fp_str (str): 指纹的字符串表示

        返回:
        Any: RDKit分子指纹对象，如果转换失败则返回None
        """
        if not fp_str:
            return None
            
        try:
            # 从base64字符串解码为字节
            fp_bytes = base64.b64decode(fp_str)
            
            # 从字节重建NumPy数组
            fp_array = np.frombuffer(fp_bytes, dtype=np.float64)
            
            # 从NumPy数组创建RDKit指纹对象
            fp = DataStructs.CreateFromBitString(''.join('1' if b else '0' for b in fp_array > 0))
            
            return fp
        except Exception as e:
            self.logger.error(f"从字符串转换为指纹时出错: {e}")
            return None

    def get_reaction_fingerprint(self, reaction: Reaction, include_agents: bool = True) -> Optional[Any]:
        """
        为整个反应生成指纹
        
        参数:
        reaction (Reaction): 反应对象
        include_agents (bool): 是否在指纹计算中包含试剂，默认为True
        
        返回:
        Any: 反应的指纹，如果无法生成则返回None
        """
        if not reaction:
            self.logger.warning("无法为空反应生成指纹")
            return None
            
        # 收集所有有效分子的指纹
        reactant_fps = []
        for mol in reaction.reactants.mols:
            if mol:
                fp = self.get_fingerprint(mol)
                if fp:
                    reactant_fps.append(fp)
        
        product_fps = []
        for mol in reaction.products.mols:
            if mol:
                fp = self.get_fingerprint(mol)
                if fp:
                    product_fps.append(fp)
        
        agent_fps = []
        if include_agents:
            for mol in reaction.agents.mols:
                if mol:
                    fp = self.get_fingerprint(mol)
                    if fp:
                        agent_fps.append(fp)
        
        # 检查是否有足够的分子来生成反应指纹
        if not reactant_fps or not product_fps:
            self.logger.warning("反应缺少有效的反应物或产物，无法生成指纹")
            return None
        
        # 合并所有指纹
        # 这里使用一个简单的方法：将所有指纹按位进行"或"操作
        all_fps = reactant_fps + product_fps + agent_fps
        if not all_fps:
            return None
            
        # 创建一个新的空指纹
        reaction_fp = DataStructs.ExplicitBitVect(self.n_bits)
        
        # 合并所有指纹
        for fp in all_fps:
            reaction_fp |= fp  # RDKit ExplicitBitVect 支持直接使用 | 或 |= 进行按位或操作
            
        return reaction_fp
    
    def calculate_component_similarity(self, comp1: ReactionComponent, comp2: ReactionComponent) -> List[Tuple[int, int, float]]:
        """
        计算两个反应组分之间所有分子对的相似度
        
        参数:
        comp1 (ReactionComponent): 第一个反应组分
        comp2 (ReactionComponent): 第二个反应组分
        
        返回:
        List[Tuple[int, int, float]]: 包含(分子1索引, 分子2索引, 相似度)的列表
        """
        similarities = []
        
        for i, mol1 in enumerate(comp1.mols):
            if mol1 is None:
                continue
                
            fp1 = self.get_fingerprint(mol1)
            if fp1 is None:
                continue
                
            for j, mol2 in enumerate(comp2.mols):
                if mol2 is None:
                    continue
                    
                fp2 = self.get_fingerprint(mol2)
                if fp2 is None:
                    continue
                    
                similarity = self.calculate_similarity(fp1, fp2)
                similarities.append((i, j, similarity))
                
        return similarities
    
    def calculate_reaction_similarity(self, reaction1: Reaction, reaction2: Reaction) -> Dict[str, List[Tuple[int, int, float]]]:
        """
        计算两个反应之间的相似度
        
        参数:
        reaction1 (Reaction): 第一个反应
        reaction2 (Reaction): 第二个反应
        
        返回:
        Dict[str, List[Tuple[int, int, float]]]: 包含各组分相似度的字典
        """
        result = {}
        
        # 计算反应物之间的相似度
        result['reactants'] = self.calculate_component_similarity(reaction1.reactants, reaction2.reactants)
        
        # 计算试剂之间的相似度
        result['agents'] = self.calculate_component_similarity(reaction1.agents, reaction2.agents)
        
        # 计算产物之间的相似度
        result['products'] = self.calculate_component_similarity(reaction1.products, reaction2.products)
        
        return result
    
    def process_smiles_list(self, smiles_list: List[str]) -> Dict[Tuple[str, str], float]:
        """
        计算一个SMILES列表中所有有效分子对之间的Tanimoto相似度
        
        参数:
        smiles_list (List[str]): 包含分子SMILES字符串的列表
        
        返回:
        Dict[Tuple[str, str], float]: 一个字典，键是分子对的元组 (smiles1, smiles2)，值是它们的相似度
        """
        if not smiles_list:
            self.logger.info("SMILES列表为空")
            return {}
            
        valid_molecules_data = []
        
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fp = self.get_fingerprint(mol)
                valid_molecules_data.append((smi, fp))
            else:
                self.logger.warning(f"跳过无效的SMILES字符串: {smi}")
                
        if len(valid_molecules_data) < 2:
            self.logger.info("有效的分子数量不足2个，无法进行两两比较")
            return {}
            
        similarities = {}
        num_valid_molecules = len(valid_molecules_data)
        
        for i in range(num_valid_molecules):
            smi1, fp1 = valid_molecules_data[i]
            for j in range(i + 1, num_valid_molecules):
                smi2, fp2 = valid_molecules_data[j]
                
                similarity_score = self.calculate_similarity(fp1, fp2)
                similarities[(smi1, smi2)] = similarity_score
                
        return similarities


# --- 示例用法 ---
if __name__ == "__main__":
    # 初始化计算器
    calculator = MolecularSimilarityCalculator()
    
    print("--- 单对分子相似度计算示例 ---")
    # 示例SMILES字符串
    smiles_A = "CCO"  # 乙醇
    smiles_B = "CCC"  # 丙烷
    smiles_C = "c1ccccc1O"  # 苯酚
    invalid_smiles = "thisisnotasmiles"
    
    # 计算两个分子之间的相似度
    similarity_AB = calculator.calculate_tanimoto_similarity(smiles_A, smiles_B)
    if similarity_AB is not None:
        print(f"'{smiles_A}' 和 '{smiles_B}' 之间的Tanimoto相似度: {similarity_AB:.4f}")
        
    similarity_AC = calculator.calculate_tanimoto_similarity(smiles_A, smiles_C)
    if similarity_AC is not None:
        print(f"'{smiles_A}' 和 '{smiles_C}' 之间的Tanimoto相似度: {similarity_AC:.4f}")
        
    # 尝试使用无效SMILES
    print(f"尝试计算 '{smiles_A}' 和无效SMILES '{invalid_smiles}' 的相似度:")
    similarity_invalid = calculator.calculate_tanimoto_similarity(smiles_A, invalid_smiles)
    if similarity_invalid is None:
        print(f"计算失败，因为 '{invalid_smiles}' 是无效的SMILES。")
    
    print("\n" + "-" * 40)
    print("--- 分子列表相似度计算示例 ---")
    
    # 处理一个分子列表
    example_smiles_list = [
        "CCO",  # 乙醇
        "CCC",  # 丙烷
        "c1ccccc1O",  # 苯酚
        "CC(=O)O", # 乙酸
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", # 咖啡因
        invalid_smiles, # 加入一个无效SMILES进行测试
        "Oc1ccccc1C(=O)OH" # 水杨酸
    ]
    
    print(f"\n处理SMILES列表: {example_smiles_list}")
    all_pairs_similarities = calculator.process_smiles_list(example_smiles_list)
    
    if all_pairs_similarities:
        print("\n所有有效分子对的相似度:")
        for (mol1_smi, mol2_smi), score in all_pairs_similarities.items():
            print(f"  '{mol1_smi}' vs '{mol2_smi}': {score:.4f}")
    else:
        print("未能计算列表中的分子相似度 (可能有效分子不足或列表为空)。")
        
    print("\n" + "-" * 40)
    print("--- 从CSV文件读取数据并计算相似度示例 ---")
    # 这里需要替换为实际的CSV文件路径
    # csv_path = "path/to/your/data.csv"
    # result_df = calculator.process_csv_data(csv_path)
    # if not result_df.empty:
    #     print(f"计算得到 {len(result_df)} 对分子的相似度")
    #     print(result_df.head())
    # else:
    #     print("CSV处理失败或没有有效的分子对")
    
    print("\n文件执行完毕。")
