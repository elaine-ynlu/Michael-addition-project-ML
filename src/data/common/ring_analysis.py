from rdkit import Chem
from .logger import ProcessLogger
from rdkit.Chem import AllChem, MolStandardize
import pandas as pd
from .logger import ProcessLogger
import re # 导入 re 模块用于更灵活的分割
from src.data.common.logger import ProcessLogger
from .smiles_utils import Reaction, ReactionComponent


class RingAnalyzer:
    """环结构分析器"""
    def __init__(self):
        self.logger = ProcessLogger('RingAnalyzer')

    def analyze_product_rings(self, product_component):
        """分析产物的环信息"""
        if not isinstance(product_component, ReactionComponent) or not product_component:
            return None, "产物组分无效"

        # 使用第一个产物进行分析
        if not product_component.mols or product_component.mols[0] is None:
            # self.logger.warning(f"无法解析产物分子: {product_component.smiles_list[0] if product_component.smiles_list else 'N/A'}")
            return None, "无法解析产物分子"

        mol = product_component.mols[0]
        ring_info = mol.GetRingInfo()
        num_rings = ring_info.NumRings()

        if num_rings == 0:
            return {
                'has_ring': False,
                'num_rings': 0,
                'is_single_ring_molecule': False,
                'has_fused_rings': False,
                'has_spiro_rings': False
            }, None

        # 检查是否所有原子都在环中
        num_atoms = mol.GetNumAtoms()
        atoms_in_rings = set()
        for ring in ring_info.AtomRings():
            atoms_in_rings.update(ring)
        is_single_ring_system = (len(atoms_in_rings) == num_atoms) if num_atoms > 0 else False

        # 检查稠合环和螺环
        has_fused_rings = self.detect_fused_rings(mol)
        has_spiro_rings = self.detect_spiro_rings(mol)

        return {
            'has_ring': True,
            'num_rings': num_rings,
            'is_single_ring_molecule': is_single_ring_system,
            'has_fused_rings': has_fused_rings,
            'has_spiro_rings': has_spiro_rings
        }, None

    def detect_fused_rings(self, mol):
        """检测分子中是否存在稠合环（共享两个或更多原子的环）"""
        if mol is None:
            return False

        ring_info = mol.GetRingInfo()
        bond_rings = ring_info.BondRings()
        if len(bond_rings) < 2:
            return False

        # 检查是否有原子属于多个SSSR环
        for i in range(mol.GetNumAtoms()):
            if ring_info.NumAtomRings(i) > 1:
                atom = mol.GetAtomWithIdx(i)
                shared_bonds = 0
                for bond in atom.GetBonds():
                    if ring_info.NumBondRings(bond.GetIdx()) > 1:
                        shared_bonds += 1
                if shared_bonds > 0:
                    return True

        return False

    def detect_spiro_rings(self, mol):
        """检测分子中是否存在螺环（仅共享一个原子的环）"""
        if mol is None:
            return False

        ring_info = mol.GetRingInfo()
        if ring_info.NumRings() < 2:
            return False

        # 检查属于多个环但不在共享键中的原子
        for i in range(mol.GetNumAtoms()):
            if ring_info.NumAtomRings(i) > 1:
                atom = mol.GetAtomWithIdx(i)
                is_fusion_atom = False
                for bond in atom.GetBonds():
                    if ring_info.NumBondRings(bond.GetIdx()) > 1:
                        is_fusion_atom = True
                        break
                if not is_fusion_atom:
                    return True

        return False

    def analyze_reaction(self, reaction: Reaction, idx: int):
        """分析反应中的环结构"""
        # 检查 Reaction 对象有效性 (即分子是否成功解析)
        if not reaction or not reaction.reactants or not reaction.products:
            self.logger.warning(f"索引 {idx}: 传入的 Reaction 对象无效或缺少关键组分。")
            return None, "无效的Reaction对象"

        if not reaction.is_valid_mols():
             failed_smiles = []
             if reaction.reactants and any(m is None for m in reaction.reactants.mols):
                 failed_smiles.append(f"Reactants: {'.'.join(reaction.reactants.smiles_list)}")
             if reaction.products and any(m is None for m in reaction.products.mols):
                 failed_smiles.append(f"Products: {'.'.join(reaction.products.smiles_list)}")
             self.logger.warning(f"索引 {idx}: 传入的Reaction对象包含无效(未解析)的分子. {', '.join(failed_smiles)}")
             return None, f"Reaction对象包含无效分子: {', '.join(failed_smiles)}"

        # 1. 分析反应物环数
        total_reactant_rings = 0
        if reaction.reactants and reaction.reactants.mols:
            try:
                total_reactant_rings = sum(mol.GetRingInfo().NumRings() for mol in reaction.reactants.mols if mol)
            except Exception as e:
                self.logger.error(f"索引 {idx}: 计算反应物环数时出错 - {str(e)}")
                return None, f"计算反应物环数时出错: {str(e)}"
        
        # 2. 分析第一个产物 (analyze_product_rings 接收 ReactionComponent)
        product_ring_analysis_data, reason = self.analyze_product_rings(reaction.products)
        if reason:
            # 如果产物无效，记录更详细的信息
            product_smiles_str = '.'.join(reaction.products.smiles_list) if reaction.products else "N/A"
            self.logger.warning(f"索引 {idx}: 产物环分析失败 - {reason}. 产物 SMILES: {product_smiles_str}")
            return None, f"索引 {idx}: 产物环分析失败 - {reason}"

        # 3. 计算环闭合标志
        is_ring_closing = False # 默认值
        if reaction.products.mols and reaction.products.mols[0]: # 确保有产物且第一个产物Mol对象有效
            product_mol = reaction.products.mols[0]
            try:
                product_rings = product_mol.GetRingInfo().NumRings()
                is_ring_closing = total_reactant_rings < product_rings
            except Exception as e:
                 self.logger.error(f"索引 {idx}: 计算产物环数时出错 - {str(e)}")
                 # 可以选择返回错误或继续，这里选择继续但标记为False
                 is_ring_closing = False
        else:
            # 如果没有有效产物分子，无法计算环闭合
             log_msg_product_smiles = '.'.join(reaction.products.smiles_list) if reaction.products and reaction.products.smiles_list else "N/A"
             if not reaction.products:
                 self.logger.warning(f"索引 {idx}: 无法计算环闭合标志，因为没有产物信息。产物SMILES: {log_msg_product_smiles}")
             elif not reaction.products.mols:
                 self.logger.warning(f"索引 {idx}: 无法计算环闭合标志，因为产物列表为空或无法解析。产物SMILES: {log_msg_product_smiles}")
             elif not reaction.products.mols[0]:
                 self.logger.warning(f"索引 {idx}: 无法计算环闭合标志，第一个产物分子无效。产物SMILES: {log_msg_product_smiles}")
        
        # 标签的构建将移到 MainFilter。这里只返回分析数据。
        # 'SMILES', 'reactants', 'products' 将由 MainFilter 从规范化步骤和重复检查步骤中获取。
        return {
            'product_ring_info': product_ring_analysis_data, # 包含 has_ring, num_rings 等.
            'is_ring_closing': is_ring_closing,
            'total_reactant_rings': total_reactant_rings
        }, None
