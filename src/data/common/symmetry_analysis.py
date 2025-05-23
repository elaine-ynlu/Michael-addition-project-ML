#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
分子对称性分析模块
"""

from rdkit import Chem
from rdkit.Chem import rdmolops
from src.data.common.smiles_utils import Reaction, ReactionComponent
from src.data.common.logger import ProcessLogger
from typing import List, Tuple, Optional, Dict
from collections import Counter

class SymmetryAnalyzer:
    """分析分子和反应中的对称性"""

    def __init__(self):
        self.logger = ProcessLogger('SymmetryAnalyzer')

    def get_atom_equivalence_classes(self, mol) -> Optional[List[Tuple[int, ...]]]:
        """
        获取分子中原子的对称等价类。
        每个元组代表一组对称等价的原子索引。

        参数:
        mol (Mol): RDKit分子对象。

        返回:
        Optional[List[Tuple[int, ...]]]: 等价类列表，如果分子无效则为None。
        """
        if not mol:
            return None
        try:
            # CanonicalRankAtoms 返回每个原子的规范秩
            # breakTies=False 使得对称等价的原子获得相同的秩
            
            # 尝试获取 CanonicalRankAtoms 函数。
            # 首先尝试标准位置 rdkit.Chem.rdmolops.CanonicalRankAtoms。
            # 如果未找到 (可能由于 RDKit 版本较旧或特定环境问题)，
            # 则尝试 rdkit.Chem.CanonicalRankAtoms，因为某些函数可能作为别名存在于 Chem 模块中。
            rank_atoms_func = None
            if hasattr(rdmolops, 'CanonicalRankAtoms'):
                rank_atoms_func = rdmolops.CanonicalRankAtoms
            elif hasattr(Chem, 'CanonicalRankAtoms'):
                self.logger.debug(
                    "rdmolops.CanonicalRankAtoms not found. Falling back to Chem.CanonicalRankAtoms."
                )
                rank_atoms_func = Chem.CanonicalRankAtoms
            
            if rank_atoms_func:
                ranks = list(rank_atoms_func(mol, breakTies=False))
            else:
                # 如果在两个常见位置都找不到该函数，则记录错误并返回 None。
                # 这通常表明 RDKit 版本过旧，不支持此功能，或者 RDKit 安装不完整。
                self.logger.error(
                    "Function 'CanonicalRankAtoms' not found in 'rdkit.Chem.rdmolops' or 'rdkit.Chem'. "
                    "This may be due to an outdated RDKit version or an incomplete installation. "
                    "Atom equivalence classes cannot be determined for this molecule."
                )
                return None
            
            # 按秩分组原子索引
            rank_to_atoms: Dict[int, List[int]] = {}
            for atom_idx, rank in enumerate(ranks):
                if rank not in rank_to_atoms:
                    rank_to_atoms[rank] = []
                rank_to_atoms[rank].append(atom_idx)
            
            # 提取等价类（只保留原子数大于1的组，因为单个原子自身不成对称关系）
            # 但为了完整表示所有原子都被分类了，我们返回所有组
            equivalence_classes = [tuple(sorted(atoms)) for atoms in rank_to_atoms.values()]
            return sorted(equivalence_classes) # 排序以保证输出一致性
        except Exception as e:
            self.logger.error(f"获取原子等价类时出错: {e}")
            return None

    def is_molecule_highly_symmetric(self, mol, min_atoms_in_symmetric_group: int = 2, min_fraction_symmetric_atoms: float = 0.5) -> bool:
        """
        判断一个分子是否高度对称。
        基于等价类中原子的数量和这些原子占分子总原子数的比例来判断。

        参数:
        mol (Mol): RDKit分子对象。
        min_atoms_in_symmetric_group (int): 一个等价类中至少包含的原子数才被认为是显著的对称组。
        min_fraction_symmetric_atoms (float): 分子中属于这类显著对称组的原子所占的最小比例。

        返回:
        bool: 如果分子被认为是高度对称的，则返回True，否则False。
        """
        if not mol:
            return False
        
        equivalence_classes = self.get_atom_equivalence_classes(mol)
        if not equivalence_classes:
            return False

        num_total_atoms = mol.GetNumAtoms()
        if num_total_atoms == 0:
            return False # 空分子无对称性

        num_symmetric_atoms = 0
        has_significant_symmetric_group = False

        for atom_group in equivalence_classes:
            if len(atom_group) >= min_atoms_in_symmetric_group:
                num_symmetric_atoms += len(atom_group)
                has_significant_symmetric_group = True
        
        if not has_significant_symmetric_group:
            return False # 没有显著的对称原子组
            
        fraction_symmetric = num_symmetric_atoms / num_total_atoms
        
        # 条件：至少存在一个对称组，并且对称原子占比达到阈值
        return fraction_symmetric >= min_fraction_symmetric_atoms

    def label_reaction_symmetry(self, reaction: Reaction) -> Dict[str, List[bool]]:
        """
        标记反应中反应物和产物分子是否高度对称。

        参数:
        reaction (Reaction): 反应对象。

        返回:
        Dict[str, List[bool]]: 一个字典，包含 'reactants_symmetry' 和 'products_symmetry' 键，
                                 对应的值是布尔列表，表示每个分子是否对称。
        """
        symmetry_labels = {
            'reactants_symmetry': [],
            'products_symmetry': []
        }

        if reaction.reactants:
            for mol in reaction.reactants.mols:
                if mol:
                    symmetry_labels['reactants_symmetry'].append(self.is_molecule_highly_symmetric(mol))
                else:
                    symmetry_labels['reactants_symmetry'].append(False) # 无效分子认为不对称
        
        if reaction.products:
            for mol in reaction.products.mols:
                if mol:
                    symmetry_labels['products_symmetry'].append(self.is_molecule_highly_symmetric(mol))
                else:
                    symmetry_labels['products_symmetry'].append(False) # 无效分子认为不对称
            
        return symmetry_labels

# 示例用法
if __name__ == '__main__':
    analyzer = SymmetryAnalyzer()

    # 对称分子示例: 苯 (C1=CC=CC=C1)
    benzene_smiles = "c1ccccc1"
    benzene_mol = Chem.MolFromSmiles(benzene_smiles)
    if benzene_mol:
        print(f"分子: {benzene_smiles}")
        eq_classes_benzene = analyzer.get_atom_equivalence_classes(benzene_mol)
        print(f"  原子等价类: {eq_classes_benzene}") # 应该所有6个碳原子在同一类
        print(f"  是否高度对称: {analyzer.is_molecule_highly_symmetric(benzene_mol)}")
        print(f"  (min_atoms_in_symmetric_group=2, min_fraction_symmetric_atoms=0.8): {analyzer.is_molecule_highly_symmetric(benzene_mol, 2, 0.8)}") # True

    # 对称分子示例: 甲烷 (C)
    methane_smiles = "C"
    methane_mol = Chem.MolFromSmiles(methane_smiles)
    if methane_mol:
        methane_mol = Chem.AddHs(methane_mol) # 需要加氢才能看到H的对称性
        print(f"\n分子: {Chem.MolToSmiles(methane_mol)} (甲烷)")
        eq_classes_methane = analyzer.get_atom_equivalence_classes(methane_mol)
        print(f"  原子等价类: {eq_classes_methane}") # C自己一类, 4个H一类
        print(f"  是否高度对称: {analyzer.is_molecule_highly_symmetric(methane_mol)}")
        print(f"  (min_atoms_in_symmetric_group=4, min_fraction_symmetric_atoms=0.7): {analyzer.is_molecule_highly_symmetric(methane_mol, 4, 0.7)}") # True

    # 不对称分子示例: 乙醇 (CCO)
    ethanol_smiles = "CCO"
    ethanol_mol = Chem.MolFromSmiles(ethanol_smiles)
    if ethanol_mol:
        print(f"\n分子: {ethanol_smiles}")
        eq_classes_ethanol = analyzer.get_atom_equivalence_classes(ethanol_mol)
        print(f"  原子等价类: {eq_classes_ethanol}") 
        print(f"  是否高度对称: {analyzer.is_molecule_highly_symmetric(ethanol_mol)}") # False

    # 手性分子示例 (通常不对称)
    chiral_smiles = "C[C@H](O)c1ccccc1"
    chiral_mol = Chem.MolFromSmiles(chiral_smiles)
    if chiral_mol:
        print(f"\n分子: {chiral_smiles}")
        eq_classes_chiral = analyzer.get_atom_equivalence_classes(chiral_mol)
        print(f"  原子等价类: {eq_classes_chiral}")
        print(f"  是否高度对称: {analyzer.is_molecule_highly_symmetric(chiral_mol)}") # False
        
    # 内消旋化合物（具有手性中心但整体对称）
    # 例如：(2R,3S)-丁烷-2,3-二醇  (meso-tartaric acid is complex, let's use a simpler meso compound if possible)
    # (2R,4S)-pentane-2,4-diol : CC(O)C(C)C(O)C  -- Simplified: OC(C)CC(C)O
    # Let's try meso-2,3-dibromobutane: BrC(C)C(C)Br (with specific stereo)
    # CC[C@H](Br)[C@@H](Br)CC (not meso)
    # For a true meso compound, e.g., (R,S)-cyclohexane-1,2-diol - more complex SMILES
    # Let's use a known simple symmetric molecule with potential for stereo: ClC(F)C(F)Cl
    # If it's (R,S) or (S,R) for the two F atoms, it could be meso.
    # F[C@H](Cl)[C@H](F)Cl (meso-1,2-dichloro-1,2-difluoroethane)
    meso_smiles = "F[C@H](Cl)[C@H](F)Cl"
    meso_mol = Chem.MolFromSmiles(meso_smiles)
    if meso_mol:
        print(f"\n分子: {meso_smiles} (内消旋)")
        eq_classes_meso = analyzer.get_atom_equivalence_classes(meso_mol)
        print(f"  原子等价类: {eq_classes_meso}") # Should show symmetry
        print(f"  是否高度对称: {analyzer.is_molecule_highly_symmetric(meso_mol)}") # Potentially True

    # 反应示例
    # 假设我们有一个反应 A + B -> C
    # A: c1ccccc1 (苯)
    # B: CC(=O)Cl (乙酰氯)
    # C: c1ccccc1C(=O)C (苯乙酮)
    print("\n--- 反应对称性分析 ---")
    r_comp = ReactionComponent(["c1ccccc1", "CC(=O)Cl"])
    p_comp = ReactionComponent(["c1ccccc1C(=O)C"])
    example_reaction = Reaction(reactants=r_comp, products=p_comp)
    
    reaction_symmetry_labels = analyzer.label_reaction_symmetry(example_reaction)
    print(f"反应物对称性: {reaction_symmetry_labels['reactants_symmetry']}") # [True, False]
    print(f"产物对称性: {reaction_symmetry_labels['products_symmetry']}")   # [False] 