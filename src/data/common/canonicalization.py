from rdkit import Chem
from rdkit.Chem import MolStandardize
from .logger import ProcessLogger
from collections import Counter
from typing import Optional


class SMILESCanonicalizer:
    """SMILES标准化处理器"""
    def __init__(self):
        self.logger = ProcessLogger('SMILESCanonicalizer')
        # 可选: 初始化 RDKit 标准化器
        # self.standardizer = MolStandardize.rdMolStandardize.Cleanup()
        # from .reaction_parser import ReactionParser # Removed
        # self.parser = ReactionParser() # Removed

    def clean_stereochemistry(self, smiles):
        """清理SMILES中的立体化学信息
        
        使用正则表达式移除SMILES中的立体化学标记
        
        Args:
            smiles (str): 输入的SMILES字符串
            
        Returns:
            str: 清理后的不含立体化学信息的SMILES字符串
        """
        if not smiles or not isinstance(smiles, str):
            return smiles
            
        import re
        # 移除@符号及其数字（手性中心标记）
        cleaned = re.sub(r'@+\d*', '', smiles)
        # 移除/和\（双键立体化学标记）
        cleaned = re.sub(r'[/\\]', '', cleaned)
        # 移除方括号中的H标记（可能表示手性氢）
        # cleaned = re.sub(r'\[([^]:]*)H([^]]*)\]', r'[\1\2]', cleaned)
        
        # self.logger.info(f"清理立体化学信息: '{smiles}' -> '{cleaned}'")
        return cleaned

    def canonicalize_single_smiles(self, smiles):
        """标准化单个SMILES字符串"""
        if not smiles or not isinstance(smiles, str):
            self.logger.warning(f"尝试标准化无效的SMILES: {smiles}")
            return None, "SMILES字符串为空或无效"

        try:
            # 检查是否有附加信息，如 |f:2.4|
            additional_info = None
            clean_smiles = smiles
            
            # 使用正则表达式查找附加信息
            import re
            match = re.search(r'(\|[^|]+\|)$', smiles)
            if match:
                additional_info = match.group(1)
                clean_smiles = smiles[:match.start()].strip()
            
            # 清理立体化学信息
            clean_smiles = self.clean_stereochemistry(clean_smiles)
            
            mol = Chem.MolFromSmiles(clean_smiles)
            if mol is None:
                self.logger.warning(f"无法使用RDKit解析SMILES: '{clean_smiles}'")
                return None, f"无法解析SMILES: '{clean_smiles}'"

            # 生成不包含立体化学信息的SMILES
            std_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            
            # 保存附加信息
            if additional_info:
                self.logger.info(f"从SMILES '{smiles}' 中提取到附加信息: {additional_info}")
            
            return std_smiles, None

        except Exception as e:
            self.logger.error(f"标准化SMILES '{smiles}' 时出错: {str(e)}")
            return None, f"标准化失败: {str(e)}"

    def canonicalize_component_smiles(self, component_smiles):
        """标准化组分SMILES字符串（如反应物、产物）"""
        if not component_smiles:
            return "", None  # 空输入是有效的（例如，无试剂）

        parts = [p.strip() for p in component_smiles.split('.') if p.strip()]
        canonical_parts = []
        for part in parts:
            cano_part, reason = self.canonicalize_single_smiles(part)
            if reason:
                return None, f"组分部分标准化失败: {reason}"
            canonical_parts.append(cano_part)

        # 排序以确保一致的顺序
        canonical_parts.sort()
        return '.'.join(canonical_parts), None

    def get_atom_counts_from_mol(self, mol):
        """
        计算单个RDKit Mol对象中每种元素类型的数量。
        包括隐式和显式氢原子。

        Args:
            mol (rdkit.Chem.Mol): RDKit分子对象。

        Returns:
            collections.Counter: 一个Counter，其中键是元素符号(str)，
                                值是它们的计数(int)。如果输入mol为None，
                                则返回None。
        """
        if mol is None:
            return None
        counts = Counter()
        for atom in mol.GetAtoms():
            # 添加重原子本身
            symbol = atom.GetSymbol()
            counts[symbol] += 1
            # 添加连接到该原子的氢原子(隐式+显式)
            num_hs = atom.GetTotalNumHs()
            if num_hs > 0:
                counts['H'] += num_hs
            # 可选：处理电荷（虽然不严格是原子平衡）
            # charge = atom.GetFormalCharge()
            # if charge != 0:
            #     counts['charge'] += charge # 使用特殊键
            # 可选：处理同位素
            # isotope = atom.GetIsotope()
            # if isotope != 0:
            #    symbol = f"{isotope}{symbol}" # 例如，"13C"
            #    counts[symbol] += 1
            # else:
            #    counts[symbol] += 1
        return counts

    def check_reaction_atom_balance_assumed_1_to_1(self, reactant_smiles, product_smiles):
        """
        检查反应是否原子平衡，假设所有提供的反应物SMILES与所有产物SMILES之间的
        化学计量比为1:1:...

        Args:
            reactant_smiles_list (list[str]): 反应物SMILES字符串列表。
            product_smiles_list (list[str]): 产物SMILES字符串列表。

        Returns:
            bool, str: 如果所有SMILES有效且反应物的总原子计数与产物的总原子计数匹配，
                      则返回True和None；否则返回False和错误原因。
        """
        reactant_smiles_list = reactant_smiles.split('.')
        product_smiles_list = product_smiles.split('.')

        if not reactant_smiles_list or not product_smiles_list:
            self.logger.warning("反应物或产物列表为空。无法检查平衡。")
            return False, "反应物或产物列表为空"

        total_reactant_counts = Counter()
        total_product_counts = Counter()

        # 处理反应物
        # self.logger.info(f"处理{len(reactant_smiles_list)}个反应物SMILES...")
        for i, smiles in enumerate(reactant_smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_msg = f"无法解析反应物SMILES #{i + 1}: '{smiles}'。反应无法平衡。"
                self.logger.error(error_msg)
                return False, error_msg
            reactant_counts = self.get_atom_counts_from_mol(mol)
            if reactant_counts is None:  # 如果mol不是None，这不应该发生，但进行防御性检查
                error_msg = f"无法获取反应物SMILES的原子计数: '{smiles}'。"
                self.logger.error(error_msg)
                return False, error_msg
            total_reactant_counts.update(reactant_counts)
            # self.logger.debug(f"  反应物 '{smiles}': {dict(reactant_counts)}")

        # 处理产物
        # self.logger.info(f"处理{len(product_smiles_list)}个产物SMILES...")
        for i, smiles in enumerate(product_smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                error_msg = f"无法解析产物SMILES #{i + 1}: '{smiles}'。反应无法平衡。"
                self.logger.error(error_msg)
                return False, error_msg
            product_counts = self.get_atom_counts_from_mol(mol)
            if product_counts is None:
                error_msg = f"无法获取产物SMILES的原子计数: '{smiles}'。"
                self.logger.error(error_msg)
                return False, error_msg
            total_product_counts.update(product_counts)
            # self.logger.debug(f"  产物 '{smiles}': {dict(product_counts)}")

        if total_reactant_counts == total_product_counts:
            # self.logger.info("原子计数匹配。在1:1:...假设下反应是平衡的。")
            return True, None
        else:
            # 找出差异以便更好地记录
            diff = Counter(total_reactant_counts)
            diff.subtract(total_product_counts)
            # mismatched_atoms = {k: v for k, v in diff.items() if v != 0}
            error_msg = f"原子计数不匹配。不匹配(反应物 - 产物): {dict(total_reactant_counts)} -- {dict(total_product_counts)} \n SMILES: {reactant_smiles} -- {product_smiles}"
            return False, error_msg


# 标准化
canonicalizer = SMILESCanonicalizer()
