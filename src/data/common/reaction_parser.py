from .logger import ProcessLogger
from .metal_filter import MetalFilter
from rdkit import Chem
from rdkit.Chem import AllChem
from .smiles_utils import ReactionComponent, Reaction
from .canonicalization import canonicalizer


class ReactionParser:
    """反应SMILES解析器 (单例模式)"""
    # 单例实例
    _instance = None
    
    def __new__(cls):
        # 如果实例不存在，创建一个新实例
        if cls._instance is None:
            cls._instance = super(ReactionParser, cls).__new__(cls)
            # 初始化单例实例
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        # 确保初始化代码只执行一次
        if self._initialized:
            return
            
        self.logger = ProcessLogger('ReactionParser')
        self.metal_filter = MetalFilter()
        # 用于存储已处理过的规范化反应字符串，检测重复
        self.processed_reactions = dict()  # SMILES : idx
        # 用于统计重复反应数量
        self.duplicate_count = 0
        self._initialized = True

    def clean_reaction_smiles(self, reaction_smiles):
        """清理反应SMILES（例如，移除注释）"""
        if not isinstance(reaction_smiles, str):
            return None, f"输入不是字符串 (类型: {type(reaction_smiles)})"

        cleaned = reaction_smiles.strip()
        if not cleaned:
            return None, "输入的SMILES为空字符串"

        # 移除注释（例如，|f:2.4|）
        cleaned = cleaned.split('|')[0].strip()
        if not cleaned:
            return None, "清理'|'后SMILES变为空"

        return cleaned, None

    def split_reaction_smiles_robust(self, reaction_smiles):
        """分割反应SMILES并移除金属组分"""
        if not isinstance(reaction_smiles, str):
            return None, "输入不是字符串"

        reactants_raw, agents_raw, products_raw = None, None, None

        # 确定分隔符并分割
        if '>>' in reaction_smiles:
            parts = reaction_smiles.split('>>', 1)
            if len(parts) == 2 and parts[0].strip() and parts[1].strip():
                left_part = parts[0].strip()
                products_raw = parts[1].strip()
                if '>' in left_part:  # 处理 R>A>>P
                    sub_parts = left_part.rsplit('>', 1)
                    if len(sub_parts) == 2 and sub_parts[0].strip() and sub_parts[1].strip():
                        reactants_raw = sub_parts[0].strip()
                        agents_raw = sub_parts[1].strip()
                    else:
                        return None, f"格式不明确 ('>>' 前有不匹配或无效的 '>'): {reaction_smiles}"
                else:  # 标准 R>>P
                    reactants_raw = left_part
                    agents_raw = None
            else:
                return None, f"无效的 '>>' 分隔符用法: {reaction_smiles}"
        elif '>' in reaction_smiles:
            parts = [p.strip() for p in reaction_smiles.split('>')]
            parts = [p for p in parts if p]
            if len(parts) >= 2:
                reactants_raw = parts[0]
                products_raw = parts[-1]
                if len(parts) > 2:
                    agents_raw = '.'.join(parts[1:-1])
                else:
                    agents_raw = None
            else:
                return None, f"无效的 '>' 分隔符用法或分割后部分不足: {reaction_smiles}"
        else:
            return None, f"未找到反应分隔符 '>>' 或 '>': {reaction_smiles}"

        # 移除金属组分
        filtered_parts, reason = self.metal_filter.filter_reaction_components({
            'reactants': reactants_raw,
            'agents': agents_raw,
            'products': products_raw
        })

        if reason:
            return None, f"金属过滤失败: {reason}"

        return filtered_parts, None

    def standardize_and_check_duplicates(self, reaction: Reaction, idx: int):
        """根据规范化的SMILES字符串检查反应是否重复"""
        if not reaction or not isinstance(reaction, Reaction):
            return None, "无效的反应组分数据"

        if not reaction.is_valid():
            return None, "反应物或产物为空"

        try:
            # 标准化并排序反应物
            reactant_mols = reaction.reactants.mols
            if None in reactant_mols:
                return None, f"无法解析部分反应物: {'.'.join(reaction.reactants.smiles_list)}"
            reactant_canon_smiles = sorted([Chem.MolToSmiles(mol, isomericSmiles=True) for mol in reactant_mols])
            standardized_reactants = '.'.join(reactant_canon_smiles)

            # 标准化并排序产物
            product_mols = reaction.products.mols
            if None in product_mols:
                return None, f"无法解析部分产物: {'.'.join(reaction.products.smiles_list)}"
            product_canon_smiles = sorted([Chem.MolToSmiles(mol, isomericSmiles=True) for mol in product_mols])
            standardized_products = '.'.join(product_canon_smiles)

            # 创建规范化的反应字符串用于重复检测
            # 规则：反应物和产物必须完全相同才算重复
            canonical_reaction_key = f"{standardized_reactants}>>{standardized_products}"

            # 检查重复并记录索引
            is_duplicate = False
            duplicate_indices = []
            
            if canonical_reaction_key in self.processed_reactions:
                is_duplicate = True
                # 获取之前记录的索引列表
                duplicate_indices = self.processed_reactions[canonical_reaction_key]
                # 添加当前索引到列表，因为要输出csv的索引，所以此处+2以正确显示
                duplicate_indices.append(idx+2)
                self.duplicate_count += 1
                f_str = f"检测到重复反应，与之前索引 {[i+2 for i in duplicate_indices[:-1]]} 重复"
                # self.logger.warning(f_str)
                return {}, f_str
            else:
                # 首次出现，创建新的索引列表
                duplicate_indices = [idx+2]

            # 更新已处理集合
            self.processed_reactions[canonical_reaction_key] = duplicate_indices

            # 更新返回的字典
            standardized_parts = {
                'reactants': standardized_reactants,
                'products': standardized_products,
                'agents': '.'.join(reaction.agents.smiles_list) if reaction.agents else None,
                'is_duplicate': duplicate_indices if is_duplicate else False
            }

            return standardized_parts, None

        except Exception as e:
            return None, f"标准化或重复检测过程中出错: {str(e)}"

    def get_duplicate_count(self):
        """获取检测到的重复反应总数"""
        return self.duplicate_count
