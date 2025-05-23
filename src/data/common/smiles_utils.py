from rdkit import Chem


class ReactionComponent:
    """反应组分数据结构，用于存储反应中的分子信息"""
    def __init__(self, smiles_list):
        # 确保输入是列表，并且移除了空字符串和None
        self.smiles_list = [s for s in smiles_list if s and isinstance(s, str)]
        # 尝试为每个有效的SMILES创建Mol对象，无效的则为None
        self.mols = []
        for s in self.smiles_list:
            try:
                mol = Chem.MolFromSmiles(s)
                self.mols.append(mol) # 保持与原来一致，允许None
            except Exception:
                self.mols.append(None) # 解析异常也添加None
    
    def __len__(self):
        """返回组分中分子的数量"""
        return len(self.smiles_list)
    
    def __bool__(self):
        """检查组分是否包含有效SMILES字符串"""
        return len(self.smiles_list) > 0
    
    def has_valid_mols(self):
        """检查是否所有SMILES都成功解析为Mol对象"""
        # 如果没有smiles，也认为mol是有效的（空的有效）
        if not self.smiles_list:
            return True
        # 检查mols列表是否包含None
        return None not in self.mols


class Reaction:
    """完整的反应数据结构，包含反应物、试剂和产物"""
    def __init__(self, reactants=None, agents=None, products=None):
        self.reactants = reactants if reactants else ReactionComponent([])
        self.agents = agents if agents else ReactionComponent([]) # 允许试剂为空
        self.products = products if products else ReactionComponent([])
    
    def is_valid(self):
        """检查反应是否有效（至少有反应物和产物SMILES）"""
        return bool(self.reactants) and bool(self.products)
    
    def is_valid_mols(self):
        """检查反应物和产物的所有SMILES是否都成功解析为Mol对象"""
        # 试剂是可选的，所以不检查它们
        return self.reactants.has_valid_mols() and self.products.has_valid_mols()

    def to_dict(self):
        """将反应转换为字典格式"""
        return {
            'reactants': self.reactants,
            'agents': self.agents,
            'products': self.products
        }

