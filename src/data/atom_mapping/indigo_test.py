from indigo import Indigo # 导入Indigo库

def get_atom_mapped_reaction_smiles(reaction_smiles):
    """
    使用Indigo进行反应SMILES的原子映射。

    参数:
    reaction_smiles (str): 反应SMILES字符串，例如 "reactant1.reactant2>>product1.product2"

    返回:
    str: 包含原子映射信息的反应SMILES字符串，如果出错则返回None。
    """
    try:
        indigo_instance = Indigo()  # 创建Indigo实例
        
        # 加载反应，Indigo会自动识别SMILES格式
        # 对于反应SMILES，通常用 loadReaction
        # 如果是反应SMARTS，则用 loadReactionSmarts
        reaction = indigo_instance.loadReaction(reaction_smiles)

        # 执行原子映射
        # mode参数可以是 "discard", "keep", "alter", "clear"
        # "discard": 丢弃现有映射，仅考虑现有反应中心（默认）
        # "keep": 保留现有映射并映射未映射的原子
        # "alter": 更改现有映射，并映射反应的其余部分，但可能会更改现有映射
        # "clear": 从反应中移除映射
        # 还可以添加 "ignore_charges", "ignore_isotopes", "ignore_valence", "ignore_radicals" 等选项
        reaction.automap("discard") 

        # 获取映射后的反应SMILES
        # canonicalSmiles() 通常会包含原子映射信息（例如 :1, :2 等）
        mapped_smiles = reaction.canonicalSmiles()
        
        return mapped_smiles
    except Exception as e:
        print(f"Indigo Atom Mapping Error: {e}")
        return None

# --- 示例用法 ---
if __name__ == "__main__":
    # 一个简单的反应SMILES
    # 溴乙烷 + 氢氧化钠 -> 乙醇 + 溴化钠 (简化示例，不考虑Na离子)
    # CCEr + [OH-] >> CCO + [Br-] (这是一个非常简化的例子，实际中OH-和Br-通常不直接这样表示在反应SMILES中，
    # 但为了演示原子映射，我们用一个更明确的有机反应)
    
    # 酯水解： 乙酸乙酯 + 水 -> 乙酸 + 乙醇
    # CCOC(=O)C.O>>CC(=O)O.CCO (这是一个更合适的例子)
    # reaction_smiles_unmapped = "CCOC(=O)C.O>>CC(=O)O.CCO"
    
    # Claisen缩合简化示例: 2 * CH3COOEt -> CH3COCH2COOEt + EtOH
    # CC(=O)OCC.CC(=O)OCC>>CC(=O)CC(=O)OCC.CCO 
    # (为了简化，我们用一个更直接的转化来展示原子映射)
    reaction_smiles_unmapped = "CC(=O)Cl.NCCO>>CC(=O)NCCO.Cl" # 酰氯与氨基醇反应

    print(f"原始反应SMILES: {reaction_smiles_unmapped}")
    
    mapped_reaction_smiles = get_atom_mapped_reaction_smiles(reaction_smiles_unmapped)
    
    if mapped_reaction_smiles:
        print(f"映射后反应SMILES: {mapped_reaction_smiles}")

    # 另一个例子 (来自Indigo文档中的反应)：
    # [I-].[Na+].C=CCBr>>[Na+].[Br-].C=CCI
    # 注意：Indigo在处理离子时，映射可能不总是直观，主要关注共价键形成和断裂的原子
    reaction_smiles_complex = "[CH3:1][NH2:2].[CH3:3][C:4](=[O:5])[Cl:6]>>[CH3:1][NH:2][C:4](=[O:5])[CH3:3].[Cl:6][H]"
    # 这个已经是映射过的，我们可以尝试清除映射再重新映射
    
    indigo_instance_for_clear = Indigo()
    rxn_to_clear = indigo_instance_for_clear.loadReaction(reaction_smiles_complex)
    rxn_to_clear.automap("clear") # 清除已有映射
    unmapped_for_remapping = rxn_to_clear.smiles() # 获取无映射的SMILES

    print(f"\n用于重新映射的SMILES (已清除原映射): {unmapped_for_remapping}")
    remapped_smiles = get_atom_mapped_reaction_smiles(unmapped_for_remapping)
    if remapped_smiles:
        print(f"重新映射后反应SMILES: {remapped_smiles}")