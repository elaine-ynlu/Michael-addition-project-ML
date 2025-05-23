from rxnmapper import RXNMapper

# 初始化 mapper
rxn_mapper = RXNMapper()

# 定义反应 SMILES 列表 (注意格式：反应物.反应物>>产物)
rxns = [
    'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F',
    'CCO>>CC(=O)O' # 示例：乙醇氧化为乙酸
]

# 获取原子映射结果
results = rxn_mapper.get_attention_guided_atom_maps(rxns)

# 打印结果
for result in results:
    print(f"Mapped RXN: {result['mapped_rxn']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("-" * 20)

# 输出示例:
# Mapped RXN: CN(C)C=O.F[c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11].O=C([O-])[O-].[CH3:1][CH:2]([CH3:3])[SH:4].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]
# Confidence: 0.9566
# --------------------
# Mapped RXN: [CH3:1][CH2:2][OH:3]>>[CH3:1][C:2](=[O:4])[OH:3]
# Confidence: 0.9983
# --------------------

# 对于大量反应，可以使用 BatchedMapper 自动处理批处理和错误
# from rxnmapper import BatchedMapper
# batch_mapper = BatchedMapper(batch_size=32)
# list(batch_mapper.map_reactions(many_rxns)) # 直接返回映射后的 SMILES 字符串列表
# list(batch_mapper.map_reactions_with_info(many_rxns)) # 返回包含置信度的字典列表
