from localmapper import localmapper

# 初始化 mapper
mapper = localmapper()

# 多个反应输入
rxns = [
    'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F',
    'CCO>>CC(=O)O' # 示例：乙醇氧化为乙酸
]
results_list = mapper.get_atom_map(rxns)
print("\nMultiple mapped RXNs:")
for r in results_list:
    print(r)

# 返回包含更多信息的字典
results_dict = mapper.get_atom_map(rxns, return_dict=True)
print("\nResults as dictionary:")
import json
print(json.dumps(results_dict, indent=2))

# 输出示例:
# Single mapped RXN: [CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]

# Multiple mapped RXNs:
# [CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]
# CCOCC.[CH3:1][Mg+].[O:3]=[CH:2][c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11].[Br-]>>[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11]

# Results as dictionary:
# [
#   {
#     "rxn": "CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F",
#     "mapped_rxn": "[CH3:1][CH:2]([CH3:3])[SH:4].CN(C)C=O.[F:11][c:10]1[cH:9][cH:8][cH:7][n:6][c:5]1F.O=C([O-])[O-].[K+].[K+]>>[CH3:1][CH:2]([CH3:3])[S:4][c:5]1[n:6][cH:7][cH:8][cH:9][c:10]1[F:11]",
#     "template": "[S:1].F-[c:2]>>[S:1]-[c:2]",
#     "confident": true
#   },
#   {
#     "rxn": "CCOCC.C[Mg+].O=Cc1ccc(F)cc1Cl.[Br-]>>CC(O)c1ccc(F)cc1Cl",
#     "mapped_rxn": "CCOCC.[CH3:1][Mg+].[O:3]=[CH:2][c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11].[Br-]>>[CH3:1][CH:2]([OH:3])[c:4]1[cH:5][cH:6][c:7]([F:8])[cH:9][c:10]1[Cl:11]",
#     "template": "[C:1]-[Mg+].[C:2]=[O:3]>>[C:1]-[C:2]-[O:3]",
#     "confident": true
#   }
# ]
