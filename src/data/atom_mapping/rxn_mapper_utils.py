"""
反应原子映射工具模块

本模块提供了使用RXNMapper进行反应原子映射的功能。
RXNMapper是一个基于注意力机制的神经网络模型，用于预测化学反应中原子的映射关系。
"""

from rxnmapper import RXNMapper
from src.data.common.logger import ProcessLogger # 假设ProcessLogger在此路径
import traceback

class ReactionAtomMapper:
    def __init__(self):
        self.logger = ProcessLogger('ReactionAtomMapper')
        self.rxn_mapper = None
        try:
            self.rxn_mapper = RXNMapper()
            self.logger.info("RXNMapper初始化成功。")
        except Exception as e:
            self.logger.error(f"RXNMapper初始化失败: {e}。请确保正确安装了rxnmapper及其依赖项(如onnxruntime)。")
            self.logger.error(traceback.format_exc())
            # self.rxn_mapper保持为None

    def get_mapped_reaction_smiles(self, reaction_smiles: str) -> tuple[dict, str]:
        """
        映射反应SMILES字符串中的原子。

        参数:
            reaction_smiles: 反应SMILES字符串(例如, "CCO.Br>>CCBr.O")。
                             rxnmapper期望格式为"反应物>>产物"或"反应物>试剂>产物"。

        返回:
            包含以下内容的元组:
            - mapped_smiles (str | None): 原子映射后的反应SMILES，如果映射失败则为None。
            - error_reason (str | None): 如果映射失败，则为错误描述，否则为None。
        """
        if self.rxn_mapper is None:
            return None, "RXNMapper未初始化或初始化失败。"
        
        if not reaction_smiles or not isinstance(reaction_smiles, str):
            return None, "提供了无效或空的反应SMILES字符串。"

        try:
            # RXNMapper的get_attention_guided_atom_maps方法需要SMILES列表作为输入
            results = self.rxn_mapper.get_attention_guided_atom_maps([reaction_smiles])
            
            if results and isinstance(results, list) and len(results) > 0:
                first_result = results[0]
                if isinstance(first_result, dict) and 'mapped_rxn' in first_result:
                    # self.logger.debug(f"映射成功: {reaction_smiles} -> {mapped_smiles}") # 成功时的调试信息
                    return first_result, None
                else:
                    error_msg = f"原子映射失败: {reaction_smiles}。第一个结果中结构异常: {first_result}"
                    self.logger.warning(error_msg)
                    return None, error_msg
            else:
                error_msg = f"原子映射失败: {reaction_smiles}。结果列表为空或异常: {results}"
                self.logger.warning(error_msg)
                return None, error_msg
        except Exception as e:
            error_msg = f"为'{reaction_smiles}'进行原子映射时出错: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return None, str(e)

if __name__ == '__main__':
    # 此部分用于直接运行文件时的示例用法和测试。
    # 需要安装rxnmapper及其依赖项。
    print("尝试初始化ReactionAtomMapper进行测试...")
    mapper = ReactionAtomMapper()

    if mapper.rxn_mapper:
        print("RXNMapper已初始化，可以进行测试。")
        test_cases = [
            ("CCO.Cl>>CCC(=O)O.HCl", "基础反应"),
            ("CC(C)OC(C)C.BrBr>>CC(C)OC(C)CBr.HBr", "更复杂的反应"),
            ("c1ccccc1O.CC(=O)Cl>>c1ccccc1OC(=O)CH3.Cl", "含芳香环的反应"),
            ("", "空SMILES"), # 预期会优雅地失败
            ("InvalidSMILES>>Product", "无效SMILES"), # 预期会失败
            ("CCO>>", "仅有反应物"), 
            (">>CCO", "仅有产物")
        ]

        for smiles, description in test_cases:
            print(f"\n测试: {description} ('{smiles}')")
            mapped_smiles, error = mapper.get_mapped_reaction_smiles(smiles)
            if error:
                print(f"  错误: {error}")
            else:
                print(f"  原始: {smiles}")
                print(f"  映射后: {mapped_smiles}")
    else:
        print("RXNMapper无法初始化。跳过直接执行测试。")
        print("请确保已安装'rxnmapper'及其依赖项(如'onnxruntime')。") 