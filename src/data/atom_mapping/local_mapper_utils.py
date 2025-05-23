"""
本地原子映射工具模块

本模块提供了一个接口，用于使用假设的 'localmapper' 工具进行反应原子映射。
用户需要根据实际的 'localmapper' 库调整实现。
"""

from localmapper import localmapper
from src.data.common.logger import ProcessLogger
import traceback


class LocalReactionAtomMapper:
    def __init__(self):
        self.logger = ProcessLogger('LocalReactionAtomMapper')
        self.local_mapper = None
        device_to_use = 'cpu'
        try:
            import torch
            if torch.cuda.is_available():
                device_to_use = 'cuda'
                self.logger.info("CUDA is available. Attempting to initialize LocalMapper on GPU.")
            else:
                self.logger.info("CUDA not available. LocalMapper will use CPU.")
        except ImportError:
            self.logger.warning("PyTorch not found. Cannot determine CUDA availability. LocalMapper will use default device (likely CPU).")

        try:
            # 尝试传递 device 参数，如果localmapper支持
            # 你可能需要查阅 localmapper 的文档以确认正确的参数名和值
            self.local_mapper = localmapper(device=device_to_use)
            self.logger.info(f"LocalMapper initialized on device: {device_to_use} (assuming 'device' parameter is supported).")
        except TypeError: # 如果 localmapper 不接受 device 参数
            self.logger.warning(f"LocalMapper initialization does not accept a 'device' parameter. Attempting default initialization.")
            try:
                self.local_mapper = localmapper()
                self.logger.info("LocalMapper initialized with default settings.")
                # 如果默认初始化后，你想确认它是否真的用了GPU（如果它内部有自动检测机制）
                # 这需要localmapper库提供相应的方法或属性，例如 self.local_mapper.device
                # 此处仅为示例，实际属性名可能不同
                if hasattr(self.local_mapper, 'device'):
                     self.logger.info(f"LocalMapper is actually using device: {self.local_mapper.device}")
            except Exception as e:
                self.logger.error(f"LocalMapper default initialization failed: {e}")
                self.logger.error(traceback.format_exc())
        except Exception as e:
            self.logger.error(f"LocalMapper initialization failed (attempting device '{device_to_use}'): {e}")


    def get_mapped_reaction_smiles(self, reaction_smiles: str) -> tuple[dict, str]:
        """
        使用假设的 localmapper 映射反应SMILES字符串中的原子。

        参数:
            reaction_smiles: 反应SMILES字符串 (例如, "CCO.Br>>CCBr.O")。

        返回:
            包含以下内容的元组:
            - mapping_result (dict | None): 包含 'mapped_rxn' (str) 和 'confidence' (float) 的字典，
                                           如果映射失败则为None。
            - error_reason (str | None): 如果映射失败，则为错误描述，否则为None。
        """
        if self.local_mapper is None:
            self.logger.info("LocalMapper (占位符) 未实际初始化或不可用。将返回模拟失败或空结果。")
            return None, "LocalMapper (占位符) 未实现或初始化。"

        if not reaction_smiles or not isinstance(reaction_smiles, str):
            return None, "提供了无效或空的反应SMILES字符串。"

        try:
            results_list = self.local_mapper.get_atom_map(reaction_smiles, return_dict=True)

            return results_list, None
        except Exception as e:
            error_msg = f"LocalMapper为 '{reaction_smiles}' 进行原子映射时出错: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            return None, str(e)

if __name__ == '__main__':
    print("尝试初始化 LocalReactionAtomMapper 进行测试...")
    mapper = LocalReactionAtomMapper()

    print("LocalReactionAtomMapper (占位符) 已初始化，可以进行测试。")
    smiles = [
        'CC(C)S.CN(C)C=O.Fc1cccnc1F.O=C([O-])[O-].[K+].[K+]>>CC(C)Sc1ncccc1F',
        'CCO>>CC(=O)O' # 示例：乙醇氧化为乙酸
    ]

    for test_cases in smiles:
        print(f"\n测试 (localmapper 占位符): ('{test_cases}')")
        # 模拟调用
        if mapper.local_mapper is not None: # 如果用户实现了local_mapper的初始化
             mapped_output, error = mapper.get_mapped_reaction_smiles(test_cases)
        else: # 当前占位符逻辑，get_mapped_reaction_smiles 会处理 local_mapper 为 None 的情况
             mapped_output, error = mapper.get_mapped_reaction_smiles(test_cases)

        if error:
            print(f"  错误: {error}")
        elif mapped_output:
            print(f"  原始: {test_cases}")
            print(f"  映射后 (local): {mapped_output.get('mapped_rxn')}")
            print(f"  置信度 (local): {mapped_output.get('confidence')}")
        else:
            print(f"  映射未成功或未返回值。")