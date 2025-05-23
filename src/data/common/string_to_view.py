from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import Draw, AllChem # 导入 AllChem
import io
from PIL import Image
import os
import pandas as pd
from src.data.common.paths import project_paths
from src.data.common.logger import ProcessLogger
from src.data.common.canonicalization import SMILESCanonicalizer

# 可选：抑制 RDKit 的警告/错误信息，如果需要的话
RDLogger.DisableLog('rdApp.*')

def smiles_to_mol_image(smiles_string, size=(300, 300), kekulize=True, wedge_bonds=True, fit_image=True, options=None):
    """
    将 SMILES 字符串转换为分子图像。

    Args:
        smiles_string (str): 输入的 SMILES 字符串。
        size (tuple): 图像的尺寸 (宽度, 高度)。
        kekulize (bool): 是否对分子进行 Kekule 化。
        wedge_bonds (bool): 是否绘制楔形键以表示立体化学。
        fit_image (bool): 是否调整分子大小以适应图像边界。
        options (Draw.MolDrawOptions or None): 可选的绘图选项对象。

    Returns:
        PIL.Image.Image or None: 如果转换成功，返回 PIL 图像对象；
                                 如果输入无效或转换失败，返回 None。
    """
    if not isinstance(smiles_string, str) or not smiles_string.strip():
        print(f"错误：输入的 SMILES 无效或为空: '{smiles_string}'")
        return None

    mol = Chem.MolFromSmiles(smiles_string.strip())

    if mol is None:
        print(f"错误：无法使用 RDKit 解析 SMILES: '{smiles_string.strip()}'")
        # 尝试返回一个表示错误的图像或文本可能更好，但现在返回 None
        return None

    try:
        # 使用 MolToImage 直接生成 PIL Image 对象
        # 注意：MolToImage 内部通常会处理 2D 坐标生成，
        # 但如果需要显式控制或在绘制前操作坐标，则需要 Compute2DCoords
        AllChem.Compute2DCoords(mol) # 确保生成2D坐标
        img = Draw.MolToImage(mol,
                              size=size,
                              kekulize=kekulize,
                              wedgeBonds=wedge_bonds,
                              fitImage=fit_image,
                              options=options)
        return img
    except Exception as e:
        print(f"错误：从分子对象生成图像时出错 (SMILES: '{smiles_string.strip()}'): {e}")
        return None

def smiles_to_mol_image_bytes(smiles_string, size=(300, 300), format='PNG', **kwargs):
    """
    将 SMILES 字符串转换为分子图像的字节流。

    Args:
        smiles_string (str): 输入的 SMILES 字符串。
        size (tuple): 图像的尺寸 (宽度, 高度)。
        format (str): 输出图像格式 (例如 'PNG', 'JPEG')。
        **kwargs: 传递给 smiles_to_mol_image 的其他参数。

    Returns:
        bytes or None: 如果转换成功，返回图像的字节流；否则返回 None。
    """
    pil_image = smiles_to_mol_image(smiles_string, size=size, **kwargs)

    if pil_image is None:
        return None

    try:
        img_byte_arr = io.BytesIO()
        pil_image.save(img_byte_arr, format=format)
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"错误：将 PIL 图像保存为字节流时出错 (SMILES: '{smiles_string.strip()}', Format: {format}): {e}")
        return None


class MoleculeVisualizer:
    def __init__(self, output_dir=None):
        self.logger = ProcessLogger('MoleculeVisualizer')
        # 确保图片输出目录存在
        if output_dir:
            self.img_dir = os.path.join(project_paths.data_michael, output_dir)
        else:
            self.img_dir = os.path.join(project_paths.data_michael, 'img')
        
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)
            self.logger.info(f"创建目录: {self.img_dir}")
        
        # 初始化SMILES处理器
        self.smiles_processor = SMILESCanonicalizer()

    def draw_molecule(self, mol, output_path, size=(300, 300)):
        """绘制分子结构并保存"""
        if mol is None:
            self.logger.error("尝试绘制一个 None 分子对象")
            return False
        try:
            # 生成2D坐标
            AllChem.Compute2DCoords(mol)
            # 绘制分子
            img = Draw.MolToImage(mol, size=size)
            if img is None:
                self.logger.error(f"无法为分子生成图像，输出路径: {output_path}")
                return False
            # 保存图片
            img.save(output_path)
            return True
        except Exception as e:
            self.logger.error(f"绘制或保存分子图像到 '{output_path}' 时出错: {str(e)}")
            return False
            
    def draw_reaction_components(self, components, output_path, size=(800, 300)):
        """绘制反应物和产物在同一张图片中"""
        if components is None:
            self.logger.error("尝试绘制 None 反应组分")
            return False
            
        try:
            reactant_mols = [mol for mol in components['reactants'].mols if mol is not None]
            product_mols = [mol for mol in components['products'].mols if mol is not None]
            
            if not reactant_mols and not product_mols:
                self.logger.error("没有有效的反应物或产物分子可绘制")
                return False
                
            # 为所有分子生成2D坐标
            for mol in reactant_mols + product_mols:
                AllChem.Compute2DCoords(mol)
                
            # 准备标签
            reactant_labels = [f"反应物 {i+1}" for i in range(len(reactant_mols))]
            product_labels = [f"产物 {i+1}" for i in range(len(product_mols))]
            
            # 合并所有分子和标签
            all_mols = reactant_mols + product_mols
            all_labels = reactant_labels + product_labels
            
            # 绘制网格图像
            img = Draw.MolsToGridImage(all_mols, 
                                      molsPerRow=min(4, len(all_mols)),
                                      subImgSize=(200, 200),
                                      legends=all_labels,
                                      useSVG=False)
            
            # 保存图片
            img.save(output_path)
            return True
            
        except Exception as e:
            self.logger.error(f"绘制或保存反应组分图像到 '{output_path}' 时出错: {str(e)}")
            return False

    def process_reaction_smiles(self, reaction_smiles):
        """解析反应SMILES字符串，返回组分"""
        try:
            # 分割反应SMILES
            parts = reaction_smiles.split('>')
            
            if len(parts) < 2 or len(parts) > 3:
                self.logger.warning(f"无效的反应SMILES格式: {reaction_smiles}")
                return None
                
            # 提取反应物和产物
            reactants_smiles = parts[0]
            
            # 处理有试剂和没有试剂的情况
            if len(parts) == 3:
                agents_smiles = parts[1]
                products_smiles = parts[2]
            else:  # len(parts) == 2
                agents_smiles = ""
                products_smiles = parts[1]
                
            # 解析各组分
            class ComponentGroup:
                def __init__(self, smiles_str):
                    self.smiles_str = smiles_str
                    self.smiles_list = [s.strip() for s in smiles_str.split('.') if s.strip()]
                    self.mols = [Chem.MolFromSmiles(s) for s in self.smiles_list]
            
            reactants = ComponentGroup(reactants_smiles)
            agents = ComponentGroup(agents_smiles)
            products = ComponentGroup(products_smiles)
            
            return {
                'reactants': reactants,
                'agents': agents,
                'products': products,
                'full_smiles': reaction_smiles
            }
            
        except Exception as e:
            self.logger.error(f"解析反应SMILES时出错: {str(e)}")
            return None

    def process_single_reaction(self, reaction_smiles, reaction_index):
        """处理单个反应，绘制所有产物"""
        # 解析反应组分
        components = self.process_reaction_smiles(reaction_smiles)
        if components is None:
            self.logger.warning(f"无法解析反应SMILES: {reaction_smiles}")
            return False
        
        # 获取产物列表
        products = components['products']
        if not products.smiles_list:
            self.logger.warning(f"反应中没有产物: {reaction_smiles}")
            return False
        
        # 为每个产物生成图像
        success_count = 0
        for prod_idx, (smiles, mol) in enumerate(zip(products.smiles_list, products.mols)):
            if mol is None:
                self.logger.warning(f"无法解析产物SMILES: {smiles}")
                continue
            
            # 生成输出文件名
            output_filename = f"reaction_{reaction_index+2:04d}_product_{prod_idx:02d}.png"
            output_path = os.path.join(self.img_dir, output_filename)
            
            if self.draw_molecule(mol, output_path):
                success_count += 1
        
        # 绘制反应物和产物在同一张图片中
        reaction_output_filename = f"reaction_{reaction_index+2:04d}_full.png"
        reaction_output_path = os.path.join(self.img_dir, reaction_output_filename)
        self.draw_reaction_components(components, reaction_output_path)
        
        return success_count > 0

    def batch_process_reactions(self, csv_file):
        """批量处理CSV文件中的反应"""
        self.logger.info(f"开始处理文件: {csv_file}")

        if not os.path.exists(csv_file):
            self.logger.error(f"输入文件不存在: {csv_file}")
            return False

        try:
            # 读取CSV文件
            df = pd.read_csv(csv_file)

            if 'SMILES' not in df.columns:
                self.logger.error(f"CSV文件 '{csv_file}' 中没有找到 'SMILES' 列")
                return False

            total = len(df)
            if total == 0:
                self.logger.warning(f"CSV文件 '{csv_file}' 为空")
                return True

            success = 0
            skipped = 0

            for idx, row in df.iterrows():
                reaction_smiles = row['SMILES']

                # 检查SMILES是否有效
                if pd.isna(reaction_smiles) or not isinstance(reaction_smiles, str) or not reaction_smiles.strip():
                    self.logger.warning(f"跳过索引 {idx}: 无效或空的反应SMILES")
                    skipped += 1
                    continue

                if self.process_single_reaction(reaction_smiles, idx):
                    success += 1

                # 每处理N个反应输出一次进度
                log_interval = 100
                if (idx + 1) % log_interval == 0 or (idx + 1) == total:
                    self.logger.info(f"已处理 {idx + 1}/{total} 个反应 (成功: {success}, 跳过: {skipped})")

            self.logger.info(f"处理完成！成功转换 {success}/{total} 个反应 (跳过 {skipped} 个)")
            return True

        except Exception as e:
            self.logger.error(f"批量处理文件 '{csv_file}' 时发生错误: {str(e)}")
            return False

    def visualize_smiles_string(self, smiles_string, output_filename=None, size=(300, 300)):
        """
        将指定的SMILES字符串转换为可视化图片并保存到指定位置
        
        Args:
            smiles_string (str): 要可视化的SMILES字符串
            output_filename (str, optional): 输出文件名，如果不指定则使用默认名称
            size (tuple, optional): 图像尺寸，默认为(300, 300)
            
        Returns:
            bool: 成功返回True，失败返回False
        """
        self.logger.info(f"开始可视化SMILES字符串: {smiles_string}")
        
        # 检查是否为反应SMILES
        if '>' in smiles_string:
            # 处理反应SMILES
            components = self.process_reaction_smiles(smiles_string)
            if components is None:
                self.logger.error(f"无法解析反应SMILES: {smiles_string}")
                return False
                
            # 设置默认输出文件名
            if output_filename is None:
                output_filename = "reaction_visualization.png"
                
            output_path = os.path.join(self.img_dir, output_filename)
            return self.draw_reaction_components(components, output_path, size=(800, 300))
        else:
            # 处理分子SMILES
            mol = Chem.MolFromSmiles(smiles_string)
            if mol is None:
                self.logger.error(f"无法解析分子SMILES: {smiles_string}")
                return False
                
            # 设置默认输出文件名
            if output_filename is None:
                output_filename = "molecule_visualization.png"
                
            output_path = os.path.join(self.img_dir, output_filename)
            success = self.draw_molecule(mol, output_path, size=size)
            
            if success:
                self.logger.info(f"成功将SMILES可视化并保存到: {output_path}")
            
            return success


def batch_visualize_molecules():
    """批量可视化分子的主函数"""
    visualizer = MoleculeVisualizer()
    input_file = os.path.join(project_paths.data_michael, 'label_structures_unique.csv')
    visualizer.batch_process_reactions(input_file)


def visualize_single_smiles(smiles_string, output_dir=None, output_filename=None):
    """
    可视化单个SMILES字符串并保存到指定目录
    
    Args:
        smiles_string (str): 要可视化的SMILES字符串
        output_dir (str, optional): 输出目录，如果不指定则使用默认目录
        output_filename (str, optional): 输出文件名，如果不指定则使用默认名称
    
    Returns:
        bool: 成功返回True，失败返回False
    """
    visualizer = MoleculeVisualizer(output_dir=output_dir)
    return visualizer.visualize_smiles_string(smiles_string, output_filename=output_filename)


if __name__ == "__main__":
    # batch_visualize_molecules()

    smiles = "Cc1ccc(C(=O)O)cc1F.O=C1CCC(=O)N1Br>>O=C(O)c1ccc(CBr)c(F)c1"
    out = 'vis_img'
    o_name = 'uspto_63.png'

    visualize_single_smiles(smiles, out, o_name)
