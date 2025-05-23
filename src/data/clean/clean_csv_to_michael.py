import os
import pandas as pd
import glob
from src.data.common.paths import project_paths
import warnings
import re # 引入 re 用于更灵活的文件名处理

# 忽略 pandas 可能产生的警告
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def get_csv_files(directory_path, pattern='*.csv'):
    """
    获取指定目录下所有匹配模式的CSV文件路径。

    Args:
        directory_path (str): 要搜索的目录路径。
        pattern (str): 文件匹配模式 (例如 '*.csv', 'prefix_*.csv')。

    Returns:
        list: 包含找到的CSV文件完整路径的列表。
    """
    search_path = os.path.join(directory_path, pattern)
    csv_files = glob.glob(search_path)
    if not csv_files:
        print(f"信息：在目录 '{directory_path}' 中没有找到匹配 '{pattern}' 的CSV文件。")
    else:
        print(f"在目录 '{directory_path}' 中找到 {len(csv_files)} 个匹配 '{pattern}' 的CSV文件。")
    return csv_files

def get_source_from_filename(filepath):
    """
    从文件名中提取数据来源标识符。
    尝试识别 'pistachio', 'reacsys', 'scifinder' 等关键字。

    Args:
        filepath (str): CSV文件的完整路径。

    Returns:
        str: 数据来源标识符 (例如 'pistachio', 'reacsys', 'unknown')。
    """
    basename = os.path.basename(filepath).lower()
    # 移除常见后缀和扩展名
    name_part = re.sub(r'(_extracted|_export|_results|_part\d*)?\.csv$', '', basename)

    if 'pistachio' in name_part:
        return 'pistachio'
    elif 'reacxy' in name_part or 'reaxy' in name_part: # 包含常见拼写错误
        return 'reacxy' # 统一为 reacxy
    elif 'scifinder' in name_part:
        return 'scifinder'
    else:
        # 如果没有特定关键字，返回清理后的文件名作为来源（或标记为未知）
        # return name_part # 或者返回一个通用标签
        print(f"  未能从文件名 '{basename}' 识别特定来源，标记为 'unknown'。")
        return 'unknown'


def process_csv_file(csv_file):
    """
    根据CSV文件的列名，应用不同的处理策略，并返回包含 'SMILES' 和 'Source' 列的DataFrame。

    Args:
        csv_file (str): 单个CSV文件的路径。

    Returns:
        pd.DataFrame or None: 处理后的数据帧 (包含 'SMILES', 'Source' 列)，如果文件无法处理或不符合任何策略，则返回None。
    """
    print(f"正在处理文件: {csv_file}")
    source = get_source_from_filename(csv_file) # 获取来源名称

    try:
        # 添加 low_memory=False 尝试解决 DtypeWarning，并指定编码
        df = pd.read_csv(csv_file, low_memory=False, encoding='utf-8')

        output_df = None # 初始化输出 DataFrame

        # 必须存在 'Reaction_SMILES' 列且不能全为空
        if 'Reaction_SMILES' not in df.columns and 'SMILES' not in df.columns:
             print(f"警告：文件 {os.path.basename(csv_file)} 中缺少 'SMILES' 列或该列数据无效，跳过此文件。")
             return None

        # 策略1：如果存在 'ReactionName' 列，筛选 'Michael addition'
        if 'ReactionName' in df.columns:
            print(f"  检测到 'ReactionName' 列，应用策略 1: 筛选 Michael addition。")
            # 确保 ReactionName 和 Reaction_SMILES 都有效
            michael_mask = (df['ReactionName'] == 'Michael addition') & df['SMILES'].notna()
            if michael_mask.any():
                # 仅选择 Reaction_SMILES 列
                output_df = df.loc[michael_mask, ['SMILES']].copy()
                print(f"  在文件 {os.path.basename(csv_file)} 中找到 {len(output_df)} 条有效的 Michael addition 反应。")
            else:
                print(f"  在文件 {os.path.basename(csv_file)} 中没有找到有效的 Michael addition 反应 (ReactionName='Michael addition' 且 Reaction_SMILES 非空)。")
                # 不直接返回None，因为可能适用策略2（如果原始逻辑是这样）
                # 但根据当前代码结构（elif），如果ReactionName存在，就不会进入策略2
                # 因此，如果Michael没找到，这个文件就处理完了（没有结果）
                return None

        # 策略2：如果不存在 'ReactionName' 列，但存在 'Reaction_SMILES' 列，提取所有有效的 'Reaction_SMILES'
        elif 'Reaction_SMILES' in df.columns: # 'Reaction_SMILES' in df.columns 已在前面检查过，这里保持结构
            print(f"  未检测到 'ReactionName' 列，应用策略 2: 提取所有有效的 Reaction_SMILES。")
            # 选择所有 Reaction_SMILES 非空的行
            smiles_mask = df['Reaction_SMILES'].notna()
            if smiles_mask.any():
                output_df = df.loc[smiles_mask, ['Reaction_SMILES']].copy()
                print(f"  从文件 {os.path.basename(csv_file)} 中提取了 {len(output_df)} 条有效的 Reaction_SMILES。")
            else:
                 # 此情况理论上已被文件开头的检查覆盖
                 print(f"  文件 {os.path.basename(csv_file)} 中的 'Reaction_SMILES' 列为空或全为NA。")
                 return None

        # 如果两个关键列都不存在 (或策略1执行了但没结果)
        else:
             # 这个分支理论上不太可能进入，因为 Reaction_SMILES 的存在性已在最开始检查
             print(f"警告：文件 {os.path.basename(csv_file)} 中既没有 'ReactionName' 列，或不满足提取条件，跳过此文件。")
             return None

        # 如果成功提取了数据，进行格式化
        if output_df is not None and not output_df.empty:
            # 重命名列为 'SMILES'
            output_df.rename(columns={'Reaction_SMILES': 'SMILES'}, inplace=True)
            # 添加 'Source' 列
            output_df['Source'] = source
            # 确保列顺序为 ['SMILES', 'Source']
            return output_df[['SMILES', 'Source']]
        else:
            # 如果经过处理后 output_df 仍然是 None 或为空
            print(f"警告：文件 {os.path.basename(csv_file)} 处理后未产生有效数据，跳过。")
            return None

    except pd.errors.EmptyDataError:
        print(f"警告：文件 {csv_file} 为空，跳过。")
        return None
    except FileNotFoundError:
        print(f"错误：文件 {csv_file} 未找到，跳过。")
        return None
    except Exception as e:
        print(f"处理文件 {csv_file} 时发生未预料的错误: {str(e)}")
        return None


def merge_and_save_results(dataframes, output_file):
    """
    合并多个只包含 'SMILES' 和 'Source' 列的DataFrame，并保存到CSV文件。

    Args:
        dataframes (list): 包含pandas DataFrame的列表 (每个应有 'SMILES', 'Source' 列)。
        output_file (str): 输出CSV文件的路径。
    """
    # 过滤掉 None 值或空的 DataFrame
    valid_dataframes = [df for df in dataframes if df is not None and not df.empty]

    if not valid_dataframes:
        print("\n没有从任何文件中收集到有效数据，无法生成合并文件。")
        return

    print(f"\n准备合并来自 {len(valid_dataframes)} 个文件的数据...")
    try:
        # 合并所有有效的 DataFrame
        # 因为所有 DataFrame 都应有相同的列 ('SMILES', 'Source')，直接 concat 即可
        final_df = pd.concat(valid_dataframes, ignore_index=True, sort=False)

        # 可选：去重，基于 SMILES 列
        # final_df.drop_duplicates(subset=['SMILES'], inplace=True)
        # print(f"去重后剩余 {len(final_df)} 条记录。")

        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存到 CSV
        final_df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"处理完成！共合并 {len(final_df)} 条记录。")
        print(f"结果已保存到: {output_file}")
    except Exception as e:
        print(f"合并或保存结果时出错: {str(e)}")


def filter_reactions_main():
    """
    主函数，协调CSV文件的获取、处理和结果合并保存。
    """
    # 定义输入目录和输出文件
    input_directory = project_paths.data_unclean
    output_filename = 'filtered_reactions.csv'  # 输出文件名保持通用
    output_file = os.path.join(project_paths.data_filtered, output_filename)

    # 1. 获取CSV文件列表
    csv_files = get_csv_files(input_directory, pattern='*.csv') # 查找所有 .csv 文件

    if not csv_files:
        return # 如果没有文件，直接退出

    # 2. 处理每个CSV文件
    processed_data = []
    for csv_file in csv_files:
        result_df = process_csv_file(csv_file)
        # 只有当返回的 DataFrame 有效时才添加到列表
        if result_df is not None and not result_df.empty:
            processed_data.append(result_df)

    # 3. 合并并保存结果
    merge_and_save_results(processed_data, output_file)


def data_michael_filter():
    """
    运行数据筛选过程的入口点。
    """
    print("开始执行反应数据筛选流程...")
    filter_reactions_main()
    print("筛选流程执行完毕。")
