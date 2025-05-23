import os
import pandas as pd
import csv
from src.data.common.paths import project_paths

def convert_cansmi_to_csv():
    # 输入和输出文件路径
    input_file = os.path.join(project_paths.origin_data, 'pistachio.cansmi')
    output_file = os.path.join(project_paths.data_origin, 'pistachio.csv')
    
    # 读取.cansmi文件
    print(f"正在读取文件: {input_file}")
    data = []
    max_cols = 0
    with open(input_file, 'r', encoding='utf8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts and parts[0]:
                data.append(parts)
                if len(parts) > max_cols:
                    max_cols = len(parts)
    
    # 设置列名
    columns = []
    if max_cols > 0:
        columns.append('SMILES')
    if max_cols > 1:
        columns.append('ID')
    if max_cols > 2:
        columns.append('Reference')
    if max_cols > 3:
        columns.append('Classification')
    if max_cols > 4:
        columns.append('ReactionName')
    for i in range(5, max_cols):
        columns.append(f'Extra_Data_{i-4}')
    
    # 填充数据
    padded_data = []
    for row in data:
        padded_row = row + [''] * (max_cols - len(row))
        padded_data.append(padded_row)
    
    # 创建DataFrame
    if padded_data:
        df = pd.DataFrame(padded_data, columns=columns)
    else:
        df = pd.DataFrame(columns=columns)
    
    # 保存为CSV文件
    print(f"正在保存到: {output_file}")
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_MINIMAL)
    print("转换完成！")


def data_clean():
    convert_cansmi_to_csv()

