# -*- coding: utf-8 -*-
"""
数据集分割脚本（仅保留指定字段）：
    - 读取指定的CSV数据集
    - 仅保留 reactants, products, rxn_mapped_SMILES, rxn_confidence, label 字段
    - 按照指定比例（默认为80/10/10）分割为训练集、验证集和测试集
    - 将分割后的数据集保存到指定目录（默认为 data/train/）

使用示例：
    python src/data/split_dataset.py --input_file data/michael/label_structures_unique.csv
"""
import csv
import os
import random
import argparse
from src.data.common.paths import project_paths
from sklearn.model_selection import train_test_split

# 只保留的字段
KEEP_FIELDS = ["reactants", "products", "rxn_mapped_SMILES", "rxn_confidence", "label"]

def split_dataset(input_file, output_dir="data/train", train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    """
    读取CSV数据集，仅保留指定字段，分割并保存到指定目录

    参数:
        input_file (str): 输入的CSV文件路径
        output_dir (str): 输出目录路径，默认为 "data/train"
        train_ratio (float): 训练集比例，默认为 0.8
        val_ratio (float): 验证集比例，默认为 0.1
        test_ratio (float): 测试集比例，默认为 0.1
        random_seed (int): 随机种子，用于复现分割结果，默认为42
    """
    if not (train_ratio + val_ratio + test_ratio == 1.0):
        raise ValueError("训练、验证和测试集的比例之和必须为1.0")

    # 读取数据，只保留需要的字段
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # 检查字段是否都存在
            missing = [field for field in KEEP_FIELDS if field not in reader.fieldnames]
            if missing:
                print(f"错误：输入文件缺少字段: {missing}")
                return
            data = []
            for row in reader:
                filtered_row = [row.get(field, "") for field in KEEP_FIELDS]
                data.append(filtered_row)
    except FileNotFoundError:
        print(f"错误：输入文件 {input_file} 未找到。")
        return
    except Exception as e:
        print(f"读取文件 {input_file} 时发生错误: {e}")
        return

    if not data:
        print(f"错误：文件 {input_file} 为空或格式不正确。")
        return

    # 设置随机种子以保证结果可复现
    random.seed(random_seed)
    random.shuffle(data)

    # 计算分割点
    train_data, temp_data = train_test_split(data, test_size=(val_ratio + test_ratio), random_state=random_seed)
    if (val_ratio + test_ratio) > 0:
        relative_test_ratio = test_ratio / (val_ratio + test_ratio)
        val_data, test_data = train_test_split(temp_data, test_size=relative_test_ratio, random_state=random_seed)
    else:
        val_data = []
        test_data = []

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 定义保存函数
    def save_to_csv(dataset, filename):
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(KEEP_FIELDS)  # 只写需要的表头
                writer.writerows(dataset)
            print(f"已保存 {len(dataset)} 条记录到 {filepath}")
        except Exception as e:
            print(f"保存文件 {filepath} 时发生错误: {e}")

    # 保存分割后的数据集
    if train_data:
        save_to_csv(train_data, "train.csv")
    if val_data:
        save_to_csv(val_data, "val.csv")
    if test_data:
        save_to_csv(test_data, "test.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分割CSV数据集为训练集、验证集和测试集，仅保留指定字段")
    parser.add_argument("--input_file", type=str, default=os.path.join(project_paths.data_michael, "labeled_reactions_with_similarity.csv"), required=False, help="输入的CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=project_paths.data_train, help="输出目录路径 (默认: data/train)")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="训练集比例 (默认: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="验证集比例 (默认: 0.1)")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="测试集比例 (默认: 0.1)")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子 (默认: 42)")

    args = parser.parse_args()

    split_dataset(
        args.input_file,
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_seed
    )
