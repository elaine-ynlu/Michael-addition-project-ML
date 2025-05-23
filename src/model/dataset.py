# -*- coding: utf-8 -*-
"""
MichaelDataset数据集类：
    - 读取CSV文件中的反应物和产物SMILES对
    - 构建字符级词表，将每个SMILES字符映射为整数索引
    - 在__getitem__中生成(src_indices, src_length, trg_indices)
    - collate_fn对批次数据进行padding，返回(src_padded, src_lengths, trg_padded)
"""
import csv
import torch
from torch.utils.data import Dataset
import re # Will be used if we need to clean SMILES, but not for atom mapping removal if 'reactants'/'products' are clean

class MichaelDataset(Dataset):
    """
    自定义数据集类，用于加载和预处理化学反应数据。
    主要功能:
    1. 从CSV文件读取反应物SMILES、产物SMILES。
    2. 读取反应的置信度（'rxn_confidence'），用作训练时的样本权重。
    3. 解析结构化的标签字符串（'label'），将其转换为数值型附加特征向量。
    4. 构建字符级的SMILES词汇表，并将SMILES字符串转换为整数索引序列。
    5. 提供 __getitem__ 方法以按索引获取单个处理好的样本数据。
    6. 提供 collate_fn 方法以将一批样本数据整理并填充为统一长度的张量，供DataLoader使用。
    """
    def __init__(self, csv_path):
        """
        数据集类的构造函数。
        Args:
            csv_path (str): 数据CSV文件的路径。
        """
        self.all_parsed_data = [] # 用于存储从CSV中解析出来的每一行原始数据的列表，每行是一个字典
        
        # --- 步骤 1: 从CSV文件读取和初步解析数据 ---
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 使用csv.DictReader按字典方式读取CSV，第一行为表头
            reader = csv.DictReader(f)
            for i, row in enumerate(reader): # 遍历CSV的每一行数据
                # 获取反应物和产物的SMILES字符串，去除首尾空格
                # 使用 .get(key, default_value) 来避免因列缺失导致的KeyError
                src_smiles = row.get('reactants', '').strip()
                trg_smiles = row.get('products', '').strip()
                
                # 如果反应物或产物SMILES为空，则打印警告并跳过该行
                if not src_smiles or not trg_smiles:
                    print(f"警告: 第 {i+1} 行缺少 'reactants' 或 'products' SMILES，已跳过。")
                    continue

                # 获取反应置信度字符串，默认为'1.0'，并去除首尾空格
                confidence_str = row.get('rxn_confidence', '1.0').strip()
                try:
                    # 尝试将置信度字符串转换为浮点数
                    confidence = float(confidence_str)
                except ValueError:
                    # 如果转换失败，打印警告并使用默认值1.0
                    print(f"警告: 第 {i+1} 行 'rxn_confidence' 值无效 ('{confidence_str}')，已使用默认值 1.0。")
                    confidence = 1.0
                
                # 获取标签字符串，默认为空字符串，并去除首尾空格
                label_str = row.get('label', '').strip()
                parsed_labels_dict = {} # 用于存储解析后的标签键值对
                if label_str: # 如果标签字符串不为空
                    parts = label_str.split('|') # 按'|'分割成多个标签部分
                    for part in parts:
                        if '=' not in part: # 如果部分不包含'='，则跳过 (格式错误)
                            continue
                        key, value = part.split('=', 1) # 按第一个'='分割成键和值
                        key = key.strip() # 去除键的首尾空格
                        value_lower = value.strip().lower() # 去除值的首尾空格并转为小写

                        # 将布尔值转换为1.0或0.0
                        if value_lower == 'true':
                            parsed_labels_dict[key] = 1.0
                        elif value_lower == 'false':
                            parsed_labels_dict[key] = 0.0
                        else:
                            # 尝试将其他值转换为浮点数
                            try:
                                parsed_labels_dict[key] = float(value.strip())
                            except ValueError:
                                # 如果转换失败，该标签值默认为0.0
                                print(f"警告: 第 {i+1} 行标签 '{key}' 的值 '{value}' 无法解析为数值，已设为0.0。")
                                parsed_labels_dict[key] = 0.0
                
                # 将解析后的数据存入列表
                self.all_parsed_data.append({
                    'src_smiles': src_smiles,        # 原始反应物SMILES
                    'trg_smiles': trg_smiles,        # 原始产物SMILES
                    'confidence': confidence,        # 反应置信度 (浮点数)
                    'label_dict': parsed_labels_dict # 解析后的标签字典
                })

        if not self.all_parsed_data:
            raise ValueError(f"未能从 {csv_path} 加载任何有效数据。请检查文件格式和内容。")

        # --- 步骤 2: 构建SMILES字符词汇表 ---
        self._build_vocab()

        # --- 步骤 3: 确定并排序所有出现过的标签键，用于统一附加特征向量的维度 ---
        all_label_keys_set = set() # 使用集合来存储所有唯一的标签键
        for item in self.all_parsed_data:
            all_label_keys_set.update(item['label_dict'].keys()) # 将每个样本的标签键加入集合
        # 对所有唯一的标签键进行排序，以确保附加特征向量中特征的顺序一致性
        self.ordered_label_keys = sorted(list(all_label_keys_set))
        # 附加特征向量的维度即为唯一标签键的数量
        self.add_feat_size = len(self.ordered_label_keys)

        # --- 步骤 4: 将所有解析后的数据转换为模型可用的张量格式 ---
        self.processed_data = [] # 存储最终处理好的样本数据，每个样本是一个包含张量的字典
        for item_data in self.all_parsed_data:
            # 将源SMILES字符串转换为整数索引序列，未知字符映射为'<unk>'的索引
            src_idx = [self.char2idx.get(c, self.unk_idx) for c in item_data['src_smiles']]
            # 将目标SMILES字符串转换为整数索引序列，并添加<sos>和<eos>标记
            # 未知字符同样映射为'<unk>'的索引
            trg_idx = [self.sos_idx] + [self.char2idx.get(c, self.unk_idx) for c in item_data['trg_smiles']] + [self.eos_idx]
            
            # 根据排序后的标签键，为当前样本构建附加特征向量
            # 如果某个键在当前样本的标签字典中不存在，则其对应特征值为0.0
            add_feats_vector = [item_data['label_dict'].get(key, 0.0) for key in self.ordered_label_keys]
            # 注意: 数值型的附加特征（如NumRings）可能需要在此处或模型外部进行归一化处理，
            # 以确保它们的值范围与其他特征（如词嵌入）不会相差过大，有助于模型训练。
            
            # 将处理好的数据存入列表
            self.processed_data.append({
                'src_idx': torch.tensor(src_idx, dtype=torch.long),         # 源SMILES索引序列张量
                'src_len': torch.tensor(len(src_idx), dtype=torch.long),  # 源SMILES原始长度的标量张量
                'trg_idx': torch.tensor(trg_idx, dtype=torch.long),         # 目标SMILES索引序列张量
                'sample_weight': torch.tensor(item_data['confidence'], dtype=torch.float), # 样本权重张量
                # 附加特征向量张量；如果add_feat_size为0，则为一个空张量
                'add_feats': torch.tensor(add_feats_vector, dtype=torch.float) if self.add_feat_size > 0 else torch.empty(0, dtype=torch.float)
            })

    def _build_vocab(self):
        """
        构建SMILES字符词汇表。
        遍历所有样本中的反应物和产物SMILES，收集所有唯一字符。
        添加特殊标记：<pad>(填充), <sos>(序列开始), <eos>(序列结束), <unk>(未知字符)。
        创建字符到索引 (char2idx) 和索引到字符 (idx2char) 的映射。
        """
        chars = set() # 用于存储所有唯一字符的集合
        # 定义特殊词汇标记
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        self.unk_token = '<unk>' # 未知字符标记

        # 从所有解析后的数据中提取SMILES字符串，收集字符
        for item in self.all_parsed_data:
            chars.update(list(item['src_smiles'])) # 添加源SMILES中的字符
            chars.update(list(item['trg_smiles'])) # 添加目标SMILES中的字符
        
        # 构建词汇表列表：特殊标记在前，后接排序后的SMILES字符
        vocab_list = [self.pad_token, self.sos_token, self.eos_token, self.unk_token] + sorted(list(chars))
        # 创建字符到索引的映射字典
        self.char2idx = {c: i for i, c in enumerate(vocab_list)}
        # 创建索引到字符的映射字典
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        
        # 存储特殊标记的索引，方便后续使用
        self.pad_idx = self.char2idx[self.pad_token]
        self.sos_idx = self.char2idx[self.sos_token]
        self.eos_idx = self.char2idx[self.eos_token]
        self.unk_idx = self.char2idx[self.unk_token] # 未知字符的索引
        
        # 词汇表大小，用于模型Embedding层
        self.src_vocab_size = len(vocab_list)
        self.trg_vocab_size = len(vocab_list) # 源和目标共享同一个词汇表

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.processed_data)

    def __getitem__(self, idx):
        """
        根据索引获取单个处理好的样本数据。
        Args:
            idx (int): 样本的索引。
        Returns:
            dict: 包含样本数据的字典，值为PyTorch张量。
                  例如: {'src_idx': tensor(...), 'src_len': tensor(...), ...}
        """
        return self.processed_data[idx]

    def collate_fn(self, batch_list_of_dicts):
        """
        自定义的批处理函数，用于DataLoader。
        将一批从__getitem__获取的样本字典列表，整理成一个包含批处理后张量的字典。
        主要功能包括：
        1. 将同一字段的数据从各个样本字典中提取出来并组合。
        2. 对变长的SMILES序列（src_idx, trg_idx）进行填充(padding)，使其在批次内长度一致。
        3. 将其他字段（如sample_weight, add_feats）堆叠(stack)成批处理张量。
        Args:
            batch_list_of_dicts (list[dict]): 一个列表，其中每个元素是__getitem__返回的样本字典。
        Returns:
            dict: 包含批处理后数据的字典，值为PyTorch张量。
                  例如: {'src_padded': tensor(...), 'src_lengths': tensor(...), ...}
        """
        # 从字典列表中提取各个字段的数据
        src_seqs = [item['src_idx'] for item in batch_list_of_dicts]
        # src_len已经是标量张量，直接取其item()得到Python标量，再转换回张量
        src_lens_list = [item['src_len'].item() for item in batch_list_of_dicts]
        trg_seqs = [item['trg_idx'] for item in batch_list_of_dicts]
        # 使用torch.stack将单个样本的权重张量堆叠成一个批次张量
        sample_weights = torch.stack([item['sample_weight'] for item in batch_list_of_dicts])
        
        # 检查批次中是否存在有效的附加特征
        has_add_feats = self.add_feat_size > 0 and \
                        all(item['add_feats'].numel() > 0 for item in batch_list_of_dicts if 'add_feats' in item)
        add_feats_collated = None
        if has_add_feats:
            # 如果存在附加特征，则将它们堆叠成一个批次张量
            add_feats_collated = torch.stack([item['add_feats'] for item in batch_list_of_dicts])

        # 对源SMILES序列和目标SMILES序列进行填充
        # 计算批次内源序列和目标序列的最大长度
        max_src = max(len(s) for s in src_seqs) if src_seqs else 0
        max_trg = max(len(t) for t in trg_seqs) if trg_seqs else 0
        batch_size = len(src_seqs)

        # 创建用于存储填充后序列的张量，初始值用pad_idx填充
        src_padded = torch.full((batch_size, max_src), self.pad_idx, dtype=torch.long)
        trg_padded = torch.full((batch_size, max_trg), self.pad_idx, dtype=torch.long)
        
        # 将每个序列填充到最大长度
        for i, seq in enumerate(src_seqs):
            src_padded[i, :len(seq)] = seq # 将原始序列数据复制到填充张量的前部
        for i, seq in enumerate(trg_seqs):
            trg_padded[i, :len(seq)] = seq
        
        # 将源序列的原始长度列表转换为张量
        src_lengths_tensor = torch.tensor(src_lens_list, dtype=torch.long)

        # 构建并返回包含所有批处理后数据的字典
        collated_batch = {
            'src_padded': src_padded,             # 填充后的源SMILES索引序列批次
            'src_lengths': src_lengths_tensor,    # 源SMILES原始长度批次
            'trg_padded': trg_padded,             # 填充后的目标SMILES索引序列批次
            'sample_weights': sample_weights      # 样本权重批次
        }
        if add_feats_collated is not None:
            # 如果存在附加特征，则将其加入到返回的字典中
            collated_batch['add_feats'] = add_feats_collated
            
        return collated_batch
