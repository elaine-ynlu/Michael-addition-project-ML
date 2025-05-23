# -*- coding: utf-8 -*-
"""
seq2seq模型基线实现，用于Michael加成产物预测
详细说明：
    - 输入与输出均为SMILES字符串序列，需先转换为索引序列后才能喂入模型
    - 使用字符级词表（vocab），将每个SMILES字符映射为整数索引
    - EncoderRNN使用GRU对源序列进行编码，生成隐藏状态向量
    - DecoderRNN使用GRU并结合Teacher Forcing机制，逐步生成目标序列
    - train_epoch：执行一个训练周期，包括前向计算、损失计算与反向传播
    - evaluate：在验证集上评估模型性能，不使用Teacher Forcing
    - main：解析命令行参数，加载数据集、构建模型并启动训练

使用示例：
    python baseline.py \
        --train_data data/michael/label_structures_unique.csv \
        --val_data data/michael/label_structures_unique.csv \
        --epochs 30 \
        --batch_size 128
"""
import torch # 导入PyTorch库
import torch.nn as nn # 导入PyTorch神经网络模块
import torch.optim as optim # 导入PyTorch优化器模块
from torch.utils.data import DataLoader # 从PyTorch导入数据加载器
import argparse # 导入参数解析模块
from tqdm import tqdm # 导入tqdm库，用于显示进度条

class EncoderRNN(nn.Module): # 定义编码器RNN类，继承自nn.Module
    """RNN编码器：将输入序列编码为隐藏表示"""
    # -----------------------------------------
    # EncoderRNN 编码器模块：将字符索引序列编码为隐藏状态
    # 输入参数:
    #   src: LongTensor [batch_size, seq_len] 输入序列索引
    #   src_lengths: List[int] 每个序列的实际长度，用于pack_padded_sequence
    # 返回值:
    #   outputs: Tensor [batch_size, seq_len, hidden_size] 每个时间步的GRU输出
    #   hidden: Tensor [num_layers, batch_size, hidden_size] 最后时刻的隐藏状态
    def __init__(self, input_size, embed_size, hidden_size, num_layers=1, dropout=0.1): # 初始化方法
        super(EncoderRNN, self).__init__() # 调用父类的初始化方法
        self.embedding = nn.Embedding(input_size, embed_size) # 定义词嵌入层
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, # 定义GRU层
                          batch_first=True, dropout=dropout if num_layers>1 else 0) # batch_first=True表示输入输出的第一个维度是batch_size，如果层数大于1则使用dropout

    def forward(self, src, src_lengths): # 定义前向传播方法
        # src: [batch, seq_len]
        embedded = self.embedding(src)  # [batch, seq_len, embed_size] # 将输入序列进行词嵌入
        # 使用pack_padded_sequence对变长序列做高效计算，enforce_sorted=False保证无需预先排序
        packed = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.cpu(), batch_first=True, enforce_sorted=False) # 对填充后的序列进行打包
        packed_outputs, hidden = self.gru(packed) # GRU层处理打包后的序列
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True) # 将打包后的输出序列解包并填充
        # outputs: [batch, seq_len, hidden_size]; hidden: [num_layers, batch, hidden_size]
        return outputs, hidden # 返回GRU的输出和最后一个隐藏状态

class DecoderRNN(nn.Module): # 定义解码器RNN类，继承自nn.Module
    """RNN解码器：基于上一时刻输入和隐藏状态生成当前时刻输出"""
    # -----------------------------------------
    # DecoderRNN 解码器模块：根据上一步输出和隐藏状态生成下一个token
    # 输入参数:
    #   input_step: LongTensor [batch_size] 上一步预测或真实的索引
    #   last_hidden: Tensor [num_layers, batch_size, hidden_size] 上一步的隐藏状态
    # 返回值:
    #   output: Tensor [batch_size, output_size] 当前时间步的输出分布
    #   hidden: Tensor [num_layers, batch_size, hidden_size] 更新后的隐藏状态
    def __init__(self, output_size, embed_size, hidden_size, num_layers=1, dropout=0.1): # 初始化方法
        super(DecoderRNN, self).__init__() # 调用父类的初始化方法
        self.embedding = nn.Embedding(output_size, embed_size) # 定义词嵌入层
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=num_layers, # 定义GRU层
                          batch_first=True, dropout=dropout if num_layers>1 else 0) # batch_first=True，如果层数大于1则使用dropout
        self.out = nn.Linear(hidden_size, output_size) # 定义线性输出层

    def forward(self, input_step, last_hidden): # 定义前向传播方法
        # input_step: [batch], last_hidden: [num_layers, batch, hidden_size]
        embedded = self.embedding(input_step).unsqueeze(1)  # [batch,1,embed_size] # 对输入进行词嵌入，并增加一个维度以匹配GRU输入格式
        output, hidden = self.gru(embedded, last_hidden) # GRU层处理嵌入后的输入和上一个隐藏状态
        output = output.squeeze(1)  # [batch, hidden_size] # 去除多余的维度
        output = self.out(output)   # [batch, output_size] # 通过线性层得到输出分布
        return output, hidden # 返回当前时间步的输出和更新后的隐藏状态

class Seq2Seq(nn.Module): # 定义Seq2Seq模型类，继承自nn.Module
    """组合编码器与解码器的序列到序列模型，支持附加特征"""
    # -----------------------------------------
    # Seq2Seq：组合编码器与解码器完成完整序列预测
    # 前向流程:
    #   1. Encoder处理源序列，得到初始隐藏状态
    #   2. 初始解码输入为<sos>标记
    #   3. 逐步调用Decoder，根据teacher_forcing_ratio决定使用真实标签还是模型预测
    #   4. 收集每个时间步的输出，生成完整序列预测
    def __init__(self, encoder, decoder, device, add_feat_size=0, encoder_hidden_size=0, num_encoder_layers=1):
        super(Seq2Seq, self).__init__() # 调用父类的初始化方法
        self.encoder = encoder # 编码器实例
        self.decoder = decoder # 解码器实例
        self.device = device # 运行设备 (CPU或GPU)
        self.add_feat_size = add_feat_size
        self.encoder_hidden_size = encoder_hidden_size # GRU的隐藏层大小
        self.num_encoder_layers = num_encoder_layers # 编码器GRU的层数

        if self.add_feat_size > 0:
            # 融合层：将每个编码器层的隐藏状态与附加特征融合
            # 输出维度应与解码器期望的隐藏状态维度一致（通常与编码器隐藏层维度相同）
            self.fusion_layers = nn.ModuleList([
                nn.Linear(self.encoder_hidden_size + self.add_feat_size, self.encoder_hidden_size)
                for _ in range(self.num_encoder_layers)
            ])
            # 或者，如果解码器的层数与编码器不同，或只想用最后一层融合，则设计会不同
            # 此处假设解码器也使用 num_encoder_layers 层，且每层隐藏维度为 encoder_hidden_size

    def forward(self, src, src_lengths, trg, teacher_forcing_ratio=0.5, add_feats=None): # 定义前向传播方法
        # src: [batch, src_len], trg: [batch, trg_len]
        batch_size = src.size(0) # 获取批次大小
        trg_len = trg.size(1) # 获取目标序列长度
        output_size = self.decoder.out.out_features  # 目标词表大小 # 获取解码器输出层的大小，即目标词表大小
        outputs = torch.zeros(batch_size, trg_len, output_size).to(self.device) # 初始化一个张量用于存储解码器每一步的输出
        # 编码器前向
        _, encoder_hidden_states = self.encoder(src, src_lengths) # [num_layers, batch, hidden_size]

        decoder_initial_hidden = encoder_hidden_states

        if add_feats is not None and self.add_feat_size > 0 and hasattr(self, 'fusion_layers'):
            # add_feats: [batch_size, add_feat_size]
            fused_hidden_list = []
            for i in range(self.num_encoder_layers):
                current_encoder_hidden_layer = encoder_hidden_states[i] # [batch, encoder_hidden_size]
                combined_input_for_fusion = torch.cat((current_encoder_hidden_layer, add_feats), dim=1)
                fused_layer_hidden = self.fusion_layers[i](combined_input_for_fusion) # [batch, encoder_hidden_size]
                fused_hidden_list.append(fused_layer_hidden)
            decoder_initial_hidden = torch.stack(fused_hidden_list, dim=0) # [num_layers, batch, hidden_size]
        
        current_decoder_hidden = decoder_initial_hidden
        input_step = trg[:, 0] # 解码器的第一个输入是目标序列的起始标记
        for t in range(1, trg_len): # 遍历目标序列的每一个时间步 (从第二个token开始)
            output, current_decoder_hidden = self.decoder(input_step, current_decoder_hidden) # 解码器根据当前输入和隐藏状态生成输出
            outputs[:, t] = output # 存储当前时间步的输出
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio # 根据teacher forcing比例决定是否使用真实标签
            top1 = output.argmax(1) # 获取当前时间步预测概率最高的token
            input_step = trg[:, t] if teacher_force else top1 # 如果使用teacher forcing，则下一个输入是真实标签；否则是模型预测的token
        return outputs # 返回所有时间步的输出


def train_epoch(model, dataloader, optimizer, criterion, device, teacher_forcing_ratio): # 定义单个训练周期的函数
    """训练一个epoch"""
    # -----------------------------------------
    # 执行一个epoch的训练流程
    # 1. 遍历dataloader获取批次数据(src, src_lengths, trg)
    # 2. 前向计算模型输出并计算loss
    # 3. 反向传播梯度并更新参数
    # 4. 累积并返回平均loss
    model.train() # 设置模型为训练模式
    epoch_loss_sum = 0.0
    total_weight_sum = 0.0

    for batch_data in tqdm(dataloader, desc="训练中"):
        src = batch_data['src_padded'].to(device)
        src_lengths = batch_data['src_lengths'].cpu() # pack_padded_sequence 需要在CPU上
        trg = batch_data['trg_padded'].to(device)
        sample_weights = batch_data['sample_weights'].to(device) # [batch_size]
        
        add_feats = None
        if 'add_feats' in batch_data:
            add_feats = batch_data['add_feats'].to(device)

        optimizer.zero_grad() # 清空优化器的梯度
        
        # 模型输出: [batch, trg_len, output_dim]
        output_logits = model(src, src_lengths, trg, teacher_forcing_ratio, add_feats=add_feats)
        
        output_dim = output_logits.shape[-1]
        # 目标和输出变形以计算损失 (忽略<sos>)
        # output_logits[:, 1:] -> [batch, trg_len-1, output_dim]
        # trg[:, 1:] -> [batch, trg_len-1]
        output_flat = output_logits[:, 1:].reshape(-1, output_dim) # [batch*(trg_len-1), output_dim]
        trg_y_flat = trg[:, 1:].reshape(-1) # [batch*(trg_len-1)]

        # criterion 的 reduction='none'
        loss_per_token = criterion(output_flat, trg_y_flat) # [batch*(trg_len-1)]
        
        # 将损失重塑回 [batch_size, trg_len-1]
        current_batch_size = src.size(0)
        # 确保 trg_len > 1，否则 trg_len-1 为0，会导致reshape错误
        seq_len_for_loss_calc = trg.size(1) - 1
        if seq_len_for_loss_calc <= 0: # 如果目标序列只有<sos>，则跳过损失计算
            continue
            
        loss_reshaped = loss_per_token.view(current_batch_size, seq_len_for_loss_calc)
        
        # 计算每个样本的平均损失 (只考虑非填充标记)
        # (trg[:, 1:] != pad_idx) -> [batch, seq_len-1], bool
        # criterion.ignore_index 应该是 dataloader.dataset.pad_idx
        # CrossEntropyLoss(reduction='none') 对于 ignore_index 的位置输出0
        # 因此，我们可以直接求和，然后除以实际目标token数量
        num_actual_tokens_per_sample = (trg[:, 1:] != criterion.ignore_index).sum(dim=1).float()
        # 防止除以零 (如果一个样本所有目标token都被忽略了)
        num_actual_tokens_per_sample = num_actual_tokens_per_sample.clamp(min=1.0)
        
        sum_loss_per_sample = loss_reshaped.sum(dim=1) # [batch_size]
        mean_loss_per_sample = sum_loss_per_sample / num_actual_tokens_per_sample # [batch_size]
        
        # 应用样本权重
        weighted_loss_per_sample = mean_loss_per_sample * sample_weights # [batch_size]
        
        # 当前批次的最终损失: 加权损失的平均值 (除以权重的和)
        # 或者，如果目标是最小化加权损失的总和，则直接 .sum()
        # 为了使不同批次/epoch的损失具有可比性，通常除以权重总和
        current_batch_loss = weighted_loss_per_sample.sum() / sample_weights.sum().clamp(min=1e-9)

        if not torch.isnan(current_batch_loss) and not torch.isinf(current_batch_loss):
            current_batch_loss.backward()
            optimizer.step()
            epoch_loss_sum += current_batch_loss.item() * sample_weights.sum().item() # 累加 (损失*权重和)
            total_weight_sum += sample_weights.sum().item() # 累加权重和
        else:
            print(f"警告: 训练中检测到 NaN/Inf 损失，跳过此批次更新。损失值: {current_batch_loss.item()}")

    if total_weight_sum == 0: return 0.0
    return epoch_loss_sum / total_weight_sum


def evaluate(model, dataloader, criterion, device): # 定义评估函数
    """在验证集上评估模型"""
    # -----------------------------------------
    # 在验证集上评估模型，不使用Teacher Forcing，保持模型参数不更新
    # 1. 设置eval模式并禁止梯度计算
    # 2. 遍历验证集计算loss并累积平均值
    model.eval() # 设置模型为评估模式
    epoch_loss_sum = 0.0
    total_weight_sum = 0.0

    with torch.no_grad(): # 在此代码块中不计算梯度
        for batch_data in tqdm(dataloader, desc="评估中"):
            src = batch_data['src_padded'].to(device)
            src_lengths = batch_data['src_lengths'].cpu()
            trg = batch_data['trg_padded'].to(device)
            sample_weights = batch_data['sample_weights'].to(device)

            add_feats = None
            if 'add_feats' in batch_data:
                add_feats = batch_data['add_feats'].to(device)

            output_logits = model(src, src_lengths, trg, teacher_forcing_ratio=0, add_feats=add_feats) # 评估时TF=0
            
            output_dim = output_logits.shape[-1]
            output_flat = output_logits[:, 1:].reshape(-1, output_dim)
            trg_y_flat = trg[:, 1:].reshape(-1)

            loss_per_token = criterion(output_flat, trg_y_flat)
            
            current_batch_size = src.size(0)
            seq_len_for_loss_calc = trg.size(1) - 1
            if seq_len_for_loss_calc <= 0:
                continue

            loss_reshaped = loss_per_token.view(current_batch_size, seq_len_for_loss_calc)
            num_actual_tokens_per_sample = (trg[:, 1:] != criterion.ignore_index).sum(dim=1).float().clamp(min=1.0)
            sum_loss_per_sample = loss_reshaped.sum(dim=1)
            mean_loss_per_sample = sum_loss_per_sample / num_actual_tokens_per_sample
            weighted_loss_per_sample = mean_loss_per_sample * sample_weights
            current_batch_loss = weighted_loss_per_sample.sum() / sample_weights.sum().clamp(min=1e-9)

            if not torch.isnan(current_batch_loss) and not torch.isinf(current_batch_loss):
                epoch_loss_sum += current_batch_loss.item() * sample_weights.sum().item()
                total_weight_sum += sample_weights.sum().item()
            # else: # 评估时可以不打印，或者只在verbose模式下打印
            # print(f"警告: 评估中检测到 NaN/Inf 损失。损失值: {current_batch_loss.item()}")

    if total_weight_sum == 0: return 0.0
    return epoch_loss_sum / total_weight_sum
