# -*- coding: utf-8 -*-
"""
带日志的训练脚本：
    - 使用ProcessLogger记录训练步骤（设备、参数、每个Epoch的损失）
    - 使用MetricsLogger将Epoch级度量写入CSV，便于后续分析
"""
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from src.data.common.paths import project_paths

from src.model.baseline import EncoderRNN, DecoderRNN, Seq2Seq, train_epoch, evaluate
from src.model.dataset import MichaelDataset
from src.model.logger_metrics import MetricsLogger
from src.data.common.logger import ProcessLogger


def main():
    parser = argparse.ArgumentParser(description="带日志、附加特征和样本权重的Seq2Seq训练脚本")
    parser.add_argument("--epochs", type=int, default=20, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--embed_size", type=int, default=256, help="Embedding维度")
    parser.add_argument("--hidden_size", type=int, default=512, help="GRU隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=1, help="编码器和解码器GRU层数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--teacher_forcing_ratio", type=float, default=0.5, help="Teacher Forcing比例")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="运行设备")
    parser.add_argument("--train_data", type=str, required=True, help="训练数据CSV路径")
    parser.add_argument("--val_data", type=str, required=True, help="验证数据CSV路径")
    parser.add_argument("--model_type", type=str, choices=["baseline", "opennmt"], help="选择训练模型类型")
    parser.add_argument("--test_data", type=str, help="测试数据CSV路径")
    parser.add_argument("--random_seed", type=int, default=42, help="随机种子")
    args = parser.parse_args()

    # 初始化日志记录器
    proc_logger_name = 'training_seq2seq_with_features'
    metrics_logger_name = 'seq2seq_metrics_with_features'
    tensorboard_log_subdir = 'seq2seq_with_features'

    proc_logger = ProcessLogger(proc_logger_name)
    proc_logger.info(f"使用设备: {args.device}")
    proc_logger.info(f"参数: {args}")

    # 初始化度量日志
    metrics_logger = MetricsLogger(process_name=metrics_logger_name)
    # 初始化TensorBoard日志
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    # 确保 project_paths.log_tensorboard 存在
    # 如果 project_paths 没有定义 log_tensorboard，则使用默认路径
    base_tb_log_dir = getattr(project_paths, 'log_tensorboard', os.path.join(os.getcwd(), 'logs', 'tensorboard'))
    tb_dir = os.path.join(base_tb_log_dir, tensorboard_log_subdir, timestamp)
    os.makedirs(tb_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_dir)
    proc_logger.info(f"TensorBoard日志目录: {tb_dir}")

    # 准备数据
    device = torch.device(args.device)
    train_dataset = MichaelDataset(args.train_data)
    val_dataset = MichaelDataset(args.val_data)

    # 从数据集中获取附加特征维度和pad_idx
    add_feat_size = train_dataset.add_feat_size
    pad_idx = train_dataset.pad_idx
    proc_logger.info(f"词汇表大小 (源/目标): {train_dataset.src_vocab_size} / {train_dataset.trg_vocab_size}")
    proc_logger.info(f"附加特征维度: {add_feat_size}")
    proc_logger.info(f"Pad token索引: {pad_idx}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
    proc_logger.info(f"数据集加载完成: 训练={len(train_dataset)} 验证={len(val_dataset)}")

    # 构建模型及优化器、损失函数
    encoder = EncoderRNN(train_dataset.src_vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    decoder = DecoderRNN(train_dataset.trg_vocab_size, args.embed_size, args.hidden_size, args.num_layers).to(device)
    
    model = Seq2Seq(encoder, decoder, device, 
                    add_feat_size=add_feat_size, 
                    encoder_hidden_size=args.hidden_size, 
                    num_encoder_layers=args.num_layers # 传递编码器层数给Seq2Seq
                    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # 关键: 损失函数的 reduction='none' 用于手动加权
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, reduction='none').to(device)
    proc_logger.info("模型和优化器初始化完成。")

    # 训练循环
    best_val_loss = float('inf')
    model_save_base_dir = getattr(project_paths, 'output_model', os.path.join(os.getcwd(), 'saved_models'))
    model_save_dir = os.path.join(model_save_base_dir, tensorboard_log_subdir, timestamp)
    os.makedirs(model_save_dir, exist_ok=True)
    proc_logger.info(f"模型将保存到: {model_save_dir}")

    if args.model_type == "baseline":
        for epoch in range(1, args.epochs + 1):
            proc_logger.info(f"Epoch {epoch} 开始")
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, args.teacher_forcing_ratio)
            proc_logger.info(f"Epoch {epoch} 训练损失 (加权平均): {train_loss:.4f}")
            val_loss = evaluate(model, val_loader, criterion, device)
            proc_logger.info(f"Epoch {epoch} 验证损失 (加权平均): {val_loss:.4f}")

            # 记录度量
            metrics_logger.log_epoch(epoch, train_loss, val_loss, args.device)
            # 向TensorBoard写入损失曲线
            writer.add_scalar('Loss/train_weighted_avg', train_loss, epoch)
            writer.add_scalar('Loss/val_weighted_avg', val_loss, epoch)

            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(model_save_dir, f"best_model_epoch{epoch}.pth")
                torch.save(model.state_dict(), best_model_path)
                proc_logger.info(f"发现新的最佳验证损失，模型已保存到: {best_model_path}")

    elif args.model_type == "opennmt":
        proc_logger.info("启动 OpenNMT 模型训练流程...")
        current_run_output_dir = os.path.join(model_save_dir, "openNMT_baseline")
        # 为 OpenNMT 创建特定的参数对象或传递 args
        # OpenNMT 的日志和模型保存将由其内部逻辑或配置文件处理，但我们传递 output_dir
        opennmt_args = argparse.Namespace(
            train_file=args.train_data,
            val_file=args.val_data,
            test_file=args.test_data,
            output_dir=current_run_output_dir, # 确保使用 current_run_output_dir
            random_seed=args.random_seed,
            device=args.device, # 传递设备信息 (cuda/cpu)
            src_col='reactants', # 源SMILES列名
            trg_col='products',  # 目标SMILES列名
            char_level_tokenization=True, # 对SMILES默认为True
            # 可选: 从args传递更多OpenNMT特定超参数
            # onmt_train_steps=getattr(args, 'onmt_train_steps', 100000),
            # onmt_batch_size=getattr(args, 'onmt_batch_size', 64)
        )

        try:
            from src.model import opennmt_baseline # 确保导入
        except ImportError as e:
            proc_logger.error(f"无法导入 src.model.opennmt_baseline。错误: {e}。请确保文件存在且路径正确。")
            raise

        opennmt_baseline.train_opennmt(opennmt_args, proc_logger) # 传递proc_logger
        proc_logger.info("OpenNMT 模型训练流程已调用。具体日志请查看 OpenNMT 内部日志。")
    
    else:
        proc_logger.error("未知的模型类型。请选择 'baseline' 或 'opennmt'。")
        return

    # 关闭日志
    metrics_logger.close()
    writer.close()
    # 保存最终模型
    final_model_path = os.path.join(model_save_dir, f"final_model_epoch{args.epochs}.pth")
    torch.save(model.state_dict(), final_model_path)
    proc_logger.info(f"训练完成。最终模型已保存到: {final_model_path}")


if __name__ == "__main__":
    main()
