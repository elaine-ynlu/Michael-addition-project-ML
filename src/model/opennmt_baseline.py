# -*- coding: utf-8 -*-
import os
import pandas as pd
import subprocess
import yaml
import torch # For torch.cuda.is_available()

# proc_logger 将从 train.py 传入

def prepare_data_for_opennmt(csv_path, src_col, trg_col, output_dir, prefix, proc_logger):
    """
    从CSV文件准备OpenNMT所需的文本数据文件。
    将指定的源列和目标列分别写入到 prefix.src 和 prefix.tgt 文件中。
    """
    proc_logger.info(f"为OpenNMT准备数据: {prefix}，来源: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        proc_logger.error(f"数据文件未找到: {csv_path}")
        raise
    except Exception as e:
        proc_logger.error(f"读取CSV文件失败 {csv_path}: {e}")
        raise

    if src_col not in df.columns:
        proc_logger.error(f"源数据列 '{src_col}' 在文件 {csv_path} 中未找到。可用列: {df.columns.tolist()}")
        raise ValueError(f"源数据列 '{src_col}' 在文件 {csv_path} 中未找到。")
    if trg_col not in df.columns:
        proc_logger.error(f"目标数据列 '{trg_col}' 在文件 {csv_path} 中未找到。可用列: {df.columns.tolist()}")
        raise ValueError(f"目标数据列 '{trg_col}' 在文件 {csv_path} 中未找到。")

    src_sents = df[src_col].astype(str).tolist() # 确保转换为字符串
    trg_sents = df[trg_col].astype(str).tolist() # 确保转换为字符串

    os.makedirs(output_dir, exist_ok=True)
    src_file_path = os.path.join(output_dir, f"{prefix}.src")
    trg_file_path = os.path.join(output_dir, f"{prefix}.tgt")

    try:
        with open(src_file_path, 'w', encoding='utf-8') as f_src:
            for line in src_sents:
                f_src.write(line + '\n')
        proc_logger.info(f"源数据已写入: {src_file_path} ({len(src_sents)}行)")

        with open(trg_file_path, 'w', encoding='utf-8') as f_trg:
            for line in trg_sents:
                f_trg.write(line + '\n')
        proc_logger.info(f"目标数据已写入: {trg_file_path} ({len(trg_sents)}行)")
    except IOError as e:
        proc_logger.error(f"写入数据文件失败: {e}")
        raise
    return src_file_path, trg_file_path


def generate_opennmt_config(config_params, config_save_path, proc_logger):
    """
    生成OpenNMT的YAML配置文件。
    config_params 应包含所有必要的路径和超参数。
    """
    proc_logger.info("生成OpenNMT YAML配置文件...")

    # 定义一些合理的默认值，特别是对于SMILES任务
    default_config = {
        # Data
        'data': {
            'corpus_1': { # 用于词表和训练
                'path_src': config_params['path_src_train'],
                'path_tgt': config_params['path_tgt_train'],
                'transforms': ['filtertoolong'], # 过滤过长序列
                'weight': 1
            },
            'valid': { # 验证数据
                'path_src': config_params['path_src_valid'],
                'path_tgt': config_params['path_tgt_valid'],
                'transforms': ['filtertoolong']
            }
        },
        # Vocabulary
        'src_vocab': config_params['path_src_vocab'],
        'tgt_vocab': config_params['path_tgt_vocab'],
        'src_vocab_size': config_params.get('src_vocab_size', 500), # 字符级词表通常较小
        'tgt_vocab_size': config_params.get('tgt_vocab_size', 500),
        'share_vocab': config_params.get('share_vocab', True), # 对SMILES，共享字符词表通常是好的

        # Tokenization (for char-level, these are important)
        'src_subword_type': 'char' if config_params.get('char_level_tokenization', True) else 'none',
        'tgt_subword_type': 'char' if config_params.get('char_level_tokenization', True) else 'none',
        'src_onmttok_kwargs': "{'mode': 'none'}", # for char level
        'tgt_onmttok_kwargs': "{'mode': 'none'}", # for char level
        
        # Model
        'encoder_type': config_params.get('onmt_encoder_type', 'rnn'),
        'decoder_type': config_params.get('onmt_decoder_type', 'rnn'),
        'enc_layers': config_params.get('onmt_enc_layers', 2),
        'dec_layers': config_params.get('onmt_dec_layers', 2),
        'hidden_size': config_params.get('onmt_hidden_size', 256), # 调整常见的SMILES模型大小
        'rnn_size': config_params.get('onmt_rnn_size', 256), # OpenNMT中 rnn_size 常用于LSTM/GRU单元数
        'word_vec_size': config_params.get('onmt_word_vec_size', 256), # 词向量维度
        'attention': config_params.get('onmt_attention', 'global'), # 'luong' or 'bahdanau' style global attention
        'global_attention': config_params.get('onmt_global_attention_type', 'general'),
        'bridge': config_params.get('onmt_bridge', True), # Add bridge layer between encoder and decoder
        'dropout': config_params.get('onmt_dropout', 0.3),

        # Optimization
        'optim': config_params.get('onmt_optim', 'adam'),
        'learning_rate': config_params.get('onmt_learning_rate', 0.001),
        'adam_beta1': 0.9,
        'adam_beta2': 0.998, # Common for Transformers, can adjust for RNNs
        'batch_size': config_params.get('onmt_batch_size', 64),
        'batch_type': 'sents', # or 'tokens'
        'max_grad_norm': config_params.get('onmt_max_grad_norm', 5), # Gradient clipping
        'train_steps': config_params.get('onmt_train_steps', 20000), # 训练步数，需要根据数据集调整
        'valid_steps': config_params.get('onmt_valid_steps', 500),  # 每N步验证一次
        'early_stopping': config_params.get('onmt_early_stopping', 5), # 早停轮数
        'early_stopping_criteria': 'accuracy', # or 'ppl', 'bleu'

        # Logging and saving
        'save_model': config_params['model_save_path_prefix'], # OpenNMT 会添加 _step_XXX.pt
        'save_checkpoint_steps': config_params.get('onmt_save_checkpoint_steps', 1000),
        'keep_checkpoint': config_params.get('onmt_keep_checkpoint', 5), # 保留最近N个检查点
        'log_file': config_params['opennmt_log_file'],
        'report_every': config_params.get('onmt_report_every', 100), # 每N步报告一次日志
        'tensorboard': True,
        'tensorboard_log_dir': config_params['tensorboard_log_dir'],

        # Other
        'seed': config_params.get('seed', 42),
        'gpu_ranks': config_params.get('gpu_ranks', []),
        'world_size': len(config_params.get('gpu_ranks', [])) if config_params.get('gpu_ranks') else 0,
        'src_seq_length': config_params.get('onmt_src_seq_length', 200), # 限制源序列长度
        'tgt_seq_length': config_params.get('onmt_tgt_seq_length', 200), # 限制目标序列长度
    }
    
    if not default_config['gpu_ranks']:
        default_config.pop('gpu_ranks', None) # Use None to avoid KeyError if already popped
        default_config.pop('world_size', None)

    try:
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(default_config, f, sort_keys=False)
        proc_logger.info(f"OpenNMT YAML配置文件已生成: {config_save_path}")
    except IOError as e:
        proc_logger.error(f"写入YAML配置文件失败: {e}")
        raise
    return config_save_path

def run_command(command, proc_logger, step_name="命令"):
    """执行shell命令并记录输出。"""
    proc_logger.info(f"执行 {step_name}: {' '.join(command)}")
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8')
        stdout, stderr = process.communicate()

        if stdout:
            proc_logger.info(f"{step_name} STDOUT:\n{stdout}")
        if stderr:
            if process.returncode != 0:
                 proc_logger.error(f"{step_name} STDERR:\n{stderr}")
            else:
                 proc_logger.info(f"{step_name} Messages (from stderr):\n{stderr}")

        if process.returncode != 0:
            proc_logger.error(f"{step_name} 失败，返回码: {process.returncode}")
            raise RuntimeError(f"{step_name} 失败。查看日志获取详细信息。 ({command[0]}) ")
        proc_logger.info(f"{step_name} 成功完成。")
        return stdout, stderr
    except FileNotFoundError:
        proc_logger.error(f"`{command[0]}` 命令未找到。请确保OpenNMT-py已安装并在PATH中。")
        raise
    except Exception as e:
        proc_logger.error(f"执行 {step_name} 时发生错误: {e}")
        raise


def train_opennmt(opennmt_args, proc_logger):
    """
    OpenNMT-py 训练主流程:
    1. 定义路径
    2. 准备数据 (CSV -> txt)
    3. 构建词表 (onmt_build_vocab)
    4. 生成训练配置文件 (YAML)
    5. 执行训练 (onmt_train)
    """
    base_run_dir = opennmt_args.output_dir
    proc_logger.info(f"启动OpenNMT训练。运行根目录: {base_run_dir}")

    data_prep_dir = os.path.join(base_run_dir, "opennmt_data")
    vocab_dir = os.path.join(base_run_dir, "opennmt_vocab")
    model_save_dir = os.path.join(base_run_dir, "opennmt_models")
    tb_log_dir = os.path.join(base_run_dir, "opennmt_tensorboard")
    opennmt_run_log_file = os.path.join(base_run_dir, "opennmt_training_run.log")
    config_file_path = os.path.join(base_run_dir, "opennmt_config.yaml")
    vocab_build_config_path = os.path.join(base_run_dir, "opennmt_vocab_config.yaml")

    os.makedirs(data_prep_dir, exist_ok=True)
    os.makedirs(vocab_dir, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(tb_log_dir, exist_ok=True)

    src_col_name = getattr(opennmt_args, 'src_col', 'reactant_smiles')
    trg_col_name = getattr(opennmt_args, 'trg_col', 'product_smiles')
    char_tokenization = getattr(opennmt_args, 'char_level_tokenization', True)

    src_train_txt, tgt_train_txt = prepare_data_for_opennmt(
        opennmt_args.train_file, src_col_name, trg_col_name, data_prep_dir, "train", proc_logger
    )
    src_val_txt, tgt_val_txt = prepare_data_for_opennmt(
        opennmt_args.val_file, src_col_name, trg_col_name, data_prep_dir, "valid", proc_logger
    )
    if hasattr(opennmt_args, 'test_file') and opennmt_args.test_file:
        prepare_data_for_opennmt(
            opennmt_args.test_file, src_col_name, trg_col_name, data_prep_dir, "test", proc_logger
        )

    src_vocab_path = os.path.join(vocab_dir, "vocab.src")
    tgt_vocab_path = os.path.join(vocab_dir, "vocab.tgt")
    shared_vocab_path = os.path.join(vocab_dir, "vocab.shared") # A common name for explicitly shared vocabs
    share_vocab_flag = getattr(opennmt_args, 'share_vocab', True)

    vocab_config_content = {
        'data': {
            'corpus_1': {'path_src': src_train_txt, 'path_tgt': tgt_train_txt, 'transforms': ['filtertoolong']}
        },
        'src_vocab': src_vocab_path,
        'tgt_vocab': tgt_vocab_path,
        'src_vocab_size': getattr(opennmt_args, 'onmt_src_vocab_size', 500),
        'tgt_vocab_size': getattr(opennmt_args, 'onmt_tgt_vocab_size', 500),
        'share_vocab': share_vocab_flag,
        'overwrite': True,
    }
    if char_tokenization:
        vocab_config_content['src_subword_type'] = 'char'
        vocab_config_content['tgt_subword_type'] = 'char'
        vocab_config_content['src_onmttok_kwargs'] = "{'mode': 'none', 'joiner_annotate': false, 'spacer_annotate': false}"
        vocab_config_content['tgt_onmttok_kwargs'] = "{'mode': 'none', 'joiner_annotate': false, 'spacer_annotate': false}"

    with open(vocab_build_config_path, 'w', encoding='utf-8') as f_vb_cfg:
        yaml.dump(vocab_config_content, f_vb_cfg, sort_keys=False)
    proc_logger.info(f"OpenNMT词表构建配置文件已生成: {vocab_build_config_path}")

    cmd_build_vocab = ["onmt_build_vocab", "-config", vocab_build_config_path, "-n_sample", "-1"]
    run_command(cmd_build_vocab, proc_logger, "OpenNMT词表构建")

    actual_src_vocab_path = src_vocab_path
    actual_tgt_vocab_path = tgt_vocab_path
    if share_vocab_flag:
        # OpenNMT with share_vocab=true in onmt_build_vocab typically produces one vocab file at the src_vocab path.
        # The tgt_vocab path might not be created, or it might be a symlink or copy.
        # We rely on the main training config to correctly interpret shared_vocab with these paths.
        if os.path.exists(src_vocab_path) and not os.path.exists(tgt_vocab_path):
             actual_tgt_vocab_path = src_vocab_path # If only src exists, assume it's the shared one
             proc_logger.info(f"共享词表模式: 目标词表路径设置为与源词表相同: {actual_tgt_vocab_path}")
        elif os.path.exists(shared_vocab_path): # Check for a specific shared vocab name
            actual_src_vocab_path = shared_vocab_path
            actual_tgt_vocab_path = shared_vocab_path
            proc_logger.info(f"共享词表模式: 检测到特定共享词表文件 {shared_vocab_path}")
        # If both exist, OpenNMT should handle it correctly with 'share_vocab: true' in main config.

    train_config_feed = {
        'path_src_train': src_train_txt,
        'path_tgt_train': tgt_train_txt,
        'path_src_valid': src_val_txt,
        'path_tgt_valid': tgt_val_txt,
        'path_src_vocab': actual_src_vocab_path,
        'path_tgt_vocab': actual_tgt_vocab_path,
        'model_save_path_prefix': os.path.join(model_save_dir, "model_onmt"),
        'opennmt_log_file': opennmt_run_log_file,
        'tensorboard_log_dir': tb_log_dir,
        'seed': opennmt_args.random_seed,
        'char_level_tokenization': char_tokenization,
        'share_vocab': share_vocab_flag
    }
    if torch.cuda.is_available() and opennmt_args.device == "cuda":
        num_gpus = torch.cuda.device_count()
        train_config_feed['gpu_ranks'] = [i for i in range(num_gpus)]
        proc_logger.info(f"检测到 {num_gpus} 个GPU，将用于训练。")
    else:
        train_config_feed['gpu_ranks'] = []
        proc_logger.info("未检测到GPU或设备设置为CPU，将使用CPU进行训练。")

    for key, value in vars(opennmt_args).items():
        if key.startswith("onmt_") and value is not None:
            train_config_feed[key] = value
            proc_logger.info(f"从命令行参数覆盖OpenNMT配置: {key} = {value}")

    generate_opennmt_config(train_config_feed, config_file_path, proc_logger)

    cmd_train = ["onmt_train", "-config", config_file_path]
    run_command(cmd_train, proc_logger, "OpenNMT模型训练")

    proc_logger.info(f"OpenNMT训练流程完成。模型保存在: {model_save_dir}")
    proc_logger.info(f"OpenNMT TensorBoard日志在: {tb_log_dir}")
    proc_logger.info(f"OpenNMT 详细运行日志在: {opennmt_run_log_file}")

if __name__ == '__main__':
    print("此脚本旨在通过 train.py 调用。")

    # 可以添加一个简单的命令行解析器和日志记录器用于独立测试
    # import argparse
    # from src.data.common.logger import ProcessLogger # 假设路径
    # parser = argparse.ArgumentParser("OpenNMT Baseline Test")
    # # 添加必要的参数: train_file, val_file, output_dir, random_seed, device
    # # ...
    # test_args = parser.parse_args()
    # test_logger = ProcessLogger("OpenNMT_Standalone_Test")
    # train_opennmt(test_args, test_logger) 