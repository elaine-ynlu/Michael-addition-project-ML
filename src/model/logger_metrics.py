# -*- coding: utf-8 -*-
"""
训练度量日志记录工具：
    - 将训练过程中每个 Epoch 的 train_loss、val_loss、device 等信息记录到 CSV 文件中
    - 日后可用于可视化和评估训练效果
"""
import os
import csv
from datetime import datetime

from src.data.common.paths import project_paths

class MetricsLogger:
    """度量日志记录器，将每个epoch的指标写入CSV"""
    def __init__(self, process_name='seq2seq', log_dir=None):
        # 默认日志目录 data/metrics
        if log_dir is None:
            log_dir = os.path.join(project_paths.data, 'metrics')
        os.makedirs(log_dir, exist_ok=True)
        # 构造日志文件名，包含时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{process_name}_metrics_{timestamp}.csv"
        self.log_path = os.path.join(log_dir, filename)
        # 打开文件并初始化CSV写入器
        self._csv_file = open(self.log_path, mode='w', newline='', encoding='utf-8')
        self._writer = csv.writer(self._csv_file)
        # 写入表头
        self._writer.writerow(['epoch', 'train_loss', 'val_loss', 'device', 'timestamp'])
        self._csv_file.flush()
        # 记录创建时的路径
        print(f"MetricsLogger: 写入日志到 {self.log_path}")

    def log_epoch(self, epoch, train_loss, val_loss, device):
        """记录单个Epoch的度量信息"""
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._writer.writerow([epoch, train_loss, val_loss, str(device), now])
        self._csv_file.flush()

    def close(self):
        """关闭日志文件"""
        self._csv_file.close() 