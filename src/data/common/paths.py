import os
import sys

# 获取项目根目录
def get_project_root():
    # 查找项目根目录（包含src文件夹的目录）
    current_path = os.path.abspath(__file__)
    while current_path:
        parent_dir = os.path.dirname(current_path)
        if os.path.basename(current_path) == 'src' and os.path.exists(os.path.join(parent_dir, 'src')):
            return parent_dir
        if parent_dir == current_path:  # 已到达文件系统根目录
            break
        current_path = parent_dir
    
    # 如果上面的方法失败，尝试从sys.path中查找
    for path in sys.path:
        if os.path.basename(path) == 'MichaelFairy' or os.path.exists(os.path.join(path, 'src')):
            return path
    
    # 最后的备选方案
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 定义项目结构
class ProjectPaths:
    def __init__(self):
        self.root = get_project_root()
        
        # 源代码目录
        self.src = os.path.join(self.root, 'src')
        
        # 数据目录结构
        self.data = os.path.join(self.root, 'data')
        self.data_origin = os.path.join(self.data, 'origin')  # 原始数据保存路径
        self.data_unclean = os.path.join(self.data, 'unclean')  # 未清洗数据保存路径
        self.data_filtered = os.path.join(self.data, 'filtered')  # 过滤数据保存路径
        self.data_michael = os.path.join(self.data, 'michael')  # 迈克尔数据保存路径
        self.data_train = os.path.join(self.data, 'train')  # 训练数据保存路径
        self.log_tensorboard = os.path.join(self.data, 'log_tensorboard')  # 日志保存路径
        self.data_heatmap = os.path.join(self.data, 'heatmap')  # 热力图保存路径
        self.output_model = os.path.join(self.data, 'output_model')  # 模型保存路径

        self.ensure_directories()
        
    def ensure_directories(self):
        """确保所有必要的目录都存在"""
        directories = [
            self.data,
            self.data_origin,
            self.data_unclean,
            self.data_filtered,
            self.data_michael,
            self.data_train,
            self.log_tensorboard,
            self.data_heatmap,
            self.output_model
        ]
        for directory in directories:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"创建目录: {directory}")



# 创建全局路径对象
project_paths = ProjectPaths() 