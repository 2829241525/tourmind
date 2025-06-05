# -*- coding: utf-8 -*-
"""
基于强化学习的文本匹配训练脚本
"""

import os
# 设置环境变量以避免tokenizers并行处理导致的死锁
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import random

from models.model import create_model
from models.rl_model import RLMatchingTrainer
from data.dataset import RoomMatchValidationDataset
from utils.logger import TrainingLogger

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'train_rl.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='基于强化学习的文本匹配训练脚本')
    parser.add_argument('--config', type=str, default='config/config_rl.json', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置文件
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 处理文件路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    def process_file_paths(file_str):
        if isinstance(file_str, str):
            return os.path.join(base_dir, file_str)
        elif isinstance(file_str, list):
            return [os.path.join(base_dir, f) for f in file_str]
        return file_str
    
    # 处理配置中的文件路径
    for key in ['train_file', 'valid_file']:
        if key in config:
            config[key] = process_file_paths(config[key])
    
    # 确保保存目录存在
    config['save_dir'] = os.path.join(base_dir, config['save_dir'])
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 设置设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载基础模型和分词器
    model_path = config['pretrained_model']
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # 根据损失函数类型加载相应的基础模型
    if config['loss_type'] == 'triplet':
        base_model = create_model('encoder', model_path)
    else:
        base_model = create_model('classifier', model_path, num_labels=2)
    
    # 将模型移至GPU
    base_model.to(device)
    base_model.eval()  # 基础模型设为评估模式
    
    # 加载数据集
    train_dataset = RoomMatchValidationDataset(config['train_file'])
    valid_dataset = RoomMatchValidationDataset(config['valid_file'])
    
    logger.info(f"加载了 {len(train_dataset)} 个训练样本")
    logger.info(f"加载了 {len(valid_dataset)} 个验证样本")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('rl_batch_size', 32),
        shuffle=True,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.get('rl_batch_size', 32),
        shuffle=False,
        num_workers=config.get('num_workers', 2),
        pin_memory=config.get('pin_memory', True)
    )
    
    # 创建强化学习训练器
    rl_trainer = RLMatchingTrainer(base_model, tokenizer, config, device)
    
    # 开始强化学习训练
    rl_trainer.train(train_loader, valid_loader)
    
    logger.info("强化学习训练完成！")

if __name__ == '__main__':
    main() 