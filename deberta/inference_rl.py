# -*- coding: utf-8 -*-
"""
基于强化学习优化的文本匹配推理脚本
"""

import os
import json
import torch
import argparse
import logging
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.model import create_model
from models.rl_model import RLMatchingOptimizer

# 默认配置
# 项目根目录路径
ROOT_DIR = '/home/maxon/disk2/roomMatch/room_match/deberta'

DEFAULT_CONFIG = {
    'base_model': os.path.join(ROOT_DIR, 'checkpoints_cross_entropy/best_model'),
    'rl_model': os.path.join(ROOT_DIR, 'checkpoints_rl/rl_model/best_precision'),
    'input': os.path.join(ROOT_DIR, 'data/processed_merged_test.csv'),
    'output': os.path.join(ROOT_DIR, 'results/predictions.csv'),
    'batch_size': 32,
    'max_length': 80,  # 文本截断最大长度
    'device': None  # 自动选择设备
}

# 配置日志记录
os.makedirs(os.path.join(os.path.dirname(__file__), 'logs'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'logs', 'inference_rl.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RLMatchingPredictor:
    """强化学习匹配预测器"""
    
    def __init__(self, base_model_path, rl_model_path, device=None, max_length=80):
        """
        初始化预测器
        
        Args:
            base_model_path: 基础模型路径
            rl_model_path: 强化学习模型路径
            device: 设备
            max_length: 文本截断最大长度
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        self.max_length = max_length
        
        # 加载基础模型和分词器
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        # 确定模型类型
        self.model_type = self._detect_model_type(base_model_path)
        logger.info(f"检测到模型类型: {self.model_type}")
        
        # 加载模型
        if self.model_type == 'encoder':
            self.base_model = create_model('encoder', base_model_path)
        else:
            self.base_model = create_model('classifier', base_model_path, num_labels=2)
            
        self.base_model.to(self.device)
        self.base_model.eval()
        
        # 创建强化学习优化器
        precision_weight = 2.0  # 默认精确率权重
        self.rl_optimizer = RLMatchingOptimizer(self.base_model, self.device, precision_weight)
        
        # 加载强化学习模型
        rl_model_loaded = self.rl_optimizer.load(rl_model_path)
        if not rl_model_loaded:
            logger.warning(f"未能加载强化学习模型，将只使用基础模型进行预测")
        else:
            logger.info(f"已加载强化学习模型，精确率权重: {self.rl_optimizer.precision_weight}")
        
        self.rl_optimizer.to(self.device)
        self.rl_optimizer.eval()
    
    def _detect_model_type(self, model_path):
        """检测模型类型"""
        if os.path.exists(os.path.join(model_path, 'config.json')):
            try:
                with open(os.path.join(model_path, 'config.json'), 'r') as f:
                    config = json.load(f)
                    # 根据配置判断是分类器还是编码器
                    if 'num_labels' in config:
                        return 'classifier'
                    return 'encoder'
            except Exception as e:
                logger.warning(f"读取模型配置时出错: {e}，默认使用分类器模型")
        return 'classifier'  # 默认为分类器模型
        
    def encode_texts(self, text1, text2):
        """编码文本对 - 与训练时逻辑一致
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            text1_embeddings, text2_embeddings: 两个文本的嵌入向量
        """
        # 将基础模型设为评估模式
        self.base_model.eval()
        
        with torch.no_grad():
            # 编码文本对
            inputs = self.tokenizer(
                text1, 
                text_pair=text2,
                padding=True, 
                truncation=True, 
                max_length=self.max_length,
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取文本表示 - 遵循与训练时相同的逻辑
            if hasattr(self.base_model, 'model'):
                # 如果是封装的模型
                if self.model_type == 'encoder':
                    outputs = self.base_model.model(**inputs)
                    # 获取[CLS]位置的表示
                    text1_embeddings = outputs.last_hidden_state[:, 0]
                else:
                    # 对于分类器模型，从中间层获取表示
                    inputs_no_labels = {k: v for k, v in inputs.items() if k != 'labels'}
                    # 获取隐藏状态
                    outputs = self.base_model.model(**inputs_no_labels, output_hidden_states=True)
                    # 使用最后一层的[CLS]表示
                    text1_embeddings = outputs.hidden_states[-1][:, 0]
            else:
                # 如果是直接的transformers模型
                if self.model_type == 'classifier':
                    # 分类器模型
                    inputs_no_labels = {k: v for k, v in inputs.items() if k != 'labels'}
                    outputs = self.base_model(**inputs_no_labels, output_hidden_states=True)
                    text1_embeddings = outputs.hidden_states[-1][:, 0]
                else:
                    # 编码器模型
                    outputs = self.base_model(**inputs)
                    text1_embeddings = outputs.last_hidden_state[:, 0]
                
            return text1_embeddings, text1_embeddings  # 对于配对输入，我们获得相同的表示
    
    def batch_predict(self, df, batch_size=32):
        """批量预测
        
        Args:
            df: 包含待预测文本对的DataFrame
            batch_size: 批处理大小
        
        Returns:
            包含预测结果的DataFrame
        """
        logger.info(f"开始批量预测，批处理大小: {batch_size}")
        
        # 数据量
        total_samples = len(df)
        logger.info(f"总样本数: {total_samples}")
        
        # 存储预测结果
        all_results = []
        
        # 使用tqdm显示进度
        for i in tqdm(range(0, total_samples, batch_size), desc="批量预测进度"):
            end_idx = min(i + batch_size, total_samples)
            batch_df = df.iloc[i:end_idx]
            
            # 提取文本
            text1_list = batch_df['spl_room_text'].tolist()
            text2_list = batch_df['s_room_text'].tolist()
            
            # 编码文本对
            text1_embeddings, text2_embeddings = self.encode_texts(text1_list, text2_list)
            
            # 预测
            with torch.no_grad():
                match_probs, actions, base_similarities = self.rl_optimizer(text1_embeddings, text2_embeddings)
            
            # 收集结果
            for j in range(len(batch_df)):
                result = {
                    'spl_room_text': batch_df.iloc[j]['spl_room_text'],
                    's_room_text': batch_df.iloc[j]['s_room_text'],
                    'match': bool(actions[j].item()),
                    'probability': match_probs[j].item(),
                    'base_similarity': base_similarities[j].item()
                }
                
                # 如果有标签的话也加入
                if 'label' in batch_df.columns:
                    result['label'] = batch_df.iloc[j]['label']
                
                all_results.append(result)
            
            # 批次处理完成后清理内存
            if torch.cuda.is_available() and (i + batch_size) % (batch_size * 10) == 0:
                torch.cuda.empty_cache()
                logger.info(f"已处理 {end_idx}/{total_samples} 条样本")
        
        logger.info(f"批量预测完成，共 {len(all_results)} 条结果")
        return pd.DataFrame(all_results)
    
    def compute_metrics(self, df):
        """计算评估指标
        
        Args:
            df: 包含预测结果和真实标签的DataFrame
            
        Returns:
            包含各项指标的字典
        """
        if 'label' not in df.columns or df['label'].max() < 0:
            logger.warning("数据中没有有效标签，无法计算评估指标")
            return None
            
        # 只考虑有标签的样本
        labeled_df = df[df['label'] >= 0]
        
        y_true = labeled_df['label'].values
        y_pred = labeled_df['match'].astype(int).values
        
        # 计算指标
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='基于强化学习优化的文本匹配推理')
    parser.add_argument('--base-model', type=str, default=DEFAULT_CONFIG['base_model'], 
                        help=f'基础模型路径（默认：{DEFAULT_CONFIG["base_model"]}）')
    parser.add_argument('--rl-model', type=str, default=DEFAULT_CONFIG['rl_model'], 
                        help=f'强化学习模型路径（默认：{DEFAULT_CONFIG["rl_model"]}）')
    parser.add_argument('--input', type=str, default=DEFAULT_CONFIG['input'], 
                        help=f'输入CSV文件路径（默认：{DEFAULT_CONFIG["input"]}）')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output'], 
                        help=f'输出CSV文件路径（默认：{DEFAULT_CONFIG["output"]}）')
    parser.add_argument('--batch-size', type=int, default=DEFAULT_CONFIG['batch_size'], 
                        help=f'批处理大小（默认：{DEFAULT_CONFIG["batch_size"]}）')
    parser.add_argument('--max-length', type=int, default=DEFAULT_CONFIG['max_length'], 
                        help=f'文本截断最大长度（默认：{DEFAULT_CONFIG["max_length"]}）')
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    # 记录配置信息
    logger.info("使用以下配置进行推理：")
    logger.info(f"  基础模型路径: {args.base_model}")
    logger.info(f"  强化学习模型路径: {args.rl_model}")
    logger.info(f"  输入文件: {args.input}")
    logger.info(f"  输出文件: {args.output}")
    logger.info(f"  批处理大小: {args.batch_size}")
    logger.info(f"  文本截断最大长度: {args.max_length}")
    
    # 加载输入数据
    logger.info(f"正在读取输入文件: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"读取完成，共 {len(df)} 条记录")
    
    # 初始化预测器
    logger.info("初始化预测器...")
    predictor = RLMatchingPredictor(
        args.base_model, 
        args.rl_model,
        max_length=args.max_length
    )
    
    # 进行批量预测
    logger.info("开始预测...")
    results_df = predictor.batch_predict(df, batch_size=args.batch_size)
    
    # 保存结果
    logger.info(f"预测完成，保存结果至: {args.output}")
    results_df.to_csv(args.output, index=False)
    
    # 如果有标签，计算指标
    metrics = predictor.compute_metrics(results_df)
    if metrics:
        logger.info("评估指标:")
        logger.info(f"  准确率: {metrics['accuracy']:.4f}")
        logger.info(f"  精确率: {metrics['precision']:.4f}")
        logger.info(f"  召回率: {metrics['recall']:.4f}")
        logger.info(f"  F1分数: {metrics['f1']:.4f}")
    
    logger.info("推理任务完成")

if __name__ == '__main__':
    main() 