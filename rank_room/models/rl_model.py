# -*- coding: utf-8 -*-
"""
基于强化学习的文本匹配精确率优化模型
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)

class RLMatchingOptimizer(nn.Module):
    """基于强化学习的文本匹配精确率优化器"""
    
    def __init__(self, base_model, device, precision_weight=2.0):
        """
        初始化强化学习优化器
        
        Args:
            base_model: 基础文本匹配模型
            device: 计算设备
            precision_weight: 精确率权重，越高越重视精确率
        """
        super().__init__()
        self.base_model = base_model
        self.device = device
        self.precision_weight = precision_weight
        
        # 创建策略网络，用于学习决策阈值
        hidden_size = 768  # DeBERTa的隐藏层大小
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 1, 128),  # 输入为两个文本表示向量拼接以及基础相似度
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # 输出匹配概率
        ).to(device)
        
        # 保存模型训练状态
        self.train_iterations = 0
        self.best_precision = 0.0
        self.best_f1 = 0.0
        
    def compute_base_similarity(self, text1_embeddings, text2_embeddings):
        """计算基础模型的文本相似度"""
        text1_norm = F.normalize(text1_embeddings, p=2, dim=1)
        text2_norm = F.normalize(text2_embeddings, p=2, dim=1)
        return torch.sum(text1_norm * text2_norm, dim=1)
    
    def forward(self, text1_embeddings, text2_embeddings):
        """
        前向传播：根据文本表示和基础相似度，预测匹配概率
        
        Args:
            text1_embeddings: 第一个文本的表示向量
            text2_embeddings: 第二个文本的表示向量
            
        Returns:
            match_probs: 匹配概率
            actions: 采样的匹配决策
            base_similarities: 基础相似度分数
        """
        # 计算基础相似度
        base_similarities = self.compute_base_similarity(text1_embeddings, text2_embeddings)
        
        # 拼接输入特征
        combined_features = torch.cat([
            text1_embeddings, 
            text2_embeddings, 
            base_similarities.unsqueeze(1)
        ], dim=1)
        
        # 通过策略网络预测匹配概率
        match_probs = self.policy_net(combined_features).squeeze(-1)
        
        # 采样匹配决策（训练时）
        if self.training:
            distribution = Bernoulli(match_probs)
            actions = distribution.sample()
            return match_probs, actions, base_similarities
        else:
            # 推理时直接使用概率
            return match_probs, (match_probs > 0.5).float(), base_similarities
    
    def compute_reward(self, predictions, labels):
        """
        计算奖励函数，平衡精确率和召回率
        
        Args:
            predictions: 预测结果
            labels: 真实标签
            
        Returns:
            rewards: 计算的奖励值
        """
        # 转换为numpy数组计算指标
        pred_np = predictions.cpu().numpy()
        label_np = labels.cpu().numpy()
        
        # 计算精确率、召回率和F1分数
        precision = precision_score(label_np, pred_np, zero_division=0)
        recall = recall_score(label_np, pred_np, zero_division=0)
        f1 = f1_score(label_np, pred_np, zero_division=0)
        
        # 记录当前指标到日志
        if self.train_iterations % 10 == 0:
            logger.debug(f"当前批次指标 - 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")
        
        # 构建更平衡的奖励函数
        # 使用β-F值作为主要奖励，可以更灵活地控制精确率和召回率的权重
        beta = max(0.5, 1.0 / self.precision_weight)  # beta<1时更注重精确率，但不会极端偏向
        if precision > 0 or recall > 0:  # 避免分母为0
            beta_f = (2 + beta**2) * precision * recall
            # 添加原始指标作为辅助奖励
            reward = beta_f + 0.2 * precision + 0.2 * recall
        else:
            reward = 0.0
        
        # 计算FP的惩罚（假阳性）
        fp_mask = (predictions == 1) & (labels == 0)
        fp_penalty = -1 * fp_mask.float().sum() / max(1, len(labels))
        
        # 计算FN的惩罚（假阴性）- 防止模型总是预测负例
        fn_mask = (predictions == 0) & (labels == 1)
        fn_penalty = -1 * fn_mask.float().sum() / max(1, len(labels))
        
        # 最终奖励
        batch_reward = torch.tensor(reward, device=self.device) + fp_penalty + fn_penalty
        
        # 防止全预测为负的情况 - 如果没有正例预测，额外惩罚
        if pred_np.sum() == 0 and label_np.sum() > 0:
            no_positive_penalty = -0.5 * (label_np.sum() / len(label_np))
            batch_reward += torch.tensor(no_positive_penalty, device=self.device)
            if self.train_iterations % 10 == 0:
                logger.warning(f"检测到全负预测，添加额外惩罚: {no_positive_penalty:.4f}")
        
        return batch_reward
    
    def save(self, path):
        """保存模型"""
        os.makedirs(path, exist_ok=True)
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'train_iterations': self.train_iterations,
            'best_precision': self.best_precision,
            'best_f1': self.best_f1,
            'precision_weight': self.precision_weight
        }, os.path.join(path, 'rl_optimizer.pt'))
        logger.info(f"已将强化学习优化器保存到: {path}")
    
    def load(self, path):
        """加载模型"""
        checkpoint_path = os.path.join(path, 'rl_optimizer.pt')
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.train_iterations = checkpoint.get('train_iterations', 0)
            self.best_precision = checkpoint.get('best_precision', 0.0)
            self.best_f1 = checkpoint.get('best_f1', 0.0)
            self.precision_weight = checkpoint.get('precision_weight', 2.0)
            logger.info(f"已加载强化学习优化器: {path}")
            return True
        else:
            logger.warning(f"未找到强化学习优化器检查点: {checkpoint_path}")
            return False

class RLMatchingTrainer:
    """强化学习匹配模型训练器"""
    
    def __init__(self, base_model, tokenizer, config, device):
        """
        初始化强化学习训练器
        
        Args:
            base_model: 基础文本匹配模型
            tokenizer: 分词器
            config: 配置信息
            device: 计算设备
        """
        self.base_model = base_model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # 创建强化学习优化器
        precision_weight = config.get('rl_precision_weight', 2.0)
        self.rl_optimizer = RLMatchingOptimizer(base_model, device, precision_weight)
        
        # 设置优化器
        self.optimizer = torch.optim.Adam(
            self.rl_optimizer.parameters(),
            lr=config.get('rl_learning_rate', 1e-4),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 训练参数
        self.num_epochs = config.get('rl_num_epochs', 5)
        self.batch_size = config.get('rl_batch_size', 32)
        self.gamma = config.get('rl_gamma', 0.99)  # 折扣因子
        self.entropy_weight = config.get('rl_entropy_weight', 0.01)  # 熵正则化权重
        
        # 保存目录
        self.save_dir = os.path.join(config['save_dir'], 'rl_model')
        os.makedirs(self.save_dir, exist_ok=True)
        
    def encode_texts(self, text1, text2):
        """编码文本对"""
        # 将基础模型设为评估模式
        self.base_model.eval()
        
        with torch.no_grad():
            # 编码文本对
            inputs = self.tokenizer(
                text1, 
                text_pair=text2,
                padding=True, 
                truncation=True, 
                max_length=self.config.get('max_length', 256),
                return_tensors='pt'
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 提取文本表示 - 修复部分开始
            if hasattr(self.base_model, 'model'):
                # 如果是封装的模型
                if self.config.get('loss_type') == 'triplet':
                    outputs = self.base_model.model(**inputs)
                    # 获取[CLS]位置的表示
                    text1_embeddings = outputs.last_hidden_state[:, 0]
                else:
                    # 对于分类器模型，我们可以从中间层获取表示
                    # 去掉labels参数，以避免计算损失
                    inputs_no_labels = {k: v for k, v in inputs.items() if k != 'labels'}
                    # 获取隐藏状态
                    outputs = self.base_model.model(**inputs_no_labels, output_hidden_states=True)
                    # 使用最后一层的[CLS]表示
                    text1_embeddings = outputs.hidden_states[-1][:, 0]
            else:
                # 如果是直接的transformers模型
                if hasattr(self.base_model, 'config') and getattr(self.base_model.config, 'problem_type', None) == 'single_label_classification':
                    # 分类器模型
                    inputs_no_labels = {k: v for k, v in inputs.items() if k != 'labels'}
                    outputs = self.base_model(**inputs_no_labels, output_hidden_states=True)
                    text1_embeddings = outputs.hidden_states[-1][:, 0]
                else:
                    # 编码器模型
                    outputs = self.base_model(**inputs)
                    text1_embeddings = outputs.last_hidden_state[:, 0]
            # 修复部分结束
                
            return text1_embeddings, text1_embeddings
    
    def train_step(self, batch):
        """执行一步训练"""
        # 将优化器设为训练模式
        self.rl_optimizer.train()
        
        # 获取文本表示
        with torch.no_grad():
            text1_embeddings, text2_embeddings = self.encode_texts(batch['text1'], batch['text2'])
        
        # 前向传播，获取匹配概率和采样结果
        match_probs, actions, _ = self.rl_optimizer(text1_embeddings, text2_embeddings)
        
        # 将标签移动到device
        labels = batch['label'].to(self.device)
        
        # 计算奖励
        rewards = self.rl_optimizer.compute_reward(actions, labels)
        
        # 策略梯度损失
        distribution = Bernoulli(match_probs)
        log_probs = distribution.log_prob(actions)
        policy_loss = -log_probs * rewards
        
        # 熵正则化，防止策略过早收敛
        entropy = distribution.entropy()
        entropy_loss = -self.entropy_weight * entropy
        
        # 总损失
        loss = policy_loss.mean() + entropy_loss.mean()
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.mean().item(),
            'entropy': entropy.mean().item(),
            'reward': rewards.item()
        }
    
    def evaluate(self, valid_loader):
        """评估模型"""
        self.rl_optimizer.eval()
        
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in valid_loader:
                # 编码文本
                text1_embeddings, text2_embeddings = self.encode_texts(batch['text1'], batch['text2'])
                
                # 预测
                _, actions, _ = self.rl_optimizer(text1_embeddings, text2_embeddings)
                labels = batch['label'].to(self.device)
                
                # 收集结果
                all_predictions.extend(actions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # 计算指标
        precision = precision_score(all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(self, train_loader, valid_loader):
        """训练模型"""
        # 尝试加载已有模型
        loaded = self.rl_optimizer.load(self.save_dir)
        
        best_precision = self.rl_optimizer.best_precision if loaded else 0.0
        best_f1 = self.rl_optimizer.best_f1 if loaded else 0.0
        
        logger.info(f"开始强化学习训练...")
        logger.info(f"精确率权重: {self.rl_optimizer.precision_weight}")
        logger.info(f"熵正则化权重: {self.entropy_weight}")
        
        for epoch in range(self.num_epochs):
            # 训练一个epoch
            train_metrics = {'loss': 0, 'policy_loss': 0, 'entropy': 0, 'reward': 0}
            
            for i, batch in enumerate(train_loader):
                step_metrics = self.train_step(batch)
                
                # 更新指标
                for k, v in step_metrics.items():
                    train_metrics[k] += v
                
                # 记录训练进度
                if (i + 1) % 10 == 0:
                    metrics_str = ', '.join([f"{k}: {v/(i+1):.4f}" for k, v in train_metrics.items()])
                    logger.info(f"Epoch {epoch+1}/{self.num_epochs}, Step {i+1}/{len(train_loader)}, {metrics_str}")
            
            # 计算平均指标
            for k in train_metrics:
                train_metrics[k] /= len(train_loader)
            
            # 验证
            val_metrics = self.evaluate(valid_loader)
            
            # 输出结果
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}:")
            logger.info(f"  Train: {', '.join([f'{k}={v:.4f}' for k, v in train_metrics.items()])}")
            logger.info(f"  Valid: {', '.join([f'{k}={v:.4f}' for k, v in val_metrics.items()])}")
            
            # 保存最佳模型
            if val_metrics['precision'] > best_precision:
                best_precision = val_metrics['precision']
                self.rl_optimizer.best_precision = best_precision
                self.rl_optimizer.save(os.path.join(self.save_dir, 'best_precision'))
                logger.info(f"  保存精确率最佳模型: {best_precision:.4f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.rl_optimizer.best_f1 = best_f1
                self.rl_optimizer.save(os.path.join(self.save_dir, 'best_f1'))
                logger.info(f"  保存F1最佳模型: {best_f1:.4f}")
            
            # 保存最新模型
            self.rl_optimizer.train_iterations += 1
            self.rl_optimizer.save(os.path.join(self.save_dir, 'latest'))
        
        logger.info("强化学习训练完成!")
        logger.info(f"最佳精确率: {best_precision:.4f}")
        logger.info(f"最佳F1分数: {best_f1:.4f}") 