# -*- coding: utf-8 -*-
"""
损失函数模块
实现了多种用于文本匹配任务的损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)

class SupervisedInfoNCELoss(nn.Module):
    """有监督的InfoNCE损失函数实现"""
    
    def __init__(self, temperature: float = 0.05, weight: float = 0.5):
        """
        初始化有监督的InfoNCE损失函数
        Args:
            temperature: 温度参数，用于调节相似度分布的平滑程度
            weight: 对比损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.weight = weight  # 对比损失权重
        self.bce_loss = nn.BCEWithLogitsLoss()        
        
    def forward(self, anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算有监督的InfoNCE损失
        Args:
            anchor_embeddings: 形状为(batch_size, hidden_size)的源文本向量
            target_embeddings: 形状为(batch_size, hidden_size)的目标文本向量
            labels: 形状为(batch_size)的标签，1表示正例对，0表示负例对
        Returns:
            loss: 有监督的InfoNCE损失值
        """
        # 对比损失（仅正样本）
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(anchor_norm, target_norm.T) / self.temperature
        pos_mask = torch.eye(labels.size(0), device=labels.device) * labels.unsqueeze(1)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        contrast_loss = -(pos_mask * log_prob).sum() / pos_mask.sum().clamp(min=1e-8)

        return contrast_loss
    
class CrossEntropyLoss(nn.Module):
    """基于交叉熵的损失函数实现"""
    
    def __init__(self, temperature: float = 0.05):
        """
        初始化交叉熵损失函数
        Args:
            temperature: 温度参数，用于调节相似度分布的平滑程度
        """
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.BCEWithLogitsLoss()
        
    def compute_similarity(self, text1_embeddings: torch.Tensor, text2_embeddings: torch.Tensor) -> torch.Tensor:
        """
        计算两个文本嵌入之间的相似度
        Args:
            text1_embeddings: 第一个文本的向量表示，形状为(batch_size, hidden_size)
            text2_embeddings: 第二个文本的向量表示，形状为(batch_size, hidden_size)
        Returns:
            logits: 相似度logits，形状为(batch_size,)
        """
        # 对向量进行L2归一化
        text1_norm = F.normalize(text1_embeddings, p=2, dim=1)
        text2_norm = F.normalize(text2_embeddings, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.sum(text1_norm * text2_norm, dim=1)
        
        # 应用温度缩放
        logits = similarities / self.temperature
        return logits
        
    def forward(self, text1_embeddings: torch.Tensor, text2_embeddings: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算交叉熵损失
        Args:
            text1_embeddings: 第一个文本的向量表示，形状为(batch_size, hidden_size)
            text2_embeddings: 第二个文本的向量表示，形状为(batch_size, hidden_size)
            labels: 标签，形状为(batch_size)，1表示正例对，0表示负例对
        Returns:
            loss: 交叉熵损失值
        """
        logits = self.compute_similarity(text1_embeddings, text2_embeddings)
        return self.criterion(logits, labels.float())
    
    def predict(self, text1_embeddings: torch.Tensor, text2_embeddings: torch.Tensor) -> torch.Tensor:
        """
        预测两个文本的相似度概率
        Args:
            text1_embeddings: 第一个文本的向量表示，形状为(batch_size, hidden_size)
            text2_embeddings: 第二个文本的向量表示，形状为(batch_size, hidden_size)
        Returns:
            probabilities: 相似度概率，形状为(batch_size,)
        """
        logits = self.compute_similarity(text1_embeddings, text2_embeddings)
        return torch.sigmoid(logits)

class CosineTripletLoss(nn.Module):
    """基于余弦距离的 Triplet Loss 实现"""
    
    def __init__(self, margin: float = 0.1, reduction: str = 'mean'):
        """
        初始化 Cosine Triplet Loss
        Args:
            margin: margin参数，用于控制正负样本之间的距离
            reduction: 损失聚合方式，'mean'或'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        
    def cosine_distance(self, tensor1: torch.Tensor, tensor2: torch.Tensor) -> torch.Tensor:
        """
        计算两个张量之间的余弦距离
        Args:
            tensor1: 第一个张量，形状为 (batch_size, hidden_size)
            tensor2: 第二个张量，形状为 (batch_size, hidden_size)
        Returns:
            余弦距离，形状为 (batch_size,)
        """
        tensor1 = F.normalize(tensor1, dim=-1)
        tensor2 = F.normalize(tensor2, dim=-1)
        cosine = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine
        
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        """
        计算 Triplet Loss
        Args:
            anchor: 锚点文本的向量表示，形状为 (batch_size, hidden_size)
            positive: 正例文本的向量表示，形状为 (batch_size, hidden_size)
            negative: 负例文本的向量表示，形状为 (batch_size, hidden_size)
        Returns:
            loss: Triplet Loss 值
        """
        # 计算正例对和负例对的余弦距离
        ap_distance = self.cosine_distance(anchor, positive)
        an_distance = self.cosine_distance(anchor, negative)
        
        # 计算带有 margin 的 triplet loss
        loss = torch.clamp(ap_distance - an_distance + self.margin, min=0)
        
        # 根据reduction方式返回损失
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss

def create_loss_fn(loss_type: str, **kwargs):
    """
    创建损失函数的工厂函数
    
    Args:
        loss_type: 损失函数类型，'triplet'、'cross_entropy'或'infoNCE'
        **kwargs: 传递给损失函数构造器的参数
        
    Returns:
        nn.Module: 创建的损失函数
    """
    if loss_type == 'triplet':
        margin = kwargs.get('margin', 0.1)
        reduction = kwargs.get('reduction', 'mean')
        logger.info(f"创建CosineTripletLoss，margin={margin}, reduction={reduction}")
        return CosineTripletLoss(margin=margin, reduction=reduction)
    elif loss_type == 'cross_entropy':
        temperature = kwargs.get('temperature', 0.05)
        logger.info(f"创建CrossEntropyLoss，temperature={temperature}")
        return CrossEntropyLoss(temperature=temperature)
    elif loss_type == 'infoNCE':
        temperature = kwargs.get('temperature', 0.05)
        weight = kwargs.get('weight', 0.5)
        logger.info(f"创建SupervisedInfoNCELoss，temperature={temperature}, weight={weight}")
        return SupervisedInfoNCELoss(temperature=temperature, weight=weight)
    else:
        raise ValueError(f"不支持的损失函数类型: {loss_type}")