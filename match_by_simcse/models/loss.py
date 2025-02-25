# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F



class SupervisedInfoNCELoss(nn.Module):
    """有监督的InfoNCE损失函数实现"""
    
    def __init__(self, temperature: float = 0.05, weight: float = 0.5):
        """
        初始化有监督的InfoNCE损失函数
        Args:
            temperature: 温度参数，用于调节相似度分布的平滑程度
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
        # 对向量进行L2归一化
        # anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        # target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
        
        # # 计算相似度矩阵
        # sim_matrix = torch.matmul(anchor_embeddings, target_embeddings.t()) / self.temperature
        
        # # 构造标签矩阵 - 只有当两个样本都是正样本(标签为1)时才认为是正例对
        # pos_mask = (labels == 1).float()
        # labels_matrix = torch.diag(pos_mask)
        
        # # 计算正例和负例的损失
        # exp_sim = torch.exp(sim_matrix)
        # log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
        
        # # 计算每个样本的损失，只考虑正样本的损失
        # mean_log_prob_pos = (labels_matrix * log_prob).sum(1) / labels_matrix.sum(1).clamp(min=1e-8)
        # # 只对正样本计算损失
        # valid_mask = (labels == 1).float()
        # loss = -(mean_log_prob_pos * valid_mask).sum() / valid_mask.sum().clamp(min=1e-8)
        # return loss 

        # 对比损失（仅正样本）
        anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
        target_norm = F.normalize(target_embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(anchor_norm, target_norm.T) / self.temperature
        pos_mask = torch.eye(labels.size(0), device=labels.device) * labels.unsqueeze(1)
        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        contrast_loss = -(pos_mask * log_prob).sum() / pos_mask.sum().clamp(min=1e-8)

        # # 二元交叉熵损失（所有样本）
        # similarity_scores = torch.diag(sim_matrix)  # 取对角相似度作为logits
        # bce_loss = self.bce_loss(similarity_scores, labels.float())
        
        return contrast_loss
    
        
    
class Cross_entropyLoss(nn.Module):
    """有监督的InfoNCE损失函数实现"""
    
    def __init__(self, temperature: float = 0.05):
        """
        初始化有监督的InfoNCE损失函数
        Args:
            temperature: 温度参数，用于调节相似度分布的平滑程度
        """
        super().__init__()
        self.temperature = temperature
    def forward(self, anchor_embeddings: torch.Tensor, target_embeddings: torch.Tensor, 
            labels: torch.Tensor) -> torch.Tensor:
        # 归一化处理
        anchor_embeddings = F.normalize(anchor_embeddings, p=2, dim=1)
        target_embeddings = F.normalize(target_embeddings, p=2, dim=1)
    
        # 计算相似度矩阵
        sim_matrix = torch.matmul(anchor_embeddings, target_embeddings.t()) / self.temperature
    
        # 构造正样本Mask（对角线且标签为1）
        pos_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device) 
        pos_mask = pos_mask & (labels == 1).unsqueeze(1)
    
        # 构造负样本Mask（非对角线且标签为0）
        neg_mask = ~torch.eye(labels.size(0), dtype=torch.bool, device=labels.device) 
        neg_mask = neg_mask & (labels == 0).unsqueeze(0)
    
        # 计算正负样本Logits
        pos_logits = sim_matrix[pos_mask]
        neg_logits = sim_matrix[neg_mask].view(sim_matrix.size(0), -1)  # 形状(batch_size, num_neg)
    
        # 计算InfoNCE损失
        logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=1)
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
    
        return loss