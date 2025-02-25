# -*- coding: utf-8 -*-
"""
SimCSE模型实现
这个模块实现了基于BERT的SimCSE模型，用于学习文本的语义表示
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class SimCSE(nn.Module):
    """
    SimCSE模型类
    
    实现了监督对比学习的编码器模型，支持两种池化策略：
    1. CLS池化：使用[CLS]token的表示作为整个序列的表示
    2. 平均池化：使用所有token的平均值作为序列表示
    
    Args:
        config (dict): 配置字典，包含模型参数和训练设置
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 加载预训练模型和分词器
        self.encoder = AutoModel.from_pretrained(config["model"]["pretrained_path"])
        self.tokenizer = AutoTokenizer.from_pretrained(config["model"]["pretrained_path"])
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播函数
        
        Args:
            input_ids (torch.Tensor): 输入序列的token ID，shape: [batch_size, seq_len]
            attention_mask (torch.Tensor, optional): 注意力掩码，shape: [batch_size, seq_len]
            
        Returns:
            torch.Tensor: 文本的语义表示向量，shape: [batch_size, hidden_size]
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # 根据配置选择池化策略
        if self.config["model"]["pooling"] == "cls":
            # 使用[CLS] token的表示
            embeddings = outputs.last_hidden_state[:, 0]  # [batch_size, hidden_size]
        elif self.config["model"]["pooling"] == "mean":
            # 使用所有token的平均值，需要考虑attention mask
            embeddings = (outputs.last_hidden_state * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
            
        return embeddings
    
    def encode(self, texts, batch_size=32, device="cuda"):
        """
        批量编码文本序列
        
        Args:
            texts (List[str]): 待编码的文本列表
            batch_size (int): 批处理大小
            device (str): 计算设备，'cuda'或'cpu'
            
        Returns:
            torch.Tensor: 文本的语义表示向量，shape: [len(texts), hidden_size]
        """
        all_embeddings = []
        
        # 分批处理文本
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # 对文本进行分词和编码
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.config["model"]["max_length"],
                return_tensors="pt"
            ).to(device)
            
            # 计算文本表示
            with torch.no_grad():
                embeddings = self.forward(**inputs)
                all_embeddings.append(embeddings.cpu())
                
        return torch.cat(all_embeddings, dim=0) 