# -*- coding: utf-8 -*-
"""
DeBERTa模型实现
这个模块实现了基于DeBERTa的文本编码器和分类器模型
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

class DeBERTaEncoder(nn.Module):
    """DeBERTa编码器封装类，用于提取文本特征"""
    def __init__(self, model_name: str):
        """
        初始化DeBERTa编码器
        
        Args:
            model_name: 预训练模型名称或路径
        """
        super().__init__()
        try:
            # 首先尝试从指定的路径加载
            if os.path.exists(model_name):
                self.model = AutoModel.from_pretrained(model_name)
                logger.info(f"成功从本地路径加载预训练模型: {model_name}")
            else:
                # 如果本地路径不存在，尝试从Hugging Face下载
                logger.warning(f"本地模型路径不存在: {model_name}")
                logger.info("尝试从Hugging Face下载模型...")
                try:
                    self.model = AutoModel.from_pretrained("microsoft/deberta-v3-base", trust_remote_code=True)
                    logger.info("成功从Hugging Face下载模型")
                    # 保存到本地
                    os.makedirs(os.path.dirname(model_name), exist_ok=True)
                    self.model.save_pretrained(model_name)
                    logger.info(f"已将模型保存到本地: {model_name}")
                except Exception as e:
                    logger.error(f"下载模型失败: {str(e)}")
                    raise
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
        
    def forward(self, **inputs):
        """
        前向传播函数
        
        Args:
            inputs: 输入数据，包括input_ids, attention_mask等
            
        Returns:
            torch.Tensor: 文本的向量表示，shape: [batch_size, hidden_size]
        """
        outputs = self.model(**inputs)
        # 使用[CLS]标记的输出作为文本表示
        return outputs.last_hidden_state[:, 0]

class DeBERTaClassifier(nn.Module):
    """DeBERTa分类器封装类，用于文本对分类任务"""
    def __init__(self, model_name: str, num_labels: int = 2, class_weights: list = None):
        """
        初始化DeBERTa分类器
        
        Args:
            model_name: 预训练模型名称或路径
            num_labels: 分类标签数量
            class_weights: 类别权重列表
        """
        super().__init__()
        try:
            # 首先尝试从指定的路径加载
            if os.path.exists(model_name):
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, 
                    num_labels=num_labels,
                    problem_type="single_label_classification"
                )
                logger.info(f"成功从本地路径加载预训练分类器: {model_name}")
            else:
                # 如果本地路径不存在，尝试从Hugging Face下载
                logger.warning(f"本地模型路径不存在: {model_name}")
                logger.info("尝试从Hugging Face下载模型...")
                try:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        "microsoft/deberta-v3-base", 
                        num_labels=num_labels,
                        trust_remote_code=True,
                        problem_type="single_label_classification"
                    )
                    logger.info("成功从Hugging Face下载模型")
                    # 保存到本地
                    os.makedirs(os.path.dirname(model_name), exist_ok=True)
                    self.model.save_pretrained(model_name)
                    logger.info(f"已将模型保存到本地: {model_name}")
                except Exception as e:
                    logger.error(f"下载模型失败: {str(e)}")
                    raise
                    
            # 设置类别权重
            if class_weights is not None:
                self.class_weights = torch.tensor(class_weights).float()
            else:
                self.class_weights = None
                
        except Exception as e:
            logger.error(f"加载模型失败: {str(e)}")
            raise
    
    def forward(self, **inputs):
        """
        前向传播函数
        
        Args:
            inputs: 输入数据，包括input_ids, attention_mask, labels等
            
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutput: 分类器输出
        """
        outputs = self.model(**inputs)
        
        # 如果有类别权重且在训练模式下，修改损失计算
        if self.class_weights is not None and self.training and 'labels' in inputs:
            logits = outputs.logits
            labels = inputs['labels']
            # 使用交叉熵损失并应用类别权重
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            outputs.loss = loss
            
        return outputs

def create_model(model_type: str, model_name: str, num_labels: int = 2, class_weights: list = None):
    """
    创建模型的工厂函数
    
    Args:
        model_type: 模型类型，'encoder'或'classifier'
        model_name: 预训练模型名称或路径
        num_labels: 分类标签数量，仅在model_type为'classifier'时使用
        class_weights: 类别权重列表，仅在model_type为'classifier'时使用
        
    Returns:
        nn.Module: 创建的模型
    """
    if model_type == 'encoder':
        return DeBERTaEncoder(model_name)
    elif model_type == 'classifier':
        return DeBERTaClassifier(model_name, num_labels, class_weights)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}") 