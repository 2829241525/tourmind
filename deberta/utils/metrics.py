

from typing import Dict, List, Union, Tuple
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class MetricsCalculator:
    """评估指标计算器"""
    
    def __init__(self):
        pass

    def calculate_metrics(self, predictions, labels):
        """
        计算评估指标
        Args:
            predictions: 模型预测的概率值
            labels: 真实标签

        Returns:
            metrics: 包含各项指标的字典
            summary: 指标摘要字符串
        """
        # 将预测概率转换为二进制预测
        binary_predictions = (predictions >= 0.5).astype(int)
        
        # 计算各项指标
        accuracy = accuracy_score(labels, binary_predictions)
        precision = precision_score(labels, binary_predictions)
        recall = recall_score(labels, binary_predictions)
        f1 = f1_score(labels, binary_predictions)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

        summary = f'Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}'

        return metrics, summary

class AverageMeter:
    """计算并存储平均值和当前值"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 