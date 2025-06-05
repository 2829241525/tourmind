# -*- coding: utf-8 -*-

import os
import json
import logging
import time
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
import threading
from queue import Queue
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class TrainingLogger:
    """训练日志记录器"""
    
    def __init__(self, log_dir: str, plot_dir: str = None, real_time_plot: bool = False):
        """
        初始化训练日志记录器
        Args:
            log_dir: 日志保存目录
            plot_dir: 图表保存目录
            real_time_plot: 是否实时绘制图表
        """
        self.log_dir = log_dir
        self.plot_dir = plot_dir or os.path.join(log_dir, 'plots')
        self.real_time_plot = real_time_plot
        
        # 创建目录
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 初始化历史记录
        self.history = {
            'step': [],
            'epoch': [],
            'train_loss': [],
            'valid_loss': [],
            'learning_rate': [],
            'batch_size': [],
            'grad_norm': [],
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # 设置日志文件
        self.log_file = os.path.join(self.log_dir, 'training.log')
        
    def log_step(self, step: int, epoch: int, train_loss: float, learning_rate: float, 
                 batch_size: int, grad_norm: float, accuracy: float = None, 
                 precision: float = None, recall: float = None, f1: float = None):
        """
        记录训练步骤信息
        Args:
            step: 当前步数
            epoch: 当前轮数
            train_loss: 训练损失
            learning_rate: 学习率
            batch_size: 批次大小
            grad_norm: 梯度范数
            accuracy: 准确率（可选）
            precision: 精确率（可选）
            recall: 召回率（可选）
            f1: F1分数（可选）
        """
        self.history['step'].append(step)
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(train_loss)
        self.history['learning_rate'].append(learning_rate)
        self.history['batch_size'].append(batch_size)
        self.history['grad_norm'].append(grad_norm)
        
        # 记录性能指标（如果提供）
        if accuracy is not None:
            self.history['accuracy'].append(accuracy)
        if precision is not None:
            self.history['precision'].append(precision)
        if recall is not None:
            self.history['recall'].append(recall)
        if f1 is not None:
            self.history['f1'].append(f1)
        
        # 保存到日志文件
        with open(self.log_file, 'a') as f:
            log_entry = {
                'step': step,
                'epoch': epoch,
                'train_loss': train_loss,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'grad_norm': grad_norm
            }
            if accuracy is not None:
                log_entry['accuracy'] = accuracy
            if precision is not None:
                log_entry['precision'] = precision
            if recall is not None:
                log_entry['recall'] = recall
            if f1 is not None:
                log_entry['f1'] = f1
            
            f.write(json.dumps(log_entry) + '\n')
        
        # 如果启用实时绘图，更新图表
        if self.real_time_plot:
            self._plot_metrics()
    
    def log_validation(self, valid_loss: float):
        """记录验证损失"""
        self.history['valid_loss'].append(valid_loss)
    
    def _plot_metrics(self):
        """绘制训练指标图表"""
        plt.figure(figsize=(15, 10))
        
        # 绘制损失曲线
        plt.subplot(2, 2, 1)
        plt.plot(self.history['step'], self.history['train_loss'], label='Train Loss')
        if self.history['valid_loss']:
            valid_steps = np.linspace(0, max(self.history['step']), len(self.history['valid_loss']))
            plt.plot(valid_steps, self.history['valid_loss'], label='Valid Loss')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        
        # 绘制学习率曲线
        plt.subplot(2, 2, 2)
        plt.plot(self.history['step'], self.history['learning_rate'])
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        
        # 绘制性能指标
        if self.history['accuracy']:
            plt.subplot(2, 2, 3)
            plt.plot(self.history['step'], self.history['accuracy'], label='Accuracy')
            plt.plot(self.history['step'], self.history['precision'], label='Precision')
            plt.plot(self.history['step'], self.history['recall'], label='Recall')
            plt.plot(self.history['step'], self.history['f1'], label='F1')
            plt.xlabel('Step')
            plt.ylabel('Score')
            plt.title('Performance Metrics')
            plt.legend()
        
        # 绘制梯度范数
        plt.subplot(2, 2, 4)
        plt.plot(self.history['step'], self.history['grad_norm'])
        plt.xlabel('Step')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()
    
    def _save_history(self):
        """保存训练历史到文件"""
        history_file = os.path.join(self.log_dir, 'history.json')
        with open(history_file, 'w') as f:
            json.dump(self.history, f)
            
    def get_summary(self):
        """获取训练总结"""
        current_time = datetime.now()
        total_time = (current_time - self.start_time).total_seconds() / 60  # 转换为分钟
        
        return {
            'total_steps': len(self.history['step']),
            'total_epochs': max(self.history['epoch']) + 1,
            'total_time_minutes': total_time,
            'avg_step_time': np.mean(self.history['time_per_step']),
            'train_loss': {
                'final': self.history['train_loss'][-1],
                'mean': np.mean(self.history['train_loss']),
                'std': np.std(self.history['train_loss']),
                'min': min(self.history['train_loss']),
                'max': max(self.history['train_loss'])
            },
            'valid_loss': {
                'final': self.history['valid_loss'][-1],
                'best': min(self.history['valid_loss']),
                'mean': np.mean(self.history['valid_loss'])
            } if self.history['valid_loss'] else None,
            'learning_rate': {
                'initial': self.history['learning_rate'][0],
                'final': self.history['learning_rate'][-1]
            }
        }
    
    def close(self):
        """关闭日志记录器"""
        # 最后一次绘图
        if self.real_time_plot:
            self._plot_metrics() 

    def plot_training_history(self, save_path=None):
        """
        绘制训练历史图表
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(12, 8))
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # 绘制loss曲线
        ax1.plot(self.history['step'], self.history['train_loss'], label='Training Loss')
        if self.history['valid_loss']:
            # 计算验证loss对应的步数
            valid_steps = np.linspace(min(self.history['step']), max(self.history['step']), 
                                    len(self.history['valid_loss']))
            ax1.plot(valid_steps, self.history['valid_loss'], label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率曲线
        if self.history['learning_rate']:
            ax2.plot(self.history['step'], self.history['learning_rate'], label='Learning Rate')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Learning Rate')
            ax2.set_title('Learning Rate Schedule')
            ax2.legend()
            ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            
    def plot_loss_distribution(self, save_path=None):
        """
        绘制loss分布图
        Args:
            save_path: 图表保存路径，如果为None则显示图表
        """
        plt.figure(figsize=(10, 6))
        
        plt.hist(self.history['train_loss'], bins=50, alpha=0.5, label='Training Loss')
        if self.history['valid_loss']:
            plt.hist(self.history['valid_loss'], bins=50, alpha=0.5, label='Validation Loss')
            
        plt.xlabel('Loss Value')
        plt.ylabel('Frequency')
        plt.title('Loss Distribution')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

class Logger:
    def __init__(self, config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """设置日志配置"""
        os.makedirs(self.config["output"]["log_dir"], exist_ok=True)
        
        # 创建带时间戳的日志文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.config["output"]["log_dir"], f"train_{timestamp}.log")
        
        # 配置日志格式
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def info(self, message):
        """记录信息级别的日志"""
        self.logger.info(message)
        
    def warning(self, message):
        """记录警告级别的日志"""
        self.logger.warning(message)
        
    def error(self, message):
        """记录错误级别的日志"""
        self.logger.error(message)
        
    def debug(self, message):
        """记录调试级别的日志"""
        self.logger.debug(message)
        
    def log_training_info(self, config):
        """记录训练配置信息"""
        self.info("=" * 50)
        self.info("训练配置信息:")
        self.info(f"模型名称: {config['model']['name']}")
        self.info(f"预训练模型路径: {config['model']['pretrained_path']}")
        self.info(f"最大序列长度: {config['model']['max_length']}")
        self.info(f"池化方式: {config['model']['pooling']}")
        self.info("-" * 30)
        self.info("训练参数:")
        self.info(f"批次大小: {config['training']['batch_size']}")
        self.info(f"训练轮数: {config['training']['num_epochs']}")
        self.info(f"学习率: {config['training']['learning_rate']}")
        self.info(f"预热步数: {config['training']['warmup_steps']}")
        self.info(f"权重衰减: {config['training']['weight_decay']}")
        self.info(f"温度参数: {config['training']['temperature']}")
        self.info("=" * 50)
        
    def log_epoch_metrics(self, epoch, num_epochs, train_loss, valid_metrics):
        """记录每轮训练的指标"""
        self.info(
            f"\nEpoch {epoch+1}/{num_epochs} 完成:\n"
            f"  训练损失: {train_loss:.4f}\n"
            f"  验证损失: {valid_metrics['loss']:.4f}\n"
            f"  验证相似度: {valid_metrics['avg_similarity']:.4f} ± {valid_metrics['std_similarity']:.4f}"
        )
        
    def log_batch_metrics(self, epoch, batch_idx, num_batches, metrics):
        """记录每个批次的指标"""
        if (batch_idx + 1) % (num_batches // 10) == 0:  # 每10%记录一次
            self.info(
                f"Epoch {epoch+1} - Batch {batch_idx+1}/{num_batches} "
                f"(进度: {(batch_idx+1)/num_batches*100:.1f}%):\n"
                f"  当前损失: {metrics['loss']:.4f}\n"
                f"  平均损失: {metrics['avg_loss']:.4f}\n"
                f"  学习率: {metrics['lr']:.2e}"
            )
            
    def log_best_model(self, epoch, valid_loss):
        """记录最佳模型信息"""
        self.info(f"✨ 保存最佳模型 (Epoch {epoch+1}, 验证损失: {valid_loss:.4f})")
        
    def log_training_complete(self, best_epoch, best_loss):
        """记录训练完成信息"""
        self.info("=" * 50)
        self.info(f"训练完成！最佳模型在第 {best_epoch} 轮，验证损失: {best_loss:.4f}")
        self.info("=" * 50) 