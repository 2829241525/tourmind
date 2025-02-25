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
    
    def __init__(
        self,
        log_dir: str,
        plot_dir: str,
        real_time_plot: bool = False,
        plot_interval: int = 1
    ):
        """
        初始化日志记录器
        Args:
            log_dir: 日志保存目录
            plot_dir: 图表保存目录
            real_time_plot: 是否实时绘制图表
            plot_interval: 实时绘制的间隔（轮数）
        """
        self.log_dir = log_dir
        self.plot_dir = plot_dir
        self.real_time_plot = real_time_plot
        self.plot_interval = plot_interval
        
        # 创建保存目录
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(plot_dir, exist_ok=True)
        
        # 初始化日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_log_{timestamp}.json")
        self.text_log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # 初始化训练记录
        self.training_history = {
            'train_loss': [],
            'valid_loss': [],
            'steps': [],
            'epochs': [],
            'learning_rates': [],
            'time_per_step': [],
            'batch_sizes': [],
            'grad_norms': [],
            'timestamp': []
        }
        
        # 记录训练开始时间
        self.start_time = datetime.now()
        self.last_step_time = self.start_time
        
        # 设置文本日志
        self.file_handler = logging.FileHandler(self.text_log_file)
        self.file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(self.file_handler)
        
        # 初始化实时绘图
        if real_time_plot:
            self.plot_queue = Queue()
            self.plot_thread = threading.Thread(target=self._plot_worker, daemon=True)
            self.plot_thread.start()
        
    def log_step(self, step: int, epoch: int, train_loss: float, learning_rate: float, 
                batch_size: int = None, grad_norm: float = None):
        """记录训练步骤信息"""
        current_time = datetime.now()
        step_time = (current_time - self.last_step_time).total_seconds()
        self.last_step_time = current_time
        
        # 更新训练历史
        self.training_history['train_loss'].append(train_loss)
        self.training_history['steps'].append(step)
        self.training_history['epochs'].append(epoch)
        self.training_history['learning_rates'].append(learning_rate)
        self.training_history['time_per_step'].append(step_time)
        self.training_history['timestamp'].append(current_time.strftime("%Y-%m-%d %H:%M:%S"))
        
        if batch_size is not None:
            self.training_history['batch_sizes'].append(batch_size)
        if grad_norm is not None:
            self.training_history['grad_norms'].append(grad_norm)
        
        # 记录文本日志
        elapsed_time = (current_time - self.start_time).total_seconds() / 60  # 转换为分钟
        log_message = (
            f"Step {step:5d} - "
            f"Epoch {epoch:3d} - "
            f"Loss: {train_loss:.4f} - "
            f"LR: {learning_rate:.2e} - "
            f"Step Time: {step_time:.2f}s - "
            f"Total Time: {elapsed_time:.1f}min"
        )
        if grad_norm is not None:
            log_message += f" - Grad Norm: {grad_norm:.4f}"
        
        logger.info(log_message)
        
        # 保存JSON日志
        self._save_history()
        
    def log_validation(self, valid_loss: float, metrics: dict = None):
        """记录验证结果"""
        self.training_history['valid_loss'].append(valid_loss)
        if metrics:
            for key, value in metrics.items():
                if key not in self.training_history:
                    self.training_history[key] = []
                self.training_history[key].append(value)
        
        # 记录文本日志
        log_message = f"Validation - Loss: {valid_loss:.4f}"
        if metrics:
            for key, value in metrics.items():
                if isinstance(value, float):
                    log_message += f" - {key}: {value:.4f}"
                else:
                    log_message += f" - {key}: {value}"
        
        logger.info(log_message)
        
        # 保存JSON日志
        self._save_history()
        
    def _save_history(self):
        """保存训练历史到JSON文件"""
        # 转换datetime对象为字符串
        history_to_save = {
            k: ([str(x) if isinstance(x, datetime) else x for x in v] 
                if isinstance(v, list) else v)
            for k, v in self.training_history.items()
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(history_to_save, f, indent=4)
            
    def get_summary(self):
        """获取训练总结"""
        current_time = datetime.now()
        total_time = (current_time - self.start_time).total_seconds() / 60  # 转换为分钟
        
        return {
            'total_steps': len(self.training_history['steps']),
            'total_epochs': max(self.training_history['epochs']) + 1,
            'total_time_minutes': total_time,
            'avg_step_time': np.mean(self.training_history['time_per_step']),
            'train_loss': {
                'final': self.training_history['train_loss'][-1],
                'mean': np.mean(self.training_history['train_loss']),
                'std': np.std(self.training_history['train_loss']),
                'min': min(self.training_history['train_loss']),
                'max': max(self.training_history['train_loss'])
            },
            'valid_loss': {
                'final': self.training_history['valid_loss'][-1],
                'best': min(self.training_history['valid_loss']),
                'mean': np.mean(self.training_history['valid_loss'])
            } if self.training_history['valid_loss'] else None,
            'learning_rate': {
                'initial': self.training_history['learning_rates'][0],
                'final': self.training_history['learning_rates'][-1]
            }
        }
    
    def _plot_metrics(self):
        """绘制训练指标图表"""
        plt.style.use('seaborn')
        
        # 创建子图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)
        
        # 绘制损失曲线
        ax1.plot(self.training_history['steps'], self.training_history['train_loss'], 'b-', label='Train Loss')
        ax1.set_title('Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制F1分数
        f1_scores = [m['f1'] for m in self.training_history['val_metrics']]
        ax2.plot(self.training_history['steps'], f1_scores, 'g-', label='F1 Score')
        ax2.set_title('F1 Score')
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('F1')
        ax2.legend()
        ax2.grid(True)
        
        # 绘制学习率
        ax3.plot(self.training_history['steps'], self.training_history['learning_rates'], 'r-', label='Learning Rate')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('LR')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # 绘制准确率和召回率
        accuracies = [m['accuracy'] for m in self.training_history['val_metrics']]
        recalls = [m['recall'] for m in self.training_history['val_metrics']]
        ax4.plot(self.training_history['steps'], accuracies, 'c-', label='Accuracy')
        ax4.plot(self.training_history['steps'], recalls, 'm-', label='Recall')
        ax4.set_title('Accuracy & Recall')
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True)
        
        # 调整布局并保存
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, 'training_metrics.png'))
        plt.close()
    
    def _plot_worker(self):
        """实时绘图工作线程"""
        while True:
            try:
                # 等待绘图信号
                self.plot_queue.get()
                # 绘制图表
                self._plot_metrics()
                # 标记任务完成
                self.plot_queue.task_done()
            except Exception as e:
                logger.error(f"绘图线程出错: {str(e)}")
    
    def close(self):
        """关闭日志记录器"""
        # 移除文件处理器
        logger.removeHandler(self.file_handler)
        self.file_handler.close()
        
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
        ax1.plot(self.training_history['steps'], self.training_history['train_loss'], label='Training Loss')
        if self.training_history['valid_loss']:
            # 计算验证loss对应的步数
            valid_steps = np.linspace(min(self.training_history['steps']), max(self.training_history['steps']), 
                                    len(self.training_history['valid_loss']))
            ax1.plot(valid_steps, self.training_history['valid_loss'], label='Validation Loss')
        ax1.set_xlabel('Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 绘制学习率曲线
        if self.training_history['learning_rates']:
            ax2.plot(self.training_history['steps'], self.training_history['learning_rates'], label='Learning Rate')
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
        
        plt.hist(self.training_history['train_loss'], bins=50, alpha=0.5, label='Training Loss')
        if self.training_history['valid_loss']:
            plt.hist(self.training_history['valid_loss'], bins=50, alpha=0.5, label='Validation Loss')
            
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