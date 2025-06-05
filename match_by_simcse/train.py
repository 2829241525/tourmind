# -*- coding: utf-8 -*-
"""
SimCSE模型训练脚本
使用princeton-nlp/sup-simcse-bert-base-uncased模型和InfoNCE损失
"""

import os
import json
import logging
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import gc
from torch.cuda.amp import autocast, GradScaler
import numpy as np

from models.loss import SupervisedInfoNCELoss
from data.dataset import RoomMatchDataset
from utils.logger import TrainingLogger

# 配置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            os.path.dirname(__file__), 'logs', 'train.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def print_gpu_memory():
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(
                i).total_memory / (1024**3)
            reserved_mem = torch.cuda.memory_reserved(i) / (1024**3)
            allocated_mem = torch.cuda.memory_allocated(i) / (1024**3)
            free_mem = total_mem - reserved_mem
            logger.info(f"GPU {i} 显存状态:")
            logger.info(f"  总显存: {total_mem:.2f} GB")
            logger.info(f"  已分配: {allocated_mem:.2f} GB")
            logger.info(f"  已预留: {reserved_mem:.2f} GB")
            logger.info(f"  可用: {free_mem:.2f} GB")


def clear_gpu_memory():
    """清理GPU显存"""
    if torch.cuda.is_available():
        # 清理 PyTorch 的 CUDA 缓存
        torch.cuda.empty_cache()
        # 强制进行垃圾回收
        gc.collect()
        logger.info("已清理GPU显存和Python缓存")


class SimCSETrainer:
    def __init__(self, config_path: str):
        """
        初始化训练器
        Args:
            config_path: 配置文件路径
        """
        # 设置 PyTorch 内存分配器
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:2048'

        with open(config_path) as f:
            self.config = json.load(f)

        # 获取当前文件所在目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 修正数据文件路径为绝对路径
        def process_file_paths(file_str):
            """处理逗号分隔的文件路径字符串"""
            if not file_str:
                return ""
            paths = file_str.split(',')
            return ','.join(os.path.join(self.base_dir, p.strip()) for p in paths)

        self.config['train_file'] = process_file_paths(
            self.config['train_file'])
        self.config['valid_file'] = process_file_paths(
            self.config['valid_file'])
        self.config['save_dir'] = os.path.join(
            self.base_dir, self.config['save_dir'])

        # 指定使用 GPU 1
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")

        # 初始化日志记录器
        log_dir = os.path.join(self.base_dir, 'logs')
        self.logger = TrainingLogger(
            log_dir=log_dir,
            plot_dir=os.path.join(log_dir, 'reports'),
            real_time_plot=False
        )

        # 训练开始前清理显存并打印状态
        clear_gpu_memory()
        print_gpu_memory()

        # 检查是否存在之前训练的最佳模型
        best_model_path = os.path.join(self.config['save_dir'], 'best_model')
        if os.path.exists(best_model_path):
            logger.info(f"找到已有的最佳模型，正在加载: {best_model_path}")
            self.model = AutoModel.from_pretrained(
                best_model_path, local_files_only=True).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(
                best_model_path, local_files_only=True)
        else:
            logger.info("未找到已有模型，从预训练模型初始化")
            # 使用本地预训练模型
            pretrained_path = os.path.join(
                self.base_dir, self.config['pretrained_model'])

            self.tokenizer = AutoTokenizer.from_pretrained(
                pretrained_path, local_files_only=True)
            self.model = AutoModel.from_pretrained(
                pretrained_path, local_files_only=True).to(self.device)

            # 启用梯度检查点以节省显存
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点以节省显存")

        # 初始化损失函数
        self.criterion = SupervisedInfoNCELoss(
            temperature=self.config.get('temperature', 0.05)).to(self.device)

        # 初始化优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-5),
            weight_decay=self.config.get('weight_decay', 0.01)
        )

        # 初始化混合精度训练的 scaler
        self.scaler = GradScaler()

    def encode_batch(self, texts, train=True):
        """将文本批量编码为向量"""
        max_length = self.config.get('max_length', 256)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        if train:
            outputs = self.model(**encoded)
        else:
            with torch.no_grad():
                outputs = self.model(**encoded)
        # 使用[CLS]标记的输出作为文本表示
        embeddings = outputs.last_hidden_state[:, 0]
        return embeddings

    def train_epoch(self, train_loader):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        # 获取梯度累积步数
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Training')):
            # 使用混合精度训练
            with autocast():
                # 获取源文本和目标文本的表示
                anchor_embeddings = self.encode_batch(
                    batch['spl_text'], train=True)
                target_embeddings = self.encode_batch(
                    batch['s_text'], train=True)

                # 获取标签并转换为tensor
                labels = batch['label'].to(self.device)

                # 计算损失
                loss = self.criterion(
                    anchor_embeddings, target_embeddings, labels)

                # 根据梯度累积步数缩放损失
                loss = loss / grad_accum_steps

            # 使用 scaler 进行反向传播
            self.scaler.scale(loss).backward()

            # 记录未缩放的loss
            total_loss += loss.item() * grad_accum_steps

            # 记录训练状态
            if batch_idx % self.config.get('logging_steps', 10) == 0:
                step = len(train_loader) * self.current_epoch + batch_idx
                # 计算梯度范数
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0).item()

                self.logger.log_step(
                    step=step,
                    epoch=self.current_epoch,
                    train_loss=loss.item() * grad_accum_steps,
                    learning_rate=self.scheduler.get_last_lr()[0],
                    batch_size=self.config.get('batch_size', 16),
                    grad_norm=grad_norm
                )

            # 在累积足够的梯度后更新参数
            if (batch_idx + 1) % grad_accum_steps == 0:
                # 使用 scaler 进行梯度裁剪和参数更新
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

        return total_loss / len(train_loader)

    def evaluate(self, valid_loader):
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        total_samples = 0

        with torch.no_grad():
            with autocast():
                for batch in tqdm(valid_loader, desc='Evaluating'):
                    anchor_embeddings = self.encode_batch(
                        batch['spl_text'], train=False)
                    target_embeddings = self.encode_batch(
                        batch['s_text'], train=False)
                    labels = batch['label'].to(self.device)

                    # 计算损失
                    loss = self.criterion(
                        anchor_embeddings, target_embeddings, labels)
                    total_loss += loss.item()

                    # 计算余弦相似度
                    similarities = torch.nn.functional.cosine_similarity(
                        anchor_embeddings, target_embeddings)

                    # 使用阈值0.5作为预测边界
                    predictions = (similarities > 0.5).float()

                    # 收集预测结果和标签
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    total_samples += len(labels)

        # 转换为numpy数组以便计算指标
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # 计算各项指标
        avg_loss = total_loss / len(valid_loader)

        # 计算正负样本数量
        positive_samples = np.sum(all_labels == 1)
        negative_samples = np.sum(all_labels == 0)
        pos_ratio = positive_samples / total_samples * 100
        neg_ratio = negative_samples / total_samples * 100

        # 计算准确率（总体、正样本、负样本）
        accuracy = np.mean(all_preds == all_labels)
        pos_accuracy = np.mean(
            all_preds[all_labels == 1] == 1) if positive_samples > 0 else 0
        neg_accuracy = np.mean(
            all_preds[all_labels == 0] == 0) if negative_samples > 0 else 0

        # 计算精确率、召回率和F1
        true_positives = np.sum((all_preds == 1) & (all_labels == 1))
        false_positives = np.sum((all_preds == 1) & (all_labels == 0))

        precision = true_positives / \
            (true_positives + false_positives) if (true_positives +
                                                   false_positives) > 0 else 0
        recall = true_positives / positive_samples if positive_samples > 0 else 0
        f1 = 2 * precision * recall / \
            (precision + recall) if (precision + recall) > 0 else 0

        # 记录评估结果
        logger.info(f"【验证集】评估结果 (Epoch {self.current_epoch + 1}):")
        logger.info(f"  样本总数: {total_samples}")
        logger.info(
            f"  正样本数: {positive_samples} ({pos_ratio:.2f}%), 负样本数: {negative_samples} ({neg_ratio:.2f}%)")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(
            f"  Accuracy: {accuracy:.4f} (正样本: {pos_accuracy:.4f}, 负样本: {neg_accuracy:.4f})")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")

        return avg_loss

    def save_checkpoint(self, epoch, valid_loss, is_best=False):
        """保存检查点"""
        # 创建保存目录
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # 保存训练状态到最新检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'valid_loss': valid_loss,
            'config': self.config
        }
        checkpoint_path = os.path.join(
            self.config['save_dir'], 'checkpoint.pt')
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存最新检查点到: {checkpoint_path}")

        # 如果是最佳模型，额外保存到best_model目录
        if is_best:
            best_model_dir = os.path.join(
                self.config['save_dir'], 'best_model')
            self.model.save_pretrained(best_model_dir)
            self.tokenizer.save_pretrained(best_model_dir)
            logger.info(f"保存最佳模型到: {best_model_dir}")

    def load_checkpoint(self):
        """加载检查点"""
        checkpoint_path = os.path.join(
            self.config['save_dir'], 'checkpoint.pt')
        if os.path.exists(checkpoint_path):
            logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if checkpoint['scheduler_state_dict'] and hasattr(self, 'scheduler'):
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_valid_loss = checkpoint['valid_loss']
            logger.info(f"从epoch {start_epoch}继续训练")
            return start_epoch, best_valid_loss
        return 0, float('inf')

    def train(self):
        """训练模型"""
        # 加载数据
        train_dataset = RoomMatchDataset(
            self.config['train_file'],
            is_train=True
        )
        valid_dataset = RoomMatchDataset(
            self.config['valid_file'],
            is_train=False
        )

        logger.info(f"Loaded {len(train_dataset)} training samples")
        logger.info(f"Loaded {len(valid_dataset)} validation samples")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        # 创建学习率调度器
        num_training_steps = len(train_loader) * \
            self.config.get('num_epochs', 10)
        num_warmup_steps = num_training_steps // 100  # 1%的步数用于warmup
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        # 加载检查点
        start_epoch, best_valid_loss = self.load_checkpoint()
        self.current_epoch = start_epoch
        self.best_valid_loss = best_valid_loss

        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting training...")
        logger.info(f"Total training steps: {num_training_steps}")
        logger.info(f"Warmup steps: {num_warmup_steps}")

        for epoch in range(start_epoch, self.config.get('num_epochs', 10)):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch(train_loader)

            # epoch 结束后清理显存
            clear_gpu_memory()
            print_gpu_memory()

            # 验证
            valid_loss = self.evaluate(valid_loader)

            # 记录验证loss
            self.logger.log_validation(valid_loss)

            # 保存训练历史
            self.logger._save_history()

            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'  Train Loss: {train_loss:.4f}')

            # 检查是否是最佳模型
            is_best = valid_loss < self.best_valid_loss
            if is_best:
                self.best_valid_loss = valid_loss
                logger.info(f'  发现最佳模型，验证损失: {self.best_valid_loss:.4f}')

            # 保存检查点（只保存最新的和最佳的）
            self.save_checkpoint(epoch, valid_loss, is_best)

            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            logger.info(f'  Current learning rate: {current_lr:.2e}')


def main():
    """主训练流程"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, 'config', 'config.json')
    trainer = SimCSETrainer(config_path)
    trainer.train()


if __name__ == '__main__':
    main()
