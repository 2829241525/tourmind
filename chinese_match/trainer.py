# -*- coding: utf-8 -*-
"""
DeBERTa训练器模块
实现了用于训练DeBERTa模型的训练器类
"""

import os
import json
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from pathlib import Path
import gc
from torch.amp import autocast, GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.model import create_model
from models.loss import create_loss_fn
from data.dataset import RoomMatchValidationDataset
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
    """打印当前GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(
                i).total_memory / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            free = total_memory - reserved
            logger.info(
                f"GPU {i}: 总内存: {total_memory:.2f}GB, 已分配: {allocated:.2f}GB, 已保留: {reserved:.2f}GB, 可用: {free:.2f}GB")
    else:
        logger.info("没有可用的GPU")


def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("已清理GPU内存")


class DeBERTaTrainer:
    """DeBERTa模型训练器"""

    def __init__(self, config_path: str, config_override=None):
        """
        初始化训练器

        Args:
            config_path: 配置文件路径
            config_override: 覆盖配置文件的参数字典
        """
        # 加载配置文件
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # 如果提供了覆盖参数，则应用这些参数
        if config_override is not None:
            self.config.update(config_override)
            logger.info(f"应用配置覆盖: {config_override}")

        # 设置基础目录
        self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 处理文件路径
        def process_file_paths(file_str):
            if isinstance(file_str, str):
                return os.path.join(self.base_dir, file_str)
            elif isinstance(file_str, list):
                return [os.path.join(self.base_dir, f) for f in file_str]
            return file_str

        # 处理配置中的文件路径
        for key in ['train_file', 'valid_file']:
            if key in self.config:
                self.config[key] = process_file_paths(self.config[key])

        # 确保保存目录存在
        self.config['save_dir'] = os.path.join(
            self.base_dir, self.config['save_dir'])
        os.makedirs(self.config['save_dir'], exist_ok=True)

        # 指定使用 GPU
        self.device = torch.device(
            'cuda:1' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")

        # 初始化日志记录器
        log_dir = os.path.join(self.base_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = TrainingLogger(
            log_dir=log_dir,
            plot_dir=os.path.join(log_dir, 'reports'),
            real_time_plot=False
        )

        # 使用更小的学习率防止梯度爆炸和NaN问题
        self.default_learning_rate = self.config.get('learning_rate', 2e-5)
        if self.config.get('reduce_learning_rate_for_nan', False):
            self.config['learning_rate'] = self.config.get(
                'learning_rate', 2e-5) * 0.1
            logger.info(
                f"启用降低学习率以防止NaN问题，学习率从 {self.default_learning_rate} 降低到 {self.config['learning_rate']}")

        # 训练开始前清理显存并打印状态
        clear_gpu_memory()
        print_gpu_memory()

        # 使用cross_entropy损失函数
        self.loss_type = 'cross_entropy'
        logger.info(f"使用损失函数类型: {self.loss_type}")

        # 检查是否存在之前训练的最佳模型
        best_model_path = os.path.join(self.config['save_dir'], 'best_model')
        if os.path.exists(best_model_path):
            logger.info(f"找到已有的最佳模型，正在加载: {best_model_path}")
            class_weights = self.config.get('class_weights', None)
            self.model = create_model(
                'classifier', best_model_path, num_labels=2, class_weights=class_weights)
            self.tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        else:
            logger.info("未找到已有模型，从预训练模型初始化")
            # 使用配置文件中指定的预训练模型路径
            model_path = self.config.get(
                'pretrained_model', "microsoft/deberta-v3-base")
            class_weights = self.config.get('class_weights', None)
            self.model = create_model(
                'classifier', model_path, num_labels=2, class_weights=class_weights)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 将模型移动到设备上
        self.model.to(self.device)

        # 使用模型的内置损失函数
        self.criterion = None

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=self.config.get('adam_epsilon', 1e-8)
        )

        # 初始化混合精度训练的 scaler
        self.scaler = GradScaler()

        # 输出fp16配置状态
        fp16_enabled = self.config.get('fp16', False)
        logger.info(f"混合精度训练(FP16): {'启用' if fp16_enabled else '禁用'}")

        # 早停相关参数
        if 'early_stopping' in self.config and isinstance(self.config['early_stopping'], dict):
            self.early_stopping_patience = self.config['early_stopping'].get(
                'patience', 3)
            self.min_delta = self.config['early_stopping'].get(
                'min_delta', 1e-4)
        else:
            self.early_stopping_patience = self.config.get(
                'early_stopping_patience', 3)
            self.min_delta = self.config.get('min_delta', 1e-4)

        self.best_valid_loss = float('inf')
        self.current_epoch = 0

    def encode_text(self, texts, train=True):
        """
        将单个文本编码为向量

        Args:
            texts: 文本列表
            train: 是否处于训练模式

        Returns:
            torch.Tensor: 文本的向量表示
        """
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
            embeddings = self.model(**encoded)
        else:
            with torch.no_grad():
                embeddings = self.model(**encoded)
        return embeddings

    def train_epoch(self, train_loader):
        """
        训练一个epoch

        Args:
            train_loader: 训练数据加载器

        Returns:
            float: 平均训练损失
        """
        self.model.train()
        total_loss = 0

        # 获取梯度累积步数
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        # 获取最大梯度范数
        max_grad_norm = self.config.get('max_grad_norm', 1.0)

        # 是否使用混合精度训练（从配置文件读取）
        use_fp16 = self.config.get('fp16', False)

        # 获取日志记录步数
        logging_steps = self.config.get('logging_steps', 100)

        # 用于计算指标的变量
        step_loss = 0.0
        all_predictions = []
        all_labels = []
        global_step = 0

        # 使用tqdm显示进度
        progress_bar = tqdm(
            train_loader,
            desc=f'Epoch {self.current_epoch+1}/{self.config.get("num_epochs", 5)}',
            position=0,
            leave=True
        )

        for batch_idx, batch in enumerate(progress_bar):
            # 使用混合精度训练
            if use_fp16:
                with autocast(device_type='cuda'):
                    # 准备输入数据
                    inputs = self.tokenizer(
                        batch['text1'],
                        text_pair=batch['text2'],
                        padding=True,
                        truncation=True,
                        max_length=self.config.get('max_length', 256),
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device)
                              for k, v in inputs.items()}
                    labels = batch['label'].to(self.device)
                    inputs['labels'] = labels

                    # 前向传播
                    outputs = self.model(**inputs)
                    loss = outputs.loss

                    # 收集预测和标签用于计算指标
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                    # 根据梯度累积步数缩放损失
                    loss = loss / grad_accum_steps
            else:
                # 准备输入数据
                inputs = self.tokenizer(
                    batch['text1'],
                    text_pair=batch['text2'],
                    padding=True,
                    truncation=True,
                    max_length=self.config.get('max_length', 256),
                    return_tensors='pt'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = batch['label'].to(self.device)
                inputs['labels'] = labels

                # 前向传播
                outputs = self.model(**inputs)
                loss = outputs.loss

                # 收集预测和标签用于计算指标
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # 根据梯度累积步数缩放损失
                loss = loss / grad_accum_steps

            # 反向传播
            if use_fp16:
                # 使用 scaler 进行反向传播
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # 记录未缩放的loss
            batch_loss = loss.item() * grad_accum_steps
            total_loss += batch_loss
            step_loss += batch_loss

            # 检查梯度是否有NaN值
            has_nan_grad = False
            for name, param in self.model.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    has_nan_grad = True
                    logger.warning(f"检测到NaN梯度，参数名: {name}")

            # 如果有NaN梯度，跳过本次更新
            if has_nan_grad:
                self.optimizer.zero_grad()
                logger.warning("由于检测到NaN梯度，已跳过本次参数更新")
                continue

            # 更新进度条
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                # 使用恒定学习率时，显示优化器的学习率
                current_lr = self.optimizer.param_groups[0]['lr']

            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })

            # 在累积足够的梯度后更新参数
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_fp16:
                    # 使用 scaler 进行梯度裁剪和参数更新
                    # 确保scale已经被使用，否则不调用unscale_
                    try:
                        # 梯度裁剪前先解除缩放
                        self.scaler.unscale_(self.optimizer)

                        # 梯度裁剪
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_grad_norm)

                        # 更新参数
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    except RuntimeError as e:
                        # 处理缩放器错误
                        logger.warning(f"梯度缩放器发生错误: {str(e)}")
                        # 清空优化器梯度
                        self.optimizer.zero_grad()
                        # 继续下一个批次
                        continue
                else:
                    # 不使用 scaler 进行梯度裁剪和参数更新
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm)
                    self.optimizer.step()

                # 更新学习率
                if hasattr(self, 'scheduler') and self.scheduler is not None:
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # 更新全局步数
                global_step += 1

                # 每logging_steps步输出一次训练指标
                if global_step % logging_steps == 0 and len(all_predictions) > 0:
                    # 计算训练指标
                    accuracy = accuracy_score(all_labels, all_predictions)
                    precision = precision_score(
                        all_labels, all_predictions, zero_division=0)
                    recall = recall_score(
                        all_labels, all_predictions, zero_division=0)
                    f1 = f1_score(all_labels, all_predictions, zero_division=0)

                    # 计算平均损失
                    avg_step_loss = step_loss / logging_steps

                    # 输出训练指标
                    logger.info(
                        f"【训练集】步数: {global_step}, 损失: {avg_step_loss:.4f}, 准确率: {accuracy:.4f}, 精确率: {precision:.4f}, 召回率: {recall:.4f}, F1: {f1:.4f}")

                    # 重置指标计算变量
                    step_loss = 0.0
                    all_predictions = []
                    all_labels = []

        return total_loss / len(train_loader)

    def evaluate(self, valid_loader):
        """
        评估模型

        Args:
            valid_loader: 验证数据加载器

        Returns:
            float: 平均验证损失
        """
        self.model.eval()

        total_loss = 0
        all_predictions = []
        all_labels = []

        # 是否使用混合精度训练
        use_fp16 = self.config.get('fp16', False)

        with torch.no_grad():
            if use_fp16:
                with autocast(device_type='cuda'):
                    for batch in tqdm(valid_loader, desc='验证中'):
                        # 准备输入数据
                        inputs = self.tokenizer(
                            batch['text1'],
                            text_pair=batch['text2'],
                            padding=True,
                            truncation=True,
                            max_length=self.config.get('max_length', 256),
                            return_tensors='pt'
                        )
                        inputs = {k: v.to(self.device)
                                  for k, v in inputs.items()}
                        labels = batch['label'].to(self.device)
                        inputs['labels'] = labels

                        # 前向传播
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                        total_loss += loss.item() * len(labels)

                        # 获取预测结果
                        logits = outputs.logits
                        predictions = torch.argmax(logits, dim=1)

                        # 收集预测结果和标签
                        all_predictions.extend(predictions.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
            else:
                for batch in tqdm(valid_loader, desc='验证中'):
                    # 准备输入数据
                    inputs = self.tokenizer(
                        batch['text1'],
                        text_pair=batch['text2'],
                        padding=True,
                        truncation=True,
                        max_length=self.config.get('max_length', 256),
                        return_tensors='pt'
                    )
                    inputs = {k: v.to(self.device)
                              for k, v in inputs.items()}
                    labels = batch['label'].to(self.device)
                    inputs['labels'] = labels

                    # 前向传播
                    outputs = self.model(**inputs)
                    loss = outputs.loss
                    total_loss += loss.item() * len(labels)

                    # 获取预测结果
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)

                    # 收集预测结果和标签
                    all_predictions.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

        # 计算平均损失
        avg_loss = total_loss / len(valid_loader.dataset)

        # 计算指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels, all_predictions, zero_division=0)
        recall = recall_score(all_labels, all_predictions, zero_division=0)
        f1 = f1_score(all_labels, all_predictions, zero_division=0)

        # 计算正负样本数量
        pos_count = sum(all_labels)
        neg_count = len(all_labels) - pos_count
        pos_ratio = pos_count / len(all_labels) if len(all_labels) > 0 else 0

        # 计算正负样本的准确率
        pos_correct = sum([1 for i, label in enumerate(
            all_labels) if label == 1 and all_predictions[i] == 1])
        neg_correct = sum([1 for i, label in enumerate(
            all_labels) if label == 0 and all_predictions[i] == 0])
        pos_accuracy = pos_correct / pos_count if pos_count > 0 else 0
        neg_accuracy = neg_correct / neg_count if neg_count > 0 else 0

        # 输出详细的验证集评估结果
        logger.info("=" * 50)
        logger.info(f"【验证集】评估结果 (Epoch {self.current_epoch+1}):")
        logger.info(f"  样本总数: {len(all_labels)}")
        logger.info(
            f"  正样本数: {pos_count} ({pos_ratio:.2%}), 负样本数: {neg_count} ({1-pos_ratio:.2%})")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(
            f"  Accuracy: {accuracy:.4f} (正样本: {pos_accuracy:.4f}, 负样本: {neg_accuracy:.4f})")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1: {f1:.4f}")
        logger.info("=" * 50)

        return avg_loss

    def save_checkpoint(self, epoch, valid_loss):
        """
        保存检查点

        Args:
            epoch: 当前epoch
            valid_loss: 验证损失
        """
        # 创建保存目录
        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        # 保存检查点
        checkpoint_path = save_dir / 'checkpoint.pt'
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') and self.scheduler is not None else None,
            'valid_loss': valid_loss,
            'config': self.config
        }
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"保存检查点到: {checkpoint_path}")

        # 如果是最佳模型，则同时保存模型和tokenizer
        if valid_loss < self.best_valid_loss:
            best_model_dir = save_dir / 'best_model'
            best_model_dir.mkdir(parents=True, exist_ok=True)

            # 保存模型
            if hasattr(self.model, 'model'):
                # 如果是封装的模型，保存内部的transformers模型
                self.model.model.save_pretrained(best_model_dir)
            else:
                # 如果是直接的transformers模型，直接保存
                self.model.save_pretrained(best_model_dir)

            # 保存tokenizer
            self.tokenizer.save_pretrained(best_model_dir)
            logger.info(f"保存最佳模型到: {best_model_dir}")
            self.best_valid_loss = valid_loss

    def load_checkpoint(self):
        """
        加载检查点

        Returns:
            tuple: (开始epoch, 最佳验证损失)
        """
        checkpoint_path = os.path.join(
            self.config['save_dir'], 'checkpoint.pt')

        if os.path.exists(checkpoint_path):
            logger.info(f"加载检查点: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # 加载模型状态
            self.model.load_state_dict(checkpoint['model_state_dict'])

            # 加载优化器状态
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            # 加载调度器状态（如果存在）
            if hasattr(self, 'scheduler') and self.scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

            # 返回上次训练的epoch和验证损失
            return checkpoint['epoch'] + 1, checkpoint.get('valid_loss', float('inf'))
        else:
            logger.info("未找到检查点，从头开始训练")
            return 0, float('inf')

    def _cleanup_old_checkpoints(self, keep_num=3):
        """
        清理旧的检查点，只保留最近的几个

        Args:
            keep_num: 保留的检查点数量
        """
        save_dir = Path(self.config['save_dir'])
        checkpoints = list(save_dir.glob('checkpoint_step_*.pt'))

        # 按修改时间排序
        checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)

        # 删除旧的检查点
        for checkpoint in checkpoints[keep_num:]:
            try:
                os.remove(checkpoint)
                logger.info(f"删除旧检查点: {checkpoint}")
            except Exception as e:
                logger.warning(f"删除检查点失败: {checkpoint}, 错误: {str(e)}")

    def train(self):
        """
        训练模型
        """
        # 使用验证数据集
        train_dataset = RoomMatchValidationDataset(
            self.config['train_file'])
        valid_dataset = RoomMatchValidationDataset(
            self.config['valid_file'])

        logger.info(f"加载了 {len(train_dataset)} 个训练样本")
        logger.info(f"加载了 {len(valid_dataset)} 个验证样本")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 64),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.get('batch_size', 64) * 2,
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )

        # 创建学习率调度器
        total_steps = len(train_dataset) // self.config.get('batch_size', 64)
        total_steps = total_steps // self.config.get(
            'gradient_accumulation_steps', 1)
        total_steps = total_steps * self.config.get('num_epochs', 5)

        # 使用配置中的步数或计算得到的步数，以较大者为准
        if 'scheduler' in self.config and isinstance(self.config['scheduler'], dict):
            num_training_steps = self.config['scheduler'].get(
                'num_training_steps', total_steps)

            # 检查是否启用warmup
            enable_warmup = self.config['scheduler'].get('enable_warmup', True)
            if enable_warmup:
                # 如果明确指定了warmup步数，使用指定值
                if 'num_warmup_steps' in self.config['scheduler']:
                    num_warmup_steps = self.config['scheduler']['num_warmup_steps']
                else:
                    # 使用warmup_ratio计算默认步数
                    num_warmup_steps = int(
                        num_training_steps * self.config.get('warmup_ratio', 0.1))
            else:
                num_warmup_steps = 0
        else:
            num_training_steps = max(
                total_steps,
                self.config.get('scheduler', {}).get(
                    'num_training_steps', 100)
            )

            # 检查是否启用warmup（全局配置）
            enable_warmup = self.config.get('enable_warmup', True)
            if enable_warmup:
                # 如果明确指定了warmup步数，使用指定值
                if 'num_warmup_steps' in self.config.get('scheduler', {}):
                    num_warmup_steps = self.config.get(
                        'scheduler', {}).get('num_warmup_steps')
                else:
                    # 使用warmup_ratio计算默认步数
                    num_warmup_steps = int(
                        num_training_steps * self.config.get('warmup_ratio', 0.1))
            else:
                num_warmup_steps = 0

        # 根据配置选择学习率调度器
        use_constant_lr = self.config.get('scheduler', {}).get(
            'use_constant_lr', False) if 'scheduler' in self.config else self.config.get('use_constant_lr', False)

        if not enable_warmup and use_constant_lr:
            # 禁用warmup且使用恒定学习率时，不设置调度器（保持恒定学习率）
            self.scheduler = None
            logger.info("使用恒定学习率调度器（无学习率衰减）")
        else:
            # 使用线性warmup和衰减调度器
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=num_training_steps
            )
            if enable_warmup:
                logger.info("使用线性warmup + 线性衰减学习率调度器")
            else:
                logger.info("使用线性衰减学习率调度器（无warmup）")

        # 加载检查点
        start_epoch, best_valid_loss = self.load_checkpoint()
        self.current_epoch = start_epoch
        self.best_valid_loss = best_valid_loss

        save_dir = Path(self.config['save_dir'])
        save_dir.mkdir(parents=True, exist_ok=True)

        logger.info("开始训练...")
        logger.info(
            f"混合精度训练(FP16): {'启用' if self.config.get('fp16', False) else '禁用'}")
        logger.info(f"总训练步数: {num_training_steps}")

        # 输出warmup状态
        enable_warmup = self.config.get('scheduler', {}).get(
            'enable_warmup', True) if 'scheduler' in self.config else self.config.get('enable_warmup', True)
        if enable_warmup:
            logger.info(f"预热设置: 启用，预热步数: {num_warmup_steps}")
            if num_warmup_steps > 0:
                warmup_ratio = num_warmup_steps / \
                    num_training_steps if num_training_steps > 0 else 0
                logger.info(f"预热比例: {warmup_ratio:.2%}")
        else:
            logger.info("预热设置: 禁用")

        logger.info(f"数据集大小: {len(train_dataset)}")
        logger.info(f"批次大小: {self.config.get('batch_size', 64)}")
        logger.info(
            f"梯度累积步数: {self.config.get('gradient_accumulation_steps', 1)}")
        logger.info(
            f"有效批次大小: {self.config.get('batch_size', 64) * self.config.get('gradient_accumulation_steps', 1)}")

        # 计算每个epoch验证的频率
        eval_steps = self.config.get('eval_steps', 1000)
        global_step = 0
        early_stopping_counter = 0

        # 检查fp16设置，在不支持时禁用
        if self.config.get('fp16', False) and not torch.cuda.is_available():
            self.config['fp16'] = False
            logger.warning("GPU不可用，已自动禁用混合精度训练")

        for epoch in range(start_epoch, self.config.get('num_epochs', 5)):
            self.current_epoch = epoch

            # 训练
            train_loss = self.train_epoch(train_loader)

            # epoch 结束后清理显存
            clear_gpu_memory()
            print_gpu_memory()

            # 每个epoch结束后完整评估
            valid_loss = self.evaluate(valid_loader)

            # 记录验证loss
            self.logger.log_validation(valid_loss)

            # 保存训练历史
            self.logger._save_history()

            logger.info(f'Epoch {epoch+1}:')
            logger.info(f'  训练损失: {train_loss:.4f}')
            logger.info(f'  验证损失: {valid_loss:.4f}')

            # 保存检查点
            self.save_checkpoint(epoch, valid_loss)

            # 保存最佳模型
            if valid_loss < self.best_valid_loss:
                self.best_valid_loss = valid_loss
                early_stopping_counter = 0

                # 保存模型
                best_model_path = save_dir / 'best_model'
                best_model_path.mkdir(parents=True, exist_ok=True)
                if hasattr(self.model, 'model'):
                    # 如果是封装的模型，保存内部的transformers模型
                    self.model.model.save_pretrained(best_model_path)
                else:
                    # 如果是直接的transformers模型，直接保存
                    self.model.save_pretrained(best_model_path)

                self.tokenizer.save_pretrained(best_model_path)
                logger.info(f'  保存最佳模型，损失: {self.best_valid_loss:.4f}')
            else:
                logger.info(
                    f'self.best_valid_loss: {self.best_valid_loss:.4f}，valid_loss: {valid_loss:.4f}')

                early_stopping_counter += 1
                logger.info(
                    f'  验证损失未改善，早停计数器: {early_stopping_counter}/{self.early_stopping_patience}')

            # 早停检查
            if early_stopping_counter >= self.early_stopping_patience:
                logger.info(f'早停触发，在第 {epoch+1} 轮后停止训练')
                break

            # 显示当前学习率
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                current_lr = self.scheduler.get_last_lr()[0]
            else:
                current_lr = self.optimizer.param_groups[0]['lr']
            logger.info(f'  当前学习率: {current_lr:.2e}')

        # 训练结束后保存最终模型
        final_model_path = save_dir / 'final_model'
        final_model_path.mkdir(parents=True, exist_ok=True)
        if hasattr(self.model, 'model'):
            # 如果是封装的模型，保存内部的transformers模型
            self.model.model.save_pretrained(final_model_path)
        else:
            # 如果是直接的transformers模型，直接保存
            self.model.save_pretrained(final_model_path)

        self.tokenizer.save_pretrained(final_model_path)
        logger.info(f'训练完成，保存最终模型到: {final_model_path}')
