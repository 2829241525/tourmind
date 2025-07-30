# -*- coding: utf-8 -*-
"""
DeBERTa训练器模块 - 房间比较三分类任务
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
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from models.model import create_model
from data.dataset import RoomComparisonDataset
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
    """DeBERTa模型训练器 - 房间比较三分类任务"""

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

        # 训练开始前清理显存并打印状态
        clear_gpu_memory()
        print_gpu_memory()

        # 设置三分类模型
        num_labels = 3  # HIGHER, LOWER, INCOMPARABLE

        # 检查是否存在之前训练的最佳模型
        best_model_path = os.path.join(self.config['save_dir'], 'best_model')
        if os.path.exists(best_model_path):
            logger.info(f"找到已有的最佳模型，正在加载: {best_model_path}")
            class_weights = self.config.get('class_weights', None)
            self.model = create_model(
                'classifier', best_model_path, num_labels=num_labels, class_weights=class_weights)
            self.tokenizer = AutoTokenizer.from_pretrained(best_model_path)
        else:
            logger.info("未找到已有模型，从预训练模型初始化")
            # 使用配置文件中指定的预训练模型路径
            model_path = self.config.get(
                'pretrained_model', "microsoft/deberta-v3-base")

            class_weights = self.config.get('class_weights', None)
            self.model = create_model(
                'classifier', model_path, num_labels=num_labels, class_weights=class_weights)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

            # 启用梯度检查点以节省显存（根据配置）
            enable_gradient_checkpointing = self.config.get(
                'enable_gradient_checkpointing', False)
            if enable_gradient_checkpointing:
                if hasattr(self.model, 'model'):
                    self.model.model.gradient_checkpointing_enable()
                else:
                    self.model.gradient_checkpointing_enable()
                logger.info("已启用梯度检查点以节省显存")
            else:
                logger.info("禁用梯度检查点")

        # 将模型移动到设备上
        self.model.to(self.device)

        # 创建优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 2e-5),
            weight_decay=self.config.get('weight_decay', 0.01),
            eps=self.config.get('adam_epsilon', 1e-8)
        )

        # 初始化混合精度训练的 scaler
        self.scaler = GradScaler(device='cuda')

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

        # 是否使用混合精度训练
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

            # 前向传播和反向传播
            if use_fp16:
                # 使用混合精度训练
                with autocast(device_type='cuda'):
                    outputs = self.model(**inputs)
                    loss = outputs.loss / grad_accum_steps  # 根据梯度累积步数缩放损失

                # 反向传播
                self.scaler.scale(loss).backward()
            else:
                # 不使用混合精度训练
                outputs = self.model(**inputs)
                loss = outputs.loss / grad_accum_steps  # 根据梯度累积步数缩放损失
                loss.backward()

            # 收集预测和标签用于计算指标
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

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
            progress_bar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}' if hasattr(self, 'scheduler') else 'N/A'
            })

            # 在累积足够的梯度后更新参数
            if (batch_idx + 1) % grad_accum_steps == 0:
                if use_fp16:
                    # 使用 scaler 进行梯度裁剪和参数更新
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
                if hasattr(self, 'scheduler'):
                    self.scheduler.step()

                self.optimizer.zero_grad()

                # 更新全局步数
                global_step += 1

                # 每logging_steps步输出一次训练指标
                if global_step % logging_steps == 0 and len(all_predictions) > 0:
                    # 计算训练指标（三分类）
                    accuracy = accuracy_score(all_labels, all_predictions)
                    precision = precision_score(
                        all_labels, all_predictions, average='weighted', zero_division=0)
                    recall = recall_score(
                        all_labels, all_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_predictions,
                                  average='weighted', zero_division=0)

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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                labels = batch['label'].to(self.device)
                inputs['labels'] = labels

                # 前向传播
                if use_fp16:
                    with autocast(device_type='cuda'):
                        outputs = self.model(**inputs)
                        loss = outputs.loss
                else:
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

        # 计算三分类指标
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(
            all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(
            all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions,
                      average='weighted', zero_division=0)

        # 各类别的指标
        precision_per_class = precision_score(
            all_labels, all_predictions, average=None, zero_division=0)
        recall_per_class = recall_score(
            all_labels, all_predictions, average=None, zero_division=0)
        f1_per_class = f1_score(
            all_labels, all_predictions, average=None, zero_division=0)

        # 统计各类别数量
        class_names = ['HIGHER', 'LOWER', 'INCOMPARABLE']

        # 输出详细的验证集评估结果
        logger.info("=" * 60)
        logger.info(f"【验证集】三分类评估结果 (Epoch {self.current_epoch+1}):")
        logger.info(f"  样本总数: {len(all_labels)}")
        logger.info(f"  Loss: {avg_loss:.4f}")
        logger.info(f"  整体准确率: {accuracy:.4f}")
        logger.info(f"  加权精确率: {precision:.4f}")
        logger.info(f"  加权召回率: {recall:.4f}")
        logger.info(f"  加权F1: {f1:.4f}")
        logger.info("")
        logger.info("各类别详细指标:")

        for i, class_name in enumerate(class_names):
            if i < len(precision_per_class):
                class_count = sum([1 for label in all_labels if label == i])
                class_correct = sum([1 for j, label in enumerate(all_labels)
                                     if label == i and all_predictions[j] == i])
                class_accuracy = class_correct / class_count if class_count > 0 else 0

                logger.info(f"  {class_name} ({i}): 数量={class_count}, "
                            f"准确率={class_accuracy:.4f}, "
                            f"精确率={precision_per_class[i]:.4f}, "
                            f"召回率={recall_per_class[i]:.4f}, "
                            f"F1={f1_per_class[i]:.4f}")

        logger.info("=" * 60)

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
            'scheduler_state_dict': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
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
            if hasattr(self, 'scheduler') and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
                self.scheduler.load_state_dict(
                    checkpoint['scheduler_state_dict'])

            # 返回上次训练的epoch和验证损失
            return checkpoint['epoch'] + 1, checkpoint.get('valid_loss', float('inf'))
        else:
            logger.info("未找到检查点，从头开始训练")
            return 0, float('inf')

    def train(self):
        """
        训练模型
        """
        # 加载房间比较数据集
        train_dataset = RoomComparisonDataset(self.config['train_file'])
        valid_dataset = RoomComparisonDataset(self.config['valid_file'])

        logger.info(f"加载了 {len(train_dataset)} 个训练样本")
        logger.info(f"加载了 {len(valid_dataset)} 个验证样本")

        # 计算类别权重
        if hasattr(train_dataset, 'get_class_weights'):
            class_weights = train_dataset.get_class_weights()
            # 更新配置中的类别权重
            self.config['class_weights'] = class_weights

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.get('batch_size', 8),
            shuffle=True,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.config.get('batch_size', 8) * 2,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4),
            pin_memory=self.config.get('pin_memory', True)
        )

        # 创建学习率调度器
        total_steps = len(train_dataset) // self.config.get('batch_size', 8)
        total_steps = total_steps // self.config.get(
            'gradient_accumulation_steps', 1)
        total_steps = total_steps * self.config.get('num_epochs', 5)

        # 使用配置中的步数或计算得到的步数，以较大者为准
        if 'scheduler' in self.config and isinstance(self.config['scheduler'], dict):
            num_training_steps = self.config['scheduler'].get(
                'num_training_steps', total_steps)
            num_warmup_steps = self.config['scheduler'].get(
                'num_warmup_steps', 0)
        else:
            num_training_steps = max(
                total_steps,
                self.config.get('scheduler', {}).get(
                    'num_training_steps', 100)
            )
            # 预热步数
            num_warmup_steps = int(
                num_training_steps * self.config.get('warmup_ratio', 0.1))
            # 如果配置中有明确指定预热步数，则使用配置中的值
            if 'num_warmup_steps' in self.config.get('scheduler', {}):
                num_warmup_steps = self.config.get('scheduler', {}).get(
                    'num_warmup_steps', num_warmup_steps)

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

        logger.info("开始训练...")
        logger.info(f"总训练步数: {num_training_steps}")
        logger.info(f"预热步数: {num_warmup_steps}")
        logger.info(f"数据集大小: {len(train_dataset)}")
        logger.info(f"批次大小: {self.config.get('batch_size', 8)}")
        logger.info(
            f"梯度累积步数: {self.config.get('gradient_accumulation_steps', 1)}")
        logger.info(
            f"有效批次大小: {self.config.get('batch_size', 8) * self.config.get('gradient_accumulation_steps', 1)}")

        # 检查fp16设置，在不支持时禁用
        if self.config.get('fp16', False) and not torch.cuda.is_available():
            self.config['fp16'] = False
            logger.warning("GPU不可用，已自动禁用混合精度训练")

        early_stopping_counter = 0

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
                early_stopping_counter += 1
                logger.info(
                    f'  验证损失未改善，早停计数器: {early_stopping_counter}/{self.early_stopping_patience}')

            # 早停检查
            if early_stopping_counter >= self.early_stopping_patience:
                logger.info(f'早停触发，在第 {epoch+1} 轮后停止训练')
                break

            # 显示当前学习率
            current_lr = self.scheduler.get_last_lr()[0]
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
