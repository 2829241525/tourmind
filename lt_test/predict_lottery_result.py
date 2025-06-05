#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
彩票预测模型 - 基于DeBERTa-v3-base的序列分类模型
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2TokenizerFast,
    DebertaV2ForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import time
import sys

# 检测是否在非交互式环境中运行
IS_INTERACTIVE = sys.stdout.isatty()

# 配置日志
log_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "predict_lottery.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
CONFIG = {
    "data_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "history_data_with_stats.csv"),
    "model_name": "/home/maxon/disk2/roomMatch/room_match/chinese_match/pretrained_models/mdeberta-v3-base",
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/lottery_prediction0428"),
    "max_length": 128,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 5,
    "seed": 42,
    "test_size": 0.1,
    "validation_size": 0.1,
    "warmup_ratio": 0.1,
    "logging_steps": 10,
    "gradient_accumulation_steps": 1,
    "num_labels": 36,  # 1-35的数字预测
    "sequence_length": 10,  # 使用前10期数据预测
}

# 设置随机种子


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(CONFIG["seed"])

# 自定义无进度条的Trainer


class NoTqdmTrainer(Trainer):
    """扩展Trainer类以添加更好的日志记录"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_loss = 0
        self._current_lr = 0
        self._global_step = 0
        self._epoch = 0
        self.use_simple_progress = not IS_INTERACTIVE
        self._last_log_time = time.time()

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.state.global_step > 0:
            self._current_loss = tr_loss / self.state.global_step

        self._current_lr = 0
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                self._current_lr = param_group['lr']
                break

        if self.state.global_step % self.args.logging_steps == 0:
            current_time = time.time()
            if current_time - self._last_log_time >= 1:
                epoch_int = int(epoch)
                epoch_display = f"{epoch_int + 1}/{self.args.num_train_epochs}"

                logger.info(f"[Epoch {epoch_display}] 步数: {self.state.global_step}, "
                            f"损失: {self._current_loss:.4f}, 学习率: {self._current_lr:.7f}")
                self._last_log_time = current_time

        return super()._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

# 数据集类


class LotteryDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(label, dtype=torch.long)
        }


def compute_metrics(pred):
    """计算评估指标"""
    predictions = pred.predictions.argmax(-1)
    labels = pred.label_ids

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    acc = accuracy_score(labels, predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


def process_lottery_data(data):
    """处理彩票数据，创建训练样本"""
    processed_data = []
    labels = []

    # 将数据按时间排序
    data = data.sort_values('lotteryDrawTime')

    # 获取开奖结果
    draw_results = data['lotteryDrawResult'].tolist()

    # 创建训练样本
    for i in range(CONFIG["sequence_length"], len(draw_results)):
        # 获取前N期的开奖结果
        history = []
        for j in range(i - CONFIG["sequence_length"], i):
            result = draw_results[j]
            history.append(f"期号{j+1}:{result}")

        # 将历史记录合并为一个字符串
        input_text = " [SEP] ".join(history)

        # 获取当前期的开奖结果作为标签
        current_result = draw_results[i].split()
        # 使用所有7个号码
        current_numbers = [int(x) for x in current_result]

        # 为每个位置创建一个样本（包括特别号）
        for pos, number in enumerate(current_numbers):
            position_type = "特别号" if pos >= 5 else "普通号"
            processed_data.append(
                f"位置{pos+1}预测_{position_type} [SEP] {input_text}")
            labels.append(number - 1)  # 将1-35映射到0-34

    return processed_data, labels


def train_model():
    """训练模型"""
    logger.info("准备开始训练...")

    # 检查CUDA是否可用
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 检查输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    logger.info(f"输出目录: {CONFIG['output_dir']}")

    # 加载数据
    logger.info("加载数据集...")
    try:
        data = pd.read_csv(CONFIG["data_path"])
        logger.info(f"成功加载数据，共 {len(data)} 条记录")

        # 处理数据
        texts, labels = process_lottery_data(data)
        logger.info(f"处理后的样本数量: {len(texts)}")

    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        return None, None, None

    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CONFIG["model_name"])

    # 划分训练集和验证集
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
    )

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=CONFIG["validation_size"], random_state=CONFIG["seed"]
    )

    logger.info(f"训练集大小: {len(train_texts)}")
    logger.info(f"验证集大小: {len(val_texts)}")
    logger.info(f"测试集大小: {len(test_texts)}")

    # 创建数据集
    train_dataset = LotteryDataset(
        train_texts, train_labels, tokenizer, CONFIG["max_length"])
    val_dataset = LotteryDataset(
        val_texts, val_labels, tokenizer, CONFIG["max_length"])
    test_dataset = LotteryDataset(
        test_texts, test_labels, tokenizer, CONFIG["max_length"])

    # 加载模型
    logger.info("加载预训练模型...")
    model = DebertaV2ForSequenceClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=CONFIG["num_labels"]
    )
    model.to(device)

    # 设置模型层的学习率
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": CONFIG["learning_rate"] * 5,
        },
        {
            "params": [p for n, p in model.named_parameters() if "classifier" not in n],
            "lr": CONFIG["learning_rate"],
        },
    ]

    # 训练参数
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["train_batch_size"],
        per_device_eval_batch_size=CONFIG["eval_batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=CONFIG["weight_decay"],
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        save_total_limit=2,
        report_to="none",
        logging_steps=CONFIG["logging_steps"],
        warmup_ratio=CONFIG["warmup_ratio"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        fp16=torch.cuda.is_available(),
        disable_tqdm=not IS_INTERACTIVE,
        dataloader_num_workers=4,
    )

    # 创建Trainer
    trainer = NoTqdmTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
        ],
        optimizers=(torch.optim.AdamW(optimizer_grouped_parameters), None)
    )

    # 训练模型
    trainer.train()

    # 保存最终模型
    model_save_path = os.path.join(CONFIG["output_dir"], "final")
    logger.info(f"保存最终模型到: {model_save_path}")
    trainer.save_model(model_save_path)

    # 评估模型
    logger.info("\n" + "="*60)
    logger.info("在测试集上评估模型...")
    test_results = trainer.evaluate(test_dataset)
    logger.info("\n最终测试结果:")
    for key, value in test_results.items():
        logger.info(f"{key}: {value:.4f}")
    logger.info("="*60)

    return model, tokenizer, test_results


def predict_next_numbers(model, tokenizer, history_data, device):
    """预测下一期号码（包括特别号）"""
    # 处理历史数据
    history_text = []
    for i in range(len(history_data) - CONFIG["sequence_length"], len(history_data)):
        result = history_data.iloc[i]['lotteryDrawResult']
        history_text.append(f"期号{i+1}:{result}")

    input_text = " [SEP] ".join(history_text)

    # 预测每个位置的号码（包括特别号）
    predictions = []
    for pos in range(7):  # 预测7个号码
        position_type = "特别号" if pos >= 5 else "普通号"
        # 准备输入
        text = f"位置{pos+1}预测_{position_type} [SEP] {input_text}"
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=CONFIG["max_length"],
            padding="max_length"
        )

        # 将输入移动到设备上
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # 获取预测
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions.append(torch.argmax(
                outputs.logits, dim=1).item() + 1)  # 将0-34映射回1-35

    return predictions


if __name__ == "__main__":
    # 检查GPU可用性
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    try:
        # 训练模型
        model, tokenizer, test_results = train_model()

        if model and tokenizer:
            # 确保模型在正确的设备上
            model = model.to(device)

            # 加载最新的历史数据进行预测
            data = pd.read_csv(CONFIG["data_path"])
            next_numbers = predict_next_numbers(model, tokenizer, data, device)

            logger.info("\n预测下一期号码:")
            # 分开显示普通号和特别号
            normal_numbers = ' '.join(map(str, next_numbers[:5]))
            special_numbers = ' '.join(map(str, next_numbers[5:]))
            logger.info(f"普通号: {normal_numbers}")
            logger.info(f"特别号: {special_numbers}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
