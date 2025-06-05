#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
房间名称中的床位信息提取 - 基于DeBERTa-v3-base的序列标注模型
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DebertaV2TokenizerFast,
    DebertaV2ForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.trainer_callback import TrainerCallback
from transformers.integrations import TensorBoardCallback
from transformers.trainer_utils import has_length
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re
from tqdm.auto import tqdm
import time
import sys
import io

# 检测是否在非交互式环境中运行（如nohup）
IS_INTERACTIVE = sys.stdout.isatty()

# 配置日志
log_file = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), "room_bed_extraction.log")
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
    "data_path": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_all_data_process_model.csv"),
    "model_name": "/home/maxon/disk2/roomMatch/room_match/deberta/pretrained_models/deberta-v3-base",
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/room_bed_extraction"),
    "max_length": 64,
    "train_batch_size": 16,
    "eval_batch_size": 16,
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "epochs": 3,
    "seed": 42,
    "test_size": 0.001,
    "validation_size": 0.001,
    "warmup_ratio": 0.1,     # 添加预热比例
    "logging_steps": 10,     # 每10步记录一次日志
    "gradient_accumulation_steps": 1,  # 梯度累积步数
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

        # 在非交互模式下使用简单进度输出
        self.use_simple_progress = not IS_INTERACTIVE

        # 跟踪最后的日志时间
        self._last_log_time = time.time()

        # 创建总进度条（每个epoch的总步数估计）
        if not self.use_simple_progress:
            total_steps = 0
            if hasattr(self, "train_dataset") and self.train_dataset is not None:
                batch_size = self.args.per_device_train_batch_size * \
                    self.args.gradient_accumulation_steps
                steps_per_epoch = len(self.train_dataset) // batch_size
                total_steps = steps_per_epoch * self.args.num_train_epochs
            self._progress_bar = tqdm(
                total=total_steps, desc="Training", position=0, leave=True)

    def _inner_training_loop(self, *args, **kwargs):
        """重写内部训练循环，记录实际的损失和学习率"""
        logger.info("="*60)
        logger.info(f"开始训练 - 将运行 {self.args.num_train_epochs} 个epoch")
        logger.info("="*60)
        return super()._inner_training_loop(*args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        """记录训练过程中的实际损失和学习率"""
        # 更新当前epoch
        self._epoch = epoch

        # 计算当前损失
        if self.state.global_step > 0:
            self._current_loss = tr_loss / self.state.global_step

        # 获取当前学习率
        self._current_lr = 0
        if self.optimizer:
            for param_group in self.optimizer.param_groups:
                self._current_lr = param_group['lr']
                break

        # 记录步数
        self._global_step = self.state.global_step

        # 如果达到记录步数，记录到日志
        if self.state.global_step % self.args.logging_steps == 0:
            # 控制日志频率，避免过于频繁
            current_time = time.time()
            if current_time - self._last_log_time >= 1:  # 至少1秒间隔
                epoch_fraction = epoch
                epoch_int = int(epoch)
                epoch_display = f"{epoch_int + 1}/{self.args.num_train_epochs}"

                logger.info(f"[Epoch {epoch_display}] 步数: {self.state.global_step}, "
                            f"损失: {self._current_loss:.4f}, 学习率: {self._current_lr:.7f}")
                self._last_log_time = current_time

        # 更新进度条
        if not self.use_simple_progress and hasattr(self, "_progress_bar"):
            self._progress_bar.update(1)
            self._progress_bar.set_postfix({
                'loss': f'{self._current_loss:.4f}',
                'lr': f'{self._current_lr:.7f}',
                'epoch': f'{epoch:.1f}'
            })
        elif self.use_simple_progress and self.state.global_step % 5 == 0:
            # 在非交互模式下，使用明确的进度显示格式（适合nohup环境）
            if hasattr(self.state, "max_steps") and self.state.max_steps:
                completed = (self.state.global_step /
                             self.state.max_steps) * 100
                progress_bar = self._generate_ascii_progress_bar(completed)

                # 使用logger而不是print，确保正确写入日志文件
                logger.info(f"进度: [{progress_bar}] {completed:.1f}% | "
                            f"步数: {self.state.global_step}/{self.state.max_steps} | "
                            f"损失: {self._current_loss:.4f} | 学习率: {self._current_lr:.7f}")

        # 调用父类方法以保持正常的保存和评估
        result = super()._maybe_log_save_evaluate(
            tr_loss, model, trial, epoch, ignore_keys_for_eval)

        # 在进行评估后记录日志
        if result and "eval_loss" in result:
            logger.info("-"*50)
            logger.info(f"验证结果 - Epoch {epoch_display}:")
            for key, value in result.items():
                logger.info(f"{key}: {value:.4f}")
            logger.info("-"*50)

        return result

    def _generate_ascii_progress_bar(self, completed, width=20):
        """生成ASCII进度条"""
        filled_length = int(width * completed / 100)
        bar = '█' * filled_length + '░' * (width - filled_length)
        return bar

    def train(self, *args, **kwargs):
        """开始训练，增加清晰的日志标记"""
        logger.info("\n" + "="*60)
        logger.info("开始模型训练")
        logger.info(f"训练集大小: {len(self.train_dataset)}")
        logger.info(
            f"验证集大小: {len(self.eval_dataset) if self.eval_dataset else 0}")
        logger.info(
            f"批次大小: {self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps}")
        logger.info(f"GPU可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"GPU类型: {torch.cuda.get_device_name(0)}")
        logger.info("="*60 + "\n")

        result = super().train(*args, **kwargs)

        # 关闭进度条
        if not self.use_simple_progress and hasattr(self, "_progress_bar"):
            self._progress_bar.close()

        return result

# 加载和处理数据


def load_data(data_path):
    """加载CSV数据并处理特定格式的spl_room_text字段"""
    logger.info(f"加载数据: {data_path}")
    df = pd.read_csv(data_path)

    # 创建新的数据框
    processed_data = pd.DataFrame({
        "s_room_name": df["spl_cleaned_room"],
        "s_room_bed_name": df["spl_pc_bed"]
    })

    logger.info(f"有效数据行数: {len(processed_data)}")
    return processed_data

# 序列标注的标签
# B-BED: 床位信息的开始
# I-BED: 床位信息的内部
# E-BED: 床位信息的结尾
# O: 非床位信息


def create_labels(room_name, bed_name, tokenizer):
    """为每个房间名称创建BIO标签序列，基于tokenizer分词结果"""
    if pd.isna(bed_name) or bed_name == "":
        # 获取分词结果
        tokens = tokenizer.tokenize(room_name)
        return ["O"] * len(tokens)

    # 先尝试直接匹配
    if bed_name in room_name:
        # 获取房间名称的分词结果
        tokenized_room = tokenizer(room_name, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_room["input_ids"])

        # 为每个token初始化为O标签
        labels = ["O"] * len(tokens)

        # 找出床位信息的起始和结束位置
        start_idx = room_name.find(bed_name)
        end_idx = start_idx + len(bed_name)

        # 获取床位信息的字符级别偏移映射
        # 这会告诉我们每个token对应原始文本中的哪些字符
        offset_mapping = tokenized_room.encodings[0].offsets

        # 标记床位信息对应的token
        bed_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            # 如果token与床位信息有重叠
            if end > start and end > start_idx and start < end_idx:
                bed_token_indices.append(i)

        # 标记床位信息的首尾和中间token
        if bed_token_indices:
            labels[bed_token_indices[0]] = "B-BED"

            for i in bed_token_indices[1:-1]:
                labels[i] = "I-BED"

            if len(bed_token_indices) > 1:
                labels[bed_token_indices[-1]] = "E-BED"

        return labels

    # 如果直接匹配失败，尝试部分匹配
    bed_patterns = [
        r'(\d+)\s*(张|beds?|double|twin|single|king|queen)',
        r'(单人|双人|大|特大)床',
        r'(King|Queen|Twin|Double|Single)\s*Bed'
    ]

    # 在床名称中寻找可能的床位信息
    bed_info = ""
    for pattern in bed_patterns:
        match = re.search(pattern, bed_name, re.IGNORECASE)
        if match:
            bed_info = match.group(0)
            break

    # 如果找到床位信息，在房间名称中寻找它
    if bed_info and bed_info in room_name:
        # 与上面相同的逻辑，但使用bed_info代替bed_name
        tokenized_room = tokenizer(room_name, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(tokenized_room["input_ids"])

        labels = ["O"] * len(tokens)

        start_idx = room_name.find(bed_info)
        end_idx = start_idx + len(bed_info)

        offset_mapping = tokenized_room.encodings[0].offsets

        bed_token_indices = []
        for i, (start, end) in enumerate(offset_mapping):
            if end > start and end > start_idx and start < end_idx:
                bed_token_indices.append(i)

        if bed_token_indices:
            labels[bed_token_indices[0]] = "B-BED"

            for i in bed_token_indices[1:-1]:
                labels[i] = "I-BED"

            if len(bed_token_indices) > 1:
                labels[bed_token_indices[-1]] = "E-BED"

        return labels

    # 如果所有匹配都失败，则返回全O标签
    tokens = tokenizer.tokenize(room_name)
    return ["O"] * len(tokens)

# 数据集类


class RoomBedDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length, bed_names=None):
        self.texts = texts
        self.bed_names = bed_names
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = {"O": 0, "B-BED": 1, "I-BED": 2, "E-BED": 3}
        self.id2label = {v: k for k, v in self.label2id.items()}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        # 如果有预先创建的标签，使用它们；否则根据bed_names创建
        if self.bed_names is not None:
            labels = create_labels(text, self.bed_names[idx], self.tokenizer)
        else:
            # 在预测时没有床位信息，所有标签初始化为O
            tokenized = self.tokenizer(text, add_special_tokens=False)
            labels = ["O"] * len(tokenized["input_ids"])

        # 使用tokenizer处理输入文本
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 将token级别的标签转换为模型需要的ID
        token_labels = [-100]  # CLS token的标签

        # 遍历非特殊token，添加相应的标签ID
        for i in range(min(len(labels), self.max_length - 2)):  # -2是因为CLS和SEP
            token_labels.append(self.label2id[labels[i]])

        # 添加SEP token和padding的标签
        token_labels.append(-100)  # SEP token的标签

        # 确保标签长度与输入长度一致
        if len(token_labels) < self.max_length:
            token_labels.extend([-100] * (self.max_length - len(token_labels)))
        else:
            token_labels = token_labels[:self.max_length]

        item = {
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "labels": torch.tensor(token_labels)
        }
        return item

# 用于评估的函数


def compute_metrics(pred):
    """计算评估指标"""
    predictions, labels = pred
    predictions = np.argmax(predictions, axis=2)

    # 定义类别映射
    id2label = {0: "O", 1: "B-BED", 2: "I-BED", 3: "E-BED"}

    # 只选择不是-100的标签和对应的预测进行评估
    true_predictions = [
        [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 展平列表以计算整体指标
    flat_predictions = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]

    # 计算精确度、召回率和F1得分
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_predictions, average='weighted'
    )
    acc = accuracy_score(flat_labels, flat_predictions)

    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# 训练模型


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
    data = load_data(CONFIG["data_path"])

    # 加载tokenizer
    logger.info("加载tokenizer...")
    tokenizer = DebertaV2TokenizerFast.from_pretrained(CONFIG["model_name"])

    # 准备训练数据
    texts = data["s_room_name"].tolist()
    bed_names = data["s_room_bed_name"].tolist()

    # 划分训练集和验证集
    train_texts, test_texts, train_beds, test_beds = train_test_split(
        texts, bed_names, test_size=CONFIG["test_size"], random_state=CONFIG["seed"]
    )

    # 进一步划分验证集
    train_texts, val_texts, train_beds, val_beds = train_test_split(
        train_texts, train_beds, test_size=CONFIG["validation_size"], random_state=CONFIG["seed"]
    )

    logger.info(f"训练集大小: {len(train_texts)}")
    logger.info(f"验证集大小: {len(val_texts)}")
    logger.info(f"测试集大小: {len(test_texts)}")

    # 创建数据集
    train_dataset = RoomBedDataset(
        train_texts, tokenizer, CONFIG["max_length"], train_beds)
    val_dataset = RoomBedDataset(
        val_texts, tokenizer, CONFIG["max_length"], val_beds)
    test_dataset = RoomBedDataset(
        test_texts, tokenizer, CONFIG["max_length"], test_beds)

    # 定义类别映射
    label2id = {"O": 0, "B-BED": 1, "I-BED": 2, "E-BED": 3}
    id2label = {v: k for k, v in label2id.items()}

    # 加载模型
    logger.info("加载预训练模型...")
    model = DebertaV2ForTokenClassification.from_pretrained(
        CONFIG["model_name"],
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )
    model.to(device)

    # 设置模型层的学习率
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "classifier" in n],
            "lr": CONFIG["learning_rate"] * 5,  # 分类层使用更高的学习率
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
        metric_for_best_model="f1",
        save_total_limit=2,
        report_to="none",
        logging_steps=CONFIG["logging_steps"],
        warmup_ratio=CONFIG["warmup_ratio"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        fp16=torch.cuda.is_available(),  # 使用混合精度训练
        disable_tqdm=not IS_INTERACTIVE,  # 在非交互环境中禁用tqdm
        dataloader_num_workers=4,  # 使用多线程加载数据
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


def predict_bed_info(model, tokenizer, room_name, device):
    """预测房间名称中的床位信息"""
    # 对输入文本进行编码
    encoding = tokenizer(
        room_name,
        return_tensors="pt",
        truncation=True,
        max_length=CONFIG["max_length"],
        return_offsets_mapping=True
    )

    # 获取原始文本中每个token的偏移位置
    offset_mapping = encoding.pop("offset_mapping").cpu().numpy()[0]

    # 将输入移动到与模型相同的设备上
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # 获取预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)[0].cpu().numpy()

    # 将预测转换为标签
    id2label = {0: "O", 1: "B-BED", 2: "I-BED", 3: "E-BED"}

    # 提取床位信息的token索引
    bed_token_indices = []
    for i, pred in enumerate(predictions):
        # 跳过特殊token和padding
        if i >= len(offset_mapping) or offset_mapping[i][0] == offset_mapping[i][1]:
            continue
        if pred in [1, 2, 3]:  # B-BED, I-BED, E-BED
            bed_token_indices.append(i)

    # 如果没有找到床位信息，返回空字符串
    if not bed_token_indices:
        return ""

    # 从原始文本中提取床位信息
    start_char = offset_mapping[bed_token_indices[0]][0]
    end_char = offset_mapping[bed_token_indices[-1]][1]

    bed_info = room_name[start_char:end_char]
    return bed_info


if __name__ == "__main__":
    # 检查GPU可用性
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    try:
        # 训练模型
        model, tokenizer, test_results = train_model()

        # 确保模型在正确的设备上
        model = model.to(device)

        # 展示一些预测示例
        sample_rooms = [
            "Standard Double Room|标准双人房（1 张双人床）",
            "Deluxe Room, 3 Twin Beds, Non Smoking",
            "Superior Double Room",
            "Deluxe Triple",
            "豪华大床房",
            "双人间",
            "家庭套房（1张大床+1张单人床）"
        ]

        logger.info("预测示例:")
        for room in sample_rooms:
            bed_info = predict_bed_info(model, tokenizer, room, device)
            logger.info(f"房间: {room}")
            logger.info(f"预测的床位信息: {bed_info}")

    except Exception as e:
        logger.error(f"训练过程中出现错误: {str(e)}", exc_info=True)
