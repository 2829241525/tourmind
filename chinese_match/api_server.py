# -*- coding: utf-8 -*-
"""
SimCSE模型推理HTTP服务
"""

import threading
import os
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any
import uvicorn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import re
from pathlib import Path
import torch.nn.functional as F
import json
import time
import logging
from concurrent.futures import ThreadPoolExecutor
import math

# 配置区域
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建日志目录
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs/api_server.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

UNKNOWN_TEXT = ""
ROOM_PREFIX = ""
BED_PREFIX = ","

# 模型路径映射/home/maxon/disk2/roomMatch/room_match/deberta
MODEL_PATHS = {
    'mdeberta': os.path.join(BASE_DIR, 'checkpoints_all_0425/best_model')
}
DEFAULT_MODEL = 'mdeberta'

MAPPING_PATH = os.path.join(BASE_DIR, 'data/mapping.xlsx')
MAX_LENGTH = 80
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEFAULT_THRESHOLD = 0.9
# 禁用半精度计算
USE_FP16 = False
# 模型加载超时时间（秒）
MODEL_LOAD_TIMEOUT = 60

# 创建PID文件路径
PID_FILE = os.path.join(BASE_DIR, "api_server.pid")

# 创建一个锁，用于控制模型加载
model_lock = threading.Lock()

# 创建全局状态对象，替代原来的全局变量


class GlobalState:
    def __init__(self):
        self.predictor = None
        self.cleaner = None
        self.model_loading = False
        self.init_attempts = 0
        self.last_error = None
        self.model_path = None
        # 添加模型加载事件
        self.model_load_event = threading.Event()


# 初始化全局状态
global_state = GlobalState()

# 创建FastAPI应用
app = FastAPI(
    title="房间匹配服务",
    description="基于SimCSE的房间文本匹配服务",
    version="1.0.0"
)

# 请求模型


class MatchRequest(BaseModel):
    spl_room_names: List[str]
    spl_room_bed_names: List[str]
    s_room_names: List[str]
    s_room_bed_names: List[str]
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL
    # 添加等待模型加载参数
    wait_for_model: bool = True

# 响应模型


class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]
    model_used: str = None  # 添加实际使用的模型信息


class SimCSEPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        logger.info(f"使用设备: {self.device}")

        # 尝试加载配置文件
        config_dir = os.path.join(os.path.dirname(
            os.path.dirname(model_path)), 'config')
        config_path = os.path.join(config_dir, 'config_cross_entropy.json')

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"成功加载配置文件: {config_path}")
            self.temperature = self.config.get('temperature', 0.05)
        else:
            logger.warning(f"未找到配置文件: {config_path}，使用默认配置")
            self.temperature = 0.05
            self.config = {}

        # 根据模型路径确定使用的模型类型

        logger.info("使用Classifier模型 (cross entropy loss)")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path).to(self.device)

        # 确保模型使用全精度计算
        if USE_FP16 is False:
            self.model = self.model.float()
            logger.info("已将模型转换为全精度(float32)计算")

        self.loss_type = 'cross_entropy'

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        self.model.eval()

        # 记录模型的数据类型信息
        for name, param in self.model.named_parameters():
            if param.dtype == torch.float16:
                logger.info(f"检测到半精度参数: {name} (dtype={param.dtype})")
                if USE_FP16 is False:
                    logger.warning(f"警告：尽管设置了使用全精度，但仍检测到半精度参数: {name}")
                break
        else:
            logger.info(
                f"模型参数数据类型: {next(self.model.parameters()).dtype}")

        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点以节省显存")

    def encode_batch(self, texts, batch_size=256):
        """编码文本 (适用于对比学习模型)"""
        embeddings = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # 移除autocast，直接使用全精度计算
                outputs = self.model(**encoded)
                if self.loss_type == 'contrastive':
                    batch_embeddings = outputs.last_hidden_state[:, 0]
                else:
                    # 获取正类的logits得分
                    batch_embeddings = outputs.logits[:, 1]

                # 检查并处理NaN值
                if torch.isnan(batch_embeddings).any():
                    batch_embeddings = torch.nan_to_num(
                        batch_embeddings, nan=-10.0)
                    logger.warning("检测到NaN值，已替换为安全值")

                # 确保返回的是float32类型
                if batch_embeddings.dtype != torch.float32:
                    batch_embeddings = batch_embeddings.float()

                embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def encode_text_pair(self, text1_list, text2_list, batch_size=256):
        """批量编码文本对，用于cross_entropy模式"""
        logits_list = []

        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 对于分类器模型，使用文本对输入
            encoded = self.tokenizer(
                batch_text1,
                text_pair=batch_text2,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors='pt'
            )

            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # 移除autocast，直接使用全精度计算
                outputs = self.model(**encoded)

                # 检查并处理NaN值
                if torch.isnan(outputs.logits).any():
                    # 将NaN值替换为一个安全的负值（表示低概率）
                    outputs.logits = torch.nan_to_num(
                        outputs.logits, nan=-10.0)
                    logger.warning("检测到NaN值，已替换为安全值")

                # 获取正类的logits得分
                batch_logits = outputs.logits[:, 1]
                # 确保返回的是float32类型
                if batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.float()
                logits_list.append(batch_logits.cpu())

        return torch.cat(logits_list, dim=0)

    def calculate_similarity(self, source_embeddings, target_embeddings=None, normalize_to_probability=True):
        """计算相似度，支持两种方式：
        1. 对比学习：计算两组嵌入之间的余弦相似度
        2. 交叉熵：将logits转换为概率值

        Args:
            source_embeddings: 源文本嵌入或logits
            target_embeddings: 目标文本嵌入 (对比学习模式需要)
            normalize_to_probability: 是否将结果归一化为0-1之间的概率值
        """

        # 交叉熵模式：直接使用logits作为相似度分数
        similarities = source_embeddings

        # 使用sigmoid函数将logits转换为0-1之间的概率值
        if normalize_to_probability:
            # 确保张量为float32类型
            if similarities.dtype != torch.float32:
                similarities = similarities.float()
            similarities = torch.sigmoid(similarities)

        pair_similarities = similarities

        return pair_similarities

    def predict(self, source_texts: List[str], target_texts: List[str], threshold: float = DEFAULT_THRESHOLD):
        """预测文本对的匹配情况"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        # 根据模型类型选择不同的预测方法

            # 使用交叉熵模型进行预测 (文本对分类)
        logits = self.encode_text_pair(source_texts, target_texts)
        similarities = self.calculate_similarity(logits)

        # 根据阈值进行预测
        predictions = (similarities > threshold).float()

        # 处理特殊浮点数，确保JSON兼容
        similarities_list = similarities.cpu().numpy().tolist()
        predictions_list = predictions.cpu().numpy().tolist()

        # 处理可能的非法浮点数值（如inf和nan）
        similarities_list = [
            float(0.0) if (not isinstance(s, (int, float)) or math.isinf(s) or math.isnan(s))
            else float(s) for s in similarities_list
        ]

        return {
            'similarities': similarities_list,
            'predictions': predictions_list,
            'threshold': threshold
        }


class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.room_rules, self.bed_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> Tuple[dict, List[Tuple[str, str, int]]]:
        # 定义映射替换模式
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        room_rules = {}
        bed_rules = []

        # 按照sheet顺序构建字典
        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)
            if not {'key', 'value'}.issubset(sheet_data.columns):
                continue

            # 处理每一行数据
            current_sheet_rules = []
            for _, row in sheet_data.iterrows():
                key = str(row['key']).strip().lower()
                values = [v.strip().strip('"')
                          for v in str(row['value']).split(',')]
                rules = [(value.lower(), key, len(value))
                         for value in values if value]
                current_sheet_rules.extend(rules)

            # 按原文本长度降序排序当前sheet的规则
            current_sheet_rules = sorted(
                current_sheet_rules, key=lambda x: x[2], reverse=True)

            # 如果是bed工作表，单独处理床型
            if sheet_name.lower() == 'bed':
                bed_rules = current_sheet_rules
                room_rules[sheet_name.lower()] = current_sheet_rules
            else:
                # 其他工作表作为房型规则，使用sheet_name作为key
                room_rules[sheet_name.lower()] = current_sheet_rules

        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        # 遍历每个sheet的规则
        for sheet_name, rules in self.room_rules.items():
            # 遍历当前sheet中的所有规则
            for value, key, _ in rules:
                pattern = re.compile(
                    rf'(?:^|(?<=[^a-zA-Z0-9-]))'  # 开始边界：确保前面是非字母数字和横杠
                    rf'{re.escape(value)}'  # 需要匹配的文本
                    rf'(?:$|(?=[^a-zA-Z0-9-]))',  # 结束边界：确保后面是非字母数字和横杠
                    re.IGNORECASE
                )
                if pattern.search(text_lower):  # 如果找到匹配
                    text_lower = pattern.sub(key, text_lower)
                    break  # 找到匹配后就停止当前sheet的搜索

            # 如果在当前sheet中找到了匹配，继续处理下一个sheet的规则
            # 这样可以让不同类型的规则（不同sheet）都能应用到文本上

        return text_lower

    def clean_bed_text(self, text: str) -> str:
        if pd.isna(text):
            return text
        if '[unknown]' in text:
            return text

        text_lower = str(text).lower()
        # 只使用bed工作表的规则
        for value, key, _ in self.bed_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'
                rf'{re.escape(value)}'
                rf'(?:$|(?=[^a-zA-Z0-9-]))',
                re.IGNORECASE
            )
            text_lower = pattern.sub(key, text_lower)

        return text_lower


def get_predictor():
    """获取当前预测器实例，如果不存在则尝试初始化"""
    if global_state.predictor is None and not global_state.model_loading:
        # 如果没有正在加载模型，且模型尚未初始化，则尝试初始化默认模型
        logger.info(f"预测器未初始化，尝试加载默认模型: {DEFAULT_MODEL}")
        init_model(DEFAULT_MODEL)
    return global_state.predictor


def get_cleaner():
    """获取当前清洗器实例，如果不存在则尝试初始化"""
    if global_state.cleaner is None and not global_state.model_loading:
        init_cleaner()
    return global_state.cleaner


def load_predictor_in_background(model_path: str):
    """在后台线程中加载预测器"""
    global global_state

    try:
        with model_lock:
            global_state.model_loading = True
            # 重置加载事件
            global_state.model_load_event.clear()
            logger.info(f"开始在后台加载模型：{model_path}")
            start_time = time.time()

            # 加载模型
            try:
                predictor = SimCSEPredictor(model_path)
                global_state.predictor = predictor
                global_state.model_path = model_path
                global_state.last_error = None
                load_time = time.time() - start_time
                logger.info(
                    f"模型加载成功，类型：{predictor.loss_type}，耗时：{load_time:.2f}秒")
            except Exception as e:
                error_msg = f"加载模型时出错：{str(e)}"
                global_state.last_error = error_msg
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())

    finally:
        with model_lock:
            global_state.model_loading = False
            # 设置加载事件，通知等待的请求
            global_state.model_load_event.set()


def init_model(model_type=DEFAULT_MODEL):
    """初始化预测器"""
    global global_state

    # 如果model_type为空，使用默认值
    model_type = model_type if model_type and model_type.strip() else DEFAULT_MODEL
    logger.info(f"初始化模型类型: {model_type}")

    # 如果已经在加载中，则直接返回
    if global_state.model_loading:
        logger.info("模型正在加载中，跳过重复初始化")
        return

    with model_lock:
        global_state.init_attempts += 1

        # 获取模型路径
        model_path = MODEL_PATHS.get(model_type)

        # 如果模型路径不在预定义字典中，则尝试构建可能的路径
        if model_path is None:
            possible_path = os.path.join(
                BASE_DIR, f'checkpoints_{model_type}/best_model')
            if os.path.exists(possible_path):
                model_path = possible_path
                logger.info(f"使用构建的模型路径: {model_path}")
            else:
                # 尝试查找可用的模型
                logger.warning(f"警告：未找到指定的模型类型: {model_type}，尝试查找可用模型")
                available_paths = []
                for model_name, path in MODEL_PATHS.items():
                    if os.path.exists(path):
                        available_paths.append((model_name, path))

                if available_paths:
                    model_name, model_path = available_paths[0]
                    logger.info(f"使用备选模型: {model_name} ({model_path})")
                else:
                    error_msg = f"错误：没有找到可用的模型，已检查以下路径: {list(MODEL_PATHS.values())}"
                    global_state.last_error = error_msg
                    logger.error(error_msg)
                    return

        # 检查模型路径是否存在
        if not os.path.exists(model_path):
            error_msg = f"错误：模型路径不存在: {model_path}"
            global_state.last_error = error_msg
            logger.error(error_msg)
            return

        # 在后台线程中加载模型
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(load_predictor_in_background, model_path)


def init_cleaner():
    """初始化文本清洗器"""
    global global_state

    try:
        if not os.path.exists(MAPPING_PATH):
            error_msg = f"文本清洗器映射文件不存在: {MAPPING_PATH}"
            global_state.last_error = error_msg
            logger.error(error_msg)
            return

        logger.info(f"加载文本清洗器：{MAPPING_PATH}")
        cleaner = DataCleaner(Path(MAPPING_PATH))
        global_state.cleaner = cleaner
        logger.info("文本清洗器加载成功")
    except Exception as e:
        error_msg = f"加载文本清洗器时出错：{str(e)}"
        global_state.last_error = error_msg
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())


def extract_english(text: str) -> str:
    if pd.isna(text):
        return UNKNOWN_TEXT
    # 首先按|分割，并去除分隔符
    parts = [part.strip() for part in text.split('|')]
    parts = [p for p in parts if p]  # 移除空字符串

    if not parts:  # 如果没有有效内容
        return UNKNOWN_TEXT

    # 检查是否所有部分都只包含中文
    def contains_only_chinese(text):
        text = text.strip()
        if not text:
            return False
        pattern = re.compile(r'^[\u4e00-\u9fff\s]+$')
        return bool(pattern.match(text))

    # 检查是否只包含英文和特殊字符
    def contains_only_english(text):
        text = text.strip()
        if not text:
            return False
        # 移除所有特殊字符和空格后，检查是否只包含英文
        cleaned_text = re.sub(r'[^a-zA-Z]', '', text)
        return bool(cleaned_text) and not re.search(r'[\u4e00-\u9fff]', text)

    # 获取文本中英文内容的数量
    def get_english_content_length(text):
        # 移除中文字符
        english_only = re.sub(
            r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', text)
        # 只保留字母和空格
        english_only = re.sub(r'[^a-zA-Z\s]', '', english_only)
        return len(english_only.strip())

    # 如果所有部分都只包含中文，返回所有中文内容
    if all(contains_only_chinese(part) for part in parts):
        return ' '.join(parts)

    # 如果所有部分都是英文，检查是否内容一致
    if all(contains_only_english(part) for part in parts):
        # 清理并比较内容
        cleaned_parts = [re.sub(
            r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', part)).lower().strip() for part in parts]
        if len(set(cleaned_parts)) == 1:  # 所有内容都相同
            return parts[0].strip()

    # 处理混合内容，选择英文内容最多的部分
    english_lengths = [(part, get_english_content_length(part))
                       for part in parts]
    max_english_length = max(length for _, length in english_lengths)

    if max_english_length > 0:  # 如果有英文内容
        # 选择英文内容最多的部分
        selected_parts = [
            part for part, length in english_lengths if length == max_english_length]
        return selected_parts[0].strip()

    # 如果没有找到英文内容，但有中文内容，返回所有内容
    has_chinese = any(contains_only_chinese(part) for part in parts)
    if has_chinese:
        return ' '.join(parts)

    # 如果既没有英文也没有中文，尝试保留所有非空内容
    if parts:
        return ' '.join(parts)

    return UNKNOWN_TEXT


def process_text_with_prefix(prefix: str, room_name: str, bed_name: str, is_supplier: bool = False, cleaner=None) -> str:
    # 提取英文
    room = extract_english(room_name)
    bed = extract_english(bed_name)
    prefix_text = prefix if prefix else UNKNOWN_TEXT

    # 只对供应商端数据进行清洗
    if is_supplier and cleaner is not None:
        if room:
            room = cleaner.clean_room_text(room)
        if bed:
            bed = cleaner.clean_bed_text(bed)

    # 组合成最终格式
    parts = []
    # if prefix:
    #     parts.append(f"{SUPPLY_PREFIX} {prefix_text}" if prefix_text else "")
    if room:
        parts.append(f"{ROOM_PREFIX} {room}")
    if bed:
        parts.append(f"{BED_PREFIX} {bed}")
    # 小写
    parts = [part.lower() for part in parts]
    return " ".join(filter(None, parts))


@app.post("/match/", response_model=MatchResponse)
async def match_rooms(request: MatchRequest, background_tasks: BackgroundTasks):
    """
    房间文本匹配接口

    - supplier_names: 供应商名称列表
    - spl_room_names: 供应商房型名称列表
    - spl_room_bed_names: 供应商床型名称列表
    - s_room_names: 母房型名称列表
    - s_room_bed_names: 母床型名称列表
    - threshold: 匹配阈值（可选，默认0.5）
    - model: 使用的模型类型（可选，默认'cross_entropy'）
    - wait_for_model: 是否等待模型加载完成（可选，默认False）

    返回：
    - similarities: 相似度列表
    - predictions: 预测结果列表（1表示匹配，0表示不匹配）
    - threshold: 使用的阈值
    - model_used: 实际使用的模型
    """
    try:
        # 获取预测器实例
        predictor = get_predictor()
        cleaner = get_cleaner()

        # 检查模型是否已加载
        if predictor is None:
            if global_state.model_loading:
                raise HTTPException(
                    status_code=503,
                    detail="模型正在加载中，请稍后再试"
                )
            else:
                error_detail = global_state.last_error or "模型尚未加载，请稍后再试"
                raise HTTPException(
                    status_code=500,
                    detail=error_detail
                )

        # 获取请求中的模型类型
        model_type = None
        if request.model:
            model_type = request.model.strip()
            if model_type:
                model_type = model_type.lower()
                logger.info(f"使用请求指定的模型类型: {model_type}")
            else:
                model_type = DEFAULT_MODEL
                logger.info(f"请求模型类型为空，使用默认模型: {model_type}")
        else:
            model_type = DEFAULT_MODEL
            logger.info(f"请求未指定模型类型，使用默认模型: {model_type}")

        # 确定模型路径
        model_path = MODEL_PATHS.get(model_type)

        if model_path is None:
            # 尝试构建可能的路径
            possible_path = os.path.join(
                BASE_DIR, f'checkpoints_{model_type}/best_model')
            if os.path.exists(possible_path):
                model_path = possible_path
                logger.info(f"使用构建的模型路径: {model_path}")
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"不支持的模型类型: {model_type}。支持的类型有: {list(MODEL_PATHS.keys())}，或确保路径 {possible_path} 存在"
                )

        # 如果模型类型与当前加载的不同，处理模型加载
        if predictor.model_path != model_path:
            logger.info(f"当前模型({predictor.model_path})与请求模型({model_path})不同")

            # 检查是否已经在加载新模型
            if global_state.model_loading:
                if request.wait_for_model:
                    logger.info(f"模型正在加载中，等待加载完成...")
                    # 等待模型加载完成，设置超时
                    wait_success = global_state.model_load_event.wait(
                        timeout=MODEL_LOAD_TIMEOUT)
                    if not wait_success:
                        logger.warning(f"等待模型加载超时，使用当前可用模型")
                    else:
                        logger.info(f"模型加载完成，使用新模型处理请求")
                        # 重新获取加载后的预测器
                        predictor = global_state.predictor
                else:
                    logger.info(f"模型正在加载中，但不等待，继续使用当前模型处理请求")
            else:
                # 开始新的模型加载
                logger.info(f"需要切换模型: {model_type} ({model_path})")

                # 检查当前使用的模型类型
                if predictor.loss_type == 'cross_entropy' and 'cross_entropy' not in model_path:
                    logger.warning(f"警告：从分类器模型切换到对比学习模型可能会影响性能！")
                elif predictor.loss_type == 'contrastive' and 'cross_entropy' in model_path:
                    logger.warning(f"警告：从对比学习模型切换到分类器模型可能会影响性能！")

                # 根据wait_for_model参数决定是否等待
                if request.wait_for_model:
                    # 清除事件，用于同步
                    global_state.model_load_event.clear()
                    # 在新线程中加载模型
                    thread = threading.Thread(
                        target=load_predictor_in_background, args=(model_path,))
                    thread.start()

                    logger.info(f"等待新模型加载完成...")
                    # 等待模型加载完成，设置超时
                    wait_success = global_state.model_load_event.wait(
                        timeout=MODEL_LOAD_TIMEOUT)
                    if not wait_success:
                        logger.warning(f"等待模型加载超时，使用当前可用模型")
                    else:
                        logger.info(f"模型加载完成，使用新模型处理请求")
                        # 重新获取加载后的预测器
                        predictor = global_state.predictor
                else:
                    # 不等待，在后台加载新模型
                    background_tasks.add_task(init_model, model_type)
                    logger.info(
                        f"在后台加载新模型，继续使用当前模型处理请求：{predictor.model_path}")

        # 检查输入列表长度是否一致
        list_lengths = [
            len(request.spl_room_names),
            len(request.spl_room_bed_names),
            len(request.s_room_names),
            len(request.s_room_bed_names)
        ]
        if len(set(list_lengths)) != 1:
            raise HTTPException(
                status_code=400,
                detail="所有输入列表的长度必须相同"
            )

        # 处理文本 - 这里注意要传入cleaner
        source_texts = []
        target_texts = []

        for spl_room, spl_bed in zip(
            request.spl_room_names,
            request.spl_room_bed_names
        ):
            source_text = process_text_with_prefix(
                None, spl_room, spl_bed, True, cleaner)
            source_texts.append(source_text)

        for s_room, s_bed in zip(
            request.s_room_names,
            request.s_room_bed_names
        ):
            target_text = process_text_with_prefix(
                None, s_room, s_bed, False, cleaner)
            target_texts.append(target_text)

        # 使用当前预测器处理请求
        result = predictor.predict(
            source_texts,
            target_texts,
            request.threshold
        )

        # 添加实际使用的模型信息
        result['model_used'] = os.path.basename(
            os.path.dirname(predictor.model_path))

        # 确保所有数值都是JSON兼容的
        def sanitize_value(value):
            if isinstance(value, (list, tuple)):
                return [sanitize_value(item) for item in value]
            elif isinstance(value, dict):
                return {k: sanitize_value(v) for k, v in value.items()}
            elif isinstance(value, float):
                if math.isnan(value) or math.isinf(value):
                    return 0.0
                return value
            return value

        # 处理结果中的所有值，确保它们是JSON兼容的
        result = sanitize_value(result)

        return result
    except Exception as e:
        # 捕获所有其他未处理的异常
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=13003)
