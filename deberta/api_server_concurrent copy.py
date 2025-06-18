# -*- coding: utf-8 -*-
"""
SimCSE模型推理HTTP服务
"""

import os
import torch
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Tuple, Dict, Any, Union
import uvicorn
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.cuda.amp import autocast
import pandas as pd
import re
from pathlib import Path
import torch.nn.functional as F
import json
import time
import logging
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
import hashlib  # 添加hashlib库导入
import asyncio  # 添加asyncio库导入
import queue
import threading
import numpy as np
from contextlib import asynccontextmanager  # 导入 asynccontextmanager 用于 lifespan

# 配置区域
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建日志目录
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# 配置日志
# 定义日志轮转配置
LOG_FILE = os.path.join(BASE_DIR, "logs/api_server.log")
MAX_LOG_SIZE = 1024 * 1024 * 1024  # 1GB
BACKUP_COUNT = 5  # 保留5个备份文件

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(
            LOG_FILE,
            maxBytes=MAX_LOG_SIZE,
            backupCount=BACKUP_COUNT
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# API令牌验证配置
API_TOKEN_HASH = "60df8f7a480258940917f0d04e74338bf76f53f449e227c771663a28bf43bd1c"  # 将令牌加密存储

# 文本处理配置
UNKNOWN_TEXT = "[unknown]"
SUPPLY_PREFIX = "[supply]"
ROOM_PREFIX = "[room]"
BED_PREFIX = "[bed]"

# 模型路径映射
MODEL_PATHS = {
    'cross_entropy': os.path.join(BASE_DIR, 'checkpoints_cross_entropy_result1_nosplct/best_model')
}
DEFAULT_MODEL = 'cross_entropy'

MAPPING_PATH = os.path.join(BASE_DIR, 'data/mapping.xlsx')
MAX_LENGTH = 80
DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
DEFAULT_THRESHOLD = 0.9

# 线程池配置
# 用于处理模型推理的线程池
MAX_WORKERS = 10  # 并发推理的最大线程数
inference_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 批处理配置
MAX_BATCH_SIZE = 1024  # 最大批处理大小
BATCH_TIMEOUT = 0.01  # 批处理超时时间（秒）
MIN_BATCH_SIZE = 32  # 最小批处理大小


class BatchProcessor:
    def __init__(self, process_fn, max_batch_size=MAX_BATCH_SIZE, timeout=BATCH_TIMEOUT):
        """
        初始化批处理器

        Args:
            process_fn: 处理批量数据的函数
            max_batch_size: 最大批处理大小
            timeout: 等待更多数据的超时时间
        """
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = queue.Queue()
        self.results = {}
        self.lock = threading.Lock()
        self.result_id = 0
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"批处理器已初始化，最大批处理大小: {max_batch_size}, 超时: {timeout}秒")

    def _worker_loop(self):
        """工作线程循环，处理批量数据"""
        while True:
            # 收集批量数据
            batch_items = []
            batch_ids = []

            # 等待第一个项目
            try:
                first_id, first_item = self.queue.get(block=True)
                batch_items.append(first_item)
                batch_ids.append(first_id)
                self.queue.task_done()
            except queue.Empty:
                continue

            # 尝试获取更多项目，直到达到最大批量大小或超时
            try:
                start_time = time.time()
                while len(batch_items) < self.max_batch_size and time.time() - start_time < self.timeout:
                    try:
                        item_id, item = self.queue.get(
                            block=True, timeout=self.timeout)
                        batch_items.append(item)
                        batch_ids.append(item_id)
                        self.queue.task_done()
                    except queue.Empty:
                        break
            except Exception as e:
                logger.error(f"批处理器收集项目时出错: {str(e)}")

            # 处理批量数据
            if batch_items:
                try:
                    batch_size = len(batch_items)
                    logger.debug(f"处理批量数据，大小: {batch_size}")
                    batch_results = self.process_fn(batch_items)

                    # 保存结果
                    with self.lock:
                        for idx, batch_id in enumerate(batch_ids):
                            self.results[batch_id] = batch_results[idx]
                except Exception as e:
                    logger.error(f"处理批量数据时出错: {str(e)}")
                    # 在错误情况下，为所有项目设置错误结果
                    with self.lock:
                        for batch_id in batch_ids:
                            self.results[batch_id] = {"error": str(e)}

    async def process(self, item):
        """
        异步处理单个项目

        Args:
            item: 要处理的项目

        Returns:
            处理结果
        """
        # 获取唯一ID
        with self.lock:
            item_id = self.result_id
            self.result_id += 1

        # 将项目放入队列
        self.queue.put((item_id, item))

        # 等待结果
        while True:
            with self.lock:
                if item_id in self.results:
                    result = self.results[item_id]
                    del self.results[item_id]
                    return result
            await asyncio.sleep(0.001)  # 短暂休眠以避免CPU过载

# 辅助函数


def verify_token(token: str):
    """验证令牌是否有效

    Args:
        token: 要验证的令牌

    Raises:
        HTTPException: 如果令牌无效则抛出401错误
    """
    if not token:
        logger.warning("API请求缺少令牌")
        raise HTTPException(
            status_code=401,
            detail="缺少API令牌"
        )

    # 对提供的令牌进行SHA-256哈希
    token_hash = hashlib.sha256(token.encode()).hexdigest()

    # 比较哈希值
    if token_hash != API_TOKEN_HASH:
        logger.warning("API令牌验证失败")
        raise HTTPException(
            status_code=401,
            detail="API令牌无效"
        )

    return True


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


# 初始化全局状态
global_state = GlobalState()

# 定义生命周期管理器，替代原有的 on_event


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行的代码，相当于原来的 startup_event
    logger.info(f"API服务启动")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(
        f"日志文件: {LOG_FILE}，最大大小: {MAX_LOG_SIZE/1024/1024/1024:.1f}GB，保留备份数: {BACKUP_COUNT}")

    # 初始化模型
    logger.info("开始初始化模型...")
    init_model()

    # 初始化清洗器
    logger.info("开始初始化文本清洗器...")
    init_cleaner()

    yield  # 这里是应用运行的部分

    # 关闭时执行的代码，相当于原来的 shutdown_event
    logger.info("API服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="房间匹配服务",
    description="基于SimCSE的房间文本匹配服务",
    version="1.0.0",
    lifespan=lifespan  # 使用新的生命周期管理器
)

# 请求模型


class MatchRequest(BaseModel):
    spl_room_names: List[str]
    spl_room_bed_names: List[str]
    s_room_names: List[str]
    s_room_bed_names: List[str]
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL
    token: str = None  # 添加token字段

# 响应模型


class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]


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

        # 设置是否使用FP16精度
        self.use_fp16 = True
        if torch.cuda.is_available():
            # 检查GPU是否支持混合精度
            gpu_cap = torch.cuda.get_device_capability(0)
            if gpu_cap[0] < 7:  # 7.0 = Volta 架构
                self.use_fp16 = False
                logger.warning(
                    f"当前GPU架构 ({gpu_cap[0]}.{gpu_cap[1]}) 不支持高效的FP16，将使用FP32")
            else:
                logger.info(f"当前GPU支持高效FP16计算，已启用混合精度")

        # 加载模型并转移到设备
        try:
            # 使用半精度加载模型节省内存
            if self.use_fp16 and torch.cuda.is_available():
                logger.info("使用FP16加载模型以节省显存")
                with torch.cuda.amp.autocast():
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16
                    ).to(self.device)
            else:
                logger.info("使用FP32加载模型")
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_path).to(self.device)

            self.loss_type = 'cross_entropy'
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise

        # 释放未使用的CUDA内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info(
                f"已清理GPU缓存，当前占用: {torch.cuda.memory_allocated()/1024/1024:.2f}MB，缓存: {torch.cuda.memory_reserved()/1024/1024:.2f}MB")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_path = model_path
        self.model.eval()

        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("已启用梯度检查点以节省显存")

        # 初始化批处理器
        self.batch_processor = BatchProcessor(
            process_fn=self._batch_process_text_pairs,
            max_batch_size=MAX_BATCH_SIZE,
            timeout=BATCH_TIMEOUT
        )
        logger.info(f"初始化批处理器，最大批大小: {MAX_BATCH_SIZE}，超时: {BATCH_TIMEOUT}秒")

        # 结果缓存，用于频繁使用的相同输入（避免重复计算）
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000  # 最大缓存条目数

        # 统计请求处理时间
        self.total_processing_time = 0
        self.total_requests = 0

        # 定期清理内存
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300  # 5分钟清理一次内存

    def encode_batch(self, texts, batch_size=256):
        """编码文本 (适用于cross_entropy模式)"""
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
                with autocast(enabled=self.use_fp16):
                    outputs = self.model(**encoded)
                    # 获取正类的logits得分
                    batch_embeddings = outputs.logits[:, 1]

                    # 确保返回的是float类型，避免后续操作出错
                    if batch_embeddings.dtype == torch.float16:
                        batch_embeddings = batch_embeddings.float()

                    embeddings.append(batch_embeddings.cpu())

        return torch.cat(embeddings, dim=0)

    def encode_text_pair(self, text1_list, text2_list, batch_size=256):
        """批量编码文本对，用于cross_entropy模式"""
        logits_list = []

        # 检查是否需要清理内存
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self.last_cleanup_time = current_time

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
                with autocast(enabled=self.use_fp16):
                    outputs = self.model(**encoded)
                    # 获取正类的logits得分
                    batch_logits = outputs.logits[:, 1]
                    # 确保返回的是float类型，避免后续sigmoid操作出错
                    if batch_logits.dtype == torch.float16:
                        batch_logits = batch_logits.float()
                    logits_list.append(batch_logits.cpu())

                # 清理当前批次的GPU内存
                del encoded
                del outputs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        return torch.cat(logits_list, dim=0)

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度：将logits转换为概率值

        Args:
            source_embeddings: 源文本logits
            normalize_to_probability: 是否将结果归一化为0-1之间的概率值
        """
        # 交叉熵模式：直接使用logits作为相似度分数
        similarities = source_embeddings

        # 使用sigmoid函数将logits转换为0-1之间的概率值
        if normalize_to_probability:
            # 确保张量为float类型，解决"sigmoid_cpu" not implemented for 'Half'的问题
            if similarities.dtype == torch.float16:
                similarities = similarities.float()
            similarities = torch.sigmoid(similarities)

        return similarities

    def predict(self, source_texts: List[str], target_texts: List[str], threshold: float = DEFAULT_THRESHOLD):
        """预测文本对的匹配情况"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        # 检查缓存
        cache_hits = 0
        cache_keys = []
        result_similarities = []
        uncached_indices = []
        uncached_sources = []
        uncached_targets = []

        for i, (source, target) in enumerate(zip(source_texts, target_texts)):
            # 创建缓存键
            cache_key = hashlib.md5(
                f"{source}:::{target}:::{threshold}".encode()).hexdigest()
            cache_keys.append(cache_key)

            # 尝试从缓存获取结果
            if cache_key in self.result_cache:
                result_similarities.append(self.result_cache[cache_key])
                cache_hits += 1
            else:
                # 未缓存的需要计算
                result_similarities.append(None)  # 占位符
                uncached_indices.append(i)
                uncached_sources.append(source)
                uncached_targets.append(target)

        # 如果有未缓存的结果，计算它们
        if uncached_sources:
            logits = self.encode_text_pair(uncached_sources, uncached_targets)
            uncached_similarities = self.calculate_similarity(
                logits).cpu().numpy().tolist()

            # 更新结果和缓存
            for i, sim, idx in zip(range(len(uncached_indices)), uncached_similarities, uncached_indices):
                result_similarities[idx] = sim

                # 更新缓存
                self.result_cache[cache_keys[idx]] = sim

        # 更新缓存统计
        self.cache_hits += cache_hits
        self.cache_misses += len(uncached_sources)

        # 如果缓存太大，移除最旧的条目
        if len(self.result_cache) > self.max_cache_size:
            # 移除10%的缓存条目
            items_to_remove = int(self.max_cache_size * 0.1)
            keys_to_remove = list(self.result_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.result_cache[key]
            logger.info(f"缓存清理：移除了 {len(keys_to_remove)} 条缓存项")

        # 根据阈值进行预测
        predictions = [1.0 if sim >
                       threshold else 0.0 for sim in result_similarities]

        return {
            'similarities': result_similarities,
            'predictions': predictions,
            'threshold': threshold
        }

    def _batch_process_text_pairs(self, items):
        """
        批量处理文本对

        Args:
            items: 批量项目列表，每个项目是(源文本, 目标文本, 阈值)的元组

        Returns:
            处理结果列表
        """
        source_texts = [item[0] for item in items]
        target_texts = [item[1] for item in items]
        thresholds = [item[2] for item in items]

        # 编码所有文本对
        logits = self.encode_text_pair(source_texts, target_texts)
        similarities = self.calculate_similarity(logits).cpu().numpy()

        # 为每个项目创建结果
        results = []
        for i, threshold in enumerate(thresholds):
            similarity = float(similarities[i])
            prediction = 1.0 if similarity > threshold else 0.0

            results.append({
                'similarities': [similarity],
                'predictions': [prediction],
                'threshold': threshold
            })

        return results

    async def predict_async(self, source_texts: List[str], target_texts: List[str], threshold: float = DEFAULT_THRESHOLD):
        """异步预测文本对的匹配情况，使用线程池避免阻塞事件循环"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        # 对于大批量数据，使用标准方法
        if len(source_texts) >= MIN_BATCH_SIZE:
            # 使用线程池运行计算密集型的推理任务
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_executor,
                self._predict_in_thread,
                source_texts, target_texts, threshold
            )
            return result
        else:
            # 对于小批量，使用批处理队列
            results = []
            for source, target in zip(source_texts, target_texts):
                # 为每个文本对创建任务并提交到批处理队列
                result = await self.batch_processor.process((source, target, threshold))
                results.append(result)

            # 合并结果
            all_similarities = []
            all_predictions = []

            for result in results:
                all_similarities.extend(result['similarities'])
                all_predictions.extend(result['predictions'])

            return {
                'similarities': all_similarities,
                'predictions': all_predictions,
                'threshold': threshold
            }

    def _predict_in_thread(self, source_texts, target_texts, threshold):
        """在线程中执行的推理函数"""
        start_time = time.time()

        # 执行推理
        logits = self.encode_text_pair(source_texts, target_texts)
        similarities = self.calculate_similarity(logits)
        predictions = (similarities > threshold).float()

        # 记录处理时间
        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.total_requests += 1

        # 记录大批量推理的性能
        if len(source_texts) > 10:
            logger.info(f"批量推理完成 - 处理了 {len(source_texts)} 对文本，耗时: {processing_time:.3f}秒，"
                        f"平均每对: {processing_time/len(source_texts)*1000:.2f}毫秒")

        return {
            'similarities': similarities.cpu().numpy().tolist(),
            'predictions': predictions.cpu().numpy().tolist(),
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

# 恢复误删的函数


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
    - token: API令牌 (必填)

    返回：
    - similarities: 相似度列表
    - predictions: 预测结果列表（1表示匹配，0表示不匹配）
    - threshold: 使用的阈值
    """
    try:
        # 验证令牌
        verify_token(request.token)

        # 记录请求输入数据
        request_id = time.strftime("%Y%m%d%H%M%S") + \
            str(int(time.time() * 1000) % 1000)

        # 获取预测器实例
        predictor = get_predictor()
        cleaner = get_cleaner()

        # 检查模型是否已加载
        if predictor is None:
            if global_state.model_loading:
                logger.warning(f"请求ID: {request_id} - 模型正在加载中，请求被拒绝")
                raise HTTPException(
                    status_code=503,
                    detail="模型正在加载中，请稍后再试"
                )
            else:
                error_detail = global_state.last_error or "模型尚未加载，请稍后再试"
                logger.error(f"请求ID: {request_id} - 模型未加载，错误: {error_detail}")
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
            else:
                model_type = DEFAULT_MODEL
        else:
            model_type = DEFAULT_MODEL
            # logger.info(f"请求ID: {request_id} - 请求未指定模型类型，使用默认模型: {model_type}")

        # 确定模型路径
        model_path = os.path.join(
            BASE_DIR, f'checkpoints_{model_type}/best_model')

        # 如果模型类型与当前加载的不同，在后台加载新模型
        if predictor.model_path != model_path and not global_state.model_loading:
            # 记录要切换的模型
            # logger.info(
            #     f"请求ID: {request_id} - 需要切换模型: {model_type} ({model_path})")

            # 在后台加载新模型
            background_tasks.add_task(init_model, model_type)

            # 返回使用当前可用模型的信息
            # logger.info(
            #     f"请求ID: {request_id} - 继续使用当前模型处理请求：{predictor.model_path}")

        # 检查输入列表长度是否一致
        list_lengths = [
            len(request.spl_room_names),
            len(request.spl_room_bed_names),
            len(request.s_room_names),
            len(request.s_room_bed_names)
        ]
        if len(set(list_lengths)) != 1:
            logger.warning(f"请求ID: {request_id} - 输入列表长度不一致: {list_lengths}")
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

        # 使用异步预测方法处理请求
        result = await predictor.predict_async(
            source_texts,
            target_texts,
            request.threshold
        )

        return result
    except Exception as e:
        # 清理当前请求所占据的显存
        try:
            import gc
            # 强制垃圾回收
            gc.collect()
            # 清理CUDA缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # 尝试释放更多显存
                torch.cuda.synchronize()
        except Exception as cleanup_e:
            logger.warning(f"清理显存时出错: {str(cleanup_e)}")

        # 捕获所有其他未处理的异常
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


if __name__ == "__main__":
    # 获取可用的worker数量
    workers = 16

    # 启动配置，使用更多的worker以提高并发处理能力
    # 修改启动方式，使用模块导入字符串而不是直接传递应用实例
    uvicorn.run(
        "api_server_concurrent:app",  # 使用导入字符串
        host="0.0.0.0",
        port=13009,
        workers=workers,  # 使用多个worker进程
        log_level="warning",  # 改为warning级别，减少info日志输出
        access_log=False,  # 关闭HTTP访问日志
        limit_concurrency=100,  # 限制并发连接数
        timeout_keep_alive=65,  # 保持连接超时时间
    )
