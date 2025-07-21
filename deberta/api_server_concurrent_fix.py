# -*- coding: utf-8 -*-
"""
SimCSE模型推理HTTP服务 - 优化版本
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
import hashlib
import asyncio
import queue
import threading
import numpy as np
from contextlib import asynccontextmanager
import gc
import weakref

# 配置区域
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建日志目录
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# 配置日志
APP_ENV = os.environ.get("APP_ENV", "development").lower()
LOG_LEVEL = "INFO" if APP_ENV == "production" else "DEBUG"
numeric_level = getattr(logging, LOG_LEVEL, logging.INFO)

# 定义日志轮转配置
LOG_FILE = os.path.join(BASE_DIR, "logs/api_server.log")
MAX_LOG_SIZE = 1024 * 1024 * 1024  # 1GB
BACKUP_COUNT = 5  # 保留5个备份文件

# 为不同环境设置不同的日志格式
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
if APP_ENV == "development":
    log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(pathname)s:%(lineno)d] - %(message)s'

logging.basicConfig(
    level=numeric_level,
    format=log_format,
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
logger.info(f"当前环境: {APP_ENV}, 日志级别: {LOG_LEVEL}")

# API令牌验证配置
API_TOKEN_HASH = "60df8f7a480258940917f0d04e74338bf76f53f449e227c771663a28bf43bd1c"

# 文本处理配置
UNKNOWN_TEXT = "[unknown]"
SUPPLY_PREFIX = "[supply]"
ROOM_PREFIX = "[room]"
BED_PREFIX = "[bed]"

# 模型路径映射
MODEL_PATHS = {
    'cross_entropy': os.path.join(BASE_DIR, 'checkpoints_cross_entropy_side_apartment_fix_5e-6/best_model')
}
DEFAULT_MODEL = 'cross_entropy'

MAPPING_PATH = os.path.join(BASE_DIR, 'data/mapping.xlsx')
MAX_LENGTH = 80
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEFAULT_THRESHOLD = 0.9

# 线程池配置 - 使用线程池而非多进程
MAX_WORKERS = 10  # 并发推理的最大线程数
inference_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# 批处理配置
MAX_BATCH_SIZE = 1024  # 最大批处理大小
BATCH_TIMEOUT = 0.01  # 批处理超时时间（秒）
MIN_BATCH_SIZE = 32  # 最小批处理大小

# 模型加载锁和信号量
model_lock = threading.Lock()
model_loading_lock = asyncio.Lock()  # 异步锁，防止并发加载

# 结果跟踪类


class Result:
    def __init__(self, timeout=10.0):
        self._event = asyncio.Event()
        self._result = None
        self._timeout = timeout

    async def get(self):
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
            return self._result
        except asyncio.TimeoutError:
            return {"error": "Result retrieval timed out"}

    def set(self, value):
        self._result = value
        self._event.set()

# 改进的批处理器类


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
        self._stop_event = threading.Event()
        self._stopped = False
        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        logger.info(f"批处理器已初始化，最大批处理大小: {max_batch_size}, 超时: {timeout}秒")

    def stop(self):
        """停止批处理器"""
        if not self._stopped:
            self._stopped = True
            self._stop_event.set()

            # 清空队列，通知所有等待的请求
            while not self.queue.empty():
                try:
                    future, _ = self.queue.get_nowait()
                    future.set({"error": "Batch processor is stopping"})
                except queue.Empty:
                    break

            # 等待工作线程结束
            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)

            logger.info("批处理器已停止")

    def _worker_loop(self):
        """工作线程循环，处理批量数据"""
        while not self._stop_event.is_set():
            batch_items = []
            batch_futures = []

            # 等待第一个项目
            try:
                first_future, first_item = self.queue.get(timeout=0.1)
                if self._stop_event.is_set():
                    first_future.set({"error": "Batch processor is stopping"})
                    break
                batch_items.append(first_item)
                batch_futures.append(first_future)
                self.queue.task_done()
            except queue.Empty:
                continue

            # 尝试获取更多项目，直到达到最大批量大小或超时
            start_time = time.time()
            while (len(batch_items) < self.max_batch_size and
                   time.time() - start_time < self.timeout and
                   not self._stop_event.is_set()):
                try:
                    future, item = self.queue.get(block=False)
                    batch_items.append(item)
                    batch_futures.append(future)
                    self.queue.task_done()
                except queue.Empty:
                    break

            # 处理批量数据
            if batch_items and not self._stop_event.is_set():
                try:
                    batch_size = len(batch_items)
                    logger.debug(f"处理批量数据，大小: {batch_size}")
                    batch_results = self.process_fn(batch_items)

                    # 设置结果
                    for future, result in zip(batch_futures, batch_results):
                        future.set(result)

                except Exception as e:
                    logger.error(f"处理批量数据时出错: {str(e)}")
                    # 在错误情况下，为所有项目设置错误结果
                    error_result = {"error": str(e)}
                    for future in batch_futures:
                        future.set(error_result)

    async def process(self, item):
        """
        异步处理单个项目

        Args:
            item: 要处理的项目

        Returns:
            处理结果
        """
        if self._stopped:
            return {"error": "Batch processor is stopped"}

        future = Result()
        self.queue.put((future, item))
        return await future.get()

# 全局状态管理类


class GlobalState:
    def __init__(self):
        self.predictor = None
        self.cleaner = None
        self.model_loading = False
        self.init_attempts = 0
        self.last_error = None
        self.model_path = None
        self.model_name = None
        self.pending_model_change = None  # 记录待加载的模型
        self._predictor_ref = None  # 使用弱引用

    def set_predictor(self, predictor):
        """设置预测器，使用弱引用"""
        if predictor is not None:
            self._predictor_ref = weakref.ref(predictor)
            self.predictor = predictor
        else:
            self._predictor_ref = None
            self.predictor = None

    def get_predictor(self):
        """获取预测器，检查弱引用是否有效"""
        if self._predictor_ref is not None:
            predictor = self._predictor_ref()
            if predictor is None:
                # 弱引用已失效
                self.predictor = None
            return predictor
        return None


# 初始化全局状态
global_state = GlobalState()

# 辅助函数


def verify_token(token: str):
    """验证令牌是否有效"""
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

# 改进的SimCSEPredictor类


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
            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # 使用半精度加载模型节省内存
            if self.use_fp16 and torch.cuda.is_available():
                logger.info("使用FP16加载模型以节省显存")
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

        # 设置模型为评估模式并禁用梯度计算
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # 初始化批处理器
        self.batch_processor = BatchProcessor(
            process_fn=self._batch_process_text_pairs,
            max_batch_size=MAX_BATCH_SIZE,
            timeout=BATCH_TIMEOUT
        )
        logger.info(f"初始化批处理器，最大批大小: {MAX_BATCH_SIZE}，超时: {BATCH_TIMEOUT}秒")

        # 统计请求处理时间
        self.total_processing_time = 0
        self.total_requests = 0

    def __del__(self):
        """析构函数，确保资源被释放"""
        self.cleanup()

    def cleanup(self):
        """清理所有资源"""
        try:
            # 停止批处理器
            if hasattr(self, 'batch_processor') and self.batch_processor is not None:
                self.batch_processor.stop()
                self.batch_processor = None

            # 删除模型
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None

            # 删除分词器
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            # 清理GPU内存
            self.cleanup_memory()

        except Exception as e:
            logger.error(f"清理预测器资源时出错: {str(e)}")

    def encode_text_pair(self, text1_list, text2_list, batch_size=256):
        """批量编码文本对，改进内存管理"""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer not available. Model may be in an invalid state.")
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError(
                "Model not available. Model may be in an invalid state.")

        total_len = len(text1_list)

        # 使用列表收集结果，避免大内存预分配
        all_logits_list = []

        for i in range(0, total_len, batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 确保在正确的设备上处理
            with torch.cuda.device(self.device if torch.cuda.is_available() else 'cpu'):
                # 对文本进行编码
                encoded = self.tokenizer(
                    batch_text1,
                    text_pair=batch_text2,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors='pt'
                )

                # 将数据移到设备
                encoded = {k: v.to(self.device) for k, v in encoded.items()}

                with torch.no_grad():
                    with autocast(enabled=self.use_fp16):
                        outputs = self.model(**encoded)
                        # 获取正类的logits得分
                        batch_logits = outputs.logits[:, 1]

                        # 立即转换到CPU并转为float32
                        batch_logits_cpu = batch_logits.float().cpu()
                        all_logits_list.append(batch_logits_cpu)

                        # 显式删除GPU张量
                        del outputs
                        del batch_logits

                # 立即清理输入张量
                for v in encoded.values():
                    del v
                del encoded

                # 每处理5个批次就清理一次GPU缓存
                if i > 0 and (i // batch_size) % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # 合并结果
        all_logits = torch.cat(all_logits_list, dim=0)

        # 清理临时列表
        del all_logits_list

        return all_logits

    def cleanup_memory(self):
        """清理GPU内存和缓存"""
        try:
            # 强制垃圾回收
            gc.collect()

            # 清理GPU缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                # 记录当前GPU使用情况
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                logger.info(
                    f"手动清理后GPU内存 - 已分配: {allocated:.2f}MB, 已预留: {reserved:.2f}MB")
        except Exception as e:
            logger.error(f"清理内存时出错: {str(e)}")

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度：将logits转换为概率值"""
        similarities = source_embeddings

        if normalize_to_probability:
            if similarities.dtype == torch.float16:
                similarities = similarities.float()
            similarities = torch.sigmoid(similarities)

        return similarities

    def _batch_process_text_pairs(self, items):
        """批量处理文本对"""
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
        """异步预测文本对的匹配情况"""
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
        with torch.no_grad():
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

        # 转换结果并清理GPU内存
        result = {
            'similarities': similarities.cpu().numpy().tolist(),
            'predictions': predictions.cpu().numpy().tolist(),
            'threshold': threshold
        }

        # 显式删除GPU张量
        del logits
        del similarities
        del predictions

        # 清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

# DataCleaner类保持不变


class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.room_rules, self.bed_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> Tuple[dict, List[Tuple[str, str, int]]]:
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        room_rules = {}
        bed_rules = []

        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)
            if not {'key', 'value'}.issubset(sheet_data.columns):
                continue

            current_sheet_rules = []
            for _, row in sheet_data.iterrows():
                key = str(row['key']).strip().lower()
                values = [v.strip().strip('"')
                          for v in str(row['value']).split(',')]
                rules = [(value.lower(), key, len(value))
                         for value in values if value]
                current_sheet_rules.extend(rules)

            current_sheet_rules = sorted(
                current_sheet_rules, key=lambda x: x[2], reverse=True)

            if sheet_name.lower() == 'bed':
                bed_rules = current_sheet_rules
                room_rules[sheet_name.lower()] = current_sheet_rules
            else:
                room_rules[sheet_name.lower()] = current_sheet_rules

        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        for sheet_name, rules in self.room_rules.items():
            for value, key, _ in rules:
                pattern = re.compile(
                    rf'(?:^|(?<=[^a-zA-Z0-9-]))'
                    rf'{re.escape(value)}'
                    rf'(?:$|(?=[^a-zA-Z0-9-]))',
                    re.IGNORECASE
                )
                if pattern.search(text_lower):
                    text_lower = pattern.sub(key, text_lower)
                    break

        return text_lower

    def clean_bed_text(self, text: str) -> str:
        if pd.isna(text):
            return text
        if '[unknown]' in text:
            return text

        text_lower = str(text).lower()
        for value, key, _ in self.bed_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'
                rf'{re.escape(value)}'
                rf'(?:$|(?=[^a-zA-Z0-9-]))',
                re.IGNORECASE
            )
            text_lower = pattern.sub(key, text_lower)

        return text_lower

# 改进的模型加载函数


def load_predictor_in_background(model_path: str, model_name: str):
    """在后台线程中加载预测器，改进内存管理"""
    global global_state

    try:
        with model_lock:
            # 检查是否已经在加载相同的模型
            if (global_state.model_loading and
                    global_state.pending_model_change == model_name):
                logger.info(f"模型 {model_name} 已在加载中，跳过重复加载")
                return

            # 如果当前模型就是要加载的模型，跳过
            if global_state.model_name == model_name and global_state.predictor is not None:
                logger.info(f"模型 {model_name} 已加载，跳过重复加载")
                return

            global_state.model_loading = True
            global_state.pending_model_change = model_name
            logger.info(f"开始在后台加载模型：{model_path}")
            start_time = time.time()

            # 在加载新模型前先进行内存清理
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                # 加载新模型
                new_predictor = SimCSEPredictor(model_path)

                # 保存旧模型引用
                old_predictor = global_state.predictor

                # 原子性地替换模型
                global_state.set_predictor(new_predictor)
                global_state.model_path = model_path
                global_state.model_name = model_name
                global_state.last_error = None
                global_state.pending_model_change = None

                # 清理旧模型
                if old_predictor is not None:
                    try:
                        logger.info("开始清理旧模型资源...")

                        # 等待正在处理的请求完成
                        time.sleep(1.0)

                        # 调用清理方法
                        old_predictor.cleanup()

                        # 删除旧模型引用
                        del old_predictor

                        # 强制垃圾回收
                        gc.collect()

                        # 清理GPU缓存
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()

                        logger.info("已释放旧模型资源")
                    except Exception as cleanup_e:
                        logger.warning(f"清理旧模型时出错: {str(cleanup_e)}")

                load_time = time.time() - start_time
                logger.info(f"模型加载成功，类型：{model_name}，耗时：{load_time:.2f}秒")

                # 记录内存使用情况
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1024 / 1024
                    reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    logger.info(
                        f"模型加载后GPU内存 - 已分配: {allocated:.2f}MB, 已预留: {reserved:.2f}MB")

            except Exception as e:
                error_msg = f"加载模型时出错：{str(e)}"
                global_state.last_error = error_msg
                global_state.pending_model_change = None
                logger.error(error_msg)
                import traceback
                logger.error(traceback.format_exc())

    finally:
        with model_lock:
            global_state.model_loading = False


def init_model(model_type=DEFAULT_MODEL):
    """初始化预测器，避免重复加载"""
    global global_state

    model_type = model_type if model_type and model_type.strip() else DEFAULT_MODEL
    logger.info(f"初始化模型类型: {model_type}")

    # 如果已经在加载中，则直接返回
    if global_state.model_loading:
        logger.info("模型正在加载中，跳过重复初始化")
        return

    # 如果当前模型就是要加载的模型，跳过
    if global_state.model_name == model_type and global_state.predictor is not None:
        logger.info(f"模型 {model_type} 已加载，跳过重复初始化")
        return

    with model_lock:
        global_state.init_attempts += 1

        # 获取模型路径
        model_name_to_load = model_type
        model_path = MODEL_PATHS.get(model_name_to_load)

        # 如果模型路径不在预定义字典中，则尝试构建可能的路径
        if model_path is None:
            possible_path = os.path.join(
                BASE_DIR, f'checkpoints_{model_name_to_load}/best_model')
            if os.path.exists(possible_path):
                model_path = possible_path
                logger.info(f"使用构建的模型路径: {model_path}")
            else:
                # 尝试查找可用的模型
                logger.warning(f"警告：未找到指定的模型类型: {model_name_to_load}，尝试查找可用模型")
                available_paths = []
                for name, path in MODEL_PATHS.items():
                    if os.path.exists(path):
                        available_paths.append((name, path))

                if available_paths:
                    model_name_to_load, model_path = available_paths[0]
                    logger.info(f"使用备选模型: {model_name_to_load} ({model_path})")
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

        # 使用单个线程执行器加载模型，避免创建过多线程
        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(load_predictor_in_background,
                        model_path, model_name_to_load)
        executor.shutdown(wait=False)  # 不等待完成


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


def get_predictor():
    """获取当前预测器实例，如果不存在则尝试初始化"""
    predictor = global_state.get_predictor()

    # 验证预测器是否有效
    if predictor is not None:
        # 检查关键属性是否存在
        if not hasattr(predictor, 'model') or predictor.model is None:
            logger.warning("预测器的model属性丢失，需要重新初始化")
            predictor = None
        elif not hasattr(predictor, 'tokenizer') or predictor.tokenizer is None:
            logger.warning("预测器的tokenizer属性丢失，需要重新初始化")
            predictor = None

    # 如果预测器无效且没有正在加载模型，则尝试初始化
    if predictor is None and not global_state.model_loading:
        logger.info(f"预测器未初始化或已失效，尝试加载默认模型: {DEFAULT_MODEL}")
        init_model(DEFAULT_MODEL)
        # 等待模型加载完成（最多等待10秒）
        for _ in range(100):
            if global_state.predictor is not None and not global_state.model_loading:
                break
            time.sleep(0.1)

    return global_state.predictor


def get_cleaner():
    """获取当前清洗器实例，如果不存在则尝试初始化"""
    if global_state.cleaner is None and not global_state.model_loading:
        init_cleaner()
    return global_state.cleaner

# 文本处理函数保持不变


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
        pattern = re.compile(r'^[\u4e00-\u9fff\s]+')
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
    if room:
        parts.append(f"{ROOM_PREFIX} {room}")
    if bed:
        parts.append(f"{BED_PREFIX} {bed}")
    # 小写
    parts = [part.lower() for part in parts]
    return " ".join(filter(None, parts))

# 定期清理任务


async def periodic_cleanup():
    """定期清理GPU内存的后台任务"""
    while True:
        try:
            await asyncio.sleep(60)  # 每分钟检查一次

            if torch.cuda.is_available():
                # 获取当前内存使用情况
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                total = torch.cuda.get_device_properties(
                    0).total_memory / 1024 / 1024
                usage_percent = (reserved / total) * 100

                # 如果内存使用超过80%，执行清理
                if usage_percent > 80:
                    logger.warning(f"GPU内存使用率过高: {usage_percent:.1f}%，执行清理...")

                    # 强制垃圾回收
                    gc.collect()

                    # 清理预测器的内存
                    if global_state.predictor is not None:
                        global_state.predictor.cleanup_memory()

                    # 再次检查内存
                    new_reserved = torch.cuda.memory_reserved() / 1024 / 1024
                    logger.info(f"清理后GPU内存: {new_reserved:.2f}MB")

        except Exception as e:
            logger.error(f"定期清理任务出错: {str(e)}")

# 生命周期管理


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行的代码
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

    # 启动定期清理任务
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield  # 这里是应用运行的部分

    # 关闭时执行的代码
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    # 清理资源
    if global_state.predictor is not None:
        global_state.predictor.cleanup()

    # 关闭线程池
    inference_executor.shutdown(wait=True)

    logger.info("API服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="房间匹配服务",
    description="基于SimCSE的房间文本匹配服务",
    version="1.0.0",
    lifespan=lifespan
)

# 请求和响应模型


class MatchRequest(BaseModel):
    spl_room_names: List[str]
    spl_room_bed_names: List[str]
    s_room_names: List[str]
    s_room_bed_names: List[str]
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL
    token: str = None


class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]

# API端点


@app.get("/health")
async def health_check():
    """健康检查接口，返回服务状态和GPU内存使用情况"""
    health_status = {
        "status": "healthy",
        "model_loaded": global_state.predictor is not None,
        "model_loading": global_state.model_loading,
        "model_path": global_state.model_path if global_state.predictor else None,
        "model_name": global_state.model_name,
        "pending_model": global_state.pending_model_change
    }

    # 添加GPU内存信息
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
        reserved = torch.cuda.memory_reserved() / 1024 / 1024
        total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024

        health_status["gpu_memory"] = {
            "allocated_mb": round(allocated, 2),
            "reserved_mb": round(reserved, 2),
            "total_mb": round(total, 2),
            "usage_percent": round((reserved / total) * 100, 2)
        }

    return health_status


@app.post("/cleanup")
async def manual_cleanup():
    """手动触发GPU内存清理"""
    if global_state.predictor is None:
        raise HTTPException(status_code=503, detail="模型未加载")

    try:
        # 执行垃圾回收
        gc.collect()

        # 清理预测器内存
        global_state.predictor.cleanup_memory()

        # 返回清理后的内存状态
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            reserved = torch.cuda.memory_reserved() / 1024 / 1024

            return {
                "status": "success",
                "message": "GPU内存清理完成",
                "gpu_memory": {
                    "allocated_mb": round(allocated, 2),
                    "reserved_mb": round(reserved, 2)
                }
            }
        else:
            return {"status": "success", "message": "内存清理完成（CPU模式）"}
    except Exception as e:
        logger.error(f"手动清理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"清理失败: {str(e)}")


@app.post("/match/", response_model=MatchResponse)
async def match_rooms(request: MatchRequest, background_tasks: BackgroundTasks):
    """
    房间文本匹配接口
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

        # 使用异步锁防止并发模型加载
        async with model_loading_lock:
            # 如果请求的模型与当前加载的不同，则在后台加载新模型
            if global_state.model_name != model_type and not global_state.model_loading:
                # 记录要切换的模型
                model_path = MODEL_PATHS.get(model_type)
                if model_path and os.path.exists(model_path):
                    logger.info(
                        f"请求ID: {request_id} - 需要切换模型: {model_type} ({model_path})")

                    # 在后台加载新模型
                    background_tasks.add_task(init_model, model_type)

                    logger.info(
                        f"请求ID: {request_id} - 继续使用当前模型处理请求：{predictor.model_path}")
                else:
                    logger.warning(
                        f"请求ID: {request_id} - 请求的模型类型无效: {model_type}")

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

        # 处理文本
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

    except HTTPException:
        raise
    except Exception as e:
        # 清理当前请求所占据的显存
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as cleanup_e:
            logger.warning(f"清理显存时出错: {str(cleanup_e)}")

        # 捕获所有其他未处理的异常
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

if __name__ == "__main__":
    # 使用单worker模式避免多进程GPU内存问题
    workers = 1  # 使用单worker

    # 根据环境设置uvicorn日志级别
    app_env = os.environ.get("APP_ENV", "development").lower()
    uvicorn_log_level = "info" if app_env == "development" else "warning"

    # 启动服务器
    uvicorn.run(
        "api_server_concurrent_fix:app",  # 直接传递app实例
        host="0.0.0.0",
        port=13003,
        workers=workers,  # 单worker模式
        log_level=uvicorn_log_level,
        access_log=(app_env == "development"),
        limit_concurrency=100,  # 限制并发连接数
        timeout_keep_alive=65,  # 保持连接超时时间
        loop="asyncio",  # 使用asyncio事件循环
    )
