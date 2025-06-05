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
import hashlib
import asyncio
import queue
import threading
import numpy as np
from contextlib import asynccontextmanager
import math

# ================= 全局配置 =================
config = {
    'host': '0.0.0.0',
    'port': 13101,
    'log_level': 'debug',
    'limit_concurrency': 100,
    'timeout_keep_alive': 65
}

# 基本配置
BASE_DIR = os.environ.get(
    'BASE_DIR', os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# 日志配置
LOG_FILE = os.path.join(BASE_DIR, "logs/api_server_mdeberta.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        RotatingFileHandler(LOG_FILE, maxBytes=1024*1024*1024, backupCount=5),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("api_server")

# API安全配置
API_TOKEN_HASH = "60df8f7a480258940917f0d04e74338bf76f53f449e227c771663a28bf43bd1c"

# 模型配置
TOKENIZER_MODEL_NAME = "microsoft/mdeberta-v3-base"
MODEL_PATHS = os.environ.get('MODEL_PATHS_JSON') and json.loads(os.environ.get('MODEL_PATHS_JSON')) or {
    'mdeberta': os.path.join(BASE_DIR, 'checkpoints_hotel_v4_12e/best_model')
}
DEFAULT_MODEL = os.environ.get('MODEL_TYPE', 'mdeberta')
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 80
DEFAULT_THRESHOLD = 0.9
USE_MIXED_PRECISION = False

# 并发和批处理配置
MAX_WORKERS = 10
MAX_BATCH_SIZE = 1024
BATCH_TIMEOUT = 0.01
MIN_BATCH_SIZE = 32

# 批处理器


class BatchProcessor:
    def __init__(self, process_fn, max_batch_size=MAX_BATCH_SIZE, timeout=BATCH_TIMEOUT):
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

            # 尝试获取更多项目
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
        # 获取唯一ID并将项目放入队列
        with self.lock:
            item_id = self.result_id
            self.result_id += 1
        self.queue.put((item_id, item))

        # 等待结果
        while True:
            with self.lock:
                if item_id in self.results:
                    result = self.results[item_id]
                    del self.results[item_id]
                    return result
            await asyncio.sleep(0.005)

# 安全工具函数


def verify_token(token: str):
    if not token:
        logger.warning("API请求缺少令牌")
        raise HTTPException(status_code=401, detail="缺少API令牌")
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if token_hash != API_TOKEN_HASH:
        logger.warning("API令牌验证失败")
        raise HTTPException(status_code=401, detail="API令牌无效")
    return True


# 模型加载和状态管理
model_lock = threading.Lock()


class GlobalState:
    def __init__(self):
        self.predictor = None
        self.cleaner = None
        self.model_loading = False
        self.init_attempts = 0
        self.last_error = None
        self.model_path = None


global_state = GlobalState()

# FastAPI生命周期管理


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"API服务启动")
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"BASE_DIR: {BASE_DIR}")

    # 初始化模型和清洗器
    logger.info("开始初始化模型...")
    init_model()

    logger.info("开始初始化文本清洗器...")
    init_cleaner()

    yield

    logger.info("API服务关闭")

# 创建FastAPI应用
app = FastAPI(
    title="房间匹配服务",
    description="基于SimCSE的房间文本匹配服务",
    version="1.0.0",
    lifespan=lifespan
)

# 请求/响应模型


class MatchRequest(BaseModel):
    hotel: List[str]
    address: List[str]
    s_hotel: List[str]
    s_address: List[str]
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL
    token: str = None


class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]

# 推理器实现


class SimCSEPredictor:
    def __init__(self, model_path: str):
        """初始化预测器"""
        self.device = torch.device(DEVICE)
        logger.info(f"使用设备: {self.device}")

        # 加载配置
        self._load_config(model_path)

        # 加载模型
        self._load_model(model_path)

        # 加载分词器
        self._load_tokenizer(model_path)

        # 初始化批处理和缓存
        self._init_batch_processor()
        self._init_cache()

    def _load_config(self, model_path):
        """加载配置文件"""
        config_dir = os.path.join(os.path.dirname(
            os.path.dirname(model_path)), 'config')
        config_path = os.path.join(config_dir, 'config_cross_entropy.json')

        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.temperature = self.config.get('temperature', 0.05)
        else:
            logger.warning(f"未找到配置文件: {config_path}，使用默认配置")
            self.temperature = 0.05
            self.config = {}

        # 配置混合精度
        self.use_mixed_precision = USE_MIXED_PRECISION and torch.cuda.is_available()
        logger.info(
            "启用半精度计算(FP16)" if self.use_mixed_precision else "使用标准精度计算(FP32)")

    def _load_model(self, model_path):
        """加载模型"""
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path).to(self.device)

            # 如果启用了混合精度，将模型转换为半精度
            if self.use_mixed_precision:
                self.model = self.model.half()
                logger.info("已将模型转换为半精度(FP16)")

            self.loss_type = 'cross_entropy'

            # 启用梯度检查点以节省显存
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                logger.info("已启用梯度检查点以节省显存")

            self.model.eval()
        except Exception as e:
            logger.error(f"加载模型时出错: {str(e)}")
            raise

    def _load_tokenizer(self, model_path):
        """加载tokenizer"""
        try:
            # 尝试从本地加载分词器
            logger.info("尝试从本地路径加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            logger.info("成功从本地路径加载分词器")
        except Exception as e:
            # 备选方案：从预训练模型加载
            logger.warning(f"从本地路径加载分词器失败: {str(e)}")
            logger.info(f"尝试从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_MODEL_NAME, use_fast=False)
            logger.info(f"成功从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model_path = model_path

    def _init_batch_processor(self):
        """初始化批处理器"""
        self.batch_processor = BatchProcessor(
            process_fn=self._batch_process_text_pairs,
            max_batch_size=MAX_BATCH_SIZE,
            timeout=BATCH_TIMEOUT
        )
        logger.info(f"初始化批处理器，最大批大小: {MAX_BATCH_SIZE}，超时: {BATCH_TIMEOUT}秒")

    def _init_cache(self):
        """初始化缓存和性能统计"""
        self.result_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.max_cache_size = 10000
        self.total_processing_time = 0
        self.total_requests = 0
        self.last_cleanup_time = time.time()
        self.cleanup_interval = 300

    def _cleanup_memory(self):
        """清理内存和缓存"""
        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("已执行内存清理")

    def encode_text_pair(self, text1_list, text2_list, batch_size=256):
        """批量编码文本对"""
        logits_list = []

        # 定期清理内存
        current_time = time.time()
        if current_time - self.last_cleanup_time > self.cleanup_interval:
            self._cleanup_memory()
            self.last_cleanup_time = current_time

        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 编码输入
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
                # 使用模型推理
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(**encoded)
                        batch_logits = outputs.logits[:, 1]
                else:
                    outputs = self.model(**encoded)
                    batch_logits = outputs.logits[:, 1]

                # 处理异常值
                if torch.isnan(batch_logits).any() or torch.isinf(batch_logits).any():
                    logger.warning("检测到NaN或inf值，替换为安全值")
                    batch_logits = batch_logits.float()
                    batch_logits = torch.nan_to_num(
                        batch_logits, nan=0.0, posinf=10.0, neginf=-10.0)

                # 在CPU上需要float32
                if batch_logits.device.type == 'cpu' and batch_logits.dtype != torch.float32:
                    batch_logits = batch_logits.float()

                logits_list.append(batch_logits.cpu().float())

                # 释放内存
                del encoded, outputs

        # 合并并返回结果
        return torch.cat(logits_list, dim=0).float()

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度并归一化"""
        # 在CPU上进行操作需要float32
        if normalize_to_probability:
            if source_embeddings.device.type == 'cpu' and source_embeddings.dtype != torch.float32:
                source_embeddings = source_embeddings.float()
            return torch.sigmoid(source_embeddings)
        return source_embeddings

    def _batch_process_text_pairs(self, items):
        """批量处理文本对"""
        source_texts = [item[0] for item in items]
        target_texts = [item[1] for item in items]
        thresholds = [item[2] for item in items]

        # 处理文本对
        logits = self.encode_text_pair(source_texts, target_texts)
        similarities = self.calculate_similarity(logits).cpu().numpy()

        # 构建结果
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

    async def predict_async(self, source_texts, target_texts, threshold=DEFAULT_THRESHOLD):
        """异步预测文本对的匹配程度"""
        if len(source_texts) != len(target_texts):
            raise ValueError("源文本和目标文本数量必须相同")

        start_time = time.time()
        total_pairs = len(source_texts)
        precision_str = "半精度FP16" if self.use_mixed_precision else "FP32全精度"
        logger.info(
            f"开始批量处理 {total_pairs} 对文本，批次大小: {MAX_BATCH_SIZE}，使用{precision_str}")

        # 分批处理
        results = []
        for i in range(0, len(source_texts), MAX_BATCH_SIZE):
            batch_sources = source_texts[i:i+MAX_BATCH_SIZE]
            batch_targets = target_texts[i:i+MAX_BATCH_SIZE]

            # 编码和计算相似度
            logits = self.encode_text_pair(batch_sources, batch_targets)
            similarities = self.calculate_similarity(logits)

            # 处理异常值
            if torch.isnan(similarities).any() or torch.isinf(similarities).any():
                similarities = torch.nan_to_num(
                    similarities, nan=0.0, posinf=1.0, neginf=0.0)

            # 转换结果
            similarities_np = similarities.cpu().float().numpy()
            similarities_list = [
                float(0.0) if (not isinstance(s, (int, float)) or math.isinf(s) or math.isnan(s))
                else float(s) for s in similarities_np.tolist()
            ]

            # 构建结果
            for similarity in similarities_list:
                prediction = 1.0 if similarity > threshold else 0.0
                results.append({
                    'similarities': [similarity],
                    'predictions': [prediction]
                })

        # 汇总结果
        all_similarities = []
        all_predictions = []
        for result in results:
            all_similarities.extend(result['similarities'])
            all_predictions.extend(result['predictions'])

        total_time = time.time() - start_time
        matched_pairs = sum(all_predictions)
        logger.info(
            f"批量处理完成，共 {total_pairs} 对文本，匹配 {matched_pairs} 对，总耗时: {total_time:.3f}秒，平均速度: {total_pairs/total_time:.1f}对/秒")

        return {
            'similarities': all_similarities,
            'predictions': all_predictions,
            'threshold': threshold
        }

# 模型和清洗器工具函数


def get_predictor():
    """获取当前预测器实例"""
    if global_state.predictor is None and not global_state.model_loading:
        logger.info(f"预测器未初始化，尝试加载默认模型: {DEFAULT_MODEL}")
        init_model(DEFAULT_MODEL)
    return global_state.predictor


def get_cleaner():
    """获取当前清洗器实例"""
    if global_state.cleaner is None and not global_state.model_loading:
        init_cleaner()
    return global_state.cleaner


def load_predictor_in_background(model_path):
    """在后台线程中加载预测器"""
    try:
        with model_lock:
            global_state.model_loading = True
            logger.info(f"开始在后台加载模型：{model_path}")
            start_time = time.time()

            try:
                predictor = SimCSEPredictor(model_path)
                global_state.predictor = predictor
                global_state.model_path = model_path
                global_state.last_error = None
                load_time = time.time() - start_time
                logger.info(
                    f"模型加载成功，类型：{predictor.loss_type}，耗时：{load_time:.2f}秒")
            except Exception as e:
                global_state.last_error = f"加载模型时出错：{str(e)}"
                logger.error(global_state.last_error)
    finally:
        with model_lock:
            global_state.model_loading = False


def init_model(model_type=None):
    """初始化预测器"""
    # 获取模型类型
    model_type = model_type or os.environ.get('MODEL_TYPE') or DEFAULT_MODEL
    logger.info(f"初始化模型类型: {model_type}")

    # 防止重复初始化
    if global_state.model_loading:
        logger.info("模型正在加载中，跳过重复初始化")
        return

    with model_lock:
        global_state.init_attempts += 1

        # 确定模型路径
        model_path = MODEL_PATHS.get(model_type)
        if model_path is None:
            possible_path = os.path.join(
                BASE_DIR, f'checkpoints_{model_type}/best_model')
            if os.path.exists(possible_path):
                model_path = possible_path
            else:
                # 尝试使用任何可用的模型
                for name, path in MODEL_PATHS.items():
                    if os.path.exists(path):
                        model_path = path
                        logger.warning(
                            f"找不到请求的模型: {model_type}，将使用可用模型: {name}")
                        break
                else:
                    global_state.last_error = f"错误：没有找到可用的模型"
                    logger.error(global_state.last_error)
                    return

        # 确认模型路径存在
        if not os.path.exists(model_path):
            global_state.last_error = f"错误：模型路径不存在: {model_path}"
            logger.error(global_state.last_error)
            return

        # 在后台加载
        ThreadPoolExecutor(max_workers=1).submit(
            load_predictor_in_background, model_path)


def init_cleaner():
    """初始化文本清洗器"""
    return


# API端点


@app.post("/hotel_match/", response_model=MatchResponse)
async def match_rooms(request: MatchRequest, background_tasks: BackgroundTasks):
    """房间文本匹配接口"""
    try:
        # 验证令牌
        # verify_token(request.token)

        request_id = time.strftime("%Y%m%d%H%M%S") + \
            str(int(time.time() * 1000) % 1000)

        # 获取预测器
        predictor = get_predictor()
        if predictor is None:
            if global_state.model_loading:
                logger.warning(f"请求ID: {request_id} - 模型正在加载中，请求被拒绝")
                raise HTTPException(status_code=503, detail="模型正在加载中，请稍后再试")
            else:
                error_detail = global_state.last_error or "模型尚未加载，请稍后再试"
                logger.error(f"请求ID: {request_id} - 模型未加载，错误: {error_detail}")
                raise HTTPException(status_code=500, detail=error_detail)

        # 检查模型类型
        model_type = request.model.strip().lower() if request.model else DEFAULT_MODEL
        if predictor.model_path != MODEL_PATHS.get(model_type) and not global_state.model_loading:
            logger.info(f"请求ID: {request_id} - 需要切换模型: {model_type}")
            background_tasks.add_task(init_model, model_type)
            logger.info(
                f"请求ID: {request_id} - 继续使用当前模型处理请求：{predictor.model_path}")

        # 检查输入列表长度
        list_lengths = [len(request.hotel), len(request.address), len(
            request.s_hotel), len(request.s_address)]
        if len(set(list_lengths)) != 1:
            logger.warning(f"请求ID: {request_id} - 输入列表长度不一致")
            raise HTTPException(status_code=400, detail="所有输入列表的长度必须相同")

        # 组装输入文本
        source_texts = [f"HotelName:{hotel},Address:{address}" for hotel, address in zip(
            request.hotel, request.address)]
        target_texts = [f"TargetHotel:{s_hotel},TargetAddress:{s_address}" for s_hotel, s_address in zip(
            request.s_hotel, request.s_address)]

        # 执行预测
        result = await predictor.predict_async(source_texts, target_texts, request.threshold)
        logger.info(
            f"请求ID: {request_id} - 匹配完成，共处理 {len(source_texts)} 对文本，预测匹配 {sum(result['predictions'])} 对")

        return result
    except Exception as e:
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

# 主函数
if __name__ == "__main__":
    import uvicorn
    import os
    # 自动设置PYTHONPATH为当前目录，兼容包导入
    os.environ['PYTHONPATH'] = os.getcwd()
    # 判断当前文件是否以包方式运行
    import sys
    if sys.argv[0].endswith("hotel_api_server_concurrent.py"):
        app_str = "hotel_api_server_concurrent:app"
    else:
        app_str = "hotel_match.hotel_api_server_concurrent:app"
    uvicorn.run(
        app_str,
        host=config['host'],
        port=config['port'],
        log_level=config['log_level'],
        limit_concurrency=config['limit_concurrency'],
        timeout_keep_alive=config['timeout_keep_alive']
    )
