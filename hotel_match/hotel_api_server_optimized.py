# -*- coding: utf-8 -*-
"""
酒店匹配推理HTTP服务 - 优化版本
"""

import os
import gc
import time
import json
import math
import hashlib
import asyncio
import threading
import weakref
from typing import List
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
import queue

import torch
from torch.cuda.amp import autocast
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn
import logging

# ============================================================================
# 配置区域
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 基础配置
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MAX_LENGTH = 80
DEFAULT_THRESHOLD = 0.9
DEFAULT_MODEL = 'mdeberta'
TOKENIZER_MODEL_NAME = "microsoft/mdeberta-v3-base"
USE_FP16 = False

# 模型路径
MODEL_PATHS = {
    'mdeberta': os.path.join(BASE_DIR, 'checkpoints_hotel_v5/best_model')
}

# 并发配置
MAX_WORKERS = 10
MAX_BATCH_SIZE = 1024
BATCH_TIMEOUT = 0.01
MIN_BATCH_SIZE = 32

# 内存管理配置
MEMORY_CHECK_INTERVAL = 60  # 秒
MEMORY_THRESHOLD = 80  # GPU使用率百分比
CLEANUP_INTERVAL = 300  # 秒

# API安全配置
API_TOKEN_HASH = "60df8f7a480258940917f0d04e74338bf76f53f449e227c771663a28bf43bd1c"

# 配置日志（精简版）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("hotel_api")

# ============================================================================
# 核心类定义
# ============================================================================


class Result:
    """异步结果跟踪器"""

    def __init__(self, timeout=10.0):
        self._event = asyncio.Event()
        self._result = None
        self._timeout = timeout

    async def get(self):
        try:
            await asyncio.wait_for(self._event.wait(), timeout=self._timeout)
            return self._result
        except asyncio.TimeoutError:
            return {"error": "Request timeout"}

    def set(self, value):
        self._result = value
        self._event.set()


class BatchProcessor:
    """智能批处理器"""

    def __init__(self, process_fn, max_batch_size=MAX_BATCH_SIZE, timeout=BATCH_TIMEOUT):
        self.process_fn = process_fn
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.queue = queue.Queue()
        self._stop_event = threading.Event()
        self._stopped = False

        self.worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self.worker_thread.start()

    def stop(self):
        if not self._stopped:
            self._stopped = True
            self._stop_event.set()

            # 清空队列并通知所有等待的请求
            while not self.queue.empty():
                try:
                    future, _ = self.queue.get_nowait()
                    future.set({"error": "Batch processor is stopping"})
                except queue.Empty:
                    break

            if self.worker_thread.is_alive():
                self.worker_thread.join(timeout=2.0)

    def _worker_loop(self):
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

            # 收集更多项目
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
                    batch_results = self.process_fn(batch_items)
                    for future, result in zip(batch_futures, batch_results):
                        future.set(result)
                except Exception as e:
                    error_result = {"error": str(e)}
                    for future in batch_futures:
                        future.set(error_result)

    async def process(self, item):
        if self._stopped:
            return {"error": "Batch processor is stopped"}

        future = Result()
        self.queue.put((future, item))
        return await future.get()


class GlobalState:
    """全局状态管理器"""

    def __init__(self):
        self.predictor = None
        self.model_loading = False
        self.last_error = None
        self.model_path = None
        self.model_name = None
        self.pending_model_change = None
        self._predictor_ref = None

    def set_predictor(self, predictor):
        if predictor is not None:
            self._predictor_ref = weakref.ref(predictor)
            self.predictor = predictor
        else:
            self._predictor_ref = None
            self.predictor = None

    def get_predictor(self):
        if self._predictor_ref is not None:
            predictor = self._predictor_ref()
            if predictor is None:
                self.predictor = None
            return predictor
        return None


class HotelPredictor:
    """酒店匹配预测器"""

    def __init__(self, model_path: str):
        """初始化预测器（参考inference.py的结构）"""
        self.device = torch.device(DEVICE)
        self.model_path = model_path
        logger.info(f"使用设备: {self.device}")

        # 配置混合精度
        self.use_fp16 = USE_FP16 and torch.cuda.is_available()
        if torch.cuda.is_available():
            gpu_cap = torch.cuda.get_device_capability(0)
            if gpu_cap[0] < 7:
                self.use_fp16 = False

        # 加载配置和模型
        self._load_config(model_path)
        self._load_tokenizer(model_path)
        self._load_model()

        # 初始化批处理器
        self.batch_processor = BatchProcessor(
            process_fn=self._batch_process_text_pairs,
            max_batch_size=MAX_BATCH_SIZE,
            timeout=BATCH_TIMEOUT
        )

        # 性能统计
        self.total_processing_time = 0
        self.total_requests = 0

    def _load_config(self, model_path):
        """加载配置文件（参考inference.py的结构）"""
        config_path = os.path.join(BASE_DIR, 'config', 'config_mdeberta.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            self.temperature = self.config.get('temperature', 0.05)
        else:
            logger.warning("未找到配置文件，使用默认配置")
            self.temperature = 0.05

    def _load_model(self):
        """加载模型（参考inference.py的结构）"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")

        logger.info("加载mDeBERTa Classifier模型")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path).to(self.device)

        # 处理精度设置（参考inference.py）
        if self.use_fp16:
            self.model = self.model.half()
            logger.info("已将模型转换为半精度(FP16)计算")
        else:
            self.model = self.model.float()
            logger.info("已将模型转换为全精度(float32)计算")

        self.loss_type = 'cross_entropy'

        # 启用梯度检查点以节省显存
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()

        self.model.eval()

    def _load_tokenizer(self, model_path):
        """加载分词器（参考inference.py的结构）"""
        try:
            logger.info("尝试从本地路径加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, use_fast=False)
            logger.info("成功从本地路径加载分词器")
        except Exception as e:
            logger.warning(f"从本地路径加载分词器失败: {str(e)}")
            logger.info(f"尝试从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                TOKENIZER_MODEL_NAME, use_fast=False)
            logger.info(f"成功从预训练模型加载分词器: {TOKENIZER_MODEL_NAME}")

        # 清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def cleanup(self):
        """清理所有资源"""
        try:
            if hasattr(self, 'batch_processor') and self.batch_processor is not None:
                self.batch_processor.stop()
                self.batch_processor = None

            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None

            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None

            self.cleanup_memory()
        except Exception as e:
            logger.error(f"清理资源时出错: {str(e)}")

    def cleanup_memory(self):
        """清理GPU内存"""
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception as e:
            logger.error(f"清理内存时出错: {str(e)}")

    def encode_text_pair(self, text1_list, text2_list, batch_size=256):
        """批量编码文本对"""
        if not hasattr(self, 'tokenizer') or self.tokenizer is None:
            raise RuntimeError("Tokenizer not available")
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("Model not available")

        logits_list = []

        for i in range(0, len(text1_list), batch_size):
            batch_text1 = text1_list[i:i + batch_size]
            batch_text2 = text2_list[i:i + batch_size]

            # 编码（参考inference.py的参数设置）
            encoded = self.tokenizer(
                batch_text1,
                text_pair=batch_text2,
                padding=True,
                truncation='only_first',
                max_length=MAX_LENGTH,
                return_tensors='pt',
                return_attention_mask=True,
                return_token_type_ids=True,
                verbose=False
            )
            encoded = {k: v.to(self.device) for k, v in encoded.items()}

            with torch.no_grad():
                # 推理
                outputs = self.model(**encoded)

                # 处理异常值（参考inference.py的处理方式）
                if torch.isnan(outputs.logits).any():
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

    def calculate_similarity(self, source_embeddings, normalize_to_probability=True):
        """计算相似度"""
        # 确保为float32类型并应用sigmoid（参考inference.py）
        if normalize_to_probability:
            if source_embeddings.dtype != torch.float32:
                source_embeddings = source_embeddings.float()
            return torch.sigmoid(source_embeddings)
        return source_embeddings

    def _batch_process_text_pairs(self, items):
        """批量处理文本对"""
        source_texts = [item[0] for item in items]
        target_texts = [item[1] for item in items]
        thresholds = [item[2] for item in items]

        logits = self.encode_text_pair(source_texts, target_texts)
        similarities = self.calculate_similarity(logits).cpu().numpy()

        results = []
        for i, threshold in enumerate(thresholds):
            similarity = float(similarities[i])
            if math.isnan(similarity) or math.isinf(similarity):
                similarity = 0.0
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

        if len(source_texts) >= MIN_BATCH_SIZE:
            # 大批量：使用线程池
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                inference_executor,
                self._predict_in_thread,
                source_texts, target_texts, threshold
            )
            return result
        else:
            # 小批量：使用批处理队列
            results = []
            for source, target in zip(source_texts, target_texts):
                result = await self.batch_processor.process((source, target, threshold))
                results.append(result)

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

        with torch.no_grad():
            logits = self.encode_text_pair(source_texts, target_texts)
            similarities = self.calculate_similarity(logits)

            # 处理异常值（参考inference.py的处理方式）
            if torch.isnan(similarities).any():
                similarities = torch.nan_to_num(
                    similarities, nan=0.0, posinf=1.0, neginf=0.0)

            predictions = (similarities > threshold).float()

        processing_time = time.time() - start_time
        self.total_processing_time += processing_time
        self.total_requests += 1

        # 安全转换结果
        similarities_np = similarities.cpu().numpy()
        similarities_list = [
            float(0.0) if (not isinstance(s, (int, float)) or math.isinf(s) or math.isnan(s))
            else float(s) for s in similarities_np.tolist()
        ]

        result = {
            'similarities': similarities_list,
            'predictions': predictions.cpu().numpy().tolist(),
            'threshold': threshold
        }

        return result

# ============================================================================
# 全局状态和工具函数
# ============================================================================


global_state = GlobalState()
model_lock = threading.Lock()
model_loading_lock = asyncio.Lock()
inference_executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


def verify_token(token: str):
    """验证API令牌"""
    if not token:
        raise HTTPException(status_code=401, detail="缺少API令牌")

    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if token_hash != API_TOKEN_HASH:
        raise HTTPException(status_code=401, detail="API令牌无效")

    return True


def load_predictor_in_background(model_path: str, model_name: str):
    """在后台线程中加载预测器"""
    try:
        with model_lock:
            if (global_state.model_loading and
                    global_state.pending_model_change == model_name):
                return

            if (global_state.model_name == model_name and
                    global_state.predictor is not None):
                return

            global_state.model_loading = True
            global_state.pending_model_change = model_name

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            try:
                new_predictor = HotelPredictor(model_path)
                old_predictor = global_state.predictor

                global_state.set_predictor(new_predictor)
                global_state.model_path = model_path
                global_state.model_name = model_name
                global_state.last_error = None
                global_state.pending_model_change = None

                if old_predictor is not None:
                    try:
                        time.sleep(1.0)
                        old_predictor.cleanup()
                        del old_predictor
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
                    except Exception:
                        pass

            except Exception as e:
                global_state.last_error = f"加载模型时出错：{str(e)}"
                global_state.pending_model_change = None
                logger.error(global_state.last_error)

    finally:
        with model_lock:
            global_state.model_loading = False


def init_model(model_type=DEFAULT_MODEL):
    """初始化预测器"""
    model_type = model_type if model_type and model_type.strip() else DEFAULT_MODEL

    if global_state.model_loading:
        return

    if (global_state.model_name == model_type and
            global_state.predictor is not None):
        return

    with model_lock:
        model_path = MODEL_PATHS.get(model_type)

        if model_path is None:
            possible_path = os.path.join(
                BASE_DIR, f'checkpoints_{model_type}/best_model')
            if os.path.exists(possible_path):
                model_path = possible_path
            else:
                available_paths = [(name, path) for name, path in MODEL_PATHS.items()
                                   if os.path.exists(path)]
                if available_paths:
                    model_type, model_path = available_paths[0]
                else:
                    global_state.last_error = "没有找到可用的模型"
                    logger.error(global_state.last_error)
                    return

        if not os.path.exists(model_path):
            global_state.last_error = f"模型路径不存在: {model_path}"
            logger.error(global_state.last_error)
            return

        executor = ThreadPoolExecutor(max_workers=1)
        executor.submit(load_predictor_in_background, model_path, model_type)
        executor.shutdown(wait=False)


def get_predictor():
    """获取当前预测器实例"""
    predictor = global_state.get_predictor()

    if predictor is not None:
        if not hasattr(predictor, 'model') or predictor.model is None:
            predictor = None
        elif not hasattr(predictor, 'tokenizer') or predictor.tokenizer is None:
            predictor = None

    if predictor is None and not global_state.model_loading:
        init_model(DEFAULT_MODEL)
        for _ in range(100):
            if global_state.predictor is not None and not global_state.model_loading:
                break
            time.sleep(0.1)

    return global_state.predictor


async def periodic_cleanup():
    """定期清理GPU内存的后台任务"""
    while True:
        try:
            await asyncio.sleep(MEMORY_CHECK_INTERVAL)

            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                reserved = torch.cuda.memory_reserved() / 1024 / 1024
                total = torch.cuda.get_device_properties(
                    0).total_memory / 1024 / 1024
                usage_percent = (reserved / total) * 100

                if usage_percent > MEMORY_THRESHOLD:
                    gc.collect()
                    if global_state.predictor is not None:
                        global_state.predictor.cleanup_memory()

        except Exception as e:
            logger.error(f"定期清理任务出错: {str(e)}")

# ============================================================================
# FastAPI应用
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("API服务启动")
    init_model()
    cleanup_task = asyncio.create_task(periodic_cleanup())

    yield

    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    if global_state.predictor is not None:
        global_state.predictor.cleanup()

    inference_executor.shutdown(wait=True)
    logger.info("API服务关闭")


app = FastAPI(
    title="酒店匹配服务",
    description="基于深度学习的酒店文本匹配服务",
    version="1.0.0",
    lifespan=lifespan
)


class MatchRequest(BaseModel):
    hotel: List[str]
    address: List[str]
    city: List[str] = None  # 添加城市参数
    s_hotel: List[str]
    s_address: List[str]
    s_city: List[str] = None  # 添加目标城市参数
    threshold: float = DEFAULT_THRESHOLD
    model: str = DEFAULT_MODEL
    token: str = None


class MatchResponse(BaseModel):
    similarities: List[float]
    predictions: List[int]


@app.get("/health")
async def health_check():
    """健康检查接口"""
    health_status = {
        "status": "healthy",
        "model_loaded": global_state.predictor is not None,
        "model_loading": global_state.model_loading,
        "model_name": global_state.model_name
    }

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
        gc.collect()
        global_state.predictor.cleanup_memory()

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


@app.post("/hotel_match/", response_model=MatchResponse)
async def match_hotels(request: MatchRequest, background_tasks: BackgroundTasks):
    """酒店匹配接口"""
    try:
        # verify_token(request.token)  # 可选择启用令牌验证

        predictor = get_predictor()
        if predictor is None:
            if global_state.model_loading:
                raise HTTPException(status_code=503, detail="模型正在加载中，请稍后再试")
            else:
                error_detail = global_state.last_error or "模型尚未加载，请稍后再试"
                raise HTTPException(status_code=500, detail=error_detail)

        # 模型切换检查
        model_type = request.model.strip().lower() if request.model else DEFAULT_MODEL
        async with model_loading_lock:
            if (global_state.model_name != model_type and
                    not global_state.model_loading):
                model_path = MODEL_PATHS.get(model_type)
                if model_path and os.path.exists(model_path):
                    background_tasks.add_task(init_model, model_type)

        # 输入验证
        required_lengths = [
            len(request.hotel), len(request.address),
            len(request.s_hotel), len(request.s_address)
        ]

        # 检查必需字段长度一致
        if len(set(required_lengths)) != 1:
            raise HTTPException(status_code=400, detail="所有必需输入列表的长度必须相同")

        # 检查可选字段长度（如果提供）
        if request.city and len(request.city) != len(request.hotel):
            raise HTTPException(status_code=400, detail="city列表长度必须与其他列表相同")
        if request.s_city and len(request.s_city) != len(request.s_hotel):
            raise HTTPException(status_code=400, detail="s_city列表长度必须与其他列表相同")

        # 组装文本（参考generate_samples.py的格式）
        source_texts = []
        target_texts = []

        for i in range(len(request.hotel)):
            # 构建源文本
            hotel = request.hotel[i]
            address = request.address[i]
            city = request.city[i] if request.city and i < len(
                request.city) else None

            source_text = f"HotelName:{hotel}"
            if city and city.strip():
                source_text += f",CityName:{city}"
            source_text += f",Address:{address}"
            source_texts.append(source_text)

            # 构建目标文本
            s_hotel = request.s_hotel[i]
            s_address = request.s_address[i]
            s_city = request.s_city[i] if request.s_city and i < len(
                request.s_city) else None

            target_text = f"TargetHotel:{s_hotel}"
            if s_city and s_city.strip():
                target_text += f",TargetCityName:{s_city}"
            target_text += f",TargetAddress:{s_address}"
            target_texts.append(target_text)

        # 执行预测
        result = await predictor.predict_async(source_texts, target_texts, request.threshold)
        return result

    except HTTPException:
        raise
    except Exception as e:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        logger.error(f"处理请求时出错: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")


if __name__ == "__main__":
    config = {
        'host': '0.0.0.0',
        'port': 13102,
        'workers': 1,
        'log_level': 'info',
        'limit_concurrency': 100,
        'timeout_keep_alive': 65
    }

    uvicorn.run(
        "hotel_api_server_optimized:app",
        host=config['host'],
        port=config['port'],
        workers=config['workers'],
        log_level=config['log_level'],
        access_log=False,
        limit_concurrency=config['limit_concurrency'],
        timeout_keep_alive=config['timeout_keep_alive'],
        loop="asyncio"
    )
