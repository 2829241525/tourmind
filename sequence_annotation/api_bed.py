#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
房间名称中的床位信息提取 - API服务
基于预训练的DeBERTa模型预测未知床位信息的FastAPI服务
支持高性能并发处理多个请求
"""

import os
import logging
import numpy as np
import torch
import time
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
from torch.cuda.amp import autocast
import uvicorn
import threading
from concurrent.futures import ThreadPoolExecutor
import psutil

# 配置区域
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 创建日志目录
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, "logs/room_bed_api.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("room_bed_api")

# 默认配置参数
CPU_COUNT = psutil.cpu_count(logical=False) or 4
CONFIG = {
    "model_dir": os.path.join(BASE_DIR, "models/room_bed_extraction/checkpoint-14439"),
    "max_length": 64,
    "batch_size": 64,
    "device": "cuda:1" if torch.cuda.is_available() else "cpu",
    "port": 8300,
    "workers": min(CPU_COUNT * 2, 16),  # 自动设置工作线程数，但不超过16
    "max_concurrent_batches": min(CPU_COUNT, 8),  # 控制最大并发批处理数
    "request_timeout": 120  # 请求超时时间（秒）
}

# 请求模型


class PredictionRequest(BaseModel):
    room_names: List[str]
    batch_size: Optional[int] = CONFIG["batch_size"]

# 响应模型


class PredictionResponse(BaseModel):
    predictions: List[Dict[str, Any]]
    model_info: Dict[str, Any]
    processing_time: float
    processed_count: int


class RoomBedPredictor:
    """床位信息提取预测器 (单例模式)"""
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RoomBedPredictor, cls).__new__(cls)
                cls._instance.initialized = False
            return cls._instance

    def __init__(self, model_dir: str = CONFIG["model_dir"]):
        """初始化预测器（单例模式）"""
        # 如果已经初始化，则不重复初始化
        if self.initialized:
            return

        with self._lock:
            if not self.initialized:
                self.device = torch.device(CONFIG["device"])
                logger.info(f"使用设备: {self.device}")

                # 加载模型和分词器
                logger.info(f"从 {model_dir} 加载模型和分词器")
                start_time = time.time()
                try:
                    # 将模型路径转为绝对路径并验证路径存在
                    abs_model_dir = os.path.abspath(model_dir)
                    if not os.path.exists(abs_model_dir):
                        raise FileNotFoundError(f"模型路径不存在: {abs_model_dir}")

                    logger.info(f"使用本地模型路径: {abs_model_dir}")

                    # 添加local_files_only=True参数告诉transformers这是本地文件而非HuggingFace仓库
                    self.tokenizer = DebertaV2TokenizerFast.from_pretrained(
                        abs_model_dir,
                        local_files_only=True,  # 表明使用本地文件
                        trust_remote_code=True   # 信任本地代码
                    )
                    self.model = DebertaV2ForTokenClassification.from_pretrained(
                        abs_model_dir,
                        local_files_only=True,  # 表明使用本地文件
                        trust_remote_code=True   # 信任本地代码
                    ).to(self.device)
                    self.model.eval()  # 设置为评估模式

                    # 启用梯度检查点以节省显存
                    if hasattr(self.model, 'gradient_checkpointing_enable'):
                        self.model.gradient_checkpointing_enable()
                        logger.info("已启用梯度检查点以节省显存")

                    # 保存模型路径和设置标签映射
                    self.model_path = abs_model_dir
                    self.id2label = {0: "O", 1: "B-BED",
                                     2: "I-BED", 3: "E-BED"}
                    self.max_length = CONFIG["max_length"]

                    # 创建模型并发锁
                    self.model_lock = threading.Lock()

                    # 创建并发控制信号量
                    self.prediction_semaphore = asyncio.Semaphore(
                        CONFIG["max_concurrent_batches"])

                    # 请求计数器
                    self.request_count = 0
                    self.request_lock = threading.Lock()

                    # 创建预测线程池
                    self.prediction_executor = ThreadPoolExecutor(
                        max_workers=CONFIG["workers"])

                    # 标记为已初始化
                    self.initialized = True

                    load_time = time.time() - start_time
                    logger.info(f"模型加载成功，耗时：{load_time:.2f}秒")
                except Exception as e:
                    self.initialized = False
                    error_msg = f"加载模型时出错：{str(e)}"
                    logger.error(error_msg)
                    import traceback
                    logger.error(traceback.format_exc())
                    raise RuntimeError(f"模型加载失败: {str(e)}")

    def predict_batch(self, room_names: List[str]):
        """批量预测房间名称中的床位信息（线程安全）"""
        # 更新请求计数
        with self.request_lock:
            self.request_count += len(room_names)
            current_count = self.request_count

        batch_start_time = time.time()

        # 分词
        tokenized = self.tokenizer(
            room_names,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # 将输入移动到模型所在设备
        input_ids = tokenized["input_ids"].to(self.device)
        attention_mask = tokenized["attention_mask"].to(self.device)

        # 获取预测 - 使用模型锁确保并发安全
        with self.model_lock:
            with torch.no_grad():
                with autocast():
                    outputs = self.model(
                        input_ids=input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=2)
                    confidence_scores = torch.softmax(outputs.logits, dim=2)

        # 初始化结果列表
        results = []

        # 处理每个预测结果
        for i, (room_name, pred, conf) in enumerate(zip(room_names, predictions, confidence_scores)):
            tokens = self.tokenizer.convert_ids_to_tokens(
                tokenized["input_ids"][i])

            # 提取床位信息
            bed_tokens = []
            confidence = []

            for j, (token, label, conf_scores) in enumerate(zip(tokens, pred, conf)):
                if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                    continue

                score = conf_scores[label.item()].item()
                confidence.append(score)

                if label.item() in [1, 2, 3]:  # B-BED, I-BED, E-BED
                    bed_tokens.append(token)

            # 合并token为文本
            bed_info = self.tokenizer.convert_tokens_to_string(
                bed_tokens).strip()

            # 计算平均置信度
            avg_confidence = np.mean(confidence) if confidence else 0

            # 添加结果
            results.append({
                "room_name": room_name,
                "predicted_bed": bed_info,
                "confidence": float(avg_confidence)
            })

        batch_time = time.time() - batch_start_time
        logger.debug(f"处理批次大小: {len(room_names)}, 耗时: {batch_time:.3f}s")

        return results

    async def predict_async(self, room_names: List[str], batch_size: int = CONFIG["batch_size"]):
        """异步预测多个房间名称的床位信息"""
        start_time = time.time()
        all_results = []

        # 创建任务列表
        tasks = []
        loop = asyncio.get_running_loop()

        # 划分批次，确保均匀分配
        batches = []
        for i in range(0, len(room_names), batch_size):
            batches.append(room_names[i:i+batch_size])

        logger.info(f"划分为 {len(batches)} 个批次进行并发处理")

        # 提交所有批次任务
        for batch in batches:
            # 使用异步信号量控制并发数量
            async with self.prediction_semaphore:
                # 提交给线程池执行
                task = loop.run_in_executor(
                    self.prediction_executor,
                    self.predict_batch,
                    batch
                )
                tasks.append(task)

        # 等待所有任务完成，设置超时保护
        batch_results = await asyncio.wait_for(
            asyncio.gather(*tasks),
            timeout=CONFIG["request_timeout"]
        )

        # 合并结果
        for results in batch_results:
            all_results.extend(results)

        # 计算处理时间
        processing_time = time.time() - start_time

        return all_results, processing_time, len(room_names)

    def __del__(self):
        """析构函数，清理资源"""
        if hasattr(self, 'prediction_executor'):
            self.prediction_executor.shutdown(wait=False)


# 创建FastAPI应用
app = FastAPI(
    title="房间床位信息提取服务",
    description="基于DeBERTa模型的房间床位信息提取API服务（高性能并发版）",
    version="1.0.0"
)

# 预先加载模型的单例对象
predictor = None


@app.on_event("startup")
async def startup_event():
    """启动时加载模型"""
    global predictor

    # 记录当前工作目录和BASE_DIR
    logger.info(f"当前工作目录: {os.getcwd()}")
    logger.info(f"BASE_DIR: {BASE_DIR}")
    logger.info(
        f"并发设置: 工作线程数 {CONFIG['workers']}, 最大并发批次 {CONFIG['max_concurrent_batches']}")

    # 初始化模型
    logger.info("开始初始化模型...")
    try:
        predictor = RoomBedPredictor()
        logger.info("模型加载完成，服务就绪")
    except Exception as e:
        logger.error(f"模型初始化失败: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict_room_bed(request: PredictionRequest):
    """
    预测房间名称中的床位信息

    - room_names: 房间名称列表
    - batch_size: 批处理大小（可选，默认64）

    返回：
    - predictions: 预测结果列表
    - model_info: 模型信息
    - processing_time: 处理时间（秒）
    - processed_count: 处理的房间数量
    """
    global predictor

    try:
        # 检查模型是否已加载
        if predictor is None or not predictor.initialized:
            # 尝试加载模型
            try:
                predictor = RoomBedPredictor()
            except Exception as e:
                logger.error(f"模型加载失败: {str(e)}")
                raise HTTPException(
                    status_code=500,
                    detail=f"模型尚未加载: {str(e)}"
                )

        # 检查请求数据是否有效
        if not request.room_names:
            raise HTTPException(
                status_code=400,
                detail="房间名称列表不能为空"
            )

        # 设置批处理大小
        batch_size = min(request.batch_size or CONFIG["batch_size"], 128)

        # 如果请求量很大，自动调整批处理大小
        if len(request.room_names) > 1000:
            batch_size = min(256, len(request.room_names) //
                             CONFIG["max_concurrent_batches"] + 1)
            logger.info(
                f"大批量请求: {len(request.room_names)} 条数据, 自动调整批处理大小为: {batch_size}")

        # 异步预测
        results, processing_time, processed_count = await predictor.predict_async(
            request.room_names,
            batch_size
        )

        # 模型信息
        model_info = {
            "model_path": predictor.model_path,
            "device": str(predictor.device),
            "batch_size": batch_size
        }

        # 记录处理时间
        items_per_second = processed_count / processing_time if processing_time > 0 else 0
        logger.info(
            f"处理了 {processed_count} 个房间名称，耗时: {processing_time:.3f}s, 速率: {items_per_second:.1f}项/秒")

        return {
            "predictions": results,
            "model_info": model_info,
            "processing_time": processing_time,
            "processed_count": processed_count
        }
    except asyncio.TimeoutError:
        logger.error(f"请求处理超时")
        raise HTTPException(status_code=408, detail="请求处理超时，请减少批量或稍后重试")
    except Exception as e:
        # 捕获所有其他未处理的异常
        logger.error(f"处理请求时出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"处理请求时出错: {str(e)}")

if __name__ == "__main__":
    # 启动服务
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=CONFIG["port"],
        workers=1  # 使用1个uvicorn工作进程，但内部有自己的线程池
    )
