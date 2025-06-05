#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
房型匹配批量处理脚本 (缓存版)
支持命令行参数和配置文件，可在后台运行
增加API请求结果缓存功能，减少重复请求
"""

# 基础模块导入
import os
import sys
import json
import logging
import argparse
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import warnings
from urllib3.exceptions import InsecureRequestWarning
import requests
from dataclasses import dataclass
import time
import numpy as np
import torch

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 忽略SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# 导入自定义模块
try:
    from deberta.inference import SimCSEPredictor
    from deberta.api_server import process_text_with_prefix, extract_english, DataCleaner
    from deberta.utils.room_cache_manager import RoomCacheManager
    # 尝试导入优化版缓存管理器
    try:
        from deberta.utils.room_cache_manager_optimized import RoomCacheManagerOptimized
        USE_OPTIMIZED_CACHE = True
        print(f"使用优化版缓存管理器")
    except ImportError:
        USE_OPTIMIZED_CACHE = False
        print(f"优化版缓存管理器未找到，使用原始版本")
except ImportError:
    print(f"导入缓存管理器失败")
    RoomCacheManager = None
    USE_OPTIMIZED_CACHE = False

# 默认配置
DEFAULT_CONFIG = {
    # 基础路径配置
    "base_dir": "/home/maxon/disk2/roomMatch/room_match",
    "output_dir": "/home/maxon/disk2/roomMatch/room_match/output",

    # API相关配置
    "supplier_id": "0",
    "room_base_type": "tourmind",
    "room_status": "0",
    "match_api_url": "http://10.0.0.110:8207/match",
    "api_base_url": "http://erp.tourmind.cn",

    # 认证相关配置
    "cookie": "admintools_user_session=MTc0NTIwMDUxNnxBVkExenFNcW5OaktJNm9wUVNyOUtDbUNPNWh3cnFWRjVhVjhKYUJqN04tOWROQzZ3aG8zTlAycGRTcjg0UWRpVVBpYW9nRDdzVlE9fHGM-dpNtI1iqSmMXZQh04-2Pk9GIjJSTRuuJ6sYURsO",
    "operator": "xiemingxuan",
    "referer": "http://erp.tourmind.cn/roomtypemapping/961935?supplierId=0&brt=tourmind&loadMaster=true",

    # 匹配相关配置
    "threshold": 0.9,  # 匹配阈值
    "max_retries": 2,  # 最大重试次数
    "retry_delay": 1,  # 重试延迟(秒)
    "batch_size": 10,  # 批处理大小

    # 缓存相关配置
    "use_cache": True,  # 是否使用缓存
    "cache_dir": "/home/maxon/disk2/roomMatch/room_match/cache",  # 缓存目录
    "cache_expire_days": 7,  # 缓存过期时间(天)

    # 数据文件配置
    "supplier_mapping_file": "/home/maxon/disk2/roomMatch/room_match/deberta/data/supplier_type.xlsx",
    "mapping_file": "/home/maxon/disk2/roomMatch/room_match/deberta/data/mapping.xlsx",

    # 日志配置
    "log_level": "INFO",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",

    # HTTP请求配置
    "timeout": 30,     # 请求超时时间(秒)
    "verify_ssl": False,  # 是否验证SSL证书

    # 其他配置
    "random_seed": 42,  # 随机种子
    "encoding": "utf-8",  # 文件编码

    # 优化版缓存相关配置
    "shard_size": 1000,  # 缓存分片大小，每个分片最多包含多少条记录，默认1000
    "save_threshold": 100,  # 缓存保存间隔，每处理多少条数据保存一次，默认100
    "no_optimize": False,  # 禁用缓存优化，对于大型数据集可能提高初始速度
    "max_batch_size": 32,  # 匹配计算时的最大批次大小，用于控制GPU内存使用，默认32
}


class RoomTypeAPI:
    """房型API接口类 (带缓存功能)"""

    def __init__(self, base_url: str, config: Dict):
        self.base_url = base_url
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RoomTypeAPI")

        # 初始化缓存管理器
        self.use_cache = config.get("use_cache", True)
        if self.use_cache:
            cache_dir = config.get(
                "cache_dir", os.path.join(config["base_dir"], "cache"))
            cache_expire_days = config.get("cache_expire_days", 7)

            # 使用优化版缓存管理器（如果可用）
            if USE_OPTIMIZED_CACHE:
                cache_config = config.copy()
                # 确保配置中包含缓存目录和过期时间
                cache_config["cache_dir"] = cache_dir
                cache_config["cache_expire_days"] = cache_expire_days

                # 获取分片配置（如果有）
                cache_config["shard_size"] = config.get("shard_size", 1000)
                cache_config["save_threshold"] = config.get(
                    "save_threshold", 100)

                self.cache_manager = RoomCacheManagerOptimized(cache_config)
                self.logger.info(
                    f"已启用优化版API缓存，缓存目录: {cache_dir}, 过期时间: {cache_expire_days}天, "
                    f"分片大小: {cache_config.get('shard_size')}, 保存阈值: {cache_config.get('save_threshold')}")
            else:
                # 使用原始缓存管理器
                self.cache_manager = RoomCacheManager(
                    cache_dir, cache_expire_days)
                self.logger.info(
                    f"已启用API缓存，缓存目录: {cache_dir}, 过期时间: {cache_expire_days}天")
        else:
            self.cache_manager = None
            self.logger.info("API缓存已禁用")

    def fetch_room_data(self, hotel_id: str) -> Optional[Dict]:
        """获取房型数据 (支持缓存)"""
        self.logger.debug(f"开始获取房型数据: hotel_id={hotel_id}")

        # 检查缓存
        if self.use_cache and self.cache_manager:
            cached_data = self.cache_manager.get_cache(hotel_id)
            if cached_data:
                self.logger.info(f"使用缓存数据: hotel_id={hotel_id}")
                return cached_data

        # 缓存未命中，从API获取数据
        params = {
            "sHotelID": hotel_id,
            "supplierID": self.config["supplier_id"],
            "sRoomBaseType": self.config["room_base_type"],
            "sroom_status": self.config["room_status"]
        }

        headers = {
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "zh-CN,zh;q=0.9",
            "Cookie": self.config["cookie"],
            "Operator": self.config["operator"],
            "Referer": self.config["referer"],
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        }

        retries = 0
        while retries < self.config["max_retries"]:
            try:
                self.logger.debug(
                    f"从API获取数据: {self.base_url}/roomtypemapping/api/get_roomtype")
                response = requests.get(
                    f"{self.base_url}/roomtypemapping/api/get_roomtype",
                    params=params,
                    headers=headers,
                    verify=False,
                    timeout=100
                )
                self.logger.debug(f"API响应状态码: {response.status_code}")

                # 解析响应数据
                data = response.json()

                # 修复可能的空数据结构问题
                if "Data" in data and data["Data"] is None:
                    data["Data"] = {}

                # 确保Success字段存在
                if "Success" not in data:
                    # 检查是否有明显的错误信息
                    if "Error" in data or "error" in data:
                        data["Success"] = False
                    else:
                        # 如果没有明显错误且返回了数据，假设成功
                        data["Success"] = "Data" in data and bool(data["Data"])

                # 缓存数据
                if self.use_cache and self.cache_manager and data:
                    self.cache_manager.set_cache(hotel_id, data)
                    self.logger.debug(f"已缓存酒店数据: hotel_id={hotel_id}")

                return data
            except Exception as e:
                retries += 1
                self.logger.warning(
                    f"获取数据失败 (重试 {retries}/{self.config['max_retries']}): {str(e)}")
                if retries < self.config["max_retries"]:
                    time.sleep(self.config["retry_delay"])
                else:
                    self.logger.error(f"获取数据最终失败: {str(e)}")
                    return None


class RoomTypeMatcher:
    """房型匹配器类"""

    def __init__(self, api_url: str, model_path: str, config: Dict):
        """初始化预测器"""
        self.api_url = api_url
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RoomTypeMatcher")
        self.predictor = None

        # 初始化数据清洗器
        mapping_path = os.path.join(
            config["base_dir"], "deberta/data/mapping.xlsx")
        self.cleaner = DataCleaner(Path(mapping_path))

        # 加载供应商映射
        self.supplier_mapping = self._load_supplier_mapping()

        # 初始化SimCSE预测器
        try:
            model_full_path = os.path.join(config["base_dir"], model_path)
            model_best_path = os.path.join(
                model_full_path, 'best_model')  # 添加best_model子目录

            # 检查是否存在best_model目录
            if os.path.exists(model_best_path) and os.path.isdir(model_best_path):
                self.logger.info(f"加载模型: {model_best_path} (best_model子目录)")
                model_to_load = model_best_path
            else:
                self.logger.info(f"加载模型: {model_full_path} (直接使用指定路径)")
                model_to_load = model_full_path

            # 初始化预测器
            self.logger.info(f"初始化预测器...")
            self.predictor = SimCSEPredictor(model_to_load)
            self.logger.info("模型加载成功")

            # 记录模型类型信息
            if hasattr(self.predictor, 'loss_type'):
                self.logger.info(f"使用模型类型: {self.predictor.loss_type}")

        except Exception as e:
            error_msg = f"加载模型失败: {str(e)}"
            self.logger.error(error_msg)
            import traceback
            self.logger.error(f"详细错误: {traceback.format_exc()}")
            raise Exception(error_msg)

    def _load_supplier_mapping(self) -> Dict[str, str]:
        """加载供应商映射关系"""
        try:
            # 尝试读取CSV文件
            csv_file_path = os.path.join(
                self.config["base_dir"], "deberta/data/supplier_type.csv")

            if os.path.exists(csv_file_path):
                self.logger.info(f"从CSV文件加载供应商映射: {csv_file_path}")
                df = pd.read_csv(csv_file_path)
                name_column = 'name' if 'name' in df.columns else 'supplier_name'
            else:
                # 尝试读取Excel文件作为备选
                excel_file_path = os.path.join(
                    self.config["base_dir"], "deberta/data/supplier_type.xlsx")

                if not os.path.exists(excel_file_path):
                    self.logger.warning(f"供应商映射文件不存在: CSV或Excel文件均未找到")
                    return {}

                self.logger.info(f"从Excel文件加载供应商映射: {excel_file_path}")
                df = pd.read_excel(excel_file_path)
                name_column = 'supplier_name'

            # 创建映射字典
            mapping = {}
            for _, row in df.iterrows():
                supplier_id = str(row.get('supplier_id', ''))
                supplier_name = row.get(name_column, '')
                if supplier_id and supplier_name:
                    mapping[supplier_id] = supplier_name

            self.logger.info(f"加载了 {len(mapping)} 个供应商映射")
            return mapping
        except Exception as e:
            self.logger.error(f"加载供应商映射失败: {str(e)}")
            return {}

    def get_supplier_name(self, supplier_id: str) -> str:
        """获取供应商名称"""
        return self.supplier_mapping.get(supplier_id, f"未知供应商({supplier_id})")

    def prepare_room_text(self, room: Dict) -> str:
        """准备房型文本用于匹配"""
        # 获取房型信息 - 处理可能的不同字段名
        room_name = room.get('RoomTypeName', room.get(
            'room_type_name', room.get('name', '')))
        bed_type = room.get('BedTypeDesc', room.get(
            'bed_type_desc', room.get('bed_type', '')))

        # 提取英文文本
        room_text = extract_english(room_name)
        bed_text = extract_english(bed_type)

        # 对供应商端数据进行清洗
        room_type = room.get('roomTypeListType', room.get('type', ''))
        if room_type == 'SPLRoomTypeList' or room_type == 'SPL':
            if room_text:
                room_text = self.cleaner.clean_room_text(room_text)
            if bed_text:
                bed_text = self.cleaner.clean_bed_text(bed_text)

        # 组合成最终格式
        parts = []

        if room_text:
            parts.append(f"{room_text}")
        if bed_text:
            parts.append(f",{bed_text}")
        else:
            parts.append("")

        # if room_text:
        #     parts.append(f"[room] {room_text}")
        # if bed_text:
        #     parts.append(f"[bed] {bed_text}")
        # else:
        #     parts.append("[bed] [unknown]")

        # 转换为小写
        parts = [part.lower() for part in parts]
        return " ".join(filter(None, parts))

    def match_room_types(
        self,
        spl_rooms: List[Dict],
        s_rooms: List[Dict],
        threshold: float,
        max_batch_size: int = 32,
        hotel_id: str = None
    ) -> Optional[Dict]:
        """
        匹配房型

        参数:
            spl_rooms: 第一个酒店供应商的房型列表
            s_rooms: 第二个酒店供应商的房型列表
            threshold: 匹配阈值
            max_batch_size: 最大批次大小，用于控制GPU内存使用

        返回:
            匹配结果字典，包含匹配的房型和分数
        """
        try:
            if not self.predictor:
                raise Exception("预测器未初始化")

            if not spl_rooms or not s_rooms:
                return None

            for room in spl_rooms:
                room["roomTypeListType"] = "SPLRoomTypeList"
            # 准备房型文本
            spl_room_texts = [self.prepare_room_text(room) for room in spl_rooms]
            s_room_texts = [self.prepare_room_text(room) for room in s_rooms]

            # 创建房型ID映射，处理不同的字段名
            spl_room_ids = []
            for i, room in enumerate(spl_rooms):
                # 尝试多种可能的ID字段
                room_id = room.get('RoomTypeID',
                                   room.get('room_type_id',
                                            room.get('id', f"spl_{i}")))
                spl_room_ids.append(str(room_id))

            s_room_ids = []
            for i, room in enumerate(s_rooms):
                # 尝试多种可能的ID字段
                room_id = room.get('SRoomID',
                                   room.get('sroom_id',
                                            room.get('id', f"s_{i}")))
                s_room_ids.append(str(room_id))

            # 使用小批次处理以避免GPU内存不足
            self.logger.info(f"使用批次大小: {max_batch_size} 进行匹配计算")
            similarities_matrix = []

            # 分批处理SPL房型
            for batch_idx in range(0, len(spl_room_texts), max_batch_size):
                # 切片当前批次的spl房型
                batch_spl_texts = spl_room_texts[batch_idx:batch_idx+max_batch_size]

                self.logger.debug(
                    f"处理批次 {batch_idx // max_batch_size + 1}/{(len(spl_room_texts) + max_batch_size - 1) // max_batch_size}, 房型数: {len(batch_spl_texts)}")

                # 为当前批次创建与所有标准房型的配对
                all_text_pairs_a = []  # 所有SPL房型文本
                all_text_pairs_b = []  # 所有标准房型文本
                pair_indices = []      # 记录每个配对对应的SPL房型索引

                for i, spl_text in enumerate(batch_spl_texts):
                    for s_text in s_room_texts:
                        all_text_pairs_a.append(spl_text)
                        all_text_pairs_b.append(s_text)
                        pair_indices.append(i)

                # 对当前批次进行匹配计算
                logits = self.predictor.encode_text_pair(
                    all_text_pairs_a, all_text_pairs_b, batch_size=1024)

                # 使用sigmoid将logits转换为概率
                similarities = self.predictor.calculate_similarity(
                    logits, normalize_to_probability=True)

                # 转换为numpy数组
                similarities_np = similarities.cpu().numpy()

                # 将扁平化的结果重组为每个SPL房型与所有标准房型的相似度矩阵
                batch_similarities_matrix = []
                for spl_idx in range(len(batch_spl_texts)):
                    # 获取当前SPL房型的所有相似度
                    spl_similarities = [similarities_np[i]
                                        for i, idx in enumerate(pair_indices) if idx == spl_idx]
                    batch_similarities_matrix.append(spl_similarities)

                # 添加到总的相似度矩阵中
                similarities_matrix.extend(batch_similarities_matrix)

                # 手动清理可能的GPU缓存
                if torch and hasattr(torch, 'cuda') and hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()

            # 转换为numpy数组
            similarities_matrix = np.array(similarities_matrix)

            # 根据阈值生成预测结果
            predictions = (similarities_matrix > threshold).astype(float)

            # 构建返回结果
            results = []
            for i, spl_room in enumerate(spl_rooms):
                # 找出最佳匹配
                best_match_idx = similarities_matrix[i].argmax()
                best_similarity = similarities_matrix[i][best_match_idx]
                s_room = s_rooms[best_match_idx]

                # 处理可能的不同字段名
                spl_room_id = spl_room.get('RoomTypeID', spl_room.get(
                    'room_type_id', spl_room.get('id', '')))
                spl_room_name = spl_room.get('RoomTypeName', spl_room.get(
                    'room_type_name', spl_room.get('name', '')))
                spl_bed_type = spl_room.get('BedTypeDesc', spl_room.get(
                    'bed_type_desc', spl_room.get('bed_type', '')))

                s_room_id = s_room.get('SRoomID', s_room.get(
                    'sroom_id', s_room.get('id', '')))
                s_room_name = s_room.get('RoomTypeName', s_room.get(
                    'room_type_name', s_room.get('name', '')))
                s_bed_type = s_room.get('BedTypeDesc', s_room.get(
                    'bed_type_desc', s_room.get('bed_type', '')))

                supplier_id = str(spl_room.get(
                    'SupplierID', spl_room.get('supplier_id', '')))

                original_match = spl_room.get("SRoomID", spl_room.get(
                    "sroom_id", spl_room.get("original_match", 0)))
                if not original_match and isinstance(original_match, str) and original_match.isdigit():
                    original_match = int(original_match)
                elif not isinstance(original_match, (int, float)):
                    original_match = 0

                # 记录匹配详情用于调试
                best_match = s_room_id
                if not best_match and isinstance(best_match, str) and best_match.isdigit():
                    best_match = int(best_match)

                # 确定匹配状态
                match_status = "已匹配" if str(original_match) == str(
                    best_match) else "原未匹配" if not original_match else "匹配变化"

                # 构建匹配信息
                result = {
                    "SPL房型ID": spl_room_id,
                    "标准房型ID": s_room_id,
                    "供应商": self.get_supplier_name(supplier_id),
                    "酒店ID": hotel_id,
                    "SPL房型": spl_room_name,
                    "SPL床型": spl_bed_type,
                    "标准房型": s_room_name,
                    "标准床型": s_bed_type,
                    "匹配分数": float(best_similarity),
                    "预测结果": 1 if best_similarity >= threshold else 0,
                    "原匹配ID": original_match,
                    "匹配状态": match_status
                }
                results.append(result)

            return {
                "similarities": similarities_matrix.tolist(),
                "predictions": predictions.tolist(),
                "matches": results
            }

        except Exception as e:
            self.logger.error(f"房型匹配过程中发生错误: {str(e)}", exc_info=True)
            return None


class BatchProcessor:
    """批量处理器类 (带缓存支持)"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")

        # 初始化API客户端 (支持缓存)
        self.room_api = RoomTypeAPI(config["api_base_url"], config)

        # 初始化房型匹配器
        self.matcher = RoomTypeMatcher(
            config["match_api_url"], config["model_path"], config)

        # 确保输出目录存在
        self.output_dir = os.path.join(
            config["base_dir"], config["output_dir"])
        os.makedirs(self.output_dir, exist_ok=True)

        # 初始化统计信息
        self.stats = {
            "total_hotels": 0,
            "processed_hotels": 0,
            "success_hotels": 0,
            "failed_hotels": 0,
            "total_rooms": 0,
            "matched_rooms": 0,
            "unmatched_rooms": 0,
            "changed_matches": 0,
            "new_matches": 0,
            "model_matched": 0,
            "model_unmatched": 0,
            "model_match_rate": 0,
            "cache_hits": 0,      # 缓存命中次数
            "cache_misses": 0     # 缓存未命中次数
        }

    def process_hotel(self, hotel_id: str, hotel_name: str = "未知", country: str = "未知") -> Optional[Dict]:
        """处理单个酒店"""
        try:
            self.logger.info(f"处理酒店: {hotel_id} ({hotel_name}, {country})")

            # 获取酒店房型数据
            hotel_data = self.room_api.fetch_room_data(hotel_id)
            if not hotel_data or not hotel_data.get("Success", False):
                self.logger.warning(f"获取酒店 {hotel_id} 数据失败")
                return None

            # 统计缓存命中情况
            if self.room_api.use_cache:
                # 检查是否来自缓存的数据
                if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                    cached_data = self.room_api.cache_manager.get_cache(
                        hotel_id)
                    if cached_data is not None:
                        self.stats["cache_hits"] += 1
                    else:
                        self.stats["cache_misses"] += 1

            # 提取房型数据，尝试多种可能的字段名
            rooms_data = hotel_data.get("Data", {})

            # 尝试不同可能的字段名获取SPL房型
            spl_rooms = None
            for field in ["SPLRoomTypeList", "SPLROOM", "SPL_ROOM", "spl_room"]:
                if field in rooms_data and rooms_data[field]:
                    spl_rooms = rooms_data[field]
                    self.logger.debug(f"从字段 {field} 获取到 SPL房型数据")
                    break

            # 如果所有尝试都失败，检查是否有通用房型字段
            if not spl_rooms and "rooms" in rooms_data:
                spl_rooms = [r for r in rooms_data["rooms"]
                             if r.get("type") == "SPL"]

            # 如果仍然没有找到，设为空列表
            if not spl_rooms:
                spl_rooms = []

            # 尝试不同可能的字段名获取标准房型
            s_rooms = None
            for field in ["SRoomTypeList", "SROOM", "S_ROOM", "s_room"]:
                if field in rooms_data and rooms_data[field]:
                    s_rooms = rooms_data[field]
                    self.logger.debug(f"从字段 {field} 获取到标准房型数据")
                    break

            # 如果所有尝试都失败，检查是否有通用房型字段
            if not s_rooms and "rooms" in rooms_data:
                s_rooms = [r for r in rooms_data["rooms"]
                           if r.get("type") == "S"]

            # 如果仍然没有找到，设为空列表
            if not s_rooms:
                s_rooms = []

            # 更新统计信息
            self.stats["total_rooms"] += len(spl_rooms)
            original_matched = sum(
                1 for room in spl_rooms if room.get("SRoomID", 0) > 0)
            self.stats["matched_rooms"] += original_matched
            self.stats["unmatched_rooms"] += len(spl_rooms) - original_matched

            if not spl_rooms or not s_rooms:
                self.logger.warning(
                    f"酒店 {hotel_id} 数据不完整: SPL房型={len(spl_rooms)}, 标准房型={len(s_rooms)}")
                return None

            # 执行房型匹配
            match_result = self.matcher.match_room_types(
                spl_rooms, s_rooms, self.config["threshold"], self.config["max_batch_size"], hotel_id)

            if not match_result or "matches" not in match_result:
                self.logger.warning(f"酒店 {hotel_id} 匹配失败")
                return None

            # 处理匹配结果
            matches = match_result["matches"]

            # 分类匹配结果
            new_matches = [m for m in matches if m["匹配状态"]
                           == "原未匹配" and m["预测结果"] == 1]
            changed_matches = [m for m in matches if m["匹配状态"]
                               == "匹配变化" and m["预测结果"] == 1]
            kept_matches = [m for m in matches if m["匹配状态"]
                            == "已匹配" and m["预测结果"] == 1]
            unmatched = [m for m in matches if m["预测结果"] == 0]

            # 更新全局统计信息
            self.stats["new_matches"] += len(new_matches)
            self.stats["changed_matches"] += len(changed_matches)
            self.stats["kept_matches"] = self.stats.get(
                "kept_matches", 0) + len(kept_matches)
            self.stats["model_matched"] += len(new_matches) + \
                len(changed_matches) + len(kept_matches)
            self.stats["model_unmatched"] += len(unmatched)

            # 更新统计信息
            hotel_stats = {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "country": country,
                "spl_rooms": len(spl_rooms),
                "s_rooms": len(s_rooms),
                "new_matches": len(new_matches),
                "changed_matches": len(changed_matches),
                "kept_matches": len(kept_matches),
                "unmatched": len(unmatched),
                "model_matched": len(new_matches) + len(changed_matches) + len(kept_matches),
                "match_rate": (len(new_matches) + len(changed_matches) + len(kept_matches)) / len(spl_rooms) if spl_rooms else 0
            }

            # 记录匹配结果
            self.logger.info(
                f"酒店 {hotel_id} 匹配结果: SPL房型={len(spl_rooms)}, 标准房型={len(s_rooms)}, "
                f"新匹配={len(new_matches)}, 匹配变化={len(changed_matches)}, "
                f"保持匹配={len(kept_matches)}, 未匹配={len(unmatched)}, "
                f"匹配率={hotel_stats['match_rate']:.2%}"
            )

            # 返回匹配结果和统计信息
            return {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "country": country,
                "match_result": match_result,
                "new_matches": new_matches,
                "changed_matches": changed_matches,
                "kept_matches": kept_matches,
                "unmatched": unmatched,
                "stats": hotel_stats
            }

        except Exception as e:
            self.logger.error(
                f"处理酒店 {hotel_id} 时发生错误: {str(e)}", exc_info=True)
            return None

    def process_hotels(self, hotels_data: pd.DataFrame):
        """批量处理酒店"""
        try:
            self.stats["total_hotels"] = len(hotels_data)

            # 创建输出文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_base = os.path.join(
                self.output_dir, f"batch_results_{timestamp}")

            all_matches = []
            new_matches = []
            changed_matches = []
            kept_matches = []
            unmatched = []
            failed_hotels = []

            # 设置输出间隔为1000个酒店
            output_interval = 1000

            # 处理每个酒店
            for idx, row in hotels_data.iterrows():
                hotel_id = str(row['hotel_id'])
                hotel_name = row.get('hotel_name', '未知酒店')
                country = row.get('country_code', '未知')

                self.logger.info(
                    f"处理进度: {idx+1}/{len(hotels_data)} - 酒店: {hotel_id}")

                result = self.process_hotel(hotel_id, hotel_name, country)

                if result:
                    self.stats["success_hotels"] += 1
                    all_matches.extend(result["match_result"]["matches"])
                    new_matches.extend(result["new_matches"])
                    changed_matches.extend(result["changed_matches"])
                    kept_matches.extend(result["kept_matches"])
                    unmatched.extend(result["unmatched"])
                else:
                    self.stats["failed_hotels"] += 1
                    failed_hotels.append({
                        "hotel_id": hotel_id,
                        "hotel_name": hotel_name,
                        "country": country
                    })

                self.stats["processed_hotels"] += 1

                # 定期保存结果 (每1000个酒店)
                if (idx + 1) % output_interval == 0:
                    self._save_results(all_matches, new_matches, changed_matches,
                                       kept_matches, unmatched, failed_hotels,
                                       output_base, is_final=False)

                    # 定期保存缓存 (每1000个酒店保存一次)
                    if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                        # 保存缓存数据，优化后的缓存管理器会自动检查是否有更新
                        save_start = time.time()
                        self.room_api.cache_manager.save()
                        save_duration = time.time() - save_start
                        self.logger.info(
                            f"已尝试保存房型数据缓存，耗时 {save_duration:.2f}秒")

                        # 如果是优化版缓存管理器，每10次输出执行一次优化
                        if ((idx + 1) // output_interval) % 10 == 0 and USE_OPTIMIZED_CACHE:
                            if hasattr(self.room_api.cache_manager, 'optimize'):
                                self.logger.info("执行缓存优化...")
                                optimize_stats = self.room_api.cache_manager.optimize()
                                self.logger.info(f"缓存优化完成: {optimize_stats}")

            # 保存最终结果
            self._save_results(all_matches, new_matches, changed_matches,
                               kept_matches, unmatched, failed_hotels,
                               output_base, is_final=True)

            # 保存统计信息
            self._save_stats(output_base)

            # 保存最终缓存
            if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                # 保存最终缓存数据
                save_start = time.time()
                self.room_api.cache_manager.save()
                save_duration = time.time() - save_start
                self.logger.info(f"已尝试保存最终房型数据缓存，耗时 {save_duration:.2f}秒")

                # 清理过期缓存
                if hasattr(self.room_api.cache_manager, 'clear_expired_cache'):
                    self.logger.info("清理过期缓存...")
                    cleared = self.room_api.cache_manager.clear_expired_cache()
                    self.logger.info(f"清理了 {cleared} 条过期缓存")

                # 如果是优化版缓存管理器，执行最终优化
                if USE_OPTIMIZED_CACHE and hasattr(self.room_api.cache_manager, 'optimize'):
                    self.logger.info("执行最终缓存优化...")
                    optimize_stats = self.room_api.cache_manager.optimize()
                    self.logger.info(f"最终缓存优化完成: {optimize_stats}")

                # 如果支持获取缓存统计，输出统计信息
                if hasattr(self.room_api.cache_manager, 'get_stats'):
                    cache_stats = self.room_api.cache_manager.get_stats()
                    self.logger.info(
                        f"缓存统计信息: {json.dumps(cache_stats, ensure_ascii=False)}")

                self.logger.info(f"已保存最终房型数据缓存")

            self.logger.info("批量处理完成")
            self._log_final_stats()

        except Exception as e:
            self.logger.error(f"批量处理过程中发生错误: {str(e)}", exc_info=True)
            # 出错时也保存缓存
            if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                save_start = time.time()
                self.room_api.cache_manager.save()
                save_duration = time.time() - save_start
                self.logger.info(f"已尝试保存房型数据缓存 (错误恢复)，耗时 {save_duration:.2f}秒")
            raise

    def _save_results(self, all_matches: List[Dict], new_matches: List[Dict],
                      changed_matches: List[Dict], kept_matches: List[Dict],
                      unmatched: List[Dict], failed_hotels: List[Dict],
                      output_base: str, is_final: bool = False):
        """保存处理结果"""
        try:
            # 转换为DataFrame (只处理all和new)
            all_df = pd.DataFrame(
                all_matches) if all_matches else pd.DataFrame()
            new_df = pd.DataFrame(
                new_matches) if new_matches else pd.DataFrame()

            # 添加时间戳后缀
            suffix = "_final" if is_final else ""

            # 保存结果 (只保存all和new)
            if not all_df.empty:
                all_df.to_csv(f"{output_base}_all{suffix}.csv",
                              index=False, encoding='utf-8')
            if not new_df.empty:
                new_df.to_csv(f"{output_base}_new{suffix}.csv",
                              index=False, encoding='utf-8')

            self.logger.info(
                f"已保存{'最终' if is_final else '中间'}结果到: {output_base} (仅all和new)")

        except Exception as e:
            self.logger.error(f"保存结果时发生错误: {str(e)}", exc_info=True)

    def _save_stats(self, output_base: str):
        """保存统计信息"""
        try:
            stats_file = f"{output_base}_stats.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"统计信息已保存到: {stats_file}")
        except Exception as e:
            self.logger.error(f"保存统计信息时发生错误: {str(e)}", exc_info=True)

    def _log_final_stats(self):
        """输出最终统计信息"""
        self.logger.info("=== 处理完成 ===")
        self.logger.info(f"总酒店数: {self.stats['total_hotels']}")
        self.logger.info(f"成功处理: {self.stats['success_hotels']}")
        self.logger.info(f"处理失败: {self.stats['failed_hotels']}")
        self.logger.info(f"总房型数: {self.stats['total_rooms']}")
        self.logger.info(f"规则匹配数: {self.stats['matched_rooms']}")
        self.logger.info(f"规则未匹配数: {self.stats['unmatched_rooms']}")
        self.logger.info(f"模型匹配数: {self.stats['model_matched']}")
        self.logger.info(f"模型未匹配数: {self.stats['model_unmatched']}")

        # 输出缓存统计
        cache_hits = self.stats.get("cache_hits", 0)
        cache_misses = self.stats.get("cache_misses", 0)
        total_requests = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_requests *
                          100) if total_requests > 0 else 0

        self.logger.info(f"缓存命中数: {cache_hits}")
        self.logger.info(f"缓存未命中数: {cache_misses}")
        self.logger.info(f"缓存命中率: {cache_hit_rate:.2f}%")

        # 修复这里的错误，使用stats中的数据
        kept_matches_count = self.stats.get("kept_matches", 0)
        self.logger.info(f"保持匹配数: {kept_matches_count}")
        self.logger.info(f"新增匹配数: {self.stats['new_matches']}")
        self.logger.info(f"匹配变化数: {self.stats['changed_matches']}")

        # 添加匹配数据验证信息
        model_matches_sum = self.stats['new_matches'] + \
            self.stats['changed_matches'] + kept_matches_count
        self.logger.info(f"模型匹配数验证: {model_matches_sum} (新增 + 变化 + 保持)")

        if model_matches_sum != self.stats['model_matched']:
            self.logger.warning(
                f"模型匹配数不一致: model_matched={self.stats['model_matched']}, 计算和={model_matches_sum}")

        if self.stats['unmatched_rooms'] > 0:
            new_match_rate = (
                self.stats['new_matches'] / self.stats['unmatched_rooms']) * 100
            self.logger.info(f"模型新增匹配率: {new_match_rate:.2f}%")


def setup_logging(log_dir: str):
    """设置日志"""
    try:
        # 确保日志目录存在
        os.makedirs(log_dir, exist_ok=True)

        # 设置日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"batch_match_{timestamp}.log")

        # 配置日志处理器
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, encoding='utf-8')
            ]
        )

        # 设置第三方库的日志级别
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

        # 返回日志文件路径，方便记录
        return log_file
    except Exception as e:
        # 如果设置日志失败，配置一个基本的控制台日志
        print(f"设置日志失败: {str(e)}，将使用基本控制台日志")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        return None


def load_hotels_data(file_path: str) -> pd.DataFrame:
    """加载酒店数据"""
    try:
        df = pd.read_csv(file_path)

        # 数据预处理
        df['hotel_id'] = df['hotel_id'].astype(str)
        df['country_code'] = df['country_code'].fillna('Unknown')
        df['country_code'] = df['country_code'].astype(str)
        df['hotel_name'] = df['hotel_name'].fillna('未知酒店名称')
        df['hotel_name'] = df['hotel_name'].astype(str)

        return df
    except Exception as e:
        logging.error(f"加载酒店数据失败: {str(e)}")
        raise


def main():
    """主函数"""
    # 初始化全局日志器，确保无论在何处都能访问
    global logger

    # 设置一个基本的控制台日志以处理早期日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(description='房型匹配批量处理工具 (带缓存功能)')

    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument(
        '--hotels', type=str, default='/home/maxon/disk2/roomMatch/room_match/deberta/data/1000sampled_hotels.csv', help='酒店数据文件路径')
    parser.add_argument('--country', type=str, default=None, help='指定处理的国家代码')
    parser.add_argument('--max_hotels', type=int,
                        default=1000, help='最大处理酒店数')
    parser.add_argument('--threshold', type=float, default=0.9, help='匹配阈值')
    # /home/maxon/disk2/roomMatch/room_match/chinese_match/checkpoints_all
    # /home/maxon/disk2/roomMatch/room_match/chinese_match/checkpoints_mdeberta_prefix_10e
    # /home/maxon/disk2/roomMatch/room_match/deberta/checkpoints_cross_entropy_result1_nosplct
    parser.add_argument(
        '--model', type=str, default='/home/maxon/disk2/roomMatch/room_match/chinese_match/checkpoints_mdeberta_prefix_10e', help='模型路径')
    parser.add_argument(
        '--output', type=str, default='/home/maxon/disk2/roomMatch/room_match/deberta/output', help='输出目录')
    parser.add_argument(
        '--cache_dir', type=str, default=None, help='缓存目录，不指定则使用默认配置')
    parser.add_argument(
        '--no_cache', action='store_true', help='禁用缓存功能')
    parser.add_argument(
        '--cache_expire_days', type=int, default=1000, help='缓存过期时间(天)')
    parser.add_argument('--shard_size', type=int, default=1000,
                        help='缓存分片大小，每个分片最多包含多少条记录，默认1000')
    parser.add_argument('--save_threshold', type=int, default=100,
                        help='缓存保存间隔，每处理多少条数据保存一次，默认100')
    parser.add_argument('--use_optimized_cache', action='store_true',
                        help='强制使用优化版缓存管理器')
    parser.add_argument('--no_optimize', action='store_true',
                        help='禁用缓存优化，对于大型数据集可能提高初始速度')
    parser.add_argument('--max_batch_size', type=int, default=1000,
                        help='匹配计算时的最大批次大小，用于控制GPU内存使用，默认32')

    args = parser.parse_args()

    try:
        # 加载配置
        config = DEFAULT_CONFIG.copy()
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))

        # 确保基础目录路径存在
        if "base_dir" not in config or not config["base_dir"]:
            config["base_dir"] = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))))
            logger.info(f"设置默认基础目录: {config['base_dir']}")

        # 确保输出目录存在
        if "output_dir" not in config or not config["output_dir"]:
            config["output_dir"] = "output"
            logger.info(f"设置默认输出目录: {config['output_dir']}")

        # 确保输出目录为绝对路径
        if not os.path.isabs(config["output_dir"]):
            config["output_dir"] = os.path.join(
                config["base_dir"], config["output_dir"])

        # 创建输出目录
        os.makedirs(config["output_dir"], exist_ok=True)

        # 更新配置
        if args.threshold:
            config["threshold"] = args.threshold
        if args.model:
            config["model_path"] = args.model
        if args.output:
            config["output_dir"] = args.output

        # 配置缓存
        config["use_cache"] = not args.no_cache
        if args.cache_dir:
            config["cache_dir"] = args.cache_dir
        if args.cache_expire_days:
            config["cache_expire_days"] = args.cache_expire_days

        # 处理缓存优化相关参数
        global USE_OPTIMIZED_CACHE
        if args.use_optimized_cache:
            USE_OPTIMIZED_CACHE = True
            logger.info("强制使用优化版缓存管理器")

        # 更新配置
        if args.shard_size:
            config["shard_size"] = args.shard_size
        if args.save_threshold:
            config["save_threshold"] = args.save_threshold
        if args.no_optimize:
            config["no_optimize"] = True
        if args.max_batch_size:
            config["max_batch_size"] = args.max_batch_size

        # 在配置更新后，记录缓存配置信息
        logger.info(
            f"缓存配置: 使用优化版={USE_OPTIMIZED_CACHE}, "
            f"分片大小={config.get('shard_size', 1000)}, "
            f"保存阈值={config.get('save_threshold', 100)}, "
            f"禁用优化={config.get('no_optimize', False)}, "
            f"最大批次大小={config.get('max_batch_size', 32)}")

        # 设置日志
        log_dir = os.path.join(
            config["base_dir"], config["output_dir"], "logs")
        log_file = setup_logging(log_dir)

        # 重新获取配置好的日志器
        logger = logging.getLogger(__name__)
        logger.info("开始批量处理")

        # 打印缓存配置信息
        if config["use_cache"]:
            logger.info(
                f"缓存已启用 - 目录: {config['cache_dir']}, 过期时间: {config['cache_expire_days']}天")
        else:
            logger.info("缓存已禁用")

        logger.info(
            f"使用配置: {json.dumps(config, ensure_ascii=False, indent=2)}")

        # 加载酒店数据
        hotels_df = load_hotels_data(args.hotels)
        logger.info(f"加载了 {len(hotels_df)} 家酒店的数据")

        # 筛选国家
        if args.country:
            hotels_df = hotels_df[hotels_df['country_code'] == args.country]
            logger.info(f"筛选 {args.country} 的酒店: {len(hotels_df)} 家")

        # 限制处理数量
        if args.max_hotels and len(hotels_df) > args.max_hotels:
            hotels_df = hotels_df.sample(n=args.max_hotels, random_state=42)
            logger.info(f"随机选择 {args.max_hotels} 家酒店处理")

        # 创建处理器并执行
        processor = BatchProcessor(config)
        processor.process_hotels(hotels_df)

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
