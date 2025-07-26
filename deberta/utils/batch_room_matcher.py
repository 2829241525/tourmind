#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
房型匹配批量处理脚本
支持命令行参数和配置文件，可在后台运行
"""
import os
import sys
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from deberta.api_server import process_text_with_prefix, extract_english, DataCleaner
from deberta.inference import SimCSEPredictor


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



# 现在可以安全地导入deberta模块

# 忽略SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# 默认配置
DEFAULT_CONFIG = {
    # 基础路径配置
    "base_dir": "/home/maxon/disk2/roomMatch/room_match",
    "output_dir": "/home/maxon/disk2/roomMatch/room_match/output",

    # API相关配置
    "supplier_id": "0",
    "room_base_type": "tourmind",
    "room_status": "0",
    "match_api_url": "http://10.0.0.110:8100/match",
    "api_base_url": "http://erp.tourmind.cn",

    # 认证相关配置admintools_user_session=MTc1MzQyODYwNHxJRzJJMV9GYmQzcTJnYk1XdkR2MWNzeW90T2htb09BdGRpWnNGVVBDRTY1SzRZMjFWYzNCNG9Sa3daNmNtb01hZGtxVTgyU21mbzA9fMvuZNO1iacMqlgld5vc1uDc0sba-FG3lLSzdHQDV7QS
    "cookie": "admintools_user_session=MTc1MzQyODYwNHxJRzJJMV9GYmQzcTJnYk1XdkR2MWNzeW90T2htb09BdGRpWnNGVVBDRTY1SzRZMjFWYzNCNG9Sa3daNmNtb01hZGtxVTgyU21mbzA9fMvuZNO1iacMqlgld5vc1uDc0sba-FG3lLSzdHQDV7QS",
    "operator": "xiemingxuan",
    "referer": "http://erp.tourmind.cn/roomtypemapping/961935?supplierId=0&brt=tourmind&loadMaster=true",

    # 匹配相关配置
    "threshold": 0.9,  # 匹配阈值
    "max_retries": 2,  # 最大重试次数
    "retry_delay": 1,  # 重试延迟(秒)
    "batch_size": 10,  # 批处理大小

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
    "encoding": "utf-8"  # 文件编码
}


class RoomTypeAPI:
    """房型API接口类"""

    def __init__(self, base_url: str, config: Dict):
        self.base_url = base_url
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.RoomTypeAPI")

    def fetch_room_data(self, hotel_id: str) -> Optional[Dict]:
        """获取房型数据"""
        self.logger.debug(f"开始获取房型数据: hotel_id={hotel_id}")

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
                    f"发送请求到: {self.base_url}/roomtypemapping/api/get_roomtype")
                response = requests.get(
                    f"{self.base_url}/roomtypemapping/api/get_roomtype",
                    params=params,
                    headers=headers,
                    verify=False,
                    timeout=240
                )
                self.logger.debug(f"API响应状态码: {response.status_code}")
                return response.json()
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

        # 初始化SimCSE预测器 - 直接使用inference中的SimCSEPredictor
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
        # 获取房型信息
        room_name = room.get('RoomTypeName', '')
        bed_type = room.get('BedTypeDesc', '')

        # 提取英文文本
        room_text = extract_english(room_name)
        bed_text = extract_english(bed_type)

        # 对供应商端数据进行清洗
        if room.get('roomTypeListType') == 'SPLRoomTypeList':
            # 获取供应商ID和国家信息
            # supplier_id = str(room.get('SupplierID', ''))
            # country_code = str(room.get('Country', '')).lower()

            # 获取供应商名称
            # supplier_name = self.get_supplier_name(supplier_id).lower()

            if room_text:
                room_text = self.cleaner.clean_room_text(room_text)
            if bed_text:
                bed_text = self.cleaner.clean_bed_text(bed_text)

        # 组合成最终格式
        parts = []

        # 为SPLRoomTypeList类型添加供应商和国家信息
        # if room.get('roomTypeListType') == 'SPLRoomTypeList':
        #     parts.append(f"[supply] {supplier_name}")
        #     parts.append(f"[country] {country_code}")

        # if room_text:
        #     parts.append(f"[room] {room_text}")
        #     #parts.append(f"{room_text}")
        # if bed_text:
        #     parts.append(f"[bed] {bed_text}")
        #     #parts.append(f",{bed_text}")
        # else:
        #     parts.append("[bed] [unknown]")
        #     #parts.append("")

        if room_text:
            parts.append(f"{room_text}")
        if bed_text:
            parts.append(f",{bed_text}")
        else:
            parts.append("")

        # 转换为小写
        parts = [part.lower() for part in parts]
        return " ".join(filter(None, parts))

    def match_room_types(
        self,
        spl_rooms: List[Dict],
        s_rooms: List[Dict],
        threshold: float
    ) -> Optional[Dict]:
        """使用SimCSE模型进行房型匹配"""
        try:
            if not self.predictor:
                raise Exception("SimCSE预测器未初始化")

            if not spl_rooms or not s_rooms:
                return None

            # 准备文本数据
            spl_texts = [self.prepare_room_text(room) for room in spl_rooms]
            s_texts = [self.prepare_room_text(room) for room in s_rooms]

            # 记录处理后的文本用于调试
            for i, (room, text) in enumerate(zip(spl_rooms, spl_texts)):
                self.logger.debug(
                    f"SPL房型[{i}] ID:{room.get('RoomTypeID', '')}, 原文本:{room.get('RoomTypeName', '')}, 处理后:{text}")

            for i, (room, text) in enumerate(zip(s_rooms, s_texts)):
                self.logger.debug(
                    f"标准房型[{i}] ID:{room.get('SRoomID', '')}, 原文本:{room.get('RoomTypeName', '')}, 处理后:{text}")

            # 创建所有SPL房型与标准房型的配对组合
            batch_size = 1000  # 使用更大的批次大小以提高处理速度

            # 准备所有配对
            all_text_pairs_a = []  # 所有SPL房型文本
            all_text_pairs_b = []  # 所有标准房型文本
            pair_indices = []      # 记录每个配对对应的SPL房型索引

            # 为每个SPL房型创建与所有标准房型的配对
            for spl_idx, spl_text in enumerate(spl_texts):
                for s_idx, s_text in enumerate(s_texts):
                    all_text_pairs_a.append(spl_text)
                    all_text_pairs_b.append(s_text)
                    pair_indices.append(spl_idx)

            # 一次性处理所有配对
            similarities_flat = []

            # 以大批次处理所有配对
            for i in range(0, len(all_text_pairs_a), batch_size):
                batch_a = all_text_pairs_a[i:i+batch_size]
                batch_b = all_text_pairs_b[i:i+batch_size]

                # 使用文本对方式计算相似度
                logits = self.predictor.encode_text_pair(
                    batch_a, batch_b, batch_size=batch_size)

                # 使用sigmoid将logits转换为概率
                similarities = self.predictor.calculate_similarity(
                    logits, normalize_to_probability=True)

                # 转换为numpy数组并添加到结果中
                similarities_np = similarities.cpu().numpy()
                similarities_flat.extend(similarities_np)

            # 将扁平化的结果重组为每个SPL房型与所有标准房型的相似度矩阵
            similarities_matrix = []
            for spl_idx in range(len(spl_texts)):
                # 获取当前SPL房型的所有相似度
                spl_similarities = [similarities_flat[i]
                                    for i, idx in enumerate(pair_indices) if idx == spl_idx]
                similarities_matrix.append(spl_similarities)

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

                # 记录匹配详情用于调试
                original_match = spl_room.get("SRoomID", 0)
                best_match = s_room.get("SRoomID", "")
                match_status = "已匹配" if original_match == best_match else "原未匹配" if original_match == 0 else "匹配变化"

                self.logger.debug(
                    f"房型匹配 - SPL房型ID: {spl_room.get('RoomTypeID', '')}, 原匹配ID: {original_match}, 最佳匹配ID: {best_match}, 分数: {best_similarity:.4f}, 状态: {match_status}")

                # 构建匹配信息，与room_all_country_evaluate_viewer.py中的格式保持一致
                result = {
                    "SPL房型ID": spl_room.get("RoomTypeID", ""),
                    "标准房型ID": s_room.get("SRoomID", ""),
                    "供应商": self.get_supplier_name(str(spl_room.get("SupplierID", ""))),
                    "酒店ID": spl_room.get("HotelID", ""),
                    "SPL房型": spl_room.get('RoomTypeName', ''),
                    "SPL床型": spl_room.get('BedTypeDesc', ''),
                    "标准房型": s_room.get('RoomTypeName', ''),
                    "标准床型": s_room.get('BedTypeDesc', ''),
                    "匹配分数": float(best_similarity),
                    "预测结果": 1 if best_similarity >= threshold else 0,
                    "原匹配ID": spl_room.get("SRoomID", 0),
                    "匹配状态": "已匹配" if spl_room.get("SRoomID", 0) == s_room.get("SRoomID", "") else
                    "原未匹配" if spl_room.get("SRoomID", 0) == 0 else "匹配变化"
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
    """批量处理器类"""

    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.BatchProcessor")
        self.room_api = RoomTypeAPI("http://erp.tourmind.cn", config)
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
            "model_match_rate": 0
        }

    def process_hotel(self, hotel_id: str, hotel_name: str = "未知", country: str = "未知") -> Optional[Dict]:
        """处理单个酒店"""
        try:
            self.logger.info(f"开始处理酒店: {hotel_id} ({hotel_name}, {country})")

            # 获取房型数据
            data = self.room_api.fetch_room_data(hotel_id)
            if not data or not isinstance(data, dict) or "Data" not in data:
                self.logger.warning(f"获取酒店 {hotel_id} 数据失败")
                return None

            # 获取房型列表
            hotel_data = data.get("Data", {})
            spl_rooms = hotel_data.get("SPLRoomTypeList", [])
            s_rooms = hotel_data.get("SRoomTypeList", [])

            if not spl_rooms or not s_rooms:
                self.logger.warning(f"酒店 {hotel_id} 没有可用的房型数据")
                return None

            # 为每个房型添加酒店信息和类型标记
            for room in spl_rooms:
                room["HotelID"] = hotel_id
                room["HotelName"] = hotel_name
                room["Country"] = country
                room["roomTypeListType"] = "SPLRoomTypeList"

            # 为标准房型设置类型标记
            for room in s_rooms:
                room["roomTypeListType"] = "SRoomTypeList"

            # 更新统计信息
            self.stats["total_rooms"] += len(spl_rooms)
            original_matched = sum(
                1 for room in spl_rooms if room.get("SRoomID", 0) > 0)
            self.stats["matched_rooms"] += original_matched
            self.stats["unmatched_rooms"] += len(spl_rooms) - original_matched

            # 执行匹配
            match_result = self.matcher.match_room_types(
                spl_rooms, s_rooms, self.config["threshold"])

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

            # 更新统计信息
            self.stats["new_matches"] += len(new_matches)
            self.stats["changed_matches"] += len(changed_matches)
            self.stats["kept_matches"] = self.stats.get(
                "kept_matches", 0) + len(kept_matches)
            self.stats["model_matched"] += len(new_matches) + \
                len(changed_matches) + len(kept_matches)
            self.stats["model_unmatched"] += len(unmatched)

            # 计算匹配率
            if self.stats["unmatched_rooms"] > 0:
                self.stats["model_match_rate"] = (
                    self.stats["new_matches"] / self.stats["unmatched_rooms"]) * 100

            self.logger.info(
                f"酒店 {hotel_id} 处理完成: {len(matches)} 个匹配结果, {len(new_matches)} 个新匹配, {len(changed_matches)} 个变化, {len(kept_matches)} 个保持匹配")

            return {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "country": country,
                "matches": matches,
                "new_matches": new_matches,
                "changed_matches": changed_matches,
                "kept_matches": kept_matches,
                "unmatched": unmatched,
                "stats": {
                    "total_rooms": len(spl_rooms),
                    "original_matched": original_matched,
                    "original_unmatched": len(spl_rooms) - original_matched,
                    "new_matches": len(new_matches),
                    "changed_matches": len(changed_matches),
                    "kept_matches": len(kept_matches),
                    "model_matched": len(new_matches) + len(changed_matches) + len(kept_matches),
                    "model_unmatched": len(unmatched)
                }
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
                    all_matches.extend(result["matches"])
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

                # 定期保存结果
                if (idx + 1) % self.config["batch_size"] == 0:
                    self._save_results(all_matches, new_matches, changed_matches,
                                       kept_matches, unmatched, failed_hotels,
                                       output_base, is_final=False)

            # 保存最终结果
            self._save_results(all_matches, new_matches, changed_matches,
                               kept_matches, unmatched, failed_hotels,
                               output_base, is_final=True)

            # 保存统计信息
            self._save_stats(output_base)

            self.logger.info("批量处理完成")
            self._log_final_stats()

        except Exception as e:
            self.logger.error(f"批量处理过程中发生错误: {str(e)}", exc_info=True)
            raise

    def _save_results(self, all_matches: List[Dict], new_matches: List[Dict],
                      changed_matches: List[Dict], kept_matches: List[Dict],
                      unmatched: List[Dict], failed_hotels: List[Dict],
                      output_base: str, is_final: bool = False):
        """保存处理结果"""
        try:
            # 转换为DataFrame
            all_df = pd.DataFrame(
                all_matches) if all_matches else pd.DataFrame()
            new_df = pd.DataFrame(
                new_matches) if new_matches else pd.DataFrame()
            changed_df = pd.DataFrame(
                changed_matches) if changed_matches else pd.DataFrame()
            kept_df = pd.DataFrame(
                kept_matches) if kept_matches else pd.DataFrame()
            unmatched_df = pd.DataFrame(
                unmatched) if unmatched else pd.DataFrame()
            failed_df = pd.DataFrame(
                failed_hotels) if failed_hotels else pd.DataFrame()

            # 添加时间戳后缀
            suffix = "_final" if is_final else ""

            # 保存结果
            if not all_df.empty:
                all_df.to_csv(f"{output_base}_all{suffix}.csv",
                              index=False, encoding='utf-8')
            if not new_df.empty:
                new_df.to_csv(f"{output_base}_new{suffix}.csv",
                              index=False, encoding='utf-8')
            if not changed_df.empty:
                changed_df.to_csv(
                    f"{output_base}_changed{suffix}.csv", index=False, encoding='utf-8')
            if not kept_df.empty:
                kept_df.to_csv(f"{output_base}_kept{suffix}.csv",
                               index=False, encoding='utf-8')
            if not unmatched_df.empty:
                unmatched_df.to_csv(
                    f"{output_base}_unmatched{suffix}.csv", index=False, encoding='utf-8')
            if not failed_df.empty:
                failed_df.to_csv(
                    f"{output_base}_failed{suffix}.csv", index=False, encoding='utf-8')

            self.logger.info(
                f"已保存{'最终' if is_final else '中间'}结果到: {output_base}")

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

        # 修复这里的错误，使用stats中的数据而不是局部变量
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
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_process_{timestamp}.log")

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
    parser = argparse.ArgumentParser(description='房型匹配批量处理工具')

    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument(
        '--hotels', type=str, default='/home/maxon/disk2/roomMatch/room_match/deberta/data/1sampled_hotels.csv', help='酒店数据文件路径')
    parser.add_argument('--country', type=str, default=None, help='指定处理的国家代码')
    parser.add_argument('--max_hotels', type=int, default=5000, help='最大处理酒店数')
    parser.add_argument('--threshold', type=float, default=0.9, help='匹配阈值')
    parser.add_argument(
        '--model', type=str, default='/home/maxon/disk2/roomMatch/room_match/deberta/checkpoints_cross_entropy_side_apartment_fix', help='模型路径')
    parser.add_argument(
        '--output', type=str, default='/home/maxon/disk2/roomMatch/room_match/deberta/output', help='输出目录')

    args = parser.parse_args()

    try:
        # 加载配置
        config = DEFAULT_CONFIG.copy()
        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))

        # 更新配置
        if args.threshold:
            config["threshold"] = args.threshold
        if args.model:
            config["model_path"] = args.model
        if args.output:
            config["output_dir"] = args.output

        # 设置日志
        log_dir = os.path.join(
            config["base_dir"], config["output_dir"], "logs")
        setup_logging(log_dir)

        logger = logging.getLogger(__name__)
        logger.info("开始批量处理")
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
