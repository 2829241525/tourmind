#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
酒店房型数据提取器
根据酒店ID获取待匹配供应商房型和标准房型列表，输出到单个CSV文件
使用多线程并发处理，所有数据合并到一个CSV文件中
"""

import pandas as pd
import requests
import json
import logging
import sys
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Any, Optional, List
from datetime import datetime

# 添加父目录到路径以支持直接运行
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 临时添加父目录到路径以导入正确的配置
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

try:
    from config import LLM_CONFIG
    from login_handler import get_login_handler
    CONFIG_AVAILABLE = True
except ImportError as e:
    print(f"导入配置失败: {e}")
    print("请确保config.py和login_handler.py文件存在")
    CONFIG_AVAILABLE = False

    # 提供模拟的配置和登录处理器
    class MockLoginHandler:
        def get_current_cookies(self):
            return "mock_cookies=test"

        def login_and_get_cookies(self):
            return "mock_cookies=test_new"

    def get_login_handler():
        return MockLoginHandler()

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hotel_room_extractor.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# 全局锁，用于线程安全
data_collection_lock = Lock()
progress_lock = Lock()
login_lock = Lock()


class HotelRoomExtractor:
    """酒店房型数据提取器"""

    # 默认API配置
    DEFAULT_API_CONFIG = {
        "base_url": "http://erp.tourmind.cn/roomtypemapping/api/get_roomtype",
        "default_params": {
            "supplierID": "0",
            "sRoomBaseType": "tourmind",
            "sroom_status": "0"
        },
        "default_headers": {
        }
    }

    def __init__(self, max_workers=10):
        """初始化提取器"""
        self.login_handler = get_login_handler()
        self.config_available = CONFIG_AVAILABLE
        self.max_workers = max_workers

        # 进度统计
        self.processed_count = 0
        self.error_count = 0
        self.total_count = 0

        # 全局数据收集器
        self.all_hotel_data = []

        if not self.config_available:
            logger.warning("配置模块不可用，使用模拟模式")
        logger.info(f"酒店房型提取器初始化完成，最大线程数: {max_workers}")

    def build_api_url(self, hotel_id: str, **override_params) -> tuple:
        """构建API URL和请求配置"""
        # 合并参数
        params = self.DEFAULT_API_CONFIG["default_params"].copy()
        params["sHotelID"] = hotel_id
        params.update(override_params)

        # 构建URL
        base_url = self.DEFAULT_API_CONFIG["base_url"]
        param_string = "&".join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{base_url}?{param_string}"

        # 更新Referer中的酒店ID
        headers = self.DEFAULT_API_CONFIG["default_headers"].copy()
        headers['Referer'] = f'http://erp.tourmind.cn/roomtypemapping/{hotel_id}?supplierId=79&brt=tourmind&loadMaster=true'

        return full_url, headers

    def process_http_request(self, url: str, method: str = "GET", headers: Optional[Dict] = None, **kwargs) -> Dict[str, Any]:
        """执行HTTP请求并获取响应，增加更长的超时时间"""
        try:
            default_headers = {
                'Accept': 'application/json, text/plain, */*',
                'Accept-Language': 'zh-CN,zh;q=0.9',
                'Connection': 'keep-alive',
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            }

            if headers:
                default_headers.update(headers)

            # 增加超时时间到60秒，并增加重试机制
            max_retries = 1
            for attempt in range(max_retries):
                try:
                    response = requests.request(
                        method=method, url=url, headers=default_headers,
                        verify=False, timeout=120, **kwargs)  # 增加超时时间到60秒

                    response_data = {
                        "status_code": response.status_code,
                        "response_data": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                        "metadata": {"url": url, "method": method, "success": response.status_code == 200}
                    }

                    return response_data

                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        logger.warning(f"HTTP请求超时，重试第{attempt + 1}次...")
                        time.sleep(2)  # 等待2秒后重试
                        continue
                    else:
                        raise

        except Exception as e:
            logger.error(f"HTTP请求失败: {str(e)}")
            return {"error": str(e)}

    def extract_room_types_for_csv(self, response_data: Dict[str, Any], hotel_id: str) -> List[Dict[str, Any]]:
        """提取房型数据并格式化为CSV需要的3列格式，只有存在待匹配供应商房型时才返回数据"""
        try:
            # 获取实际的响应数据
            actual_response = response_data.get("response_data", {})

            if isinstance(actual_response, str):
                try:
                    actual_response = json.loads(actual_response)
                except json.JSONDecodeError:
                    return []

            if not isinstance(actual_response, dict) or actual_response is None:
                return []

            api_data = actual_response.get("Data", {})
            if api_data is None:
                return []

            spl_room_type_list = api_data.get("SPLRoomTypeList", [])
            s_room_type_list = api_data.get("SRoomTypeList", [])

            # 先检查是否有待匹配供应商房型
            has_unmatched_supplier_rooms = False
            unmatched_supplier_rooms = []

            for room in spl_room_type_list or []:
                if room and room.get("SRoomID") == 0:  # 未匹配的房型
                    room_name = room.get("RoomTypeName", "").strip()
                    bed_desc = room.get("BedTypeDesc", "").strip()
                    supplier_room = f"{room_name} {bed_desc}".strip()

                    if supplier_room:
                        has_unmatched_supplier_rooms = True
                        unmatched_supplier_rooms.append({
                            "酒店ID": hotel_id,
                            "待匹配供应商房型+床型": supplier_room,
                            "标准房型+床型": ""
                        })

            # 如果没有待匹配供应商房型，返回空列表
            if not has_unmatched_supplier_rooms:
                return []

            csv_rows = unmatched_supplier_rooms

            # 处理标准房型
            for room in s_room_type_list or []:
                if room:
                    room_name = room.get("RoomTypeName", "").strip()
                    bed_desc = room.get("BedTypeDesc", "").strip()
                    standard_room = f"{room_name} {bed_desc}".strip()

                    if standard_room:
                        csv_rows.append({
                            "酒店ID": hotel_id,
                            "待匹配供应商房型+床型": "",
                            "标准房型+床型": standard_room
                        })

            return csv_rows

        except Exception as e:
            logger.error(f"房型数据提取失败: {str(e)}")
            return []

    def process_single_hotel_thread_safe(self, hotel_row: Dict) -> Dict[str, Any]:
        """线程安全的单个酒店处理函数"""
        hotel_id = str(hotel_row['hotel_id'])
        hotel_name = hotel_row.get('hotel_name', '')

        try:
            # 构建API请求
            api_url, api_headers = self.build_api_url(hotel_id)

            # 线程安全地获取cookies
            with login_lock:
                current_cookies = self.login_handler.get_current_cookies()
                if current_cookies:
                    api_headers['Cookie'] = current_cookies

            # 执行HTTP请求
            response_data = self.process_http_request(
                url=api_url, headers=api_headers)

            if "error" in response_data:
                return {"hotel_id": hotel_id, "status": "error", "error": response_data["error"]}

            # 检查是否需要重新登录
            if response_data.get("status_code") == 502 or "管理后台登录" in str(response_data.get("response_data")):
                with login_lock:  # 线程安全地重新登录
                    logger.warning(f"酒店{hotel_id}检测到需要重新登录")
                    new_cookies = self.login_handler.login_and_get_cookies()

                    if new_cookies:
                        api_headers['Cookie'] = new_cookies
                        response_data = self.process_http_request(
                            url=api_url, headers=api_headers)

                        if "error" in response_data or response_data.get("status_code") != 200:
                            return {"hotel_id": hotel_id, "status": "error", "error": "重新登录后请求仍然失败"}
                    else:
                        return {"hotel_id": hotel_id, "status": "error", "error": "重新登录失败"}

            elif response_data.get("status_code") != 200:
                return {"hotel_id": hotel_id, "status": "error", "error": f"HTTP请求失败，状态码: {response_data.get('status_code')}"}

            # 提取房型数据
            csv_rows = self.extract_room_types_for_csv(response_data, hotel_id)

            # 线程安全地添加到全局数据收集器
            with data_collection_lock:
                self.all_hotel_data.extend(csv_rows)

            # 更新进度统计
            with progress_lock:
                self.processed_count += 1
                if self.processed_count % 10 == 0:
                    logger.info(
                        f"进度: {self.processed_count}/{self.total_count} 已完成")

            return {
                "hotel_id": hotel_id,
                "hotel_name": hotel_name,
                "status": "success",
                "rows_count": len(csv_rows)
            }

        except Exception as e:
            logger.error(f"处理酒店{hotel_id}失败: {str(e)}")
            with progress_lock:
                self.error_count += 1
            return {"hotel_id": hotel_id, "status": "error", "error": str(e)}

    def process_hotels_from_csv_multithreaded(self, csv_file_path: str, output_dir: str = "output",
                                              start_index: int = 0, end_index: int = None,
                                              delay_seconds: float = 1.0) -> str:
        """使用多线程从CSV文件批量处理酒店，输出单个CSV文件"""
        try:
            # 读取酒店列表CSV
            logger.info(f"正在读取酒店列表文件: {csv_file_path}")
            hotels_df = pd.read_csv(csv_file_path)

            if 'hotel_id' not in hotels_df.columns:
                raise ValueError("CSV文件必须包含'hotel_id'列")

            total_hotels = len(hotels_df)
            logger.info(f"共读取到{total_hotels}个酒店")

            # 应用索引范围
            if end_index is None:
                end_index = total_hotels

            start_index = max(0, start_index)
            end_index = min(total_hotels, end_index)

            if start_index >= end_index:
                raise ValueError(
                    f"无效的索引范围: start_index={start_index}, end_index={end_index}")

            selected_hotels = hotels_df.iloc[start_index:end_index]
            self.total_count = len(selected_hotels)
            logger.info(
                f"选择处理酒店范围: {start_index}-{end_index-1}，共{self.total_count}个酒店")

            # 创建输出目录
            os.makedirs(output_dir, exist_ok=True)

            # 重置计数器和数据收集器
            self.processed_count = 0
            self.error_count = 0
            self.all_hotel_data = []

            # 使用线程池处理酒店
            logger.info(f"开始使用{self.max_workers}个线程并发处理酒店...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # 提交所有任务
                future_to_hotel = {}
                for index, row in selected_hotels.iterrows():
                    hotel_dict = row.to_dict()
                    future = executor.submit(
                        self.process_single_hotel_thread_safe, hotel_dict)
                    future_to_hotel[future] = hotel_dict

                    # 控制提交速度，增加间隔时间
                    if delay_seconds > 0:
                        time.sleep(delay_seconds)

                # 收集结果
                results = []
                for future in as_completed(future_to_hotel):
                    hotel_dict = future_to_hotel[future]
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        logger.error(f"获取结果失败: {str(e)}")
                        with progress_lock:
                            self.error_count += 1

            # 保存所有数据到一个CSV文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                output_dir, f"hotel_room_types_{timestamp}.csv")

            if self.all_hotel_data:
                df = pd.DataFrame(self.all_hotel_data)
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(
                    f"所有房型数据已保存到: {output_file} (共{len(self.all_hotel_data)}条记录)")
            else:
                logger.warning("没有提取到任何房型数据")

            # 输出最终统计
            logger.info("="*80)
            logger.info("多线程批量处理完成汇总:")
            logger.info(f"总处理酒店数量: {self.total_count}")
            logger.info(f"成功处理: {self.processed_count}")
            logger.info(f"失败处理: {self.error_count}")
            logger.info(f"总房型记录数: {len(self.all_hotel_data)}")
            logger.info(f"输出文件: {output_file}")
            logger.info("="*80)

            return output_file

        except Exception as e:
            logger.error(f"多线程批量处理失败: {str(e)}")
            raise


def main():
    """主函数 - 示例用法"""
    try:
        # 初始化提取器（使用10个线程）
        extractor = HotelRoomExtractor(max_workers=4)

        # CSV文件路径
        csv_file = "1000sampled_hotels.csv"

        # 检查文件是否存在
        if not os.path.exists(csv_file):
            logger.error(f"CSV文件不存在: {csv_file}")
            return

        # 多线程批量处理酒店（示例：处理前20个酒店）
        output_file = extractor.process_hotels_from_csv_multithreaded(
            csv_file_path=csv_file,
            output_dir="output",
            start_index=0,
            end_index=200,  # 处理前20个酒店
            delay_seconds=1.0  # 每个请求间隔1秒，增加等待时间
        )

        logger.info(f"多线程处理完成，输出文件: {output_file}")

    except Exception as e:
        logger.error(f"主函数执行失败: {str(e)}")


if __name__ == "__main__":
    main()
