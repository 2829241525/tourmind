#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import logging
from typing import List

# ===== 全局配置 =====
CONFIG = {
    # 服务器配置
    "BASE_URL": "http://10.0.0.110:13009",
    "ENDPOINT": "/match/",

    # 请求配置
    "TIMEOUT": 30,
    "VERIFY_SSL": False,

    # 数组大小配置
    "ARRAY_SIZE": 3,

    # 默认数据
    "DEFAULT_SPL_ROOM_NAMES": [
        "Standard Family Room",
        "Run Of House, Non Smoking - Late Check-In 20:00 (3 Twin Bunk Beds) - Package Rate",
        "NoSmoke RunOfHouse CleaningEvery3days"
    ],

    "DEFAULT_S_ROOM_NAMES": [
        "Standard Family Room",
        "Run of House, Non Smoking - Late Check-In 20:00 2 Large Twin Beds/3 Twin Bunk Beds",
        "Run of House, Non Smoking 2 Large Twin Beds/3 Twin Bunk Beds"
    ],

    # 模型配置
    "MODEL": "cross_entropy_result1_nosplct",
    "TOKEN": "tourmind_room_match",

    # Cookie配置
    "COOKIE": "admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN",

    # 日志配置
    "LOG_LEVEL": logging.INFO,
    "LOG_FILE": "/home/maxon/disk2/roomMatch/room_match/logs/room_match_request.log"
}

# ===== 日志配置 =====
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RoomMatchClient:
    """房间匹配API客户端"""

    def __init__(self):
        self.base_url = CONFIG["BASE_URL"]
        self.endpoint = CONFIG["ENDPOINT"]
        self.timeout = CONFIG["TIMEOUT"]
        self.verify_ssl = CONFIG["VERIFY_SSL"]

        # 构建请求头
        self.headers = {
            'Cookie': CONFIG["COOKIE"],
            'cookie': CONFIG["COOKIE"],  # 两个cookie头，保持与原curl一致
            'User-Agent': 'Python Room Match Client/1.0.0',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '10.0.0.110:13009',
            'Connection': 'keep-alive'
        }

        logger.info(f"初始化房间匹配客户端，服务器: {self.base_url}")

    def extend_array(self, base_array: List[str], target_size: int, fill_value: str = "") -> List[str]:
        """
        扩展数组到指定大小

        Args:
            base_array: 基础数组
            target_size: 目标大小
            fill_value: 填充值

        Returns:
            扩展后的数组
        """
        if len(base_array) >= target_size:
            return base_array[:target_size]

        extended_array = base_array.copy()
        while len(extended_array) < target_size:
            # 循环使用基础数组中的元素，如果基础数组为空则使用填充值
            if base_array:
                extended_array.append(
                    base_array[len(extended_array) % len(base_array)])
            else:
                extended_array.append(fill_value)

        return extended_array

    def build_request_data(self,
                           array_size: int = None,
                           spl_room_names: List[str] = None,
                           s_room_names: List[str] = None,
                           model: str = None,
                           token: str = None) -> dict:
        """
        构建请求数据

        Args:
            array_size: 数组大小，默认使用配置中的值
            spl_room_names: 供应商房间名称列表
            s_room_names: 标准房间名称列表
            model: 模型名称
            token: 访问令牌

        Returns:
            请求数据字典
        """
        if array_size is None:
            array_size = CONFIG["ARRAY_SIZE"]

        if spl_room_names is None:
            spl_room_names = CONFIG["DEFAULT_SPL_ROOM_NAMES"]

        if s_room_names is None:
            s_room_names = CONFIG["DEFAULT_S_ROOM_NAMES"]

        if model is None:
            model = CONFIG["MODEL"]

        if token is None:
            token = CONFIG["TOKEN"]

        # 扩展数组到指定大小
        extended_spl_room_names = self.extend_array(spl_room_names, array_size)
        extended_s_room_names = self.extend_array(s_room_names, array_size)

        # 床型名称数组，默认为空字符串
        spl_room_bed_names = [""] * array_size
        s_room_bed_names = [""] * array_size

        request_data = {
            "spl_room_names": extended_spl_room_names,
            "spl_room_bed_names": spl_room_bed_names,
            "s_room_names": extended_s_room_names,
            "s_room_bed_names": s_room_bed_names,
            "model": model,
            "token": token
        }

        logger.info(f"构建请求数据，数组大小: {array_size}")
        logger.debug(
            f"请求数据: {json.dumps(request_data, ensure_ascii=False, indent=2)}")

        return request_data

    def send_match_request(self, request_data: dict) -> dict:
        """
        发送匹配请求

        Args:
            request_data: 请求数据

        Returns:
            响应数据
        """
        url = f"{self.base_url}{self.endpoint}"

        try:
            logger.info(f"发送请求到: {url}")

            response = requests.post(
                url,
                headers=self.headers,
                json=request_data,
                timeout=self.timeout,
                verify=self.verify_ssl
            )

            response.raise_for_status()

            result = response.json()
            logger.info(f"请求成功，状态码: {response.status_code}")
            logger.info(
                f"响应数据: {json.dumps(result, ensure_ascii=False, indent=2)}")

            return result

        except requests.exceptions.RequestException as e:
            logger.error(f"请求失败: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"解析响应JSON失败: {e}")
            raise

    def match_rooms(self,
                    array_size: int = None,
                    spl_room_names: List[str] = None,
                    s_room_names: List[str] = None,
                    model: str = None,
                    token: str = None) -> dict:
        """
        执行房间匹配

        Args:
            array_size: 数组大小
            spl_room_names: 供应商房间名称列表
            s_room_names: 标准房间名称列表
            model: 模型名称
            token: 访问令牌

        Returns:
            匹配结果
        """
        request_data = self.build_request_data(
            array_size=array_size,
            spl_room_names=spl_room_names,
            s_room_names=s_room_names,
            model=model,
            token=token
        )

        return self.send_match_request(request_data)


def main():
    """主函数"""
    try:
        # 创建客户端
        client = RoomMatchClient()

        # 示例1: 使用默认配置（3个元素）
        logger.info("=== 示例1: 使用默认配置 ===")
        result1 = client.match_rooms()
        print("默认配置结果:")
        print(json.dumps(result1, ensure_ascii=False, indent=2))

        # 示例2: 扩展到5个元素
        logger.info("=== 示例2: 扩展到5个元素 ===")
        result2 = client.match_rooms(array_size=5)
        print("\n扩展到5个元素结果:")
        print(json.dumps(result2, ensure_ascii=False, indent=2))

        # 示例3: 自定义房间名称和数组大小
        logger.info("=== 示例3: 自定义房间名称 ===")
        custom_spl_rooms = ["Deluxe Room", "Suite Room"]
        custom_s_rooms = ["Deluxe Standard Room", "Executive Suite"]

        result3 = client.match_rooms(
            array_size=4,
            spl_room_names=custom_spl_rooms,
            s_room_names=custom_s_rooms
        )
        print("\n自定义房间名称结果:")
        print(json.dumps(result3, ensure_ascii=False, indent=2))

    except Exception as e:
        logger.error(f"执行失败: {e}")
        print(f"错误: {e}")


if __name__ == "__main__":
    main()
