#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试优化版缓存管理器的性能
比较原版和优化版缓存管理器在大规模数据下的性能差异
"""

from deberta.utils.room_cache_manager_optimized import RoomCacheManagerOptimized
from deberta.utils.room_cache_manager import RoomCacheManager
import os
import sys
import json
import time
import logging
import argparse
import random
from datetime import datetime

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (deberta目录)
parent_dir = os.path.dirname(current_dir)
# 获取项目根目录
BASE_DIR = os.path.dirname(parent_dir)

# 将项目根目录添加到Python路径
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 导入缓存管理器


def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def generate_test_data(count=10000):
    """生成测试数据"""
    test_data = {}
    for i in range(count):
        hotel_id = f"test_hotel_{i:05d}"
        # 每个酒店10-30个房型
        room_count = random.randint(10, 30)
        rooms = []
        for j in range(room_count):
            rooms.append({
                "room_id": f"room_{i:05d}_{j:02d}",
                "name": f"房型 {j+1}",
                "size": random.randint(15, 50),
                "price": random.randint(100, 2000),
                "features": [
                    "wifi",
                    "breakfast" if random.random() > 0.5 else "no_breakfast",
                    "window" if random.random() > 0.3 else "no_window"
                ]
            })

        test_data[hotel_id] = {
            "hotel_id": hotel_id,
            "Data": {
                "Rooms": rooms,
                "UpdateTime": datetime.now().isoformat()
            },
            "Success": True
        }

    return test_data


def test_original_cache(cache_dir, data, logger):
    """测试原始缓存管理器性能"""
    config = {
        "cache_dir": cache_dir,
        "cache_expire_days": 7,
        "base_dir": BASE_DIR
    }

    # 初始化原始缓存管理器
    start_time = time.time()
    cache_manager = RoomCacheManager(config)
    init_time = time.time() - start_time
    logger.info(f"原始缓存管理器初始化时间: {init_time:.3f}秒")

    # 写入测试数据
    total_items = len(data)
    logger.info(f"开始写入 {total_items} 条测试数据...")

    write_start_time = time.time()
    for i, (key, value) in enumerate(data.items()):
        cache_manager.set_cache(key, value)

        # 每1000条保存一次
        if (i + 1) % 1000 == 0:
            save_start = time.time()
            cache_manager.save()
            save_time = time.time() - save_start
            progress = (i + 1) / total_items * 100
            logger.info(
                f"原始缓存 - 进度: {i+1}/{total_items} ({progress:.1f}%), 保存耗时: {save_time:.3f}秒")

    # 最终保存
    final_save_start = time.time()
    cache_manager.save()
    final_save_time = time.time() - final_save_start
    write_time = time.time() - write_start_time

    logger.info(
        f"原始缓存 - 写入完成，总耗时: {write_time:.3f}秒，最终保存耗时: {final_save_time:.3f}秒")

    # 测试读取性能
    read_start_time = time.time()
    read_count = min(1000, len(data))
    test_keys = list(data.keys())[:read_count]

    for i, key in enumerate(test_keys):
        cache_manager.get_cache(key)
        if (i + 1) % 100 == 0:
            progress = (i + 1) / read_count * 100
            elapsed = time.time() - read_start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"原始缓存 - 读取进度: {i+1}/{read_count} ({progress:.1f}%), 速率: {rate:.1f}项/秒")

    read_time = time.time() - read_start_time
    logger.info(f"原始缓存 - 读取完成，读取 {read_count} 条数据耗时: {read_time:.3f}秒")

    # 清理过期缓存性能
    clear_start_time = time.time()
    cache_manager.clear_expired_cache()
    clear_time = time.time() - clear_start_time
    logger.info(f"原始缓存 - 清理过期缓存耗时: {clear_time:.3f}秒")

    return {
        "init_time": init_time,
        "write_time": write_time,
        "final_save_time": final_save_time,
        "read_time": read_time,
        "clear_time": clear_time
    }


def test_optimized_cache(cache_dir, data, logger):
    """测试优化版缓存管理器性能"""
    config = {
        "cache_dir": cache_dir,
        "cache_expire_days": 7,
        "base_dir": BASE_DIR,
        "shard_size": 1000,
        "save_threshold": 1000
    }

    # 初始化优化版缓存管理器
    start_time = time.time()
    cache_manager = RoomCacheManagerOptimized(config)
    init_time = time.time() - start_time
    logger.info(f"优化版缓存管理器初始化时间: {init_time:.3f}秒")

    # 写入测试数据
    total_items = len(data)
    logger.info(f"开始写入 {total_items} 条测试数据...")

    write_start_time = time.time()
    for i, (key, value) in enumerate(data.items()):
        cache_manager.set_cache(key, value)

        # 增量保存由优化版缓存管理器自动处理
        # 但我们依然每1000条记录一下状态
        if (i + 1) % 1000 == 0:
            progress = (i + 1) / total_items * 100
            elapsed = time.time() - write_start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"优化版缓存 - 进度: {i+1}/{total_items} ({progress:.1f}%), 写入速率: {rate:.1f}项/秒")

    # 最终保存
    final_save_start = time.time()
    cache_manager.save()
    final_save_time = time.time() - final_save_start
    write_time = time.time() - write_start_time

    # 获取缓存统计
    cache_stats = cache_manager.get_stats()
    logger.info(
        f"优化版缓存 - 写入完成，总耗时: {write_time:.3f}秒，最终保存耗时: {final_save_time:.3f}秒")
    logger.info(
        f"优化版缓存 - 统计信息: 共 {cache_stats['total_cache']} 条缓存，{cache_stats['total_shards']} 个分片")

    # 测试读取性能
    read_start_time = time.time()
    read_count = min(1000, len(data))
    test_keys = list(data.keys())[:read_count]

    for i, key in enumerate(test_keys):
        cache_manager.get_cache(key)
        if (i + 1) % 100 == 0:
            progress = (i + 1) / read_count * 100
            elapsed = time.time() - read_start_time
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            logger.info(
                f"优化版缓存 - 读取进度: {i+1}/{read_count} ({progress:.1f}%), 速率: {rate:.1f}项/秒")

    read_time = time.time() - read_start_time
    logger.info(f"优化版缓存 - 读取完成，读取 {read_count} 条数据耗时: {read_time:.3f}秒")

    # 测试优化功能
    optimize_start_time = time.time()
    optimize_stats = cache_manager.optimize()
    optimize_time = time.time() - optimize_start_time
    logger.info(f"优化版缓存 - 优化耗时: {optimize_time:.3f}秒，结果: {optimize_stats}")

    # 清理过期缓存性能
    clear_start_time = time.time()
    cache_manager.clear_expired_cache()
    clear_time = time.time() - clear_start_time
    logger.info(f"优化版缓存 - 清理过期缓存耗时: {clear_time:.3f}秒")

    return {
        "init_time": init_time,
        "write_time": write_time,
        "final_save_time": final_save_time,
        "read_time": read_time,
        "optimize_time": optimize_time,
        "clear_time": clear_time,
        "stats": cache_stats
    }


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试缓存管理器性能')
    parser.add_argument('--data_count', type=int,
                        default=5000, help='测试数据量，默认5000条')
    parser.add_argument('--cache_dir', type=str,
                        default=None, help='缓存目录，不指定则使用临时目录')
    parser.add_argument('--skip_original',
                        action='store_true', help='跳过测试原始缓存管理器')
    parser.add_argument('--skip_optimized',
                        action='store_true', help='跳过测试优化版缓存管理器')

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()
    logger.info("开始缓存性能测试")

    # 设置缓存目录
    if args.cache_dir:
        cache_dir = args.cache_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cache_dir = os.path.join(BASE_DIR, "test_cache", timestamp)

    os.makedirs(cache_dir, exist_ok=True)
    logger.info(f"使用缓存目录: {cache_dir}")

    # 生成测试数据
    data_count = args.data_count
    logger.info(f"正在生成 {data_count} 条测试数据...")
    test_data = generate_test_data(data_count)
    logger.info(f"测试数据生成完成")

    results = {}

    # 测试原始缓存管理器
    if not args.skip_original:
        original_cache_dir = os.path.join(cache_dir, "original")
        os.makedirs(original_cache_dir, exist_ok=True)

        try:
            logger.info("=== 开始测试原始缓存管理器 ===")
            original_results = test_original_cache(
                original_cache_dir, test_data, logger)
            results["original"] = original_results
            logger.info("=== 原始缓存管理器测试完成 ===")
        except Exception as e:
            logger.error(f"测试原始缓存管理器失败: {str(e)}")

    # 测试优化版缓存管理器
    if not args.skip_optimized:
        optimized_cache_dir = os.path.join(cache_dir, "optimized")
        os.makedirs(optimized_cache_dir, exist_ok=True)

        try:
            logger.info("=== 开始测试优化版缓存管理器 ===")
            optimized_results = test_optimized_cache(
                optimized_cache_dir, test_data, logger)
            results["optimized"] = optimized_results
            logger.info("=== 优化版缓存管理器测试完成 ===")
        except Exception as e:
            logger.error(f"测试优化版缓存管理器失败: {str(e)}")

    # 输出对比结果
    if "original" in results and "optimized" in results:
        logger.info("=== 性能对比结果 ===")

        # 初始化时间对比
        original_init = results["original"]["init_time"]
        optimized_init = results["optimized"]["init_time"]
        init_speedup = original_init / optimized_init if optimized_init > 0 else 0
        logger.info(
            f"初始化时间: 原始版={original_init:.3f}秒, 优化版={optimized_init:.3f}秒, 加速比={init_speedup:.2f}x")

        # 写入时间对比
        original_write = results["original"]["write_time"]
        optimized_write = results["optimized"]["write_time"]
        write_speedup = original_write / optimized_write if optimized_write > 0 else 0
        logger.info(
            f"写入时间: 原始版={original_write:.3f}秒, 优化版={optimized_write:.3f}秒, 加速比={write_speedup:.2f}x")

        # 最终保存时间对比
        original_save = results["original"]["final_save_time"]
        optimized_save = results["optimized"]["final_save_time"]
        save_speedup = original_save / optimized_save if optimized_save > 0 else 0
        logger.info(
            f"最终保存时间: 原始版={original_save:.3f}秒, 优化版={optimized_save:.3f}秒, 加速比={save_speedup:.2f}x")

        # 读取时间对比
        original_read = results["original"]["read_time"]
        optimized_read = results["optimized"]["read_time"]
        read_speedup = original_read / optimized_read if optimized_read > 0 else 0
        logger.info(
            f"读取时间: 原始版={original_read:.3f}秒, 优化版={optimized_read:.3f}秒, 加速比={read_speedup:.2f}x")

        # 清理过期缓存时间对比
        original_clear = results["original"]["clear_time"]
        optimized_clear = results["optimized"]["clear_time"]
        clear_speedup = original_clear / optimized_clear if optimized_clear > 0 else 0
        logger.info(
            f"清理过期缓存时间: 原始版={original_clear:.3f}秒, 优化版={optimized_clear:.3f}秒, 加速比={clear_speedup:.2f}x")

        # 总体加速比
        original_total = original_init + original_write + original_read + original_clear
        optimized_total = optimized_init + optimized_write + optimized_read + \
            optimized_clear + results["optimized"].get("optimize_time", 0)
        total_speedup = original_total / optimized_total if optimized_total > 0 else 0
        logger.info(
            f"总体时间: 原始版={original_total:.3f}秒, 优化版={optimized_total:.3f}秒, 加速比={total_speedup:.2f}x")

        # 保存结果到文件
        results["comparison"] = {
            "init_speedup": init_speedup,
            "write_speedup": write_speedup,
            "save_speedup": save_speedup,
            "read_speedup": read_speedup,
            "clear_speedup": clear_speedup,
            "total_speedup": total_speedup
        }

        result_file = os.path.join(cache_dir, "performance_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"性能测试结果已保存到: {result_file}")

    logger.info("测试完成")


if __name__ == "__main__":
    main()
