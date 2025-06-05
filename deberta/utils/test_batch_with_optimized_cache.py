#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试批处理工具与优化版缓存的集成
验证批处理工具能够正确使用优化版缓存管理器
"""

import os
import sys
import json
import time
import logging
import argparse
import pandas as pd
from datetime import datetime
import traceback

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (deberta目录)
parent_dir = os.path.dirname(current_dir)
# 获取项目根目录
BASE_DIR = os.path.dirname(parent_dir)

# 将项目根目录添加到Python路径
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 导入批处理和缓存管理器
try:
    from deberta.utils.room_cache_manager_optimized import RoomCacheManagerOptimized
    from deberta.utils.batch_room_matcher_cached import BatchProcessor, DEFAULT_CONFIG
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 全局日志器
logger = None

def setup_logging():
    """设置日志"""
    global logger
    
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(BASE_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"test_batch_cache_{timestamp}.log")

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
        
        logger = logging.getLogger(__name__)
        return logger
    except Exception as e:
        print(f"设置日志失败: {str(e)}，将使用基本控制台日志")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        logger = logging.getLogger(__name__)
        return logger


def load_hotel_sample(file_path, max_hotels=10):
    """从CSV文件加载测试用的酒店样本"""
    try:
        df = pd.read_csv(file_path)

        # 数据预处理
        df['hotel_id'] = df['hotel_id'].astype(str)
        df['country_code'] = df['country_code'].fillna('Unknown')
        df['country_code'] = df['country_code'].astype(str)
        df['hotel_name'] = df['hotel_name'].fillna('未知酒店名称')
        df['hotel_name'] = df['hotel_name'].astype(str)

        # 只取少量样本用于测试
        if max_hotels and len(df) > max_hotels:
            df = df.head(max_hotels)

        return df
    except Exception as e:
        if logger:
            logger.error(f"加载酒店数据失败: {str(e)}")
        else:
            print(f"加载酒店数据失败: {str(e)}")
        return pd.DataFrame()


def test_batch_with_optimized_cache(test_config, hotels_df, logger):
    """测试批处理工具与优化版缓存的集成"""
    # 获取并打印优化版缓存统计（如果有）
    cache_dir = test_config.get("cache_dir")
    if os.path.exists(cache_dir):
        try:
            cache_manager = RoomCacheManagerOptimized(test_config)
            stats = cache_manager.get_stats()
            logger.info(f"测试前缓存统计: {json.dumps(stats, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"获取缓存统计失败: {str(e)}")

    # 创建并运行批处理器
    try:
        start_time = time.time()

        # 初始化批处理器
        batch_processor = BatchProcessor(test_config)

        # 处理酒店数据
        batch_processor.process_hotels(hotels_df)

        # 统计总耗时
        total_time = time.time() - start_time
        logger.info(f"批处理测试完成，总耗时: {total_time:.2f}秒")

        # 获取并打印最终缓存统计
        try:
            if hasattr(batch_processor.room_api, 'cache_manager'):
                cache_manager = batch_processor.room_api.cache_manager
                if hasattr(cache_manager, 'get_stats'):
                    stats = cache_manager.get_stats()
                    logger.info(
                        f"测试后缓存统计: {json.dumps(stats, ensure_ascii=False)}")
        except Exception as e:
            logger.error(f"获取最终缓存统计失败: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"批处理测试失败: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def main():
    """主函数"""
    # 设置日志
    global logger
    logger = setup_logging()
    
    try:
        # 解析命令行参数
        parser = argparse.ArgumentParser(description='测试批处理工具与优化版缓存的集成')
        parser.add_argument('--hotels', type=str, default=os.path.join(
            BASE_DIR, 'deberta/data/10000sampled_hotels.csv'), help='酒店数据文件路径')
        parser.add_argument('--max_hotels', type=int, default=10, help='最大处理酒店数')
        parser.add_argument('--cache_dir', type=str, default=None, help='缓存目录')
        parser.add_argument('--clean_cache', action='store_true', help='清空现有缓存')
        parser.add_argument('--shard_size', type=int,
                            default=5, help='缓存分片大小，默认5（便于测试）')
        parser.add_argument('--save_threshold', type=int,
                            default=3, help='缓存保存阈值，默认3（便于测试）')

        args = parser.parse_args()

        logger.info("开始测试批处理工具与优化版缓存的集成")

        # 设置测试配置
        test_config = DEFAULT_CONFIG.copy()
        test_config["base_dir"] = BASE_DIR
        test_config["batch_size"] = 3  # 小批次便于测试
        test_config["threshold"] = 0.7  # 匹配阈值

        # 设置缓存相关配置
        if args.cache_dir:
            test_config["cache_dir"] = args.cache_dir
        else:
            # 默认使用测试专用缓存目录
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            test_config["cache_dir"] = os.path.join(
                BASE_DIR, "test_cache", timestamp)

        # 确保缓存目录存在
        os.makedirs(test_config["cache_dir"], exist_ok=True)

        # 设置为使用优化版缓存
        test_config["use_cache"] = True
        test_config["shard_size"] = args.shard_size
        test_config["save_threshold"] = args.save_threshold

        # 清空缓存（如果需要）
        if args.clean_cache and os.path.exists(test_config["cache_dir"]):
            logger.info(f"清空缓存目录: {test_config['cache_dir']}")
            for file in os.listdir(test_config["cache_dir"]):
                file_path = os.path.join(test_config["cache_dir"], file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    logger.error(f"删除文件 {file_path} 失败: {str(e)}")

        # 加载测试用酒店数据
        logger.info(f"加载测试用酒店数据: {args.hotels}")
        hotels_df = load_hotel_sample(args.hotels, args.max_hotels)
        if hotels_df.empty:
            logger.error("加载酒店数据失败，测试终止")
            return

        logger.info(f"成功加载 {len(hotels_df)} 条酒店数据")

        # 运行测试
        result = test_batch_with_optimized_cache(test_config, hotels_df, logger)

        if result:
            logger.info("测试成功完成")
        else:
            logger.error("测试失败")
            
    except Exception as e:
        if logger:
            logger.error(f"测试过程中发生错误: {str(e)}")
            logger.error(traceback.format_exc())
        else:
            print(f"测试过程中发生错误: {str(e)}")
            print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
