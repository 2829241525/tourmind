#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
房型数据批量预缓存工具
支持多线程并发获取酒店房型数据并缓存
减少批量处理时的API请求延迟
"""

# 首先导入基础模块
import os
import sys
import json
import time
import queue
import logging
import argparse
import threading
import pandas as pd
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
import warnings
from urllib3.exceptions import InsecureRequestWarning

# 忽略SSL警告
warnings.filterwarnings('ignore', category=InsecureRequestWarning)

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (deberta目录)
parent_dir = os.path.dirname(current_dir)
# 获取项目根目录
BASE_DIR = os.path.dirname(parent_dir)

# 将项目根目录添加到Python路径
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# 确保能正确导入deberta模块
try:
    from deberta.utils.batch_room_matcher_cached import RoomTypeAPI, DEFAULT_CONFIG
    # 导入旧的缓存管理器作为备用
    from deberta.utils.room_cache_manager import RoomCacheManager
    # 优先使用优化版缓存管理器
    try:
        from deberta.utils.room_cache_manager_optimized import RoomCacheManagerOptimized
        USE_OPTIMIZED_CACHE = True
        print(f"使用优化版缓存管理器")
    except ImportError:
        USE_OPTIMIZED_CACHE = False
        print(f"优化版缓存管理器未找到，使用原始版本")

    print(f"成功导入deberta模块，项目根目录: {BASE_DIR}")
except ImportError as e:
    print(f"导入deberta模块失败: {e}")
    print(f"当前Python路径: {sys.path}")
    raise


# 优化配置，添加默认的分片配置
DEFAULT_CACHE_CONFIG = {
    "shard_size": 1000,        # 每个分片的最大缓存条目数
    "save_threshold": 500      # 多少条记录保存一次
}


class RoomTypePrecacher:
    """房型数据预缓存器"""

    def __init__(self, config: Dict, max_threads: int = 20):
        """
        初始化预缓存器

        参数:
            config: 配置信息
            max_threads: 最大线程数
        """
        self.config = config
        self.max_threads = max_threads
        self.logger = logging.getLogger(f"{__name__}.RoomTypePrecacher")

        # 增加缓存配置
        if "cache_config" not in self.config:
            self.config["cache_config"] = DEFAULT_CACHE_CONFIG.copy()

        # 确保分片大小合理
        if config.get("total_hotels", 0) > 0:
            # 根据总酒店数调整分片大小，避免分片过多
            total_hotels = config.get("total_hotels", 10000)
            # 确保每个分片至少有500条数据，且分片总数不超过50个
            suggested_shard_size = max(500, total_hotels // 50)
            # 更新配置但不小于默认值
            cache_config = self.config["cache_config"]
            cache_config["shard_size"] = max(
                cache_config.get("shard_size", 1000),
                suggested_shard_size
            )
            self.logger.info(f"调整分片大小为: {cache_config['shard_size']}")

        # 初始化API客户端
        self.room_api = RoomTypeAPI(config["api_base_url"], config)

        # 初始化统计信息
        self.stats = {
            "total_hotels": 0,
            "processed_hotels": 0,
            "success_hotels": 0,
            "failed_hotels": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "start_time": time.time(),
            "end_time": 0,
            "duration": 0
        }

        # 线程安全的计数器和队列
        self.hotel_queue = queue.Queue()
        self.lock = threading.Lock()

        # 初始化线程池
        self.threads = []

    def load_hotels_data(self, file_path: str) -> pd.DataFrame:
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
            self.logger.error(f"加载酒店数据失败: {str(e)}")
            raise

    def worker(self):
        """工作线程，从队列获取酒店并处理"""
        # 本地统计，减少锁竞争
        local_stats = {
            "processed": 0,
            "success": 0,
            "failed": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }

        # 批量处理列表
        batch_size = 10
        processed_hotels = []

        while True:
            try:
                # 从队列获取酒店信息
                hotel_info = self.hotel_queue.get(block=False)
                if hotel_info is None:
                    break

                hotel_id = hotel_info["hotel_id"]
                hotel_name = hotel_info["hotel_name"]
                country = hotel_info["country"]

                # 处理单个酒店
                result = self.process_hotel(
                    hotel_id, hotel_name, country, local_stats)

                # 更新本地统计
                local_stats["processed"] += 1
                if result:
                    local_stats["success"] += 1
                else:
                    local_stats["failed"] += 1

                # 记录处理完成的酒店
                processed_hotels.append(hotel_id)

                # 标记任务完成
                self.hotel_queue.task_done()

                # 每处理一定数量的酒店，批量更新全局统计信息
                if len(processed_hotels) >= batch_size:
                    self._update_global_stats(
                        local_stats, len(processed_hotels))
                    processed_hotels = []
                    # 重置本地统计中的计数器
                    local_stats["processed"] = 0
                    local_stats["success"] = 0
                    local_stats["failed"] = 0

            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"工作线程发生错误: {str(e)}")
                # 标记任务完成
                if 'hotel_info' in locals():
                    self.hotel_queue.task_done()

        # 处理剩余的统计数据
        if local_stats["processed"] > 0:
            self._update_global_stats(local_stats, len(processed_hotels))

    def _update_global_stats(self, local_stats, batch_size):
        """批量更新全局统计信息，减少锁竞争"""
        with self.lock:
            self.stats["processed_hotels"] += local_stats["processed"]
            self.stats["success_hotels"] += local_stats["success"]
            self.stats["failed_hotels"] += local_stats["failed"]
            self.stats["cache_hits"] += local_stats["cache_hits"]
            self.stats["cache_misses"] += local_stats["cache_misses"]

            # 每处理50个酒店，才输出进度，减少日志I/O
            if self.stats["processed_hotels"] % 50 == 0 or self.stats["processed_hotels"] == self.stats["total_hotels"]:
                progress = (self.stats["processed_hotels"] /
                            self.stats["total_hotels"]) * 100
                self.logger.info(
                    f"进度: {self.stats['processed_hotels']}/{self.stats['total_hotels']} ({progress:.1f}%)")

                # 仅当处理了一定数量的酒店后才保存缓存
                # 使用缓存管理器配置的保存阈值，而不是固定值
                save_threshold = self.config.get(
                    "cache_config", {}).get("save_threshold", 100)
                if self.stats["processed_hotels"] % save_threshold == 0:
                    if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                        # 使用优化后的增量保存，不需要每次都保存全部数据
                        self.room_api.cache_manager.save()

    def process_hotel(self, hotel_id: str, hotel_name: str = "未知", country: str = "未知", local_stats: Dict = None) -> bool:
        """处理单个酒店，获取和缓存房型数据"""
        max_retries = 3  # 最大重试次数
        retry_delay = 0.1  # 初始重试间隔秒数，改小以减少等待时间
        attempt = 0

        while attempt < max_retries:
            try:
                if attempt > 0:
                    # 仅在DEBUG日志级别下输出重试信息
                    self.logger.debug(
                        f"重试 ({attempt}/{max_retries}) 获取酒店 {hotel_id} 数据")

                self.logger.debug(
                    f"预缓存酒店: {hotel_id} ({hotel_name}, {country})")

                # 检查缓存是否已存在
                if hasattr(self.room_api, 'use_cache') and self.room_api.use_cache:
                    if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                        cached_data = self.room_api.cache_manager.get_cache(
                            hotel_id)
                        if cached_data is not None:
                            # 使用本地统计而非全局锁
                            if local_stats is not None:
                                local_stats["cache_hits"] += 1
                            return True

                # 缓存不存在，从API获取数据
                # 使用本地统计而非全局锁
                if local_stats is not None:
                    local_stats["cache_misses"] += 1

                # 获取房型数据
                data = self.room_api.fetch_room_data(hotel_id)

                if data and isinstance(data, dict) and "Data" in data:
                    # 缓存将在fetch_room_data内部处理
                    return True
                else:
                    attempt += 1
                    if attempt >= max_retries:
                        self.logger.warning(f"获取酒店 {hotel_id} 数据失败，已达到最大重试次数")
                        return False
                    else:
                        # 线性增长的重试延迟，而非指数增长
                        time.sleep(retry_delay)
                        retry_delay += 1  # 每次增加1秒，而不是乘以2
                        continue

            except Exception as e:
                self.logger.error(f"处理酒店 {hotel_id} 时发生错误: {str(e)}")
                attempt += 1
                if attempt >= max_retries:
                    return False
                else:
                    # 线性增长的重试延迟
                    time.sleep(retry_delay)
                    retry_delay += 0.2  # 每次增加1秒

    def cache_hotels_data(self, hotels_data: pd.DataFrame, country_code: Optional[str] = None, max_hotels: Optional[int] = None):
        """批量缓存酒店房型数据"""
        try:
            # 筛选国家
            if country_code:
                hotels_data = hotels_data[hotels_data['country_code']
                                          == country_code]
                self.logger.info(
                    f"筛选 {country_code} 的酒店: {len(hotels_data)} 家")

            # 限制处理数量
            if max_hotels and len(hotels_data) > max_hotels:
                hotels_data = hotels_data.sample(n=max_hotels, random_state=42)
                self.logger.info(f"随机选择 {max_hotels} 家酒店处理")

            # 初始化统计信息
            self.stats["total_hotels"] = len(hotels_data)
            self.stats["start_time"] = time.time()

            # 更新配置中的总酒店数，用于优化缓存分片大小
            self.config["total_hotels"] = self.stats["total_hotels"]

            self.logger.info(f"开始预缓存 {self.stats['total_hotels']} 家酒店的房型数据")
            self.logger.info(f"使用 {self.max_threads} 个线程进行并发处理")

            # 在处理前优化缓存
            if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                if hasattr(self.room_api.cache_manager, 'optimize'):
                    self.logger.info("正在优化缓存存储结构...")
                    optimize_stats = self.room_api.cache_manager.optimize()
                    self.logger.info(f"缓存优化完成: {optimize_stats}")

            # 将酒店数据加入队列
            for idx, row in hotels_data.iterrows():
                hotel_info = {
                    "hotel_id": str(row['hotel_id']),
                    "hotel_name": row.get('hotel_name', '未知酒店'),
                    "country": row.get('country_code', '未知')
                }
                self.hotel_queue.put(hotel_info)

            # 创建并启动工作线程
            # 确保非测试模式下线程数不低于5个
            test_mode = self.config.get("test_mode", False)
            if test_mode:
                # 测试模式只使用1个线程
                thread_count = 1
            else:
                # 正常模式下，线程数至少5个，但不超过设定的最大线程数和酒店数量
                thread_count = max(5, min(self.max_threads, len(hotels_data)))

            self.logger.info(f"最终使用 {thread_count} 个线程进行处理")
            self.threads = []

            for i in range(thread_count):
                t = threading.Thread(target=self.worker)
                t.daemon = True
                t.start()
                self.threads.append(t)
                self.logger.debug(f"启动工作线程 {i+1}")

            # 等待队列处理完成
            self.hotel_queue.join()

            # 添加终止信号并等待所有线程结束
            for i in range(thread_count):
                self.hotel_queue.put(None)

            for t in self.threads:
                t.join()

            # 保存最终缓存
            if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                self.room_api.cache_manager.save()
                self.logger.info(f"已保存最终房型数据缓存")

                # 清理过期缓存
                self.room_api.cache_manager.clear_expired_cache()

                # 如果是优化版缓存管理器，再次执行优化
                if hasattr(self.room_api.cache_manager, 'optimize'):
                    self.logger.info("处理完成后正在优化缓存存储...")
                    optimize_stats = self.room_api.cache_manager.optimize()
                    self.logger.info(f"最终缓存优化完成: {optimize_stats}")

                # 如果支持获取统计，输出缓存统计信息
                if hasattr(self.room_api.cache_manager, 'get_stats'):
                    cache_stats = self.room_api.cache_manager.get_stats()
                    self.logger.info(
                        f"缓存统计信息: {json.dumps(cache_stats, ensure_ascii=False)}")

            # 更新统计信息
            self.stats["end_time"] = time.time()
            self.stats["duration"] = self.stats["end_time"] - \
                self.stats["start_time"]

            # 输出统计信息
            self._log_final_stats()

        except Exception as e:
            self.logger.error(f"批量缓存过程中发生错误: {str(e)}")
            # 出错时也保存缓存
            if hasattr(self.room_api, 'cache_manager') and self.room_api.cache_manager:
                self.room_api.cache_manager.save()
                self.logger.info(f"已保存房型数据缓存 (错误恢复)")
            raise

    def _log_final_stats(self):
        """输出最终统计信息"""
        self.logger.info("=== 预缓存完成 ===")
        self.logger.info(f"总酒店数: {self.stats['total_hotels']}")
        self.logger.info(f"成功处理: {self.stats['success_hotels']}")
        self.logger.info(f"处理失败: {self.stats['failed_hotels']}")

        # 缓存统计
        cache_hits = self.stats.get("cache_hits", 0)
        cache_misses = self.stats.get("cache_misses", 0)
        total_requests = cache_hits + cache_misses
        cache_hit_rate = (cache_hits / total_requests *
                          100) if total_requests > 0 else 0

        self.logger.info(f"已有缓存数: {cache_hits}")
        self.logger.info(f"新增缓存数: {cache_misses}")
        self.logger.info(f"缓存命中率: {cache_hit_rate:.2f}%")

        # 时间统计
        duration_seconds = self.stats["duration"]
        duration_minutes = duration_seconds / 60
        avg_time = duration_seconds / \
            self.stats['total_hotels'] if self.stats['total_hotels'] > 0 else 0

        self.logger.info(
            f"总用时: {duration_seconds:.2f}秒 ({duration_minutes:.2f}分钟)")
        self.logger.info(f"平均每个酒店处理时间: {avg_time:.2f}秒")

        # 保存统计信息到文件
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 确保使用正确的缓存目录路径
            if os.path.isabs(self.config['cache_dir']):
                stats_dir = self.config['cache_dir']
            else:
                stats_dir = os.path.join(BASE_DIR, self.config['cache_dir'])

            # 确保目录存在
            os.makedirs(stats_dir, exist_ok=True)

            stats_file = os.path.join(
                stats_dir, f"precache_stats_{timestamp}.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats, f, ensure_ascii=False, indent=2)
            self.logger.info(f"统计信息已保存到: {stats_file}")
        except Exception as e:
            self.logger.error(f"保存统计信息时发生错误: {str(e)}")


def setup_logging(log_dir: str):
    """设置日志"""
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"precache_{timestamp}.log")

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


# 修改RoomTypeAPI类以支持优化版缓存管理器
def patch_room_type_api():
    """
    动态替换RoomTypeAPI中的缓存管理器初始化逻辑，
    优先使用优化版缓存管理器
    """
    original_init = RoomTypeAPI.__init__

    def new_init(self, api_base_url, config=None):
        original_init(self, api_base_url, config)

        # 如果启用了缓存，尝试使用优化版缓存管理器
        if getattr(self, 'use_cache', False) and USE_OPTIMIZED_CACHE:
            try:
                # 清理原有缓存管理器
                if hasattr(self, 'cache_manager') and self.cache_manager:
                    # 保存现有缓存再关闭
                    self.cache_manager.save()

                # 使用优化版缓存管理器
                logger = logging.getLogger(f"{__name__}.RoomTypeAPI")
                logger.info("正在切换到优化版缓存管理器...")

                # 复制配置并添加缓存特定配置
                cache_config = config.copy()
                if "cache_config" in config:
                    for k, v in config["cache_config"].items():
                        cache_config[k] = v

                self.cache_manager = RoomCacheManagerOptimized(cache_config)
                logger.info("已启用优化版缓存管理器")
            except Exception as e:
                logger = logging.getLogger(f"{__name__}.RoomTypeAPI")
                logger.error(f"启用优化版缓存管理器失败: {str(e)}，将使用原始版本")

    # 替换初始化方法
    if USE_OPTIMIZED_CACHE:
        RoomTypeAPI.__init__ = new_init


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='房型数据批量预缓存工具')

    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument(
        '--hotels', type=str, default=os.path.join(BASE_DIR, 'deberta/data/10000sampled_hotels.csv'), help='酒店数据文件路径')
    parser.add_argument('--country', type=str, default=None, help='指定处理的国家代码')
    parser.add_argument('--max_hotels', type=int, default=None, help='最大处理酒店数')
    parser.add_argument('--threads', type=int, default=20,
                        help='最大线程数，默认20，最小建议为5')
    parser.add_argument(
        '--cache_dir', type=str, default=None, help='缓存目录，不指定则使用默认配置')
    parser.add_argument(
        '--cache_expire_days', type=int, default=7, help='缓存过期时间(天)')
    parser.add_argument(
        '--test_mode', action='store_true', help='测试模式，只处理1个酒店，验证配置是否正确')
    parser.add_argument(
        '--debug', action='store_true', help='启用DEBUG级别日志')
    parser.add_argument(
        '--save_interval', type=int, default=100, help='缓存保存间隔，每处理多少条数据保存一次，默认100')
    parser.add_argument(
        '--batch_size', type=int, default=10, help='批处理大小，每次向线程池提交多少任务，默认10')
    parser.add_argument(
        '--quiet', action='store_true', help='安静模式，减少日志输出')
    parser.add_argument(
        '--perf_monitor', action='store_true', help='启用性能监控，记录CPU和内存使用情况')
    parser.add_argument(
        '--shard_size', type=int, default=1000, help='缓存分片大小，每个分片最多包含多少条记录，默认1000')
    parser.add_argument(
        '--use_optimized_cache', action='store_true', help='强制使用优化版缓存管理器')
    parser.add_argument(
        '--no_optimize', action='store_true', help='禁用缓存优化，对于大型数据集可能提高初始速度')

    args = parser.parse_args()

    try:
        # 启用性能监控
        if args.perf_monitor:
            try:
                import psutil
                pid = os.getpid()
                process = psutil.Process(pid)
                start_cpu_percent = process.cpu_percent(interval=0.1)
                start_memory = process.memory_info().rss / (1024 * 1024)  # MB
                start_time = time.time()
                logger = logging.getLogger(__name__)
                logger.info(
                    f"启用性能监控: PID={pid}, 初始CPU使用率={start_cpu_percent}%, 内存占用={start_memory:.2f}MB")

                # 创建性能监控线程
                def monitor_performance():
                    while True:
                        try:
                            cpu_percent = process.cpu_percent(interval=1)
                            memory = process.memory_info().rss / (1024 * 1024)  # MB
                            elapsed = time.time() - start_time
                            logger.info(
                                f"性能监控 [{elapsed:.1f}秒]: CPU={cpu_percent}%, 内存={memory:.2f}MB")
                            time.sleep(30)  # 每30秒记录一次
                        except Exception:
                            break

                monitor_thread = threading.Thread(
                    target=monitor_performance, daemon=True)
                monitor_thread.start()
            except ImportError:
                logger.warning("无法启用性能监控，需要安装psutil库: pip install psutil")

        # 设置日志
        log_dir = os.path.join(BASE_DIR, "logs")
        log_file = setup_logging(log_dir)

        # 设置日志级别
        if args.debug:
            logging.getLogger().setLevel(logging.DEBUG)
            logging.getLogger("deberta").setLevel(logging.DEBUG)

        logger = logging.getLogger(__name__)
        logger.info("开始批量预缓存")
        logger.info(f"日志文件: {log_file}")
        logger.info(f"项目基础目录: {BASE_DIR}")

        # 强制使用优化版缓存管理器
        global USE_OPTIMIZED_CACHE
        if args.use_optimized_cache:
            USE_OPTIMIZED_CACHE = True
            logger.info("强制使用优化版缓存管理器")

        # 加载配置
        config = DEFAULT_CONFIG.copy()
        # 确保基础目录正确
        config["base_dir"] = BASE_DIR

        if args.config:
            with open(args.config, 'r', encoding='utf-8') as f:
                config.update(json.load(f))

        # 配置缓存
        config["use_cache"] = True
        if args.cache_dir:
            config["cache_dir"] = args.cache_dir
        else:
            # 默认缓存目录
            config["cache_dir"] = os.path.join(BASE_DIR, "cache")

        if args.cache_expire_days:
            config["cache_expire_days"] = args.cache_expire_days

        # 配置缓存优化选项
        cache_config = config.get("cache_config", {})

        # 设置缓存分片大小
        if args.shard_size:
            cache_config["shard_size"] = args.shard_size

        # 设置缓存保存间隔
        if args.save_interval:
            cache_config["save_threshold"] = args.save_interval

        # 设置缓存优化标识
        if args.no_optimize:
            cache_config["no_optimize"] = True

        # 更新配置中的缓存配置
        config["cache_config"] = cache_config

        # 设置测试模式标志
        config["test_mode"] = args.test_mode

        # 设置安静模式
        if args.quiet:
            config["quiet_mode"] = True
            # 提高日志级别，减少输出
            logging.getLogger().setLevel(logging.WARNING)

        # 设置批处理大小
        if args.batch_size:
            config["batch_size"] = args.batch_size

        # 输出配置信息
        logger.info(
            f"缓存目录: {config['cache_dir']}, 过期时间: {config['cache_expire_days']}天")
        logger.info(f"缓存分片大小: {cache_config.get('shard_size', 1000)}")
        logger.info(f"缓存保存间隔: 每 {cache_config.get('save_threshold', 100)} 条记录")

        # 如果使用优化版缓存，修补RoomTypeAPI
        if USE_OPTIMIZED_CACHE:
            patch_room_type_api()

        # 限制线程数
        max_threads = min(args.threads, 20)  # 最大不超过20个线程
        # 非测试模式下确保至少有5个线程
        if not args.test_mode and max_threads < 5:
            max_threads = 5
            logger.info(f"调整线程数至最小值5")

        # 检查各个路径是否存在
        if not os.path.exists(BASE_DIR):
            logger.warning(f"警告: 基础目录不存在 - {BASE_DIR}")

        # 确保缓存目录存在
        os.makedirs(config['cache_dir'], exist_ok=True)

        # 检查酒店数据文件是否存在
        if not os.path.exists(args.hotels):
            logger.error(f"错误: 酒店数据文件不存在 - {args.hotels}")
            sys.exit(1)
        else:
            logger.info(f"酒店数据文件: {args.hotels}")

        # 创建预缓存器
        start_time = time.time()
        precacher = RoomTypePrecacher(config, max_threads=max_threads)

        # 加载酒店数据
        hotels_df = precacher.load_hotels_data(args.hotels)
        logger.info(f"加载了 {len(hotels_df)} 家酒店的数据")

        # 测试模式下只处理一个酒店
        if args.test_mode:
            logger.info("测试模式: 只处理1个酒店")
            if not hotels_df.empty:
                hotels_df = hotels_df.head(1)

        # 开始批量缓存
        precacher.cache_hotels_data(
            hotels_df,
            country_code=args.country,
            max_hotels=1 if args.test_mode else args.max_hotels
        )

        # 打印总耗时
        total_time = time.time() - start_time
        logger.info(f"任务总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")

        # 如果启用了性能监控，再次记录
        if args.perf_monitor and 'process' in locals():
            try:
                end_cpu_percent = process.cpu_percent(interval=0.1)
                end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                logger.info(
                    f"性能监控最终结果: CPU使用率={end_cpu_percent}%, 内存占用={end_memory:.2f}MB")
                logger.info(f"内存变化: {end_memory - start_memory:.2f}MB")
            except Exception as e:
                logger.error(f"获取性能监控数据失败: {str(e)}")

    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
