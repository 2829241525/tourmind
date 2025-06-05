#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
优化版房型数据缓存管理器
- 实现分片存储，减少单个缓存文件大小
- 支持增量更新，避免频繁写入整个缓存
- 使用二进制格式提高读写效率
- 添加索引加速查找
"""

import os
import time
import pickle
import logging
import hashlib
import threading
from typing import Dict, Any, Optional, Set, List, Tuple
from datetime import datetime, timedelta
import shutil


class CacheShard:
    """缓存分片类，管理单个缓存分片文件"""

    def __init__(self, shard_id: str, cache_dir: str, expire_days: int = 7):
        """
        初始化缓存分片

        参数:
            shard_id: 分片ID
            cache_dir: 缓存目录
            expire_days: 缓存过期天数
        """
        self.shard_id = shard_id
        self.cache_dir = cache_dir
        self.expire_days = expire_days
        self.cache_file = os.path.join(
            cache_dir, f"room_cache_shard_{shard_id}.pkl")
        self.cache_data: Dict[str, Dict] = {}
        self.modified = False
        self.last_access_time: Dict[str, float] = {}
        self.logger = logging.getLogger(f"{__name__}.CacheShard.{shard_id}")

        # 加载缓存数据
        self._load()

    def _load(self) -> None:
        """从文件加载缓存数据"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'rb') as f:
                    data = pickle.load(f)
                    self.cache_data = data.get('data', {})
                    self.last_access_time = data.get('access_time', {})
                self.logger.debug(
                    f"分片 {self.shard_id} 已加载 {len(self.cache_data)} 条缓存")
            else:
                self.logger.debug(f"分片 {self.shard_id} 不存在，创建新分片")
        except Exception as e:
            self.logger.error(f"加载分片 {self.shard_id} 失败: {str(e)}")
            # 如果加载失败，使用空缓存
            self.cache_data = {}
            self.last_access_time = {}

    def save(self) -> bool:
        """保存缓存数据到文件"""
        if not self.modified:
            return False  # 如果没有修改，不需要保存

        try:
            # 创建临时文件避免直接写入造成的损坏风险
            temp_file = f"{self.cache_file}.tmp"
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'data': self.cache_data,
                    'access_time': self.last_access_time,
                    'update_time': time.time()
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            # 安全地替换原文件
            if os.path.exists(self.cache_file):
                os.replace(temp_file, self.cache_file)
            else:
                os.rename(temp_file, self.cache_file)

            self.modified = False
            return True
        except Exception as e:
            self.logger.error(f"保存分片 {self.shard_id} 失败: {str(e)}")
            return False

    def get(self, key: str) -> Optional[Dict]:
        """获取缓存数据"""
        if key in self.cache_data:
            # 更新访问时间
            self.last_access_time[key] = time.time()
            return self.cache_data[key]
        return None

    def set(self, key: str, value: Dict) -> None:
        """设置缓存数据"""
        self.cache_data[key] = value
        self.last_access_time[key] = time.time()
        self.modified = True

    def clear_expired(self) -> int:
        """清理过期缓存"""
        now = time.time()
        expire_time = now - (self.expire_days * 24 * 60 * 60)
        expired_keys = [
            k for k, t in self.last_access_time.items() if t < expire_time]

        count = 0
        for key in expired_keys:
            if key in self.cache_data:
                del self.cache_data[key]
                del self.last_access_time[key]
                count += 1
                self.modified = True

        if count > 0:
            self.logger.debug(f"分片 {self.shard_id} 清理了 {count} 条过期缓存")

        return count

    def get_size(self) -> int:
        """获取当前分片中的缓存条目数"""
        return len(self.cache_data)


class RoomCacheManagerOptimized:
    """优化版房型数据缓存管理器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化缓存管理器

        参数:
            config: 配置信息，包含以下字段:
                - cache_dir: 缓存目录
                - cache_expire_days: 缓存过期天数
                - shard_size: 每个分片的最大缓存条目数，默认1000
                - save_threshold: 增量保存阈值，默认100
        """
        self.config = config
        self.cache_dir = self._get_cache_dir()
        self.expire_days = config.get('cache_expire_days', 7)
        self.shard_size = config.get('shard_size', 1000)
        self.save_threshold = config.get('save_threshold', 100)

        # 分片映射
        self.shards: Dict[str, CacheShard] = {}
        self.shard_index: Dict[str, str] = {}  # key -> shard_id

        # 用于跟踪新增或修改的缓存项
        self.modified_keys: Set[str] = set()
        self.modified_count = 0

        # 线程锁
        self.lock = threading.RLock()

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        self.logger = logging.getLogger(
            f"{__name__}.RoomCacheManagerOptimized")
        self.logger.info(
            f"初始化优化版缓存管理器: 目录={self.cache_dir}, 过期天数={self.expire_days}, 分片大小={self.shard_size}")

        # 加载分片索引
        self._load_shards()

    def _get_cache_dir(self) -> str:
        """获取缓存目录的绝对路径"""
        cache_dir = self.config.get('cache_dir', 'cache')
        if os.path.isabs(cache_dir):
            return cache_dir

        # 相对于BASE_DIR的路径
        base_dir = self.config.get('base_dir', os.getcwd())
        return os.path.join(base_dir, cache_dir)

    def _get_shard_id(self, key: str) -> str:
        """根据键获取分片ID"""
        if key in self.shard_index:
            return self.shard_index[key]

        # 为新键计算分片ID
        # 使用MD5哈希的前8位作为分片前缀
        hash_prefix = hashlib.md5(key.encode()).hexdigest()[:8]

        # 查找可用的分片（未满的分片）
        available_shards = [shard_id for shard_id, shard in self.shards.items()
                            if shard.get_size() < self.shard_size]

        if available_shards:
            # 如果有可用分片，选择第一个
            shard_id = available_shards[0]
        else:
            # 否则创建新分片
            shard_id = f"{hash_prefix}_{len(self.shards)}"
            self.shards[shard_id] = CacheShard(
                shard_id, self.cache_dir, self.expire_days)

        # 记录键到分片的映射
        self.shard_index[key] = shard_id
        return shard_id

    def _load_shards(self) -> None:
        """加载所有分片"""
        try:
            # 扫描缓存目录，加载所有分片文件
            count = 0
            for file in os.listdir(self.cache_dir):
                if file.startswith("room_cache_shard_") and file.endswith(".pkl"):
                    shard_id = file[len("room_cache_shard_"):-4]
                    self.shards[shard_id] = CacheShard(
                        shard_id, self.cache_dir, self.expire_days)
                    count += 1

            # 构建索引
            for shard_id, shard in self.shards.items():
                for key in shard.cache_data.keys():
                    self.shard_index[key] = shard_id

            self.logger.info(
                f"加载了 {count} 个分片，包含 {len(self.shard_index)} 条缓存记录")
        except Exception as e:
            self.logger.error(f"加载分片失败: {str(e)}")

    def get_cache(self, key: str) -> Optional[Dict]:
        """获取缓存数据"""
        with self.lock:
            # 如果键不在索引中，直接返回None
            if key not in self.shard_index:
                return None

            # 获取分片
            shard_id = self.shard_index[key]
            shard = self.shards.get(shard_id)
            if not shard:
                return None

            # 从分片获取数据
            return shard.get(key)

    def set_cache(self, key: str, value: Dict) -> None:
        """设置缓存数据"""
        with self.lock:
            # 获取分片ID
            shard_id = self._get_shard_id(key)
            shard = self.shards.get(shard_id)
            if not shard:
                shard = CacheShard(shard_id, self.cache_dir, self.expire_days)
                self.shards[shard_id] = shard

            # 设置数据
            shard.set(key, value)

            # 记录修改
            self.modified_keys.add(key)
            self.modified_count += 1

            # 如果达到保存阈值，执行增量保存
            if self.modified_count >= self.save_threshold:
                self._save_incremental()

    def save(self) -> None:
        """
        保存所有缓存数据
        仅当有缓存更新时才实际执行保存，避免不必要的I/O操作
        """
        with self.lock:
            # 如果没有修改，直接返回
            if not self.modified_keys:
                self.logger.debug("缓存数据未修改，跳过保存操作")
                return

            start_time = time.time()
            saved_count = 0

            # 保存所有有修改的分片
            modified_shards = {
                self.shard_index[key] for key in self.modified_keys if key in self.shard_index}
            for shard_id in modified_shards:
                shard = self.shards.get(shard_id)
                if shard and shard.modified:
                    if shard.save():
                        saved_count += 1

            # 重置修改计数
            self.modified_keys.clear()
            self.modified_count = 0

            # 记录保存统计
            duration = time.time() - start_time
            total_cache = sum(shard.get_size()
                              for shard in self.shards.values())
            self.logger.info(
                f"缓存数据已保存，共 {total_cache} 条记录，保存了 {saved_count} 个分片，耗时 {duration:.2f}秒")

    def _save_incremental(self) -> None:
        """
        增量保存修改的缓存数据
        仅当有实际修改时才执行保存操作
        """
        with self.lock:
            # 如果没有修改，直接返回
            if not self.modified_keys:
                return

            # 获取需要保存的分片
            modified_shards = {
                self.shard_index[key] for key in self.modified_keys if key in self.shard_index}
            saved_count = 0

            # 保存修改的分片
            for shard_id in modified_shards:
                shard = self.shards.get(shard_id)
                if shard and shard.modified:
                    if shard.save():
                        saved_count += 1

            # 重置修改计数
            self.modified_keys.clear()
            self.modified_count = 0

            # 只在DEBUG级别记录增量保存
            if saved_count > 0:
                self.logger.debug(f"增量保存了 {saved_count} 个分片")

    def clear_expired_cache(self) -> int:
        """清理所有过期缓存"""
        with self.lock:
            total_cleared = 0
            for shard_id, shard in list(self.shards.items()):
                cleared = shard.clear_expired()
                total_cleared += cleared

                # 如果分片已经清空，可以删除它
                if shard.get_size() == 0:
                    shard.save()  # 保存空分片
                    # 删除分片文件
                    try:
                        if os.path.exists(shard.cache_file):
                            os.remove(shard.cache_file)
                            self.logger.debug(f"删除空分片: {shard_id}")
                    except Exception as e:
                        self.logger.error(f"删除空分片 {shard_id} 失败: {str(e)}")

                    # 从内存中移除分片
                    del self.shards[shard_id]

                    # 更新索引
                    keys_to_remove = [
                        k for k, s in self.shard_index.items() if s == shard_id]
                    for k in keys_to_remove:
                        del self.shard_index[k]

                # 如果分片有修改，保存它
                elif shard.modified:
                    shard.save()

            if total_cleared > 0:
                self.logger.info(f"清理了 {total_cleared} 条过期缓存")

            return total_cleared

    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        with self.lock:
            total_cache = 0
            shard_stats = {}

            for shard_id, shard in self.shards.items():
                size = shard.get_size()
                total_cache += size
                shard_stats[shard_id] = {
                    "size": size,
                    "modified": shard.modified
                }

            return {
                "total_shards": len(self.shards),
                "total_cache": total_cache,
                "shards": shard_stats,
                "modified_count": self.modified_count
            }

    def optimize(self) -> Dict[str, Any]:
        """优化缓存存储"""
        with self.lock:
            start_time = time.time()

            # 记录初始状态
            initial_shards = len(self.shards)
            initial_cache = sum(shard.get_size()
                                for shard in self.shards.values())

            # 第一步：清理过期缓存
            cleared = self.clear_expired_cache()

            # 第二步：合并小分片
            small_shards = [shard_id for shard_id, shard in self.shards.items()
                            if shard.get_size() < self.shard_size // 2]

            # 如果小分片数量超过2个，进行合并
            merged_count = 0
            if len(small_shards) >= 2:
                # 按大小排序
                small_shards.sort(key=lambda sid: self.shards[sid].get_size())

                while len(small_shards) >= 2:
                    # 取出最小的两个分片
                    sid1 = small_shards.pop(0)
                    sid2 = small_shards.pop(0)

                    # 获取分片
                    shard1 = self.shards.get(sid1)
                    shard2 = self.shards.get(sid2)

                    if not shard1 or not shard2:
                        continue

                    # 如果合并后大小不超过分片大小，进行合并
                    if shard1.get_size() + shard2.get_size() <= self.shard_size:
                        # 创建新分片
                        new_sid = f"merged_{int(time.time())}_{merged_count}"
                        new_shard = CacheShard(
                            new_sid, self.cache_dir, self.expire_days)

                        # 复制数据
                        for k, v in shard1.cache_data.items():
                            new_shard.set(k, v)
                            self.shard_index[k] = new_sid

                        for k, v in shard2.cache_data.items():
                            new_shard.set(k, v)
                            self.shard_index[k] = new_sid

                        # 保存新分片
                        new_shard.save()

                        # 添加到分片集合
                        self.shards[new_sid] = new_shard

                        # 删除旧分片文件
                        try:
                            if os.path.exists(shard1.cache_file):
                                os.remove(shard1.cache_file)
                            if os.path.exists(shard2.cache_file):
                                os.remove(shard2.cache_file)
                        except Exception as e:
                            self.logger.error(f"删除合并后的旧分片失败: {str(e)}")

                        # 从内存中移除旧分片
                        del self.shards[sid1]
                        del self.shards[sid2]

                        merged_count += 1
                    else:
                        # 如果不能合并，将sid2放回列表
                        small_shards.append(sid2)

            # 统计最终状态
            final_shards = len(self.shards)
            final_cache = sum(shard.get_size()
                              for shard in self.shards.values())
            duration = time.time() - start_time

            stats = {
                "initial_shards": initial_shards,
                "final_shards": final_shards,
                "initial_cache": initial_cache,
                "final_cache": final_cache,
                "cleared_cache": cleared,
                "merged_shards": merged_count,
                "duration": duration
            }

            self.logger.info(
                f"缓存优化完成: 初始分片数={initial_shards}, 最终分片数={final_shards}, "
                f"清理过期缓存={cleared}, 合并小分片={merged_count}, 耗时={duration:.2f}秒"
            )

            return stats

    def backup(self, backup_dir: Optional[str] = None) -> str:
        """备份缓存数据"""
        with self.lock:
            # 如果未指定备份目录，使用默认目录
            if not backup_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_dir = os.path.join(
                    self.cache_dir, f"backup_{timestamp}")

            # 确保备份目录存在
            os.makedirs(backup_dir, exist_ok=True)

            # 先保存所有修改
            self.save()

            # 复制所有分片文件
            copied_files = 0
            for shard_id, shard in self.shards.items():
                if os.path.exists(shard.cache_file):
                    dest_file = os.path.join(
                        backup_dir, os.path.basename(shard.cache_file))
                    shutil.copy2(shard.cache_file, dest_file)
                    copied_files += 1

            self.logger.info(f"已备份 {copied_files} 个分片文件到: {backup_dir}")
            return backup_dir

    def restore(self, backup_dir: str) -> bool:
        """从备份恢复缓存数据"""
        with self.lock:
            if not os.path.exists(backup_dir):
                self.logger.error(f"备份目录不存在: {backup_dir}")
                return False

            try:
                # 先清空当前缓存
                self.clear()

                # 复制备份文件到缓存目录
                copied_files = 0
                for file in os.listdir(backup_dir):
                    if file.startswith("room_cache_shard_") and file.endswith(".pkl"):
                        src_file = os.path.join(backup_dir, file)
                        dest_file = os.path.join(self.cache_dir, file)
                        shutil.copy2(src_file, dest_file)
                        copied_files += 1

                # 重新加载分片
                self.shards.clear()
                self.shard_index.clear()
                self._load_shards()

                self.logger.info(f"已从备份恢复 {copied_files} 个分片文件")
                return True
            except Exception as e:
                self.logger.error(f"从备份恢复失败: {str(e)}")
                return False

    def clear(self) -> None:
        """清空所有缓存"""
        with self.lock:
            # 删除所有分片文件
            for shard_id, shard in self.shards.items():
                try:
                    if os.path.exists(shard.cache_file):
                        os.remove(shard.cache_file)
                except Exception as e:
                    self.logger.error(f"删除分片文件 {shard_id} 失败: {str(e)}")

            # 清空内存中的分片和索引
            self.shards.clear()
            self.shard_index.clear()
            self.modified_keys.clear()
            self.modified_count = 0

            self.logger.info("缓存已清空")


# 测试函数
def test_cache_manager():
    """测试缓存管理器"""
    config = {
        "cache_dir": "./test_cache",
        "cache_expire_days": 7,
        "shard_size": 10,  # 小分片便于测试
        "save_threshold": 5
    }

    # 确保测试目录存在
    os.makedirs(config["cache_dir"], exist_ok=True)

    # 初始化缓存管理器
    cache_manager = RoomCacheManagerOptimized(config)

    # 添加测试数据
    for i in range(30):
        key = f"test_hotel_{i}"
        value = {
            "hotel_id": key,
            "data": f"测试数据 {i}",
            "timestamp": time.time()
        }
        cache_manager.set_cache(key, value)

    # 获取缓存
    for i in range(5):
        key = f"test_hotel_{i}"
        data = cache_manager.get_cache(key)
        print(f"获取缓存 {key}: {data is not None}")

    # 保存缓存
    cache_manager.save()

    # 获取统计信息
    stats = cache_manager.get_stats()
    print(f"缓存统计: {stats}")

    # 测试清理过期缓存
    # 设置一些过期数据
    for i in range(5):
        key = f"expired_hotel_{i}"
        value = {
            "hotel_id": key,
            "data": f"过期数据 {i}",
            "timestamp": time.time() - (8 * 24 * 60 * 60)  # 8天前的数据
        }
        cache_manager.set_cache(key, value)

        # 模拟过期
        shard_id = cache_manager._get_shard_id(key)
        shard = cache_manager.shards.get(shard_id)
        if shard:
            shard.last_access_time[key] = time.time() - (8 * 24 * 60 * 60)

    # 清理过期缓存
    cleared = cache_manager.clear_expired_cache()
    print(f"清理了 {cleared} 条过期缓存")

    # 优化缓存
    optimize_stats = cache_manager.optimize()
    print(f"优化结果: {optimize_stats}")

    # 备份和恢复
    backup_dir = cache_manager.backup()
    print(f"备份到: {backup_dir}")

    # 清空当前缓存
    cache_manager.clear()

    # 恢复备份
    restored = cache_manager.restore(backup_dir)
    print(f"恢复结果: {restored}")

    # 获取恢复后的统计信息
    stats = cache_manager.get_stats()
    print(f"恢复后的缓存统计: {stats}")

    # 清理测试目录
    shutil.rmtree(config["cache_dir"], ignore_errors=True)


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行测试
    test_cache_manager()
