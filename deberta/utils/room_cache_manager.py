#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
房型数据缓存管理器
用于缓存API请求结果，减少重复请求
增加性能优化，降低I/O开销
"""

import os
import json
import time
import logging
import threading
import pickle
from typing import Dict, Optional, Any
from pathlib import Path


class RoomCacheManager:
    """房型数据缓存管理器类 (性能优化版)"""

    def __init__(self, cache_dir: str, cache_expire_days: int = 7):
        """
        初始化缓存管理器

        参数:
            cache_dir: 缓存目录路径
            cache_expire_days: 缓存过期时间(天)
        """
        self.cache_dir = cache_dir
        self.cache_expire_days = cache_expire_days
        self.logger = logging.getLogger(f"{__name__}.RoomCacheManager")

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 缓存文件路径 - 使用二进制pickle格式提高读写效率
        self.json_cache_file = os.path.join(
            self.cache_dir, "room_data_cache.json")
        self.pickle_cache_file = os.path.join(
            self.cache_dir, "room_data_cache.pickle")

        # 添加线程锁，保证多线程访问安全
        self.lock = threading.RLock()

        # 添加脏标记，只有数据变更时才写入磁盘
        self.dirty = False

        # 添加内存缓存标志，避免频繁加载
        self.is_loaded = False

        # 加载缓存
        self.cache_data = self._load_cache()

        # 定期自动保存的计数器
        self.save_counter = 0
        # 设置触发自动保存的阈值
        self.save_threshold = 50

    def _load_cache(self) -> Dict:
        """从文件加载缓存数据，优先尝试pickle格式"""
        try:
            # 先尝试读取二进制格式(速度更快)
            if os.path.exists(self.pickle_cache_file):
                with open(self.pickle_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.logger.info(
                        f"已加载缓存数据(pickle格式)，包含 {len(cache_data)} 条记录")
                    self.is_loaded = True
                    return cache_data

            # 再尝试读取JSON格式(兼容旧数据)
            elif os.path.exists(self.json_cache_file):
                with open(self.json_cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    self.logger.info(
                        f"已加载缓存数据(JSON格式)，包含 {len(cache_data)} 条记录")
                    # 标记为脏数据，下次保存时转换为pickle格式
                    self.dirty = True
                    self.is_loaded = True
                    return cache_data
            else:
                self.logger.info("缓存文件不存在，创建新缓存")
                self.is_loaded = True
                return {}
        except Exception as e:
            self.logger.error(f"加载缓存数据失败: {str(e)}")
            self.is_loaded = True
            return {}

    def _save_cache(self) -> bool:
        """保存缓存数据到文件，使用pickle提高性能"""
        if not self.dirty:
            # 数据未变更，无需保存
            return True

        try:
            # 使用锁保护写入操作
            with self.lock:
                # 使用缓存数据的副本，避免在多线程环境中迭代时修改字典导致错误
                cache_data_copy = self.cache_data.copy()

                # 使用pickle格式保存(更高效)
                with open(self.pickle_cache_file, 'wb') as f:
                    pickle.dump(cache_data_copy, f,
                                protocol=pickle.HIGHEST_PROTOCOL)

                # 同时保存一份JSON格式备份(可读性好)
                with open(self.json_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(cache_data_copy, f, ensure_ascii=False, indent=2)

                self.logger.info(f"缓存数据已保存，共 {len(cache_data_copy)} 条记录")
                self.dirty = False  # 重置脏标记
                return True
        except Exception as e:
            self.logger.error(f"保存缓存数据失败: {str(e)}")
            return False

    def get_cache(self, hotel_id: str) -> Optional[Dict]:
        """
        获取酒店房型数据缓存

        参数:
            hotel_id: 酒店ID

        返回:
            缓存的房型数据，如果不存在或过期则返回None
        """
        # 确保数据已加载
        if not self.is_loaded:
            self.cache_data = self._load_cache()

        try:
            # 使用锁保护读取操作，但尽量减小锁的范围
            with self.lock:
                # 检查缓存是否存在
                if hotel_id not in self.cache_data:
                    return None

                # 获取缓存数据
                cache_entry = self.cache_data[hotel_id]
                cache_time = cache_entry.get("timestamp", 0)

            # 检查缓存是否过期 - 锁外操作提高并发性
            current_time = time.time()
            if current_time - cache_time > self.cache_expire_days * 86400:
                self.logger.debug(f"酒店 {hotel_id} 的缓存数据已过期")
                return None

            return cache_entry.get("data")
        except Exception as e:
            self.logger.error(f"获取缓存数据时发生错误: {str(e)}")
            return None

    def set_cache(self, hotel_id: str, data: Dict) -> bool:
        """
        设置酒店房型数据缓存

        参数:
            hotel_id: 酒店ID
            data: 房型数据

        返回:
            设置是否成功
        """
        # 确保数据已加载
        if not self.is_loaded:
            self.cache_data = self._load_cache()

        try:
            # 创建缓存条目
            cache_entry = {
                "timestamp": time.time(),
                "data": data
            }

            # 使用锁保护写入操作，但尽量减小锁的范围
            with self.lock:
                # 更新缓存
                self.cache_data[hotel_id] = cache_entry
                self.dirty = True  # 标记数据已变更
                self.save_counter += 1

            # 仅当累计变更达到阈值时才保存（减少I/O）
            if self.save_counter >= self.save_threshold:
                with self.lock:
                    if self.save_counter >= self.save_threshold:
                        self._save_cache()
                        self.save_counter = 0

            return True
        except Exception as e:
            self.logger.error(f"设置缓存数据时发生错误: {str(e)}")
            return False

    def clear_expired_cache(self) -> int:
        """
        清理过期缓存

        返回:
            清理的缓存条目数量
        """
        # 确保数据已加载
        if not self.is_loaded:
            self.cache_data = self._load_cache()

        try:
            # 使用锁保护读写操作
            with self.lock:
                current_time = time.time()
                expire_time = self.cache_expire_days * 86400

                # 首先收集过期键，避免在迭代过程中修改字典
                expired_keys = []
                # 使用字典的副本进行迭代，避免运行时错误
                for hotel_id, cache_entry in self.cache_data.copy().items():
                    cache_time = cache_entry.get("timestamp", 0)
                    if current_time - cache_time > expire_time:
                        expired_keys.append(hotel_id)

                # 删除过期缓存
                for hotel_id in expired_keys:
                    del self.cache_data[hotel_id]

                # 标记数据变更
                if expired_keys:
                    self.dirty = True

                # 保存缓存
                if self.dirty:
                    self._save_cache()

                self.logger.info(f"已清理 {len(expired_keys)} 条过期缓存")
                return len(expired_keys)
        except Exception as e:
            self.logger.error(f"清理过期缓存时发生错误: {str(e)}")
            return 0

    def save(self) -> bool:
        """
        强制保存缓存数据
        仅当缓存有修改时才实际写入磁盘，避免不必要的I/O操作

        返回:
            保存是否成功
        """
        with self.lock:
            self.save_counter = 0

            # 检查是否有修改，只有脏数据才需要保存
            if not self.dirty:
                self.logger.debug("缓存数据未修改，跳过保存操作")
                return True

            return self._save_cache()
