import pandas as pd
import logging
import os
import requests
import json
from datetime import datetime
from collections import OrderedDict
import argparse
import sys
import traceback
from tqdm import tqdm
import numpy as np
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import time
import glob
from typing import List, Dict, Any
import shutil

###########################
#根据酒店信息获取供应商信息
###########################

class SupplierProcessor:
    def __init__(self, csv_path=None, limit=1000000, mode='all', max_workers=20):
        # 获取项目根目录
        self.root_dir = self._get_project_root()
        
        # 设置基本属性
        self.mode = mode  # 运行模式：'update', 'verify', 'all'
        self.limit = limit  # 限制处理的数据条数
        self.max_workers = max_workers  # 最大并发数
        
        # 初始化日志记录器
        self.logger = None
        self.log_dir = os.path.join(self.root_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 初始化基本配置
        self.csv_path = csv_path or os.path.join(self.root_dir, "room_match", "data", "source_artificial_data.csv")
        self.output_path = os.path.join(self.root_dir, "room_match", "data", "supply_artificial_info.csv")
        self.cache_dir = os.path.join(self.root_dir, "room_match", "data", "cache_supply")
        self.cache_file_pattern = os.path.join(self.cache_dir, 'api_cache_{}.json')
        
        # 缓存文件大小限制（1GB）
        self.CACHE_FILE_SIZE_LIMIT = 1024 * 1024 * 1024
        self.current_cache_file_index = 0
        
        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 设置日志
        self.setup_logger()
        
        self.base_url = "http://119.23.73.59:8080/api/room"
        self.batch_size = 10000  # 每10000条数据保存一次
        self.chunk_size = 50000  # 每次读取50000条数据
        self.max_retries = 3  # 最大重试次数
        self.retry_delay = 1  # 重试延迟（秒）
        self.session = None  # aiohttp session
        self.lock = asyncio.Lock()  # 用于并发控制
        self.request_timeout = 60  # API请求超时时间（秒）
        self.cache_data = {}  # 缓存数据
        self.cache_modified = False  # 缓存是否被修改
        
        # TCP连接池配置
        self.conn_limit = max_workers * 2  # 连接池大小
        self.conn_timeout = 30  # 连接超时时间
        self.read_timeout = 30  # 读取超时时间
        self.conn_retry_attempts = 3  # 连接重试次数
        
        # 请求速率限制
        self.request_interval = 0.1  # 请求间隔（秒）
        self.batch_interval = 1.0  # 批次间隔（秒）
        self.last_request_time = 0  # 上次请求时间
        
        # 性能统计
        self.performance_stats = {
            'start_time': None,
            'total_requests': 0,
            'success_requests': 0,
            'failed_requests': 0,
            'retry_count': 0,
            'total_time': 0,
            'last_minute_requests': 0,
            'last_minute_time': time.time()
        }

        # 创建必要的目录
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        # 在初始化时进行一次缓存文件整理
        self.organize_cache_files()
        self.logger.info("缓存文件初始化完成")

        self.logger.info(f"初始化处理器: mode={mode}, limit={limit}, input={csv_path}")
        self.logger.info(f"项目根目录: {self.root_dir}")
        self.logger.info(f"输出目录: {os.path.dirname(self.output_path)}")

    def _get_project_root(self):
        """获取项目根目录"""
        # 获取当前文件的绝对路径
        current_file = os.path.abspath(__file__)
        # 获取当前文件所在目录
        current_dir = os.path.dirname(current_file)
        # 向上回溯三级目录（room_match/process_data/interface -> room_match -> root）
        root_dir = os.path.dirname(os.path.dirname(current_dir))
        return root_dir

    def setup_logger(self):
        """设置日志配置"""
        if self.logger is not None:
            return
            
        self.logger = logging.getLogger('supplier_processor')
        self.logger.setLevel(logging.INFO)
        
        # 清除已存在的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()

        # 创建处理器和格式化器
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 文件处理器
        log_file = os.path.join(self.log_dir, f'hotel_api_{self.mode}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # 添加处理器
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # 设置日志不传播到父记录器
        self.logger.propagate = False

    def get_cache_files(self) -> List[str]:
        """获取所有缓存文件"""
        return sorted(glob.glob(self.cache_file_pattern.format('*')))

    def get_next_cache_file(self) -> str:
        """获取下一个缓存文件名"""
        cache_files = self.get_cache_files()
        if not cache_files:
            self.current_cache_file_index = 0
        else:
            # 从现有文件名中提取最大索引
            indices = [int(f.split('_')[-1].split('.')[0]) for f in cache_files]
            self.current_cache_file_index = max(indices) + 1
        return self.cache_file_pattern.format(self.current_cache_file_index)

    def check_and_split_cache_files(self):
        """检查并分割大缓存文件"""
        cache_files = self.get_cache_files()
        for cache_file in cache_files:
            try:
                file_size = os.path.getsize(cache_file)
                if file_size > self.CACHE_FILE_SIZE_LIMIT:
                    self.logger.info(f"缓存文件 {cache_file} 超过大小限制，开始分割...")
                    self.split_cache_file(cache_file)
            except Exception as e:
                self.logger.error(f"检查缓存文件大小时出错: {str(e)}")

    def split_cache_file(self, cache_file: str):
        """分割大缓存文件"""
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 计算每个文件应该包含的键数量
            total_keys = len(data)
            keys_per_file = max(1, int(total_keys * (self.CACHE_FILE_SIZE_LIMIT / os.path.getsize(cache_file))))
            
            # 分割数据
            keys = list(data.keys())
            for i in range(0, total_keys, keys_per_file):
                chunk_keys = keys[i:i + keys_per_file]
                chunk_data = {k: data[k] for k in chunk_keys}
                
                # 创建新的缓存文件
                new_cache_file = self.get_next_cache_file()
                with open(new_cache_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False)
                
                self.logger.info(f"已创建新的缓存文件: {new_cache_file}")
            
            # 删除原始文件
            os.remove(cache_file)
            self.logger.info(f"已删除原始缓存文件: {cache_file}")
            
        except Exception as e:
            self.logger.error(f"分割缓存文件时出错: {str(e)}")

    def load_cache(self):
        """加载所有API响应缓存并去重"""
        self.cache_data = {}
        cache_files = self.get_cache_files()
        
        if not cache_files:
            self.logger.info("未找到缓存文件")
            return
            
        self.logger.info(f"开始加载 {len(cache_files)} 个缓存文件...")
        total_entries = 0
        duplicate_entries = 0
        loaded_files = 0
        
        # 第一次遍历：验证文件完整性
        valid_cache_files = []
        for cache_file in cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                valid_cache_files.append(cache_file)
                loaded_files += 1
            except json.JSONDecodeError:
                self.logger.warning(f"警告：缓存文件 {cache_file} 已损坏，将重命名为 {cache_file}.corrupted")
                try:
                    os.rename(cache_file, cache_file + '.corrupted')
                except:
                    pass
                continue
            except Exception as e:
                self.logger.error(f"验证缓存文件 {cache_file} 时出错: {str(e)}")
                continue
        
        if not valid_cache_files:
            self.logger.info("没有有效的缓存文件")
            return
            
        # 记录每个键首次出现的文件
        key_first_appearance = {}
        self.logger.info("正在统计缓存数据...")
        
        # 第二次遍历：统计每个键首次出现的文件
        for cache_file in valid_cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    for key in cache_data.keys():
                        if key not in key_first_appearance:
                            key_first_appearance[key] = cache_file
            except Exception as e:
                self.logger.error(f"统计缓存文件 {cache_file} 时出错: {str(e)}")
                continue
        
        self.logger.info("正在加载缓存数据...")
        # 第三次遍历：只加载每个键在首次出现的文件中的数据
        for cache_file in valid_cache_files:
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    for key, value in cache_data.items():
                        total_entries += 1
                        if key_first_appearance[key] == cache_file:
                            self.cache_data[key] = value
                        else:
                            duplicate_entries += 1
            except Exception as e:
                self.logger.error(f"加载缓存文件 {cache_file} 出错: {str(e)}")
                continue
        
        self.logger.info(f"缓存加载完成: {len(self.cache_data)} 条唯一数据，{duplicate_entries} 条重复数据")

    def save_cache(self):
        """保存API缓存数据，实现增量更新到最新的缓存文件"""
        if not self.cache_modified:
            return
            
        try:
            # 获取所有缓存文件
            cache_files = self.get_cache_files()
            
            # 如果没有缓存文件，创建新文件
            if not cache_files:
                current_cache_file = self.cache_file_pattern.format(0)
                self.save_to_cache_file(current_cache_file, self.cache_data)
                self.cache_modified = False
                return
            
            # 获取最新的缓存文件
            latest_file = cache_files[-1]
            
            try:
                # 读取最新文件的数据
                with open(latest_file, 'r', encoding='utf-8') as f:
                    current_data = json.load(f)
                
                # 估算合并后的大小
                merged_data = current_data.copy()
                new_data = {}
                
                # 找出需要更新的数据
                for hotel_id, hotel_data in self.cache_data.items():
                    if hotel_id not in current_data:
                        new_data[hotel_id] = hotel_data
                    elif ('update_time' in hotel_data and 
                          'update_time' in current_data[hotel_id] and 
                          datetime.fromisoformat(hotel_data['update_time']) > 
                          datetime.fromisoformat(current_data[hotel_id]['update_time'])):
                        new_data[hotel_id] = hotel_data
                
                if not new_data:
                    self.logger.info("没有新的数据需要保存")
                    self.cache_modified = False
                    return
                
                merged_data.update(new_data)
                estimated_size = len(json.dumps(merged_data, ensure_ascii=False).encode('utf-8'))
                
                if estimated_size > self.CACHE_FILE_SIZE_LIMIT:
                    # 如果合并后超过大小限制，创建新的缓存文件
                    self.current_cache_file_index += 1
                    new_cache_file = self.cache_file_pattern.format(self.current_cache_file_index)
                    success = self.save_to_cache_file(new_cache_file, new_data)
                    if success:
                        self.logger.info(f"创建新的缓存文件: {new_cache_file}, 包含 {len(new_data)} 条新数据")
                else:
                    # 更新现有文件
                    success = self.save_to_cache_file(latest_file, merged_data)
                    if success:
                        self.logger.info(f"更新现有缓存文件: {latest_file}, 新增/更新 {len(new_data)} 条数据")
                
                if success:
                    self.cache_modified = False
                
            except Exception as e:
                self.logger.error(f"处理最新缓存文件时发生错误: {str(e)}")
                # 创建新的缓存文件
                self.current_cache_file_index += 1
                new_cache_file = self.cache_file_pattern.format(self.current_cache_file_index)
                success = self.save_to_cache_file(new_cache_file, self.cache_data)
                if success:
                    self.logger.info(f"创建新的缓存文件: {new_cache_file}, 包含 {len(self.cache_data)} 条数据")
                    self.cache_modified = False
            
        except Exception as e:
            self.logger.error(f"保存缓存文件出错: {str(e)}")
            self.logger.error(traceback.format_exc())

    def save_to_cache_file(self, cache_file: str, data: dict) -> bool:
        """安全地保存数据到缓存文件，返回是否保存成功"""
        temp_file = cache_file + '.tmp'
        backup_file = cache_file + '.bak'
        
        try:
            # 在写入之前确保所有数据都被正确转换
            converted_data = self.convert_numpy_types(data)
            
            # 验证转换后的数据
            for key, value in converted_data.items():
                if not isinstance(key, (str, int, float)):
                    converted_data = {str(k): v for k, v in converted_data.items()}
                    break
            
            # 先写入临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(converted_data, f, ensure_ascii=False, indent=None)
            
            # 验证临时文件
            with open(temp_file, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)
                # 验证数据完整性
                if len(loaded_data) != len(data):
                    raise ValueError("数据完整性验证失败：数据条数不匹配")
            
            # 如果原文件存在，先创建备份
            if os.path.exists(cache_file):
                os.rename(cache_file, backup_file)
            
            # 重命名临时文件为正式文件
            os.rename(temp_file, cache_file)
            
            # 再次验证新文件
            with open(cache_file, 'r', encoding='utf-8') as f:
                final_data = json.load(f)
                if len(final_data) != len(data):
                    raise ValueError("最终文件数据完整性验证失败")
            
            # 验证成功后删除备份文件
            if os.path.exists(backup_file):
                os.remove(backup_file)
                
            self.logger.info(f"成功保存缓存文件: {cache_file}, 包含 {len(data)} 条数据")
            return True
            
        except Exception as e:
            self.logger.error(f"保存缓存文件 {cache_file} 时出错: {str(e)}")
            # 如果出错，尝试恢复备份
            if os.path.exists(backup_file):
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                os.rename(backup_file, cache_file)
                self.logger.info(f"已恢复备份文件: {cache_file}")
            
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return False

    def convert_numpy_types(self, obj):
        """递归转换NumPy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {str(self.convert_numpy_types(key)): self.convert_numpy_types(value) 
                   for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self.convert_numpy_types(item) for item in obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                            np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.string_, np.unicode_)):
            return str(obj)
        elif isinstance(obj, (int, float, bool, str)):
            return obj
        elif obj is None:
            return obj
        return str(obj)  # 对于其他类型，转换为字符串

    def merge_cache_data(self, existing_data, new_data):
        """合并现有缓存数据和新数据"""
        try:
            # 如果现有数据为空，直接返回新数据
            if not existing_data:
                return new_data
                
            # 合并数据，保留最新的更新时间
            merged_data = existing_data.copy()
            for hotel_id, new_info in new_data.items():
                if hotel_id not in merged_data:
                    merged_data[hotel_id] = new_info
                else:
                    # 比较更新时间，保留最新的数据
                    existing_time = datetime.fromisoformat(merged_data[hotel_id]['update_time'])
                    new_time = datetime.fromisoformat(new_info['update_time'])
                    if new_time > existing_time:
                        merged_data[hotel_id] = new_info
            
            return merged_data
        except Exception as e:
            self.logger.error(f"合并缓存数据时发生错误: {str(e)}")
            return new_data

    def extract_room_info(self, hotel_id, api_response):
        """提取房间信息的关键字段，并合并现有缓存"""
        if not api_response or not api_response.get('success'):
            return None
        
        room_list = api_response.get('data', {}).get('SPLRoomTypeList', [])
        extracted_rooms = []
        
        # 获取现有缓存数据
        cache_key = str(hotel_id)  # 确保键是字符串
        existing_rooms = []
        if cache_key in self.cache_data:
            existing_rooms = self.cache_data[cache_key].get('rooms', [])
        
        # 创建现有房间的索引，用于快速查找
        existing_room_index = {
            (str(room.get('RoomTypeID')), str(room.get('SupplierID'))): room 
            for room in existing_rooms
        }
        
        # 处理新的API响应数据
        for room in room_list:
            # 确保所有值都被转换为原生Python类型
            room_type_id = str(self.convert_numpy_types(room.get('RoomTypeID')))
            supplier_id = str(self.convert_numpy_types(room.get('SupplierID')))
            room_key = (room_type_id, supplier_id)
            
            extracted_room = {
                'RoomTypeID': room_type_id,
                'RoomTypeName': str(room.get('RoomTypeName', '')),
                'BedTypeDesc': str(room.get('BedTypeDesc', '')),
                'SupplierID': supplier_id,
                'SupplierName': str(room.get('SupplierName', ''))
            }
            
            # 如果这个房间在现有缓存中不存在，或者数据有更新，则添加/更新
            if room_key not in existing_room_index or existing_room_index[room_key] != extracted_room:
                extracted_rooms.append(extracted_room)
                # 从索引中移除这个房间，这样最后剩下的就是API中没有返回的房间
                existing_room_index.pop(room_key, None)
            else:
                # 如果数据完全相同，保留现有数据
                extracted_rooms.append(existing_room_index[room_key])
                existing_room_index.pop(room_key, None)
        
        # 添加API没有返回但在缓存中存在的房间
        for remaining_room in existing_room_index.values():
            extracted_rooms.append(remaining_room)
        
        # 确保所有数据都是可序列化的
        result = {
            'hotel_id': str(hotel_id),
            'update_time': datetime.now().isoformat(),
            'rooms': extracted_rooms
        }
        
        # 最后一次确保所有数据都被正确转换
        return self.convert_numpy_types(result)

    async def create_session(self):
        """创建带有连接池的会话"""
        try:
            if self.session is None or self.session.closed:
                conn = aiohttp.TCPConnector(
                    limit=self.conn_limit,
                    ttl_dns_cache=300,
                    enable_cleanup_closed=True,
                    force_close=True
                )
                timeout = aiohttp.ClientTimeout(
                    total=None,  # 整体超时时间，None表示不限制
                    connect=self.conn_timeout,
                    sock_connect=self.conn_timeout,
                    sock_read=self.read_timeout
                )
                self.session = aiohttp.ClientSession(
                    connector=conn,
                    timeout=timeout,
                    raise_for_status=False
                )
                self.logger.info("创建新的会话连接")
        except Exception as e:
            self.logger.error(f"创建会话时发生错误: {str(e)}")
            if self.session and not self.session.closed:
                await self.session.close()
            self.session = None
            raise

    async def get_hotel_info_async(self, s_hotel_id):
        """异步获取酒店信息"""
        # 检查缓存
        cache_key = str(s_hotel_id)
        try:
            if cache_key in self.cache_data:
                cache_info = self.cache_data[cache_key]
                
                # 验证缓存数据的完整性
                if not isinstance(cache_info, dict) or 'update_time' not in cache_info or 'rooms' not in cache_info:
                    self.logger.warning(f"缓存数据格式不正确: hotel_id={s_hotel_id}")
                    del self.cache_data[cache_key]
                    self.cache_modified = True
                else:
                    try:
                        cache_time = datetime.fromisoformat(cache_info['update_time'])
                        if datetime.now().timestamp() - cache_time.timestamp() < 24 * 3600:  # 24小时内的缓存有效
                            # 验证rooms数据的完整性
                            rooms = cache_info['rooms']
                            if isinstance(rooms, list) and all(
                                isinstance(room, dict) and 
                                all(key in room for key in ['RoomTypeID', 'RoomTypeName', 'SupplierID', 'SupplierName'])
                                for room in rooms
                            ):
                                self.performance_stats['success_requests'] += 1
                                return {'success': True, 'data': {'SPLRoomTypeList': rooms}}
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"缓存时间格式不正确: hotel_id={s_hotel_id}, error={str(e)}")
                        del self.cache_data[cache_key]
                        self.cache_modified = True
        except Exception as e:
            self.logger.error(f"检查缓存时发生错误: hotel_id={s_hotel_id}, error={str(e)}")

        # 请求速率限制
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_interval:
            await asyncio.sleep(self.request_interval - time_since_last)
        self.last_request_time = time.time()

        retry_count = 0
        last_error = None
        
        while retry_count < self.max_retries:
            try:
                # 确保session可用
                await self.create_session()
                if self.session is None:
                    raise Exception("无法创建会话连接")

                headers = {
                    'Content-Type': 'application/json',
                    'Accept': '*/*',
                    'User-Agent': 'Python/3.8'
                }
                data = {
                    "password": "your-secure-password",
                    "id": 123,
                    "hotel_id": int(s_hotel_id)
                }
                
                self.performance_stats['total_requests'] += 1
                async with self.session.post(
                    self.base_url, 
                    json=data, 
                    headers=headers,
                    timeout=self.request_timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        # 提取关键信息并保存到缓存，同时保留原有缓存数据
                        if result and result.get('success'):
                            extracted_info = self.extract_room_info(s_hotel_id, result)
                            if extracted_info:
                                async with self.lock:
                                    self.cache_data[cache_key] = extracted_info
                                    self.cache_modified = True
                            self.performance_stats['success_requests'] += 1
                        return result
                    else:
                        error_msg = await response.text()
                        last_error = f"API请求失败: status={response.status}, error={error_msg}"
                        
            except asyncio.TimeoutError:
                last_error = "请求超时"
                if self.session and not self.session.closed:
                    await self.session.close()
                self.session = None
                
            except aiohttp.ClientError as e:
                last_error = f"客户端错误: {str(e)}"
                if self.session and not self.session.closed:
                    await self.session.close()
                self.session = None
                
            except Exception as e:
                last_error = f"未知错误: {str(e)}"
                if self.session and not self.session.closed:
                    await self.session.close()
                self.session = None

            retry_count += 1
            self.performance_stats['retry_count'] += 1
            if retry_count < self.max_retries:
                retry_delay = self.retry_delay * (2 ** retry_count)
                self.logger.warning(f"重试请求 hotel_id={s_hotel_id}, 第{retry_count}次重试, 等待{retry_delay}秒")
                await asyncio.sleep(retry_delay)
                continue
            
        self.performance_stats['failed_requests'] += 1
        self.logger.error(f"获取酒店信息失败: s_hotel_id={s_hotel_id}, 重试{retry_count}次后失败, 最后错误: {last_error}")
        return None

    def process_cached_data(self):
        """处理缓存的API数据"""
        try:
            if os.path.exists(self.output_path):
                os.remove(self.output_path)
                self.logger.info(f"已清除旧的输出文件: {self.output_path}")
            
            # 读取原始CSV
            df = pd.read_csv(self.csv_path)
            if self.limit:
                df = df.head(self.limit)
            
            total_rows = len(df)
            success_count = 0
            failure_count = 0
            current_batch = []
            
            self.logger.info(f"开始处理缓存数据，总计{total_rows}条")
            
            with tqdm(total=total_rows, desc="数据处理进度") as pbar:
                for _, row in df.iterrows():
                    row_data = OrderedDict(row)
                    
                    if pd.isna(row.get("s_hotel_id")):
                        row_data['supplier_name'] = ''
                        failure_count += 1
                    else:
                        cache_key = str(row['s_hotel_id'])
                        if cache_key in self.cache_data:
                            try:
                                cached_hotel = self.cache_data[cache_key]
                                room_list = cached_hotel['rooms']
                                
                                # 查找匹配的房间
                                matched_room = next(filter(
                                    lambda x: str(x.get('RoomTypeID')) == str(row["spl_room_type_id"]) and
                                            str(x.get('SupplierID')) == str(row["supplier_id"]),
                                    room_list), None)
                                
                                if matched_room:
                                    row_data['supplier_name'] = matched_room.get('SupplierName', '')
                                    success_count += 1
                                else:
                                    row_data['supplier_name'] = ''
                                    failure_count += 1
                                    self.logger.debug(
                                        f"未找到匹配的房型: hotel_id={row['s_hotel_id']}, "
                                        f"room_type_id={row['spl_room_type_id']}, "
                                        f"supplier_id={row['supplier_id']}"
                                    )
                            except Exception as e:
                                self.logger.error(f"处理缓存数据时发生错误: s_hotel_id={row['s_hotel_id']}, error={str(e)}")
                                row_data['supplier_name'] = ''
                                failure_count += 1
                        else:
                            row_data['supplier_name'] = ''
                            failure_count += 1
                            self.logger.debug(f"缓存中未找到酒店数据: hotel_id={row['s_hotel_id']}")
                    
                    current_batch.append(row_data)
                    if len(current_batch) >= self.batch_size:
                        df_batch = pd.DataFrame(current_batch)
                        mode = 'w' if not os.path.exists(self.output_path) else 'a'
                        header = not os.path.exists(self.output_path)
                        df_batch.to_csv(self.output_path, mode=mode, header=header, index=False, encoding='utf-8')
                        self.logger.info(f"成功保存{len(current_batch)}条数据到文件")
                        current_batch = []
                    
                    pbar.update(1)
            
            # 保存最后的批次
            if current_batch:
                df_batch = pd.DataFrame(current_batch)
                mode = 'w' if not os.path.exists(self.output_path) else 'a'
                header = not os.path.exists(self.output_path)
                df_batch.to_csv(self.output_path, mode=mode, header=header, index=False, encoding='utf-8')
                self.logger.info(f"成功保存{len(current_batch)}条数据到文件")
            
            # 打印处理统计信息
            self.logger.info(f"数据处理完成：总数据 {total_rows} 条")
            self.logger.info(f"成功：{success_count} 条")
            self.logger.info(f"失败：{failure_count} 条")
            self.logger.info(f"成功率：{(success_count/total_rows)*100:.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理数据时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def fetch_all_api_data(self):
        """获取所有需要的API数据"""
        try:
            await self.create_session()
            
            # 读取CSV并获取唯一的酒店ID
            self.logger.info("开始读取CSV文件并提取酒店ID...")
            df = pd.read_csv(self.csv_path)
            if self.limit:
                df = df.head(self.limit)
            
            # 获取非空的唯一酒店ID
            unique_hotel_ids = df['s_hotel_id'].dropna().unique()
            total_hotels = len(unique_hotel_ids)
            self.logger.info(f"共找到 {total_hotels} 个唯一酒店ID")
            
            # 过滤出缓存中不存在的酒店ID
            uncached_hotel_ids = []
            for hotel_id in unique_hotel_ids:
                cache_key = str(hotel_id)
                if cache_key not in self.cache_data:
                    uncached_hotel_ids.append(hotel_id)
            
            total_uncached = len(uncached_hotel_ids)
            self.logger.info(f"其中 {total_uncached} 个酒店ID在缓存中不存在，需要从API获取")
            
            if total_uncached == 0:
                self.logger.info("所有酒店数据都在缓存中，无需调用API")
                return True
            
            # 初始化性能统计
            self.performance_stats['start_time'] = time.time()
            self.performance_stats['total_requests'] = 0
            self.performance_stats['success_requests'] = 0
            self.performance_stats['failed_requests'] = 0
            self.performance_stats['retry_count'] = 0
            
            # 初始化计数器
            processed_count = 0
            last_save_time = time.time()
            last_progress_time = time.time()
            save_threshold = 100  # 每获取100条数据保存一次
            processed_since_last_save = 0
            
            self.logger.info(f"开始获取API数据，需要请求{len(uncached_hotel_ids)}个酒店")
            
            with tqdm(total=len(uncached_hotel_ids), desc="API请求进度") as pbar:
                tasks = []
                for hotel_id in uncached_hotel_ids:
                    task = asyncio.create_task(self.get_hotel_info_async(hotel_id))
                    tasks.append((hotel_id, task))
                    
                    if len(tasks) >= self.max_workers:
                        # 等待一批任务完成
                        for hotel_id, task in tasks:
                            try:
                                result = await task
                                processed_count += 1
                                processed_since_last_save += 1
                                
                                # 检查是否需要保存缓存
                                current_time = time.time()
                                if (processed_since_last_save >= save_threshold or 
                                    current_time - last_save_time > 300):  # 最长5分钟保存一次
                                    if self.cache_modified:
                                        self.save_cache()
                                        last_save_time = current_time
                                        processed_since_last_save = 0
                                        self.logger.info(
                                            f"已保存缓存，当前进度: {processed_count}/{len(uncached_hotel_ids)} "
                                            f"({(processed_count/len(uncached_hotel_ids)*100):.2f}%)"
                                        )
                                
                                # 检查是否需要显示详细进度
                                if current_time - last_progress_time > 5:  # 每5秒显示一次详细进度
                                    self.update_performance_stats()
                                    self.logger.info(
                                        f"当前进度: {processed_count}/{len(uncached_hotel_ids)} "
                                        f"({(processed_count/len(uncached_hotel_ids)*100):.2f}%), "
                                        f"成功率: {(self.performance_stats['success_requests']/self.performance_stats['total_requests']*100):.1f}%, "
                                        f"平均速度: {processed_count/self.performance_stats['total_time']:.1f} 请求/秒, "
                                        f"最近一分钟请求数: {self.performance_stats['last_minute_requests']}"
                                    )
                                    last_progress_time = current_time
                                
                            except Exception as e:
                                self.logger.error(f"处理API响应时发生错误: hotel_id={hotel_id}, error={str(e)}")
                            
                            pbar.update(1)
                        
                        tasks = []
                        # 批次间隔
                        await asyncio.sleep(self.batch_interval)
                
                # 处理剩余的任务
                if tasks:
                    for hotel_id, task in tasks:
                        try:
                            result = await task
                            processed_count += 1
                            processed_since_last_save += 1
                        except Exception as e:
                            self.logger.error(f"处理API响应时发生错误: hotel_id={hotel_id}, error={str(e)}")
                        
                        pbar.update(1)
            
            # 保存最终的缓存
            if self.cache_modified:
                self.save_cache()
            
            # 在所有数据处理完成后，进行一次缓存整理
            self.organize_cache_files()
            
            # 打印最终统计信息
            self.logger.info(self.get_performance_summary())
            
            return True
            
        except Exception as e:
            self.logger.error(f"获取API数据时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    async def process_data_async(self):
        """分两阶段处理数据：1. 获取API数据 2. 处理数据"""
        try:
            # 第一阶段：获取API数据
            self.logger.info("开始第一阶段：获取API数据")
            if not await self.fetch_all_api_data():
                self.logger.error("获取API数据失败")
                return False
            self.logger.info("API数据获取完成")
            
            # 第二阶段：处理数据
            self.logger.info("开始第二阶段：处理数据")
            if not self.process_cached_data():
                self.logger.error("数据处理失败")
                return False
            self.logger.info("数据处理完成")
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    async def verify_data_async(self, sample_ratio=0.1):
        """异步校验数据一致性"""
        try:
            self.logger.info("开始数据校验")
            # 创建新的session
            self.session = aiohttp.ClientSession()
            
            # 分块读取原始CSV
            total_rows = sum(1 for _ in open(self.csv_path, 'r', encoding='utf-8')) - 1
            if self.limit and self.limit < total_rows:
                total_rows = self.limit
            
            if not os.path.exists(self.output_path):
                self.logger.error(f"更新后的文件不存在: {self.output_path}")
                return False
            
            # 读取更新后的CSV
            updated_df = pd.read_csv(self.output_path)
            
            # 只验证有供应商名称的数据
            updated_df = updated_df[updated_df['supplier_name'].notna() & (updated_df['supplier_name'] != '')]
            
            # 基本检查
            if len(updated_df) == 0:
                self.logger.warning("没有找到有效的供应商名称数据进行验证")
                return True

            # 抽样验证
            sample_size = max(int(len(updated_df) * sample_ratio), 1)
            self.logger.info(f"将随机抽取 {sample_size} 条数据进行校验")
            sample_indices = np.random.choice(len(updated_df), sample_size, replace=False)
            sample_df = updated_df.iloc[sample_indices]

            # 初始化验证结果统计
            verification_results = {
                'total': sample_size,
                'success': 0,
                'failure': 0,
                'failures': []  # 存储失败详情
            }

            # 创建异步任务
            tasks = []
            with tqdm(total=sample_size, desc="数据校验进度") as pbar:
                for _, row in sample_df.iterrows():
                    task = asyncio.create_task(self.verify_row_async(row, verification_results))
                    tasks.append((task, pbar))
                    
                    if len(tasks) >= self.max_workers:
                        for task, progress_bar in tasks:
                            await task
                            progress_bar.update(1)
                        tasks = []
                
                # 处理剩余的任务
                for task, progress_bar in tasks:
                    await task
                    progress_bar.update(1)

            # 输出验证结果统计
            success_rate = (verification_results['success'] / verification_results['total']) * 100
            self.logger.info(f"\n验证结果统计:")
            self.logger.info(f"总验证数量: {verification_results['total']}")
            self.logger.info(f"验证成功: {verification_results['success']}")
            self.logger.info(f"验证失败: {verification_results['failure']}")
            self.logger.info(f"成功率: {success_rate:.2f}%")

            # 如果有失败案例，保存到文件
            if verification_results['failures']:
                failure_file = os.path.join(self.log_dir, f'verification_failures_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
                with open(failure_file, 'w', encoding='utf-8') as f:
                    json.dump(verification_results['failures'], f, ensure_ascii=False, indent=2)
                self.logger.info(f"失败详情已保存到: {failure_file}")

            return success_rate >= 80  # 返回是否达到80%的成功率要求

        except Exception as e:
            self.logger.error(f"数据校验时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False
        finally:
            if self.session:
                await self.session.close()
                self.session = None

    async def verify_row_async(self, row, verification_results):
        """异步验证单行数据"""
        try:
            response = await self.get_hotel_info_async(row['s_hotel_id'])
            if response and response.get('success'):
                room_type_list = response.get('data', {}).get('SPLRoomTypeList', [])
                resp_item = next(filter(
                    lambda x: str(x.get('RoomTypeID')) == str(row["spl_room_type_id"]) and
                            str(x.get('SupplierID')) == str(row["supplier_id"]),
                    room_type_list), None)

                if resp_item:
                    api_supplier_name = resp_item.get('SupplierName', '')
                    csv_supplier_name = row['supplier_name']
                    
                    async with self.lock:
                        if api_supplier_name != csv_supplier_name:
                            verification_results['failure'] += 1
                            failure_info = {
                                's_room_id': row['s_room_id'],
                                'supplier_id': row['supplier_id'],
                                'api_supplier_name': api_supplier_name,
                                'csv_supplier_name': csv_supplier_name,
                                'error': 'supplier_name不匹配'
                            }
                            verification_results['failures'].append(failure_info)
                            self.logger.error(
                                f"supplier_name不匹配: s_room_id={row['s_room_id']}, "
                                f"supplier_id={row['supplier_id']}, "
                                f"API={api_supplier_name}, CSV={csv_supplier_name}"
                            )
                        else:
                            verification_results['success'] += 1
                else:
                    async with self.lock:
                        verification_results['failure'] += 1
                        failure_info = {
                            's_room_id': row['s_room_id'],
                            'supplier_id': row['supplier_id'],
                            'error': 'API中未找到匹配的房型'
                        }
                        verification_results['failures'].append(failure_info)
            else:
                async with self.lock:
                    verification_results['failure'] += 1
                    failure_info = {
                        's_room_id': row['s_room_id'],
                        'supplier_id': row['supplier_id'],
                        'error': 'API请求失败'
                    }
                    verification_results['failures'].append(failure_info)

        except Exception as e:
            async with self.lock:
                verification_results['failure'] += 1
                failure_info = {
                    's_room_id': row['s_room_id'],
                    'supplier_id': row['supplier_id'],
                    'error': f'验证过程发生错误: {str(e)}'
                }
                verification_results['failures'].append(failure_info)
                self.logger.error(f"验证过程发生错误: {str(e)}")

    async def run_async(self):
        """异步运行处理器"""
        try:
            if self.mode in ['update', 'all']:
                self.logger.info("开始更新数据...")
                if not await self.process_data_async():
                    self.logger.error("数据更新失败")
                    return False
                self.logger.info("数据更新完成")

            # if self.mode in ['verify', 'all']:
            #     self.logger.info("开始校验数据...")
            #     if not await self.verify_data_async(sample_ratio=0.1):
            #         self.logger.error("数据校验失败")
            #         return False
            #     self.logger.info("数据校验完成")

            return True

        except Exception as e:
            self.logger.error(f"处理过程中发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def update_performance_stats(self):
        """更新性能统计信息"""
        current_time = time.time()
        
        # 更新每分钟请求数
        if current_time - self.performance_stats['last_minute_time'] >= 60:
            self.performance_stats['last_minute_requests'] = (
                self.performance_stats['total_requests'] - 
                self.performance_stats['last_minute_requests']
            )
            self.performance_stats['last_minute_time'] = current_time
        
        # 更新总时间
        if self.performance_stats['start_time']:
            self.performance_stats['total_time'] = current_time - self.performance_stats['start_time']

    def get_performance_summary(self):
        """获取性能统计摘要"""
        self.update_performance_stats()
        
        total_time = self.performance_stats['total_time']
        total_requests = self.performance_stats['total_requests']
        success_requests = self.performance_stats['success_requests']
        failed_requests = self.performance_stats['failed_requests']
        retry_count = self.performance_stats['retry_count']
        
        return (
            f"\n性能统计:\n"
            f"总运行时间: {total_time:.1f}秒\n"
            f"总请求数: {total_requests}\n"
            f"成功请求: {success_requests}\n"
            f"失败请求: {failed_requests}\n"
            f"重试次数: {retry_count}\n"
            f"平均速度: {total_requests/total_time:.1f} 请求/秒\n"
            f"成功率: {(success_requests/total_requests*100):.1f}%\n"
            f"最近一分钟请求数: {self.performance_stats['last_minute_requests']}"
        )

    def organize_cache_files(self):
        """程序启动时加载缓存文件，不进行排序和整理"""
        try:
            self.logger.info("开始加载缓存文件...")
            cache_files = self.get_cache_files()
            
            if not cache_files:
                self.logger.info("没有找到缓存文件")
                return
            
            self.logger.info(f"找到 {len(cache_files)} 个缓存文件")
            
            # 加载所有缓存文件的数据
            loaded_files = 0
            total_entries = 0
            duplicate_entries = 0
            
            # 加载并处理所有文件
            for cache_file in cache_files:
                try:
                    # 检查文件大小
                    file_size = os.path.getsize(cache_file)
                    self.logger.info(f"正在加载缓存文件: {cache_file}, 大小: {file_size/1024/1024/1024:.2f}GB")
                    
                    # 读取并验证文件内容
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        file_data = json.load(f)
                        if isinstance(file_data, dict):
                            loaded_files += 1
                            file_entries = len(file_data)
                            total_entries += file_entries
                            
                            # 处理每条数据
                            for hotel_id, hotel_data in file_data.items():
                                # 跳过rooms为空的数据
                                if not hotel_data.get('rooms'):
                                    continue
                                    
                                # 检查是否已存在该酒店的数据
                                if hotel_id not in self.cache_data:
                                    self.cache_data[hotel_id] = hotel_data
                                else:
                                    duplicate_entries += 1
                                    # 比较更新时间，只保留最新的数据
                                    existing_time = datetime.fromisoformat(self.cache_data[hotel_id]['update_time'])
                                    new_time = datetime.fromisoformat(hotel_data['update_time'])
                                    if new_time > existing_time:
                                        self.cache_data[hotel_id] = hotel_data
                        else:
                            self.logger.warning(f"缓存文件格式不正确: {cache_file}")
                except json.JSONDecodeError:
                    self.logger.warning(f"缓存文件损坏: {cache_file}")
                    try:
                        os.rename(cache_file, cache_file + '.corrupted')
                    except:
                        pass
                    continue
                except Exception as e:
                    self.logger.error(f"读取缓存文件时发生错误: {cache_file}, 错误: {str(e)}")
                    continue
            
            # 设置当前缓存文件索引为最新文件的索引
            if cache_files:
                latest_file = cache_files[-1]
                try:
                    self.current_cache_file_index = int(latest_file.split('_')[-1].split('.')[0])
                except:
                    self.current_cache_file_index = len(cache_files) - 1
            
            # 输出处理结果
            self.logger.info(f"完成缓存加载: 加载了 {loaded_files} 个文件")
            self.logger.info(f"总条目数: {total_entries}, 重复条目: {duplicate_entries}")
            self.logger.info(f"当前缓存中的唯一酒店数: {len(self.cache_data)}")
            
        except Exception as e:
            self.logger.error(f"加载缓存文件时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())

    def restore_from_backup(self, backup_dir, original_files):
        """从备份目录恢复文件"""
        self.logger.info("开始从备份恢复文件...")
        try:
            for original_file in original_files:
                backup_file = os.path.join(backup_dir, os.path.basename(original_file))
                if os.path.exists(backup_file):
                    os.rename(backup_file, original_file)
                    self.logger.info(f"已恢复文件: {backup_file} -> {original_file}")
            self.logger.info("文件恢复完成")
        except Exception as e:
            self.logger.error(f"恢复文件时发生错误: {str(e)}")
            raise

def main():
    parser = argparse.ArgumentParser(description='酒店供应商数据处理工具')
    parser.add_argument('--mode', choices=['update', 'verify', 'all'], default='all',
                      help='处理模式：update=仅更新数据，verify=仅校验数据，all=更新并校验（默认）')
    parser.add_argument('--limit', type=int, default=600000)
    parser.add_argument('--input', type=str, default=None,
                      help='输入CSV文件路径（相对于项目根目录或绝对路径）')
    parser.add_argument('--sample-ratio', type=float, default=0.1,
                      help='校验时的抽样比例（默认：0.1）')
    parser.add_argument('--max-workers', type=int, default=20,
                      help='最大并发数（默认：50）')

    args = parser.parse_args()

    # 如果提供了输入文件路径，确保它是绝对路径
    if args.input:
        if not os.path.isabs(args.input):
            # 获取项目根目录
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            args.input = os.path.join(root_dir, args.input)

    processor = SupplierProcessor(
        csv_path=args.input,
        limit=args.limit,
        mode=args.mode,
        max_workers=args.max_workers
    )

    # 运行异步主程序
    success = asyncio.run(processor.run_async())
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
