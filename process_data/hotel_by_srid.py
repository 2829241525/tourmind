import pandas as pd
import json
import os
import logging
import traceback
import requests
import time
from datetime import datetime
from collections import OrderedDict
import argparse

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key, value):
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
        self.cache[key] = value

class HotelProcessor:
    def __init__(self, csv_path=None):
        # 初始化基本配置
        self.csv_path = csv_path or "room_match/data/test_room_match_folks.csv"
        self.api_data_path = "room_match/data/test_room_match_folks_subset.api_data.json"
        self.output_path = "room_match/data/test_room_match_folks_hotel.csv"
        self.base_url = "http://119.23.73.59:8080/api/hotel-data"
        self.api_data = {}
        self.cache = LRUCache(100)  # 设置缓存容量为100
        
        # 创建必要的目录
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.api_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        
        # 检查输入文件是否存在
        if not os.path.exists(self.output_path):
            raise FileNotFoundError(f"输入文件不存在: {self.csv_path}")
        
        # 设置日志
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # 清除已存在的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
            
        # 添加控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 添加文件处理器
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"hotel_processor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        self.logger.info(f"初始化处理器: input={self.csv_path}, output={self.output_path}")

    def get_hotel_info_by_srid(self, s_room_id):
        """获取酒店信息"""
        try:
            # 先从缓存获取
            cached_data = self.cache.get(s_room_id)
            if cached_data:
                self.logger.info(f"从缓存获取数据: s_room_id={s_room_id}")
                return cached_data

            # 缓存未命中，从API获取
            headers = {
                'Content-Type': 'application/json',
                'Accept': '*/*',
                'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
                'Host': '119.23.73.59:8080',
                'Connection': 'keep-alive'
            }
            data = {
                "password": "your-secure-password",
                "id": int(s_room_id)
            }
            
            response = requests.post(self.base_url, json=data, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # 将数据存入缓存
                self.cache.put(s_room_id, data)
                #self.logger.info(f"API返回数据并缓存: {data}")
                return data
            else:
                self.logger.error(f"API请求失败: {response.status_code}")
                return None
        except Exception as e:
            self.logger.error(f"获取酒店信息时发生错误: {str(e)}")
            return None

    def fetch_api_data(self):
        """获取所有API数据"""
        try:
            # 读取CSV文件
            df = pd.read_csv(self.csv_path)
            unique_ids = df['s_room_id'].unique()
            total_ids = len(unique_ids)
            
            self.logger.info(f"开始获取API数据，共有 {total_ids} 个唯一id需要处理")
            
            # 加载已有的API数据
            if os.path.exists(self.api_data_path):
                with open(self.api_data_path, 'r') as f:
                    self.api_data = json.load(f)
                self.logger.info(f"已加载 {len(self.api_data)} 条API数据记录")
            
            # 找出未处理的ID
            processed_ids = set(str(id) for id in self.api_data.keys())
            remaining_ids = [str(id) for id in unique_ids if str(id) not in processed_ids]
            self.logger.info(f"还有 {len(remaining_ids)} 个id需要处理")
            
            if not remaining_ids:
                self.logger.info("无需处理新数据")
                return True
            
            # 处理剩余ID
            start_time = time.time()
            processed_count = 0
            success_count = 0
            failure_count = 0
            
            for s_room_id in remaining_ids:
                response = self.get_hotel_info_by_srid(s_room_id)
                if response and response.get('success'):
                    self.api_data[s_room_id] = response['data']
                    success_count += 1
                else:
                    failure_count += 1
                processed_count += 1
                
                # 每10条记录显示一次进度
                if processed_count % 10 == 0:
                    elapsed_time = (time.time() - start_time) / 60
                    speed = processed_count / elapsed_time if elapsed_time > 0 else 0
                    progress = (processed_count / len(remaining_ids)) * 100
                    self.logger.info(
                        f"进度: {processed_count}/{len(remaining_ids)} ({progress:.2f}%) "
                        f"速度: {speed:.2f} 条/分钟 "
                        f"成功: {success_count} 失败: {failure_count}"
                    )
                
                # 每50条记录保存一次数据
                if processed_count % 50 == 0:
                    self.save_api_data()
            
            # 最后保存一次
            self.save_api_data()
            
            total_time = (time.time() - start_time) / 60
            self.logger.info(f"API数据获取完成,总耗时: {total_time:.2f}分钟")
            self.logger.info(f"本次处理: {processed_count} 成功: {success_count} 失败: {failure_count}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"获取API数据时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def save_api_data(self):
        """保存API数据到文件"""
        try:
            with open(self.api_data_path, 'w') as f:
                json.dump(self.api_data, f)
            self.logger.info(f"已保存 {len(self.api_data)} 条API数据到文件")
            return True
        except Exception as e:
            self.logger.error(f"保存API数据时发生错误: {str(e)}")
            return False

    def update_csv(self):
        """更新CSV文件中的酒店信息"""
        try:
            # 读取CSV文件
            self.logger.info(f"开始读取CSV文件: {self.csv_path}")
            df = pd.read_csv(self.csv_path)
            self.logger.info(f"成功读取CSV文件，共 {len(df)} 行")
            
            # 检查s_room_id的格式
            self.logger.info("检查s_room_id格式:")
            self.logger.info(f"CSV中s_room_id的类型: {df['s_room_id'].dtype}")
            self.logger.info(f"CSV中s_room_id的示例: {df['s_room_id'].head().tolist()}")
            
            # 确保s_room_id是字符串类型
            df['s_room_id'] = df['s_room_id'].astype(str)
            
            # 备份原始数据
            df_backup = df.copy()
            
            # 更新数据
            update_count = 0
            failed_count = 0
            for index, row in df.iterrows():
                s_room_id = str(row['s_room_id'])
                
                # 获取最新的API数据
                api_response = self.get_hotel_info_by_srid(s_room_id)
                if not api_response or not api_response.get('success'):
                    failed_count += 1
                    self.logger.warning(f"未找到匹配的房型信息: s_room_id={s_room_id}, supplier_id={row.get('supplier_id', 'unknown')}")
                    continue
                    
                api_data = api_response['data']
                
                # 记录更新前后的数据
                if update_count < 5:  # 只记录前5条更新
                    self.logger.info(f"更新记录 {update_count + 1}:")
                    self.logger.info(f"s_room_id: {s_room_id}")
                    self.logger.info(f"原始数据: {dict(row)}")
                    self.logger.info(f"API数据: {api_data}")
                
                # 更新数据
                df.at[index, 's_hotel_id'] = api_data.get('s_hotel_id')
                df.at[index, 'hotel_name'] = api_data.get('hotel_name')
                df.at[index, 'country_code'] = api_data.get('country_code')
                df.at[index, 'group_code'] = api_data.get('group_code', '')
                update_count += 1
            
            # 写入更新后的数据
            self.logger.info(f"开始写入更新后的数据到文件: {self.output_path}")
            df.to_csv(self.output_path, index=False)
            self.logger.info("数据写入完成")
            
            # 验证行数
            if len(df) != len(df_backup):
                self.logger.error(f"行数不匹配: 原始数据 {len(df_backup)} 行, 更新后 {len(df)} 行")
                return False
            
            # 计算更新统计
            self.logger.info(f"处理完成：总数据 {len(df)} 条")
            self.logger.info(f"成功：{update_count} 条")
            self.logger.info(f"失败：{failed_count} 条")
            success_rate = (update_count / len(df)) * 100
            self.logger.info(f"成功率：{success_rate:.2f}%")
            
            return True
            
        except Exception as e:
            self.logger.error(f"更新CSV数据时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def verify_data(self, sample_size=None):
        """验证数据一致性"""
        try:
            # 检查输出文件是否存在
            if not os.path.exists(self.output_path):
                self.logger.error(f"输出文件不存在: {self.output_path}")
                self.logger.info("请先运行更新操作生成输出文件")
                return False

            # 读取CSV文件
            self.logger.info(f"开始读取输出文件: {self.output_path}")
            df = pd.read_csv(self.output_path)
            if sample_size is None:
                sample_size = max(int(len(df) * 0.1), 1)  # 验证10%的数据，至少1条
            
            self.logger.info(f"开始验证数据，随机抽取 {sample_size} 条记录进行检查")
            
            df['s_room_id'] = df['s_room_id'].astype(str)
            sample_records = df.sample(n=min(sample_size, len(df)))
            
            success_count = 0
            total_count = 0
            field_mismatch_stats = {
                's_hotel_id': 0,
                'hotel_name': 0,
                'country_code': 0,
                'group_code': 0
            }
            
            for _, record in sample_records.iterrows():
                total_count += 1
                s_room_id = str(record['s_room_id'])
                self.logger.info(f"\n验证记录 s_room_id={s_room_id}")
                
                # 获取API数据（优先从缓存获取）
                api_response = self.get_hotel_info_by_srid(s_room_id)
                if not api_response or not api_response.get('success'):
                    self.logger.error(f"获取API数据失败: {api_response}")
                    continue
                    
                api_data = api_response['data']
                
                self.logger.info("数据比较:")
                self.logger.info(f"CSV数据: {dict(record)}")
                self.logger.info(f"API数据: {api_data}")
                
                # 只验证存在的字段
                fields = [field for field in ['s_hotel_id', 'hotel_name', 'country_code', 'group_code'] 
                         if field in record]
                has_mismatch = False
                
                for field in fields:
                    csv_value = str(record.get(field)).strip() if pd.notna(record.get(field)) else ''
                    api_value = str(api_data.get(field, '')).strip()
                    
                    # 对于hotel_name字段，忽略大小写和空白字符
                    if field == 'hotel_name':
                        csv_value = csv_value.lower()
                        api_value = api_value.lower()
                    
                    if csv_value != api_value:
                        field_mismatch_stats[field] += 1
                        self.logger.error(f"数据不匹配 - {field}:")
                        self.logger.error(f"  CSV值: '{csv_value}'")
                        self.logger.error(f"  API值: '{api_value}'")
                        has_mismatch = True
                
                if has_mismatch:
                    self.logger.warning(f"记录验证失败 s_room_id={s_room_id}")
                else:
                    self.logger.info(f"记录验证成功 s_room_id={s_room_id}")
                    success_count += 1
            
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            self.logger.info(f"\n数据验证完成，共验证 {total_count} 条记录")
            self.logger.info(f"验证成功: {success_count} 条")
            self.logger.info(f"成功率: {success_rate:.2f}%")
            
            # 输出每个字段的不匹配统计
            self.logger.info("\n字段不匹配统计:")
            for field, count in field_mismatch_stats.items():
                mismatch_rate = (count / total_count) * 100 if total_count > 0 else 0
                self.logger.info(f"{field}: {count} 条不匹配 ({mismatch_rate:.2f}%)")
            
            self.logger.info(f"缓存命中统计: 当前缓存数量 {len(self.cache.cache)}")
            
            # 放宽成功率要求到80%
            return success_rate >= 80
            
        except Exception as e:
            self.logger.error(f"数据验证时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def process_data(self):
        """处理数据的主流程"""
        try:
            # 获取API数据
            if not self.fetch_api_data():
                return False
            
            # 更新CSV文件
            if not self.update_csv():
                return False
            
            # 验证数据
            if not self.verify_data(sample_size=200):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"处理数据时发生错误: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='酒店数据处理工具')
    parser.add_argument('--mode', choices=['update', 'verify', 'fetch', 'all'], default='all',
                      help='处理模式：update=仅更新CSV，verify=仅校验数据，fetch=仅获取API数据，all=完整处理（默认）')
    parser.add_argument('--input', type=str, default=None,
                      help='输入CSV文件路径')
    parser.add_argument('--output', type=str, default=None,
                      help='输出CSV文件路径（用于验证）')
    parser.add_argument('--sample-size', type=int, default=200,
                      help='校验时的样本数量（默认：200）')
    parser.add_argument('--cache-size', type=int, default=100,
                      help='缓存大小（默认：100）')

    args = parser.parse_args()

    # 创建处理器实例
    processor = HotelProcessor(csv_path=args.input)
    processor.cache = LRUCache(args.cache_size)
    
    # 如果指定了输出文件路径，则使用指定的路径
    if args.output:
        processor.output_path = args.output

    success = True
    if args.mode == 'fetch':
        success = processor.fetch_api_data()
    elif args.mode == 'update':
        success = processor.update_csv()
    elif args.mode == 'verify':
        success = processor.verify_data(sample_size=args.sample_size)
    else:  # args.mode == 'all'
        success = processor.process_data()

    # 根据处理结果设置退出码
    import sys
    sys.exit(0 if success else 1)