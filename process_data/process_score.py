import pandas as pd
import http.client
import json
from typing import Dict, List, Any, Tuple
import time
import random
import os
import argparse
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading
import socket
from urllib3.util.retry import Retry
from urllib3.exceptions import MaxRetryError
import glob
import requests
import sys
import pandas as pd
###########################
#构造负样本，通过分数判断，小于0.2以上的数据，作为负样本
###########################

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)


# 路径常量定义
DATA_DIR = 'room_match/data'
CHECKPOINT_FILE = os.path.join(DATA_DIR, 'checkpoint_all_room_match_data1.json')
RESULT_FILE = os.path.join(DATA_DIR, 'all_artificial_data.csv')
SOURCE_FILE = os.path.join(DATA_DIR, 'positive_artificial.csv')
CACHE_FILE_PATTERN = os.path.join(DATA_DIR, 'score_cache','api_response_cache_{}.json')
CACHE_FILE_SIZE_LIMIT = 1024 * 1024 * 1024

# 处理参数定义
BATCH_SIZE = 10000  # 每处理10000行保存一次
MAX_WORKERS = 200  # 最大线程数
MAX_RETRIES = 2  # 最大重试次数
TIMEOUT = 10  # 超时时间（秒）
RETRY_DELAY = 1  # 重试延迟（秒）
SAVE_INTERVAL = 1000  # 每处理50000条数据保存一次
LOG_INTERVAL = 10000  # 每处理10000条数据打印一次日志

# 全局变量定义
pbar = None
thread_local = threading.local()
api_cache = {}
cache_lock = threading.Lock()
current_cache_file_index = 0
api_params_queue = Queue()
api_responses_dict = {}


def get_cache_files() -> List[str]:
    """获取所有缓存文件"""
    return sorted(glob.glob(CACHE_FILE_PATTERN.format('*')))


def get_next_cache_file() -> str:
    """获取下一个缓存文件名"""
    global current_cache_file_index
    cache_files = get_cache_files()
    if not cache_files:
        current_cache_file_index = 0
    else:
        # 从现有文件名中提取最大索引
        indices = [int(f.split('_')[-1].split('.')[0]) for f in cache_files]
        current_cache_file_index = max(indices) + 1
    return CACHE_FILE_PATTERN.format(current_cache_file_index)


def optimize_response_data(response_data: Dict) -> Dict:
    """优化API响应数据，只保留必要字段"""
    if not response_data or not response_data.get('data'):
        return response_data

    optimized_data = {
        'success': response_data['success'],
        'data': []
    }

    # 只保留必要的字段
    needed_fields = {'s_room_id', 's_room_name', 's_room_bed_desc', 'points'}
    for item in response_data['data']:
        optimized_item = {k: item[k] for k in needed_fields if k in item}
        optimized_data['data'].append(optimized_item)

    return optimized_data


def load_api_cache():
    """加载所有API响应缓存并去重"""
    global api_cache
    api_cache = {}
    cache_files = get_cache_files()

    if not cache_files:
        print("未找到缓存文件")
        return

    print(f"开始加载 {len(cache_files)} 个缓存文件...")
    total_entries = 0
    duplicate_entries = 0
    loaded_files = 0
    valid_cache_files = []

    # 第一次遍历：验证文件完整性
    for cache_file in cache_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                json.load(f)
            valid_cache_files.append(cache_file)
            loaded_files += 1
        except json.JSONDecodeError:
            print(f"警告：缓存文件 {cache_file} 已损坏，将重命名为 {cache_file}.corrupted")
            try:
                os.rename(cache_file, cache_file + '.corrupted')
            except:
                pass
            continue
        except Exception as e:
            print(f"验证缓存文件 {cache_file} 时出错: {str(e)}")
            continue

    if not valid_cache_files:
        print("没有有效的缓存文件")
        return

    # 记录每个键首次出现的文件
    key_first_appearance = {}
    print("正在统计缓存数据...")

    # 第二次遍历：统计每个键首次出现的文件
    for cache_file in valid_cache_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                for key in cache_data.keys():
                    if key not in key_first_appearance:
                        key_first_appearance[key] = cache_file
        except Exception as e:
            print(f"统计缓存文件 {cache_file} 时出错: {str(e)}")
            continue

    print("正在加载缓存数据...")
    # 第三次遍历：只加载每个键在首次出现的文件中的数据
    for cache_file in valid_cache_files:
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                for key, value in cache_data.items():
                    total_entries += 1
                    if key_first_appearance[key] == cache_file:
                        api_cache[key] = value
                    else:
                        duplicate_entries += 1
        except Exception as e:
            print(f"加载缓存文件 {cache_file} 出错: {str(e)}")
            continue

    print(f"缓存加载完成: {len(api_cache)} 条唯一数据，{duplicate_entries} 条重复数据")


def save_api_cache():
    """保存API响应缓存，只进行增量更新"""
    if not api_cache:
        return

    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(CACHE_FILE_PATTERN), exist_ok=True)

        # 获取现有缓存文件
        cache_files = get_cache_files()

        # 如果没有缓存文件，创建新文件
        if not cache_files:
            current_cache_file = get_next_cache_file()
            temp_file = current_cache_file + '.tmp'
            try:
                with open(temp_file, 'w', encoding='utf-8') as f:
                    json.dump(api_cache, f, ensure_ascii=False)
                # 验证临时文件
                with open(temp_file, 'r', encoding='utf-8') as f:
                    json.load(f)
                # 验证成功后重命名
                os.rename(temp_file, current_cache_file)
            except Exception as e:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if pbar:
                    pbar.write(f"创建新缓存文件出错: {str(e)}")
            return

        # 获取最新的缓存文件
        latest_cache_file = cache_files[-1]
        
        try:
            # 读取最新缓存文件
            with open(latest_cache_file, 'r', encoding='utf-8') as f:
                existing_cache = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            if pbar:
                pbar.write(f"警告：最新缓存文件损坏或不存在，创建新文件")
            latest_cache_file = get_next_cache_file()
            existing_cache = {}

        # 计算新增数据
        new_keys = set(api_cache.keys()) - set(existing_cache.keys())
        if not new_keys:
            return

        # 创建新的缓存文件（只包含新增数据）
        new_cache_file = get_next_cache_file()
        new_data = {k: api_cache[k] for k in new_keys}

        # 使用临时文件进行安全写入
        temp_file = new_cache_file + '.tmp'
        try:
            # 写入临时文件
            with open(temp_file, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False)
            
            # 验证临时文件
            with open(temp_file, 'r', encoding='utf-8') as f:
                json.load(f)
            
            # 重命名临时文件为正式文件
            os.rename(temp_file, new_cache_file)
            
            if pbar:
                pbar.write(f"成功保存 {len(new_data)} 条新增缓存数据")

        except Exception as e:
            if os.path.exists(temp_file):
                os.remove(temp_file)
            if pbar:
                pbar.write(f"保存新增缓存数据出错: {str(e)}")

    except Exception as e:
        if pbar:
            pbar.write(f"保存缓存文件出错: {str(e)}")


def get_cache_key(s_hotel_id: int, supplier_id: int, spl_room_id: int) -> str:
    """生成缓存键"""
    return f"{s_hotel_id}_{supplier_id}_{spl_room_id}"


def get_from_cache(s_hotel_id: int, supplier_id: int, spl_room_id: int) -> Dict[str, Any]:
    """从缓存中获取数据"""
    cache_key = get_cache_key(s_hotel_id, supplier_id, spl_room_id)
    return api_cache.get(cache_key)


def update_cache(s_hotel_id: int, supplier_id: int, spl_room_id: int, response_data: Dict[str, Any]):
    """更新缓存"""
    if response_data.get('success'):
        cache_key = get_cache_key(s_hotel_id, supplier_id, spl_room_id)
        with cache_lock:
            # 优化响应数据
            optimized_data = optimize_response_data(response_data)
            api_cache[cache_key] = optimized_data
            # 每处理一定数量的数据保存一次缓存
            if len(api_cache) % SAVE_INTERVAL == 0:
                if pbar:
                    pbar.write(f"保存缓存数据: {len(api_cache)} 条")
                save_api_cache()


def get_connection():
    """获取线程本地的连接"""
    if not hasattr(thread_local, "conn"):
        thread_local.conn = http.client.HTTPConnection("119.23.73.59", 8080)
    return thread_local.conn


def call_api_with_retry(params: Tuple[int, int, int], retry_count: int = 0) -> Tuple[Dict[str, Any], int]:
    """带重试机制的API调用函数"""
    s_hotel_id, supplier_id, spl_room_id, idx = params

    # 首先检查缓存
    cached_response = get_from_cache(s_hotel_id, supplier_id, spl_room_id)
    if cached_response is not None:
        return cached_response, idx

    try:
        # 创建新的连接
        conn = http.client.HTTPConnection("119.23.73.59", 8080, timeout=TIMEOUT)

        payload = json.dumps({
            "password": "your-secure-password",
            "s_hotel_id": int(s_hotel_id),
            "supplier_id": int(supplier_id),
            "spl_room_id": int(spl_room_id)
        })
        headers = {
            'Content-Type': 'application/json',
        }

        # 发送请求
        conn.request("POST", "/api/room-mapping", payload, headers)

        # 设置响应超时
        res = conn.getresponse()
        data = res.read()
        response_data = json.loads(data.decode("utf-8"))

        # 检查响应状态
        if not response_data.get('success'):
            raise Exception(f"API返回错误: {str(response_data)},调用参数为：{params}")

        # 更新缓存
        update_cache(s_hotel_id, supplier_id, spl_room_id, response_data)

        return response_data, idx

    except (socket.timeout, socket.error, http.client.HTTPException, json.JSONDecodeError) as e:
        # 处理可重试的错误
        if retry_count < MAX_RETRIES:
            if pbar:
                pbar.write(f"API调用失败 [idx={idx}]: {str(e)}，{retry_count + 1}秒后重试...")
            time.sleep(RETRY_DELAY * (retry_count + 1))  # 指数退避
            return call_api_with_retry(params, retry_count + 1)
        else:
            if pbar:
                pbar.write(f"API调用失败 [idx={idx}]: {str(e)}，已达到最大重试次数")
            return {"success": False, "message": str(e), "data": None}, idx

    except Exception as e:
        # 处理其他错误
        if pbar:
            pbar.write(f"API调用出错 [idx={idx}]: {str(e)}")
        return {"success": False, "message": str(e), "data": None}, idx

    finally:
        if 'conn' in locals():
            try:
                conn.close()
            except:
                pass


def call_api(params: Tuple[int, int, int]) -> Tuple[Dict[str, Any], int]:
    """调用API的函数，返回API响应和行索引"""
    return call_api_with_retry(params)


def process_batch(df_batch: pd.DataFrame, start_idx: int) -> List[Dict]:
    """处理一批数据"""
    global api_params_queue, api_responses_dict
    batch_results = []

    # 分离supplier_id=98的数据和其他数据
    df_supplier_98 = df_batch[df_batch['supplier_id'] == 98]
    df_other = df_batch[df_batch['supplier_id'] != 98]

    # 处理supplier_id=98的数据
    for idx, row in df_supplier_98.iterrows():
        result_row = row.to_dict()
        result_row.update({
            'neg_room_name': '',
            'neg_bed_name': '',
            'point': '',
            'label': 1
        })
        batch_results.append(result_row)
        if pbar:
            pbar.update(1)

    # 处理其他数据
    if not df_other.empty:
        # 先检查缓存中是否有数据
        cached_results = []
        current_batch_params = []

        for idx, row in df_other.iterrows():
            cache_key = get_cache_key(row['s_hotel_id'], row['supplier_id'], row['spl_room_type_id'])
            cached_data = api_cache.get(cache_key)

            if cached_data is not None:
                try:
                    processed_rows = process_cached_data(row, cached_data)
                    if processed_rows:
                        cached_results.extend(processed_rows)
                    if pbar:
                        pbar.update(1)
                except Exception as e:
                    if pbar:
                        pbar.write(f"处理缓存数据出错 [idx={idx}]: {str(e)}")
                        pbar.write(f"缓存数据内容: {cached_data}")
                        pbar.write(f"行数据内容: {row.to_dict()}")
            else:
                # 将需要调用API的参数加入队列
                params = (row['s_hotel_id'], row['supplier_id'], row['spl_room_type_id'], idx)
                current_batch_params.append(params)
                api_params_queue.put((params, row))

        # 添加缓存处理的结果
        batch_results.extend(cached_results)

        # 处理需要调用API的数据
        if current_batch_params:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_params = {
                    executor.submit(call_api, params): (params, idx)
                    for params, idx in [(p, p[3]) for p in current_batch_params]
                }

                for future in as_completed(future_to_params):
                    params, idx = future_to_params[future]
                    try:
                        response, _ = future.result()
                        if response.get('data') is not None:
                            # 只存储API响应，不立即处理
                            api_responses_dict[idx] = response
                            if pbar:
                                pbar.update(1)
                        elif response.get('success') and start_idx % LOG_INTERVAL == 0:
                            if pbar:
                                pbar.write(f"API响应成功但无数据 [idx={idx}]")
                    except Exception as e:
                        if pbar:
                            pbar.write(f"处理API响应出错 [idx={idx}]: {str(e)}")

    return batch_results


def save_checkpoint(processed_rows: int):
    """保存处理进度"""
    checkpoint_data = {
        'processed_rows': processed_rows,
        'timestamp': datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint_data, f)


def load_checkpoint() -> int:
    """加载上次处理进度"""
    try:
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, 'r') as f:
                checkpoint_data = json.load(f)
                return checkpoint_data.get('processed_rows', 0)
    except Exception as e:
        if pbar:
            pbar.write(f"读取断点文件出错: {str(e)}")
    return 0


def load_existing_results() -> pd.DataFrame:
    """加载已有的处理结果"""
    try:
        if os.path.exists(RESULT_FILE):
            return pd.read_csv(RESULT_FILE)
    except Exception as e:
        if pbar:
            pbar.write(f"读取已有结果文件出错: {str(e)}")
    return pd.DataFrame()


def sort_and_shuffle_by_points(data_list: List[Dict]) -> List[Dict]:
    if not data_list:
        return []

    # 先按points排序
    sorted_data = sorted(data_list, key=lambda x: float(x.get('points', 0)), reverse=True)

    # 获取最高分和最低分
    max_point = float(sorted_data[0].get('points', 0))
    min_point = float(sorted_data[-1].get('points', 0))

    # 如果所有分数都相同，直接打乱整个列表
    if max_point == min_point:
        random.shuffle(sorted_data)
        return sorted_data

    # 处理不同分数的情况
    current_point = None
    start_idx = 0
    result = []

    for i, item in enumerate(sorted_data):
        point = float(item.get('points', 0))

        # 如果是新的分数，或者是最后一个元素
        if point != current_point or i == len(sorted_data) - 1:
            # 如果不是第一个分数，处理之前的区间
            if current_point is not None:
                # 确定区间结束位置
                end_idx = i if point != current_point else i + 1
                # 打乱相同分数的数据
                same_point_data = sorted_data[start_idx:end_idx]
                if len(same_point_data) > 1:  # 只有多于一个元素时才打乱
                    random.shuffle(same_point_data)
                result.extend(same_point_data)

            # 更新新区间的起始位置和分数
            start_idx = i
            current_point = point

    # 处理最后一组数据（如果还没处理）
    if start_idx < len(sorted_data) and current_point == point:
        last_group = sorted_data[start_idx:]
        if len(last_group) > 1:  # 只有多于一个元素时才打乱
            random.shuffle(last_group)
        result.extend(last_group)

    return result


def process_row(row: pd.Series, api_response, s_hotel_id: Dict[str, Any]) -> List[Dict]:
    global pbar
    results = []

    # 处理API返回的数据
    if 'data' in api_response and api_response['data']:
        data_list = api_response['data']

        # 按points排序，相同分数随机打乱
        data_list = sort_and_shuffle_by_points(data_list)

        # 找到当前sroomid对应的数据及其分数
        current_match = None
        for i, item in enumerate(data_list):
            if str(item.get('s_room_id')) == str(row['s_room_id']):
                current_match = item
                break

        # 生成label=1的数据
        # 1. 如果有当前房型的匹配，生成一条
        if current_match:
            current_row = row.to_dict()

            # 找到分数仅次于当前分数的数据
            next_best_match = None
            current_score = float(current_match.get('points', 0))

            # 遍历所有数据找到分数仅次于当前分数的匹配
            for item in data_list:
                if str(item.get('s_room_id')) != str(row['s_room_id']):
                    item_score = float(item.get('points', 0))
                    if item_score <= (current_score-0.01):
                        next_best_match = item
                        break

            # 如果找到了分数仅次于当前分数的匹配，使用它的数据
            if next_best_match:
                current_row.update({
                    'neg_room_name': next_best_match.get('s_room_name', ''),
                    'neg_bed_name': next_best_match.get('s_room_bed_desc', ''),
                    'point': float(next_best_match.get('points', 0)),
                    'label': 1
                })
            else:
                # 如果没有找到分数更低的匹配，使用空值
                current_row.update({
                    'neg_room_name': '',
                    'neg_bed_name': '',
                    'point': '',
                    'label': 1
                })
            results.append(current_row)
        # 如果没有任何好的匹配
        else:
            empty_row = row.to_dict()
            empty_row.update({
                'neg_room_name': '',
                'neg_bed_name': '',
                'point': 0.0,
                'label': 1
            })
            results.append(empty_row)

        # 2. 如果当前分数不是最高分，且最高分>0.83，再生成一条
        highest_match = data_list[0] if data_list else None
        if (current_match and
                float(highest_match.get('points', 0)) > float(current_match.get('points', 0)) and
                str(highest_match.get('s_room_id')) != str(row['s_room_id']) and
                float(highest_match.get('points', 0)) > 0.83):
            highest_row = row.to_dict().copy()
            # 找到分数仅次于最高分的数据
            next_highest = None
            highest_score = float(highest_match.get('points', 0))
            for item in data_list:
                if str(item.get('s_room_id')) != str(highest_match.get('s_room_id')):
                    item_score = float(item.get('points', 0))
                    if highest_score-0.01 >= item_score:
                        next_highest = item
                        break

            if next_highest:
                highest_row.update({
                    's_room_name': highest_match.get('s_room_name', ''),
                    's_room_bed_name': highest_match.get('s_room_bed_desc', ''),
                    'neg_room_name': next_highest.get('s_room_name', ''),
                    'neg_bed_name': next_highest.get('s_room_bed_desc', ''),
                    'point': float(next_highest.get('points', 0)),
                    'label': 1
                })
            else:
                highest_row.update({
                    's_room_name': highest_match.get('s_room_name', ''),
                    's_room_bed_name': highest_match.get('s_room_bed_desc', ''),
                    'neg_room_name': '',
                    'neg_bed_name': '',
                    'point': '',
                    'label': 1
                })
            results.append(highest_row)

        # 处理负样本（label=0）
        low_points_data = [item for item in data_list
                           if float(item.get('points', 0)) < 0.8
                           and str(item.get('s_room_id')) != str(row['s_room_id'])][:3]

        # 为低分数据创建新行
        for item in low_points_data:
            neg_row = row.to_dict()
            neg_row.update({
                'neg_room_name': item.get('s_room_name', ''),
                'neg_bed_name': item.get('s_room_bed_desc', ''),
                'point': float(item.get('points', 0)),
                'label': 0
            })
            results.append(neg_row)

    else:
        # 如果API没有返回数据，生成一条空的label=1数据
        empty_row = row.to_dict()
        empty_row.update({
            'neg_room_name': '',
            'neg_bed_name': '',
            'point': 0.0,
            'label': 1
        })
        results.append(empty_row)

    return results


def process_cached_data(row: pd.Series, cached_data: Dict[str, Any]) -> List[Dict]:
    """
    处理缓存数据并返回排序后的结果
    
    Args:
        row: 当前处理的数据行
        cached_data: 缓存的API响应数据
        
    Returns:
        List[Dict]: 处理后的结果列表，按similar属性排序
    """
    results = []
    
    # 检查缓存数据的有效性
    if not cached_data or not cached_data.get('data'):
        return []
        
    try:
        # 从cached_data中获取系统房型数据
        supplier_names = []
        spl_room_names = []
        spl_room_bed_names = []
        s_room_names = []
        s_room_bed_names = []
        s_room_data = []
        
        supplier_names.append(row['supplier_name'] if pd.notna(row['supplier_name']) and str(row['supplier_name']) != "nan" else "")
        spl_room_names.append(row['spl_room_name'])
        spl_room_bed_names.append(row['spl_room_bed_name'] if pd.notna(row['spl_room_bed_name']) and str(row['spl_room_bed_name']) != "nan" else "")
        s_room_names.append(row['s_room_name'])
        s_room_bed_names.append(row['s_room_bed_name'] if pd.notna(row['s_room_bed_name']) and str(row['s_room_bed_name']) != "nan" else "")
        s_room_data.append({
            's_room_name': row['s_room_name'],
            's_room_bed_desc': row['s_room_bed_name'],
            's_room_id': row['s_room_id'],
        })


        # 为每个系统房型添加一组供应商数据
        for item in cached_data['data']:
            supplier_names.append(row['supplier_name'] if pd.notna(row['supplier_name']) and str(row['supplier_name']) != "nan" else "")
            spl_room_names.append(row['spl_room_name'])
            spl_room_bed_names.append(row['spl_room_bed_name'] if pd.notna(row['spl_room_bed_name']) and str(row['spl_room_bed_name']) != "nan" else "")
            s_room_names.append(item.get('s_room_name', ''))
            s_room_bed_names.append(item.get('s_room_bed_desc', ''))
            s_room_data.append({
                's_room_name': item.get('s_room_name', ''),
                's_room_bed_desc': item.get('s_room_bed_desc', ''),
                's_room_id': item.get('s_room_id', ''),
            })
            
        # 准备调用API的数据
        api_data = {
            "supplier_names": supplier_names,
            "spl_room_names": spl_room_names,
            "spl_room_bed_names": spl_room_bed_names,
            "s_room_names": s_room_names,
            "s_room_bed_names": s_room_bed_names,
            "threshold": 0.65
        }
        
        # 调用API获取相似度
        response = requests.post(
            'http://10.0.0.110:8100/match',
            headers={
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            json=api_data,
            timeout=100
        )
        
        reference_similarities = None;
        if response.status_code == 200:
            response_data = response.json()
            similarities = response_data.get('similarities', [])
            
            reference_similarities = similarities[0]
            if reference_similarities is None or reference_similarities <0.65:
                return ValueError("基准值相似度小于0.65，数据异常")

            # 将相似度与原始数据组合
            combined_data = []
            for i, item in enumerate(s_room_data):
                if i < len(similarities):
                    combined_data.append({
                        **item,
                        'similar': similarities[i]
                    })
            
            # 按相似度排序
            sorted_data = sorted(combined_data, key=lambda x: float(x.get('similar', 0)), reverse=True)
            
            ###lable 为1的样本
            nagetive_data = [item for item in sorted_data if float(item.get('similar', 0)) < reference_similarities-0.1]
            # 随机选择1个样本
            if len(nagetive_data) > 0:
                nagetive_data = nagetive_data[0]

            result = row.to_dict()
            result.update({
                'neg_room_name': nagetive_data.get('s_room_name', '') if len(nagetive_data) > 0 else '',
                'neg_bed_name': nagetive_data.get('s_room_bed_desc', '') if len(nagetive_data) > 0 else '',
                'similar': nagetive_data.get('similar', 0) if len(nagetive_data) > 0 else 0,
                'label': 1
            })
            results.append(result)

            ### 添加母房型和子房型，文本一致数据，令文本一致的情况更加贴近
            # result = row.to_dict()
            # result.update({
            #     's_room_name': result.get('spl_room_name', ''),
            #     's_room_bed_name': result.get('spl_room_bed_name', ''),
            #     'neg_room_name': '',
            #     'neg_bed_name': '',
            #     'similar': '',
            #     'label': 1
            # })
            # results.append(result)


            ###lable 为1的样本
            filtered_data = [item for item in sorted_data if float(item.get('similar', 0)) < reference_similarities-0.2]
            # 随机选择5个样本
            if len(filtered_data) > 5:
                filtered_data = random.sample(filtered_data, 5)

            # 转换结果格式
            for item in filtered_data:
                result = row.to_dict()
                result.update({
                    's_room_name': item.get('s_room_name', ''),
                    's_room_bed_name': item.get('s_room_bed_desc', ''),
                    'neg_room_name': '',
                    'neg_bed_name': '',
                    'similar': float(item.get('similar', 0)),
                    'label': 0
                })
                results.append(result)
            
                
    except Exception as e:
        if pbar:
            pbar.write(f"处理positive接口数据时出错: {str(e)},row: {row}")
        # 如果调用positive接口失败，使用原始points进行排序
        data_list = cached_data['data']
        sorted_data = sorted(data_list, key=lambda x: float(x.get('points', 0)), reverse=True)
        
        for item in sorted_data:
            result = row.to_dict()
            result.update({
                'neg_room_name': item.get('s_room_name', ''),
                'neg_bed_name': item.get('s_room_bed_desc', ''),
                'point': float(item.get('points', 0)),
                'similar': float(item.get('points', 0)),  # 如果获取similar失败，使用points作为similar
                'label': 1 if str(item.get('s_room_id')) == str(row['s_room_id']) else 0
            })
            results.append(result)
    
    return results


def update_cache_from_queue():
    """从队列中更新缓存并处理数据"""
    global api_params_queue, api_responses_dict
    
    if api_params_queue.empty():
        return
    
    print("开始从队列更新缓存...")
    total_processed = 0
    all_processed_rows = []
    
    try:
        while not api_params_queue.empty():
            params, row = api_params_queue.get()
            cache_key = get_cache_key(params[0], params[1], params[2])
            
            # 如果已经有API响应，更新缓存并处理数据
            if params[3] in api_responses_dict:
                response = api_responses_dict[params[3]]
                if response.get('success'):
                    # 更新缓存
                    api_cache[cache_key] = optimize_response_data(response)
                    total_processed += 1
                    
                    # 处理数据
                    try:
                        processed_rows = process_cached_data(row, response)
                        if processed_rows:
                            all_processed_rows.extend(processed_rows)
                    except Exception as e:
                        print(f"处理API数据出错: {str(e)}")
                        print(f"API响应内容: {response}")
                        print(f"行数据内容: {row.to_dict()}")
            
            api_params_queue.task_done()
        
        # 保存更新后的缓存
        if total_processed > 0:
            print(f"处理完成，共更新 {total_processed} 条缓存数据")
            save_api_cache()
        else:
            print("没有需要更新的缓存数据")
            
        return all_processed_rows
            
    except Exception as e:
        print(f"更新缓存时出错: {str(e)}")
        return []
    finally:
        # 清空响应字典
        api_responses_dict.clear()
        
        # 确保队列为空
        while not api_params_queue.empty():
            try:
                api_params_queue.get_nowait()
                api_params_queue.task_done()
            except:
                pass


def main():
    global pbar

    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='处理房间匹配数据')
    parser.add_argument('--use-checkpoint', action='store_true',
                        help='是否使用断点续传功能（默认关闭）')
    parser.add_argument('--process-rows', type=int, default=1000000,
                        help='要处理的行数（默认200000行）')
    parser.add_argument('--test-cache', action='store_true',
                        help='是否进行缓存测试（默认关闭）')
    args = parser.parse_args()

    # 设置随机种子以保证结果可复现
    random.seed(42)

    # 添加标签统计计数器
    label_counts = {0: 0, 1: 0}

    # 加载API响应缓存
    load_api_cache()

    # 读取CSV文件
    df = pd.read_csv(SOURCE_FILE)
    
    # 获取所有action_type为del的数据组合
    del_combinations = df[df['action_type'] == 'del'][['spl_room_type_id', 's_room_id', 'supplier_id']].drop_duplicates()
    
    # 创建一个合并键，用于匹配
    del_combinations['merge_key'] = del_combinations.apply(lambda x: f"{x['spl_room_type_id']}_{x['s_room_id']}_{x['supplier_id']}", axis=1)
    df['merge_key'] = df.apply(lambda x: f"{x['spl_room_type_id']}_{x['s_room_id']}_{x['supplier_id']}", axis=1)
    
    # # 过滤掉在删除集合中的数据
    df = df[~df['merge_key'].isin(del_combinations['merge_key'])]
    
    # # 删除临时的merge_key列
    df = df.drop('merge_key', axis=1)
    
    print(f"过滤删除数据后，剩余行数: {len(df)}")

    print(f"开始处理数据，总行数: {len(df)}")

    # 获取原始列名
    original_columns = df.columns.tolist()
    # 新增的列名
    new_columns = ['neg_room_name', 'neg_bed_name', 'point', 'label']
    # 合并列名
    all_columns = original_columns + new_columns

    # 根据命令行参数决定是否使用断点续传
    start_row = load_checkpoint() if args.use_checkpoint else 0
    if start_row > 0:
        print(f"从断点继续处理，起始行: {start_row + 1}")

    # 根据命令行参数决定是否加载已有结果
    all_results = []
    if args.use_checkpoint:
        existing_results = load_existing_results()
        all_results = existing_results.to_dict('records') if not existing_results.empty else []
        print(f"已加载 {len(all_results)} 条现有结果")

    # 只处理指定行数
    df = df.head(args.process_rows)
    if start_row >= len(df):
        print("所有数据已处理完成")
        return

    # 创建进度条
    pbar = tqdm(total=len(df) - start_row, desc="处理进度", unit="行")

    try:
        # 按批次处理数据
        for batch_start in range(start_row, len(df), BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, len(df))
            df_batch = df.iloc[batch_start:batch_end]

            # 处理当前批次
            batch_results = process_batch(df_batch, batch_start)
            if batch_results:
                # 统计本批次的标签分布
                for result in batch_results:
                    label = result.get('label')
                    if label is not None:
                        label_counts[label] = label_counts.get(label, 0) + 1

                all_results.extend(batch_results)

            # 每处理完一个批次就保存结果
            if all_results and batch_start % (BATCH_SIZE * 10) == 0:  # 每10个批次保存一次
                result_df = pd.DataFrame(all_results)
                # 确保列的顺序与原始数据一致
                result_df = result_df.reindex(columns=all_columns)
                result_df.to_csv(RESULT_FILE, index=False)

                if args.use_checkpoint:
                    save_checkpoint(batch_end)

                # 计算标签比例
                total_labels = sum(label_counts.values())
                label_percentages = {k: f"{v / total_labels * 100:.2f}%" for k, v in label_counts.items()}

                pbar.set_postfix({
                    "已处理": f"{batch_end}/{len(df)}",
                    "生成数据": len(all_results),
                    "缓存数量": len(api_cache),
                    "缓存文件数": len(get_cache_files()),
                    "标签分布": label_percentages
                })

    except Exception as e:
        pbar.write(f"处理过程出错: {str(e)}")
        if args.use_checkpoint and all_results:
            result_df = pd.DataFrame(all_results)
            result_df = result_df.reindex(columns=all_columns)
            result_df.to_csv(RESULT_FILE, index=False)
            save_checkpoint(batch_start)

    finally:
        # 在程序结束时更新缓存并处理数据
        processed_rows = update_cache_from_queue()
        if processed_rows:
            all_results.extend(processed_rows)
            
            # 更新标签统计
            for result in processed_rows:
                label = result.get('label')
                if label is not None:
                    label_counts[label] = label_counts.get(label, 0) + 1

        # 关闭进度条
        if pbar:
            pbar.close()

        # 保存最终结果
        if all_results:
            result_df = pd.DataFrame(all_results)
            # 确保列的顺序与原始数据一致
            result_df = result_df.reindex(columns=all_columns)
            result_df.to_csv(RESULT_FILE, index=False)
            print(f"处理完成，共生成 {len(result_df)} 条数据")
            print(f"API缓存数量: {len(api_cache)} 条")
            print(f"缓存文件数量: {len(get_cache_files())} 个")

            # 打印最终的标签分布
            total_labels = sum(label_counts.values())
            print("\n标签分布统计:")
            for label, count in label_counts.items():
                percentage = count / total_labels * 100
                print(f"Label {label}: {count} 条 ({percentage:.2f}%)")
        else:
            print("处理完成，未生成数据")


if __name__ == "__main__":
    import os

    os.chdir('/home/maxon/disk2/roomMatch/')
    main()
