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
import urllib3

# 禁用SSL警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

'''
负样本构造
'''

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

DATA_DIR = 'room_match/data'
RESULT_FILE = os.path.join(DATA_DIR, 'all_cn_data.csv')
SOURCE_FILE = os.path.join(DATA_DIR, 'positive_cn_filtered.csv')
ROOM_TYPES_FILE = os.path.join(DATA_DIR, 'score_cache','hotel_room_types.json')

# API配置
API_HEADERS = {
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'zh-CN,zh;q=0.9',
    'Connection': 'keep-alive',
    'Cookie': 'admintools_user_session=MTc0NDg3NTk2M3xWMDNZb3llbWw1LUMzc1lPeVZuYzlSckxkMVhoalQwVEQ2dVNFV1RMM28ta0Z2Z2RmcHhSNU5rY21uUHVnNTJlTmlaS21haWlfM2s9fKrtwhHO8Bo025Fl4idZaI_DpNqaV4WcWecugvboloaK',
    'Operator': 'xiemingxuan',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36'
}

# 处理参数配置
BATCH_SIZE = 10000
MAX_WORKERS = 20
MAX_RETRIES = 2
TIMEOUT = 10
RETRY_DELAY = 1
SAVE_INTERVAL = 1000
LOG_INTERVAL = 10000
CACHE_FILE_SIZE_LIMIT = 1024 * 1024 * 1024

###################
# 全局变量
###################

pbar = None
thread_local = threading.local()
api_cache = {}
cache_lock = threading.Lock()
current_cache_file_index = 0
api_params_queue = Queue()
api_responses_dict = {}


###################
# API相关函数
###################

def get_room_type_data(s_hotel_id: int) -> List[Dict]:
    """获取指定酒店的房型数据"""
    url = 'http://erp.tourmind.cn/roomtypemapping/api/get_roomtype'
    params = {
        'sHotelID': int(s_hotel_id),
        'supplierID': 0,
        'sRoomBaseType': 'tourmind',
        'sroom_status': 0
    }
    
    try:
        response = requests.get(url, params=params, headers=API_HEADERS, verify=False, timeout=100)
        data = response.json()
        
        if data.get('Success'):
            room_types = []
            for room in data.get('Data', {}).get('SRoomTypeList', []):
                room_types.append({
                    's_room_id': room.get('SRoomID'),
                    's_room_name': room.get('RoomTypeName'),
                    's_room_bed_desc': room.get('BedTypeDesc')
                })
            return room_types
        return None
    except Exception as e:
        print(f"获取酒店 {s_hotel_id} 房型数据失败: {str(e)}")
        return None


###################
# 数据处理函数
###################

def process_cached_data(row: pd.Series, cached_data: Dict[str, Any]) -> List[Dict]:
    """处理缓存数据并返回排序后的结果"""
    results = []
    
    if not cached_data or not cached_data.get('data'):
        return []
        
    try:
        # 准备数据
        supplier_names = []
        spl_room_names = []
        spl_room_bed_names = []
        s_room_names = []
        s_room_bed_names = []
        s_room_data = []
        
        # 添加当前行数据
        supplier_names.append(row['supplier_name'] if pd.notna(row['supplier_name']) else "")
        spl_room_names.append(row['spl_room_name'])
        spl_room_bed_names.append(row['spl_room_bed_name'] if pd.notna(row['spl_room_bed_name']) else "")
        s_room_names.append(row['s_room_name'])
        s_room_bed_names.append(row['s_room_bed_name'] if pd.notna(row['s_room_bed_name']) else "")
        s_room_data.append({
            's_room_name': row['s_room_name'],
            's_room_bed_desc': row['s_room_bed_name'],
            's_room_id': row['s_room_id'],
        })

        # 添加缓存数据
        for item in cached_data['data']:
            supplier_names.append(row['supplier_name'] if pd.notna(row['supplier_name']) else "")
            spl_room_names.append(row['spl_room_name'])
            spl_room_bed_names.append(row['spl_room_bed_name'] if pd.notna(row['spl_room_bed_name']) else "")
            s_room_names.append(item.get('s_room_name', ''))
            s_room_bed_names.append(item.get('s_room_bed_desc', ''))
            s_room_data.append({
                's_room_name': item.get('s_room_name', ''),
                's_room_bed_desc': item.get('s_room_bed_desc', ''),
                's_room_id': item.get('s_room_id', ''),
            })
            
        # 调用匹配API
        api_data = {
            "spl_room_names": spl_room_names,
            "spl_room_bed_names": spl_room_bed_names,
            "s_room_names": s_room_names,
            "s_room_bed_names": s_room_bed_names,
            "threshold": 0.65,
            "model": "cross_entropy_0414_ev1"
        }
        
        response = requests.post(
            'http://10.0.0.110:8203/match',
            headers={'Content-Type': 'application/json'},
            json=api_data,
            timeout=100
        )
        
        if response.status_code == 200:
            response_data = response.json()
            similarities = response_data.get('similarities', [])
            
            reference_similarities = similarities[0]
            if reference_similarities is None or reference_similarities < 0.65:
                return []

            # 处理相似度数据
            combined_data = []
            for i, item in enumerate(s_room_data):
                if i < len(similarities):
                    combined_data.append({**item, 'similar': similarities[i]})
            
            # 按相似度排序
            sorted_data = sorted(combined_data, key=lambda x: float(x.get('similar', 0)), reverse=True)
            
            # 处理正样本
            negative_data = sorted_data
            # negative_data = [item for item in sorted_data if float(item.get('similar', 0)) < reference_similarities-0.1]
            if negative_data:
                negative_data = negative_data[0]
                if negative_data.get('similar')>0.5:
                    result = row.to_dict()
                    result.update({
                        'neg_room_name': negative_data.get('s_room_name', ''),
                        'neg_bed_name': negative_data.get('s_room_bed_desc', ''),
                        'similar': negative_data.get('similar', 0),
                        'label': 1
                    })
                    results.append(result)

            # 处理负样本
            filtered_data = [item for item in sorted_data if float(item.get('similar', 0)) < reference_similarities-0.2]
            if len(filtered_data) > 4:
                filtered_data = random.sample(filtered_data, 4)

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
            pbar.write(f"处理positive接口数据时出错: {str(e)}, row: {row}")
    
    return results


###################
# 主函数
###################

def main():
    global pbar

    # 解析命令行参数
    parser = argparse.ArgumentParser(description='处理房间匹配数据')
    parser.add_argument('--use-checkpoint', action='store_true', help='是否使用断点续传功能')
    parser.add_argument('--process-rows', type=int, default=1000000, help='要处理的行数')
    parser.add_argument('--test-cache', action='store_true', help='是否进行缓存测试')
    args = parser.parse_args()

    # 设置随机种子
    random.seed(42)

    # 读取和预处理数据
    df = pd.read_csv(SOURCE_FILE)
    # del_combinations = df[df['action_type'] == 'del'][['spl_room_type_id', 's_room_id', 'supplier_id']].drop_duplicates()
    
    # # 创建合并键并过滤数据
    # del_combinations['merge_key'] = del_combinations.apply(lambda x: f"{x['spl_room_type_id']}_{x['s_room_id']}_{x['supplier_id']}", axis=1)
    # df['merge_key'] = df.apply(lambda x: f"{x['spl_room_type_id']}_{x['s_room_id']}_{x['supplier_id']}", axis=1)
    # df = df[~df['merge_key'].isin(del_combinations['merge_key'])]
    # df = df.drop('merge_key', axis=1)
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df = df.head(300000)
    print(f"过滤删除数据后，剩余行数: {len(df)}")

    # 获取并处理酒店房型数据
    unique_hotel_ids = sorted(df['s_hotel_id'].unique())
    print(f"总共有 {len(unique_hotel_ids)} 个不同的酒店ID")
    
    # 读取已存在的缓存数据
    existing_hotel_room_types = {}
    if os.path.exists(ROOM_TYPES_FILE):
        try:
            with open(ROOM_TYPES_FILE, 'r', encoding='utf-8') as f:
                existing_hotel_room_types = json.load(f)
            print(f"已读取缓存数据，包含 {len(existing_hotel_room_types)} 个酒店的房型数据")
            
            # 过滤掉已经存在于缓存中的酒店ID
            cached_hotel_ids = set(existing_hotel_room_types.keys())
            unique_hotel_ids = [hotel_id for hotel_id in unique_hotel_ids 
                              if str(hotel_id) not in cached_hotel_ids]
            print(f"过滤后需要处理 {len(unique_hotel_ids)} 个新的酒店ID")
        except Exception as e:
            print(f"读取缓存文件出错: {str(e)}")
            existing_hotel_room_types = {}
    

    # 处理新的酒店数据
    hotel_room_types = {}
    thread_lock = threading.Lock()

    def process_hotel(hotel_id):
        """处理单个酒店的房型数据"""
        try:
            room_types = get_room_type_data(hotel_id)
            if room_types:
                with thread_lock:
                    hotel_room_types[str(hotel_id)] = room_types
                return True
            return False
        except Exception as e:
            print(f"处理酒店 {hotel_id} 时出错: {str(e)}")
            return False

    # 使用线程池并发处理
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for hotel_id in unique_hotel_ids:
            futures.append(executor.submit(process_hotel, hotel_id))
        
        # 使用tqdm显示进度
        successful = 0
        failed = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="获取酒店房型数据"):
            try:
                if future.result():
                    successful += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"执行任务时出错: {str(e)}")
                failed += 1
        
        print(f"处理完成: 成功 {successful} 个，失败 {failed} 个")
    
    # 合并新旧数据
    if existing_hotel_room_types:
        hotel_room_types.update(existing_hotel_room_types)
    
    # 保存合并后的数据
    print(f"正在写入房型数据，共 {len(hotel_room_types)} 个酒店")
    with open(ROOM_TYPES_FILE, 'w', encoding='utf-8') as f:
        # 写入开始标记
        f.write('{\n')
        
        # 获取所有酒店ID并排序，以确保写入顺序一致
        hotel_ids = sorted(hotel_room_types.keys())
        
        # 使用tqdm显示进度
        for i, hotel_id in enumerate(tqdm(hotel_ids, desc="写入房型数据")):
            room_data = hotel_room_types[hotel_id]
            # 写入酒店ID和房型数据
            f.write(f'  "{hotel_id}": {json.dumps(room_data, ensure_ascii=False)}')
            # 除了最后一个元素，都需要添加逗号
            if i < len(hotel_ids) - 1:
                f.write(',\n')
            else:
                f.write('\n')
        
        # 写入结束标记
        f.write('}\n')
    
    print(f"房型数据已保存到: {ROOM_TYPES_FILE}")
    print(f"成功获取和更新共 {len(hotel_room_types)} 个酒店的房型数据")
    print(f"开始处理数据，总行数: {len(df)}")

    # 处理缓存数据
    all_results = []
    print("开始处理房型匹配数据...")
    
    # 使用tqdm显示进度
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理房型匹配"):
        try:
            # 获取当前酒店的房型数据
            hotel_id = str(row['s_hotel_id'])
            if hotel_id not in hotel_room_types:
                print(f"警告: 酒店 {hotel_id} 的房型数据不存在")
                continue
                
            cached_data = {
                'success': True,
                'data': hotel_room_types[hotel_id]
            }
            
            # 处理当前行数据
            results = process_cached_data(row, cached_data)
            if results:
                all_results.extend(results)
                
            # 每处理一定数量的数据保存一次
            if len(all_results) > 0 and len(all_results) % SAVE_INTERVAL == 0:
                print(f"\n已处理 {len(all_results)} 条数据，正在保存...")
                result_df = pd.DataFrame(all_results)
                result_df.to_csv(RESULT_FILE, index=False, encoding='utf-8')
                print(f"数据已保存到: {RESULT_FILE}")
                
        except Exception as e:
            print(f"\n处理行 {idx} 时出错: {str(e)}")
            print(f"行数据: {row.to_dict()}")
            continue
    
    # 保存最终结果
    if all_results:
        print(f"\n处理完成，共生成 {len(all_results)} 条数据，正在保存...")
        result_df = pd.DataFrame(all_results)
        result_df.to_csv(RESULT_FILE, index=False, encoding='utf-8')
        print(f"所有数据已保存到: {RESULT_FILE}")
    else:
        print("\n没有生成任何匹配数据")

if __name__ == "__main__":
    os.chdir('/home/maxon/disk2/roomMatch/')
    main()
