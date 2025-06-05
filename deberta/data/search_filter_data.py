#!/usr/bin/env python
# -*- coding: utf-8 -*-

# 首先添加项目根目录到路径，确保可以正确导入模块
import torch
import math  # 添加math模块用于计算分批数量
import queue  # 添加queue模块用于线程间通信
import threading  # 添加threading模块用于多线程处理
import time  # 添加time模块用于请求间隔
import json
import random  # 添加random模块用于随机插入
import re
import logging
import pandas as pd
import os
import sys
import requests  # 添加requests模块用于HTTP请求
from tqdm import tqdm  # 导入tqdm进度条库

# 获取项目根目录并添加到系统路径
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
print(f"项目根目录: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"已添加项目根目录到系统路径: {project_root}")
print(f"当前系统路径: {sys.path}")

# 不再需要直接导入SimCSEPredictor
print("将使用API请求替代直接导入 SimCSEPredictor")

# 配置信息
CONFIG = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_merged_cleaned.csv',  # 输入CSV文件路径
    'output_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_merged_cleaned_filtered.csv',  # 输出CSV文件路径
    # 合并后的输出文件路径
    'merged_output_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/merged_filtered_result.csv',
    'target_columns': ['spl_room_text', 's_room_text'],  # 需要处理的列名
    'label_column': 'label',  # 标签列名
    'log_file': os.path.abspath('process_room.log'),  # 日志文件路径
    'replace_original': True,  # 是否用过滤结果替换原始文件
    # 新添加的配置
    'view_input_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/view_similarity_results_merged_cleaned.csv',  # 视图数据输入文件
    'view_output_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/view_match_results.csv',  # 视图数据输出文件
    'batch_size': 32,  # 批处理大小

    # API 配置
    'api_url': 'http://10.0.0.110:8202/match/',  # API接口地址
    'model_name': 'cross_entropy_0414_ev1',  # 模型名称
    'threshold': 0.5,  # 匹配阈值
    'max_retries': 3,  # 最大重试次数
    'retry_delay': 2,  # 重试等待时间（秒）

    # 运行模式配置
    'return_hard_samples': True,  # 是否返回困难样本
    'max_hard_samples': 10000  # 最大困难样本数量
}

# API请求头
API_HEADERS = {
    'Cookie': 'admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': '10.0.0.110:8202',
    'Connection': 'keep-alive'
}

# 地名列表，用于在"view"前添加
PLACE_NAMES = [
    "gulangyu", "lijiang", "dujiang", "emei", "huangshan", "lushan", "putuo",
    "jiuzhaigou", "yulong", "qiandao", "mogao", "longmen", "yungang", "chengde",
    "shanhaiguan", "zhangye", "shaolin", "kaiping", "ciqikou", "huangguoshu",
    "tianjin", "xiaoqikong", "wuyi", "fuxian", "sun", "shangrila", "kanas",
    "namtso", "tianshan", "sanfangqixiang"
]

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def insert_text_randomly(original_text, text_to_insert):
    """
    在原始文本的空格位置随机插入文本

    Args:
        original_text: 原始文本
        text_to_insert: 要插入的文本

    Returns:
        插入文本后的新文本
    """
    # 检查输入是否为有效字符串
    if not isinstance(original_text, str) or not original_text.strip():
        return str(original_text) + ' ' + text_to_insert

    # 分割文本为单词列表
    words = original_text.split()

    # 如果没有足够的单词，则在末尾添加
    if len(words) <= 1:
        return original_text + ' ' + text_to_insert

    # 随机选择插入位置（在单词之间）
    insert_position = random.randint(0, len(words))

    # 在选定位置插入新文本
    words.insert(insert_position, text_to_insert)

    # 重新组合文本
    return ' '.join(words)


def filter_main_data(df):
    """
    过滤label为1，s_room_text包含main，spl_room_text不包含main的数据
    并为每条记录创建新的样本

    Args:
        df: 原始DataFrame

    Returns:
        processed_df: 处理后的DataFrame
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始过滤main相关数据...")

    # 过滤条件
    # filter_mask = (
    #     (df[CONFIG['label_column']] == 1) &
    #     (df['s_room_text'].str.contains('main', case=False, na=False)) &
    #     (~df['spl_room_text'].str.contains('main', case=False, na=False))
    # )
    filter_mask = (
        (df[CONFIG['label_column']] == 1)
    )
    # 获取被过滤数据的索引filtered_df.index.tolist()
    original_indices = df[filter_mask].index.tolist()

    # 复制被过滤的数据
    filtered_df = df[filter_mask].copy().head(1000)

    filtered_count = len(filtered_df)
    logging.info(f"共找到 {filtered_count} 条符合条件的main相关数据")

    if filtered_count == 0:
        return pd.DataFrame(), []

    # 修改原始数据的label为0
    filtered_df[CONFIG['label_column']] = 1

    # 创建新样本
    new_samples = filtered_df.copy()
    new_samples[CONFIG['label_column']] = 0
    new_samples['s_room_text'] = new_samples['s_room_text'].apply(
        lambda x: insert_text_randomly(x, 'main')
    )

    # 合并原始和新样本
    processed_df = pd.concat([filtered_df, new_samples], ignore_index=True)

    new_samples = filtered_df.copy()
    new_samples[CONFIG['label_column']] = 0
    new_samples['s_room_text'] = new_samples['s_room_text'].apply(
        lambda x: insert_text_randomly(x, '[main]')
    )

    processed_df = pd.concat([processed_df, new_samples], ignore_index=True)

    logging.info(f"处理完成，生成了 {len(processed_df)} 条main相关数据")
    return processed_df, filtered_df.index.tolist()


def filter_side_view_data(df):
    """
    过滤label为1，s_room_text包含side(前后不能都为字母)不包含view，
    spl_room_text不包含side但包含view的数据，并为每条记录创建新的样本

    Args:
        df: 原始DataFrame

    Returns:
        processed_df: 处理后的DataFrame
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始过滤side和view相关数据...")

    # 过滤条件
    filter_mask = (
        (df[CONFIG['label_column']] == 1) &
        # 使用正则表达式确保"side"前后不能都为字母
        (df['s_room_text'].str.contains('((?<!\w)side|side(?!\w))', case=False, regex=True, na=False)) &
        (~df['s_room_text'].str.contains('view', case=False, na=False)) &
        (~df['spl_room_text'].str.contains('side', case=False, na=False)) &
        (df['spl_room_text'].str.contains('view', case=False, na=False))
    )

    # 获取被过滤数据的索引
    original_indices = df[filter_mask].index.tolist()

    # 复制被过滤的数据
    filtered_df = df[filter_mask].copy()

    filtered_count = len(filtered_df)
    logging.info(f"共找到 {filtered_count} 条符合条件的side和view相关数据")

    if filtered_count == 0:
        return pd.DataFrame(), []

    # 修改原始数据的label为0
    filtered_df[CONFIG['label_column']] = 0

    # 创建新样本1: s_room_text中的side改为view，spl_room_text不变
    new_samples1 = filtered_df.copy()
    new_samples1[CONFIG['label_column']] = 1
    new_samples1['s_room_text'] = new_samples1['s_room_text'].apply(
        lambda x: re.sub(r'(?<!\w)side|side(?!\w)', 'view', x,
                         flags=re.IGNORECASE) if isinstance(x, str) else x
    )

    # 创建新样本2: s_room_text不变，spl_room_text中的view改为side
    new_samples2 = filtered_df.copy()
    new_samples2[CONFIG['label_column']] = 1
    new_samples2['spl_room_text'] = new_samples2['spl_room_text'].apply(
        lambda x: re.sub(r'view', 'side', x,
                         flags=re.IGNORECASE) if isinstance(x, str) else x
    )

    # 合并所有样本
    processed_df = pd.concat(
        [filtered_df, new_samples1, new_samples2], ignore_index=True)

    logging.info(f"处理完成，生成了 {len(processed_df)} 条side和view相关数据")
    return processed_df, original_indices


def filter_floor_data(df):
    """
    过滤label为1，s_room_text包含floor，spl_room_text不包含floor的数据
    并为每条记录创建新的样本

    Args:
        df: 原始DataFrame

    Returns:
        processed_df: 处理后的DataFrame
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始过滤floor相关数据...")

    # 过滤条件
    filter_mask = (
        (df[CONFIG['label_column']] == 1) &
        (df['s_room_text'].str.contains('floor', case=False, na=False)) &
        (~df['spl_room_text'].str.contains('flr', case=False, na=False)) &
        (~df['spl_room_text'].str.contains(' fl', case=False, na=False)) &
        (~df['spl_room_text'].str.contains('floor', case=False, na=False))
    )

    # 获取被过滤数据的索引
    original_indices = df[filter_mask].index.tolist()

    # 复制被过滤的数据
    filtered_df = df[filter_mask].copy()

    filtered_count = len(filtered_df)
    logging.info(f"共找到 {filtered_count} 条符合条件的floor相关数据")

    if filtered_count == 0:
        return pd.DataFrame(), []

    # 修改原始数据的label为0
    filtered_df[CONFIG['label_column']] = 0

    # 创建新样本
    new_samples = filtered_df.copy()
    new_samples[CONFIG['label_column']] = 1
    new_samples['spl_room_text'] = new_samples['spl_room_text'].apply(
        lambda x: insert_text_randomly(x, 'floor')
    )

    # 合并原始和新样本
    processed_df = pd.concat([filtered_df, new_samples], ignore_index=True)

    logging.info(f"处理完成，生成了 {len(processed_df)} 条floor相关数据")
    return processed_df, original_indices


def filter_western_style_data(df):
    """
    过滤label为1，s_room_text不包含western style，
    spl_room_text包含western style的数据，并为每条记录创建新的样本

    Args:
        df: 原始DataFrame

    Returns:
        processed_df: 处理后的DataFrame
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始过滤western style相关数据...")

    # 过滤条件
    filter_mask = (
        (df[CONFIG['label_column']] == 1) &
        (~df['s_room_text'].str.contains('western style', case=False, na=False)) &
        (~df['s_room_text'].str.contains('western-style', case=False, na=False)) &
        (df['spl_room_text'].str.contains('western style', case=False, na=False))
    )

    # 获取被过滤数据的索引
    original_indices = df[filter_mask].index.tolist()

    # 复制被过滤的数据
    filtered_df = df[filter_mask].copy()

    filtered_count = len(filtered_df)
    logging.info(f"共找到 {filtered_count} 条符合条件的western style相关数据")

    if filtered_count == 0:
        return pd.DataFrame(), []

    # 修改原始数据的label为0
    filtered_df[CONFIG['label_column']] = 0

    # 创建新样本
    new_samples = filtered_df.copy()
    new_samples[CONFIG['label_column']] = 1
    new_samples['s_room_text'] = new_samples['s_room_text'].apply(
        lambda x: insert_text_randomly(x, 'western style')
    )

    # 合并原始和新样本
    processed_df = pd.concat([filtered_df, new_samples], ignore_index=True)

    logging.info(f"处理完成，生成了 {len(processed_df)} 条western style相关数据")
    return processed_df, original_indices


def process_room_text(text):
    """
    不再处理房间文本，直接返回原始文本
    原功能已移除:
    1. 移除所有[room]标记
    2. 如果包含[bed][unknown]，则移除这些标记
    3. 如果只包含[bed]，则替换为逗号
    """
    return text


def standardize_dataframe(df, complete_columns=None):
    """
    标准化DataFrame结构，确保包含所有必需的列

    Args:
        df: 需要标准化的DataFrame
        complete_columns: 完整的列列表，默认使用s_hotel_id,spl_room_type_id,spl_room_text,s_room_text,label

    Returns:
        标准化后的DataFrame
    """
    if df.empty:
        return df

    if complete_columns is None:
        complete_columns = ['s_hotel_id', 'spl_room_type_id',
                            'spl_room_text', 's_room_text', 'label']

    # 检查每个必需的列是否存在，如果不存在则添加空列
    for col in complete_columns:
        if col not in df.columns:
            df[col] = ''

    # 保留DataFrame中任何额外存在的列
    existing_columns = set(df.columns)
    required_columns = set(complete_columns)
    extra_columns = existing_columns - required_columns

    # 最终列顺序：先是必需列，然后是额外列
    final_columns = complete_columns + list(extra_columns)

    # 返回带有所有必需列和额外列的DataFrame
    return df[final_columns]


def process_batch_with_predictor(predictor, batch_data, batch_id, sample_type='hard_negative'):
    """使用API服务处理单批数据

    Args:
        predictor: 不再使用，保留参数仅为兼容现有代码
        batch_data: 包含这批数据的字典
        batch_id: 批次ID
        sample_type: 样本类型，'hard_negative'为困难负样本（评分>0.5），'hard_positive'为困难正样本（评分<0.9）

    Returns:
        处理结果字典，包含batch_id, results, match_count, status字段
    """
    try:
        start_time = time.time()
        # 获取文本对，并将房型与床型信息用逗号拼接
        spl_texts = []
        s_texts = []
        spl_bed_texts = []
        s_bed_texts = []

        for record in batch_data['data_records']:
            # 获取房型和床型信息
            spl_room = record['spl_room_text']
            s_room = record['s_room_text']
            spl_bed = record.get('spl_bed_text', '')
            s_bed = record.get('s_bed_text', '')

            # 不再拼接房型和床型，而是分别添加到列表中
            spl_texts.append(spl_room)
            s_texts.append(s_room)
            spl_bed_texts.append(
                spl_bed if spl_bed and not pd.isna(spl_bed) else "")
            s_bed_texts.append(s_bed if s_bed and not pd.isna(s_bed) else "")

        # 准备API请求数据
        payload = {
            "spl_room_names": spl_texts,
            "spl_room_bed_names": spl_bed_texts,
            "s_room_names": s_texts,
            "s_room_bed_names": s_bed_texts,
            "model": CONFIG['model_name'],
            "threshold": CONFIG['threshold']
        }

        # 发送API请求
        retries = 0
        max_retries = CONFIG['max_retries']

        while retries < max_retries:
            try:
                logging.info(f"[批次 {batch_id}] 发送API请求，数据量: {len(spl_texts)}条")
                response = requests.post(
                    CONFIG['api_url'],
                    headers=API_HEADERS,
                    json=payload,
                    timeout=90  # 设置超时时间为90秒
                )

                # 检查响应状态码
                if response.status_code != 200:
                    logging.error(
                        f"[批次 {batch_id}] API请求失败，状态码: {response.status_code}")
                    logging.error(f"响应内容: {response.text}")
                    retries += 1
                    if retries < max_retries:
                        time.sleep(CONFIG['retry_delay'])
                        continue
                    return {
                        'batch_id': batch_id,
                        'status': 'error',
                        'error_message': f"API请求失败，状态码: {response.status_code}"
                    }

                # 解析响应结果
                response_data = response.json()

                if 'similarities' not in response_data or 'predictions' not in response_data:
                    logging.error(
                        f"[批次 {batch_id}] API响应格式不正确: {response_data}")
                    retries += 1
                    if retries < max_retries:
                        time.sleep(CONFIG['retry_delay'])
                        continue
                    return {
                        'batch_id': batch_id,
                        'status': 'error',
                        'error_message': "API响应格式不正确"
                    }

                # 提取相似度结果
                similarities = response_data['similarities']

                # 处理结果
                batch_results = []
                match_count = 0

                for i, similarity in enumerate(similarities):
                    # 从batch_data获取对应的记录
                    record = batch_data['data_records'][i].copy()

                    # 添加预测结果
                    similarity_value = float(similarity)

                    # 根据样本类型判断是否为所需样本
                    if sample_type == 'hard_negative':
                        # 困难负样本：评分 > 0.5
                        is_match = similarity_value > 0.5
                    else:  # sample_type == 'hard_positive'
                        # 困难正样本：评分 < 0.9
                        is_match = similarity_value < 0.9

                    # 只有is_match为True时才添加记录
                    if is_match:
                        record['api_score'] = similarity_value
                        batch_results.append(record)
                        match_count += 1

                # 记录API请求耗时
                end_time = time.time()
                elapsed = end_time - start_time
                logging.info(
                    f"[批次 {batch_id}] API响应耗时: {elapsed:.2f}秒，匹配数: {match_count}")

                # 返回处理结果
                return {
                    'batch_id': batch_id,
                    'results': batch_results,
                    'match_count': match_count,
                    'status': 'success'
                }

            except Exception as e:
                logging.error(f"[批次 {batch_id}] API请求异常: {str(e)}")
                retries += 1
                if retries < max_retries:
                    time.sleep(CONFIG['retry_delay'])
                else:
                    return {
                        'batch_id': batch_id,
                        'status': 'error',
                        'error_message': f"API请求异常: {str(e)}"
                    }

    except Exception as e:
        logging.error(f"[批次 {batch_id}] 处理过程中发生错误: {str(e)}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")
        return {
            'batch_id': batch_id,
            'status': 'error',
            'error_message': str(e)
        }


def filter_negative_view_data():
    """
    筛选SPL房型和标准房型都包含view，且匹配分数小于0.5的数据，
    在view前添加地名前缀，并通过模型预测匹配关系

    Returns:
        processed_df: 处理后的DataFrame
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始筛选view相关数据...")

    try:
        # 检查输入文件是否存在
        if not os.path.exists(CONFIG['view_input_file']):
            logging.error(f"视图数据输入文件不存在: {CONFIG['view_input_file']}")
            return pd.DataFrame(), []

        # 读取CSV文件
        logging.info(f"开始读取视图数据CSV文件: {CONFIG['view_input_file']}")
        df = pd.read_csv(CONFIG['view_input_file'])

        # 检查必要的列是否存在
        required_columns = ['SPL房型', '标准房型', '匹配分数']
        recommended_columns = ['SPL床型', '标准床型']  # 建议的列，非必须
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"以下列在CSV中不存在: {missing_columns}")
            return pd.DataFrame(), []

        # 记录可用的列
        available_columns = df.columns.tolist()
        has_spl_bed = 'SPL床型' in available_columns
        has_s_bed = '标准床型' in available_columns

        if not has_spl_bed or not has_s_bed:
            logging.warning(f"CSV中没有床型信息列，将使用空字符串代替")

        # 筛选条件：SPL房型和标准房型都包含view，且匹配分数小于0.5
        filter_mask = (
            (df['SPL房型'].str.contains('view', case=False, na=False)) &
            (df['标准房型'].str.contains('view', case=False, na=False)) &
            (df['api_score'] < 0.5)
        )

        # 获取符合条件的数据
        filtered_df = df[filter_mask].copy()

        # 获取被过滤数据的索引
        original_indices = filtered_df.index.tolist()

        filtered_count = len(filtered_df)
        logging.info(f"共找到 {filtered_count} 条符合条件的view相关数据")

        if filtered_count == 0:
            return pd.DataFrame(), []

        # 如果需要限制数据量，可以取消注释下面的代码
        # filtered_df = filtered_df.head(100)  # 测试时使用小数据量
        # logging.info(f"限制处理数据，只处理前 20 条")

        # 不再需要初始化SimCSEPredictor，替换为API请求
        logging.info(f"将使用API调用: {CONFIG['api_url']}")
        predictor = None  # 仍然传递None作为predictor参数，保持代码兼容性

        # 记录每条数据的原始信息
        data_records = []

        # 对每条原始数据进行处理，先记录原始数据
        for index, row in filtered_df.iterrows():
            spl_room_name = row['SPL房型']
            s_room_name = row['标准房型']
            original_score = row['匹配分数']

            # 获取床型信息（如果有）
            spl_bed_name = row['SPL床型'] if has_spl_bed else ""
            s_bed_name = row['标准床型'] if has_s_bed else ""

            # 记录原始信息
            data_records.append({
                'spl_room_text': spl_room_name,
                's_room_text': s_room_name,
                'original_score': original_score,
                'prefix': 'original',
                'api_prediction': 0,
                'api_score': 0,
                'spl_bed_text': spl_bed_name,
                's_bed_text': s_bed_name
            })

        # 随机选择10个地名用于替换
        if len(PLACE_NAMES) > 10:
            selected_places = random.sample(PLACE_NAMES, 10)
        else:
            selected_places = PLACE_NAMES

        # 准备批量处理数据
        all_batches = []  # 存储所有批次的请求数据
        all_api_records = []  # 存储所有需要预测的记录

        # 对每条原始数据，生成地名修改后的数据
        logging.info("生成数据批次...")
        for i, record in enumerate(data_records):
            if record['prefix'] != 'original':  # 跳过非原始数据
                continue

            spl_room_name = record['spl_room_text']
            s_room_name = record['s_room_text']
            original_score = record['original_score']
            spl_bed_name = record['spl_bed_text']
            s_bed_name = record['s_bed_text']

            # 当前批次的数据
            batch = {
                'data_records': []  # 存储这批数据的记录
            }
            new_record = {
                'spl_room_text': spl_room_name,
                's_room_text': s_room_name,
                'original_score': original_score,
                'prefix': "",
                'spl_bed_text': spl_bed_name,
                's_bed_text': s_bed_name,
                'api_prediction': 0,
                'api_score': 0
            }
            batch['data_records'].append(new_record)
            # 遍历地名列表，为每个地名创建一条新数据
            for place_name in selected_places:
                # 在SPL房型和标准房型的view前添加地名
                modified_spl = re.sub(
                    r'view', f' {place_name} view', spl_room_name, flags=re.IGNORECASE)
                modified_s = re.sub(
                    r'view', f' {place_name} view', s_room_name, flags=re.IGNORECASE)

                # 记录数据信息
                new_record = {
                    'spl_room_text': modified_spl,
                    's_room_text': modified_s,
                    'original_score': original_score,
                    'prefix': place_name,
                    'spl_bed_text': spl_bed_name,
                    's_bed_text': s_bed_name,
                    'api_prediction': 0,
                    'api_score': 0
                }

                all_api_records.append(new_record)
                batch['data_records'].append(new_record)

                # 如果当前批次数据达到批量上限，添加到请求列表并创建新批次
                if len(batch['data_records']) >= CONFIG['batch_size']:
                    all_batches.append(batch)
                    batch = {
                        'data_records': []
                    }

            # 如果当前批次还有数据，添加到请求列表
            if len(batch['data_records']) > 0:
                all_batches.append(batch)

        total_batches = len(all_batches)
        total_samples = sum(len(batch['data_records'])
                            for batch in all_batches)
        logging.info(
            f"共生成 {total_batches} 个批次，总共 {total_samples} 条数据，每批次最多 {CONFIG['batch_size']} 条数据")

        # 使用单线程顺序处理所有批次
        logging.info("使用单线程顺序处理所有批次")

        # 处理所有批次
        all_results = []
        total_matches = 0
        start_time = time.time()

        # 创建进度条，用于显示处理进度
        progress_bar = tqdm(
            total=total_batches,
            desc="处理数据批次",
            unit="批次",
            ncols=100,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # 单线程顺序处理
        for batch_id, batch in enumerate(all_batches):
            result = process_batch_with_predictor(predictor, batch, batch_id)
            if result['status'] == 'success':
                all_results.extend(result['results'])
                match_count = result['match_count']
                total_matches += match_count

                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "匹配数": total_matches,
                    "批次": f"{batch_id+1}/{total_batches}",
                    "成功率": f"{total_matches/((batch_id+1)*CONFIG['batch_size'])*100:.1f}%"
                })

        # 关闭进度条
        progress_bar.close()

        total_time = time.time() - start_time
        samples_per_second = total_samples / total_time if total_time > 0 else 0
        logging.info(
            f"全部批次处理完成，总耗时: {total_time:.2f}秒，平均每秒处理 {samples_per_second:.2f} 个样本")

        # 释放模型
        logging.info("API模式无需释放模型")

        # 筛选预测为1的结果作为处理后的数据
        matched_records = all_results
        # 构建处理后的数据
        processed_rows = []
        for record in matched_records:
            # 获取房型和床型信息
            spl_room = record['spl_room_text']
            s_room = record['s_room_text']
            spl_bed = record.get('spl_bed_text', '')
            s_bed = record.get('s_bed_text', '')

            # 拼接房型和床型信息，如果床型存在则添加
            spl_full_text = spl_room
            if spl_bed and not pd.isna(spl_bed) and spl_bed.strip():
                spl_full_text += ", " + spl_bed.strip()

            s_full_text = s_room
            if s_bed and not pd.isna(s_bed) and s_bed.strip():
                s_full_text += ", " + s_bed.strip()

            # 确保数据结构与其他函数一致
            row_data = {
                's_hotel_id': '',   # 空值，保持列结构一致
                'spl_room_type_id': '',  # 空值，保持列结构一致
                'spl_room_text': spl_full_text,  # 使用拼接后的文本
                's_room_text': s_full_text,  # 使用拼接后的文本
                'label': 0,  # 设置为不匹配
                'hard_sample_type': 'hard_negative',  # 标记为困难负样本
                'api_score': record.get('api_score', 0)  # 保存模型评分
            }
            processed_rows.append(row_data)

        # 如果没有匹配结果，添加一些原始数据作为负样本
        if not processed_rows:
            for record in data_records[:10]:  # 仅使用前10个原始数据
                # 获取房型和床型信息
                spl_room = record['spl_room_text']
                s_room = record['s_room_text']
                spl_bed = record.get('spl_bed_text', '')
                s_bed = record.get('s_bed_text', '')

                # 拼接房型和床型信息，如果床型存在则添加
                spl_full_text = spl_room
                if spl_bed and not pd.isna(spl_bed) and spl_bed.strip():
                    spl_full_text += ", " + spl_bed.strip()

                s_full_text = s_room
                if s_bed and not pd.isna(s_bed) and s_bed.strip():
                    s_full_text += ", " + s_bed.strip()

                row_data = {
                    's_hotel_id': '',   # 空值，保持列结构一致
                    'spl_room_type_id': '',  # 空值，保持列结构一致
                    'spl_room_text': spl_full_text,  # 使用拼接后的文本
                    's_room_text': s_full_text,  # 使用拼接后的文本
                    'label': 0,  # 设置为不匹配
                    'hard_sample_type': 'hard_negative',  # 标记为困难负样本
                    'api_score': record.get('api_score', 0)  # 保存模型评分
                }
                processed_rows.append(row_data)

        # 将结果保存到输出文件
        if matched_records:
            # 保存完整的结果（包括所有字段）到view_output_file
            result_df = pd.DataFrame(all_results)
            result_df.to_csv(CONFIG['view_output_file'], index=False)
            logging.info(f"已将所有详细结果保存到: {CONFIG['view_output_file']}")
            logging.info(f"共找到 {total_matches} 条预测匹配的记录")

        # 创建处理后的DataFrame
        processed_df = pd.DataFrame(processed_rows)

        if not processed_df.empty:
            logging.info(f"处理完成，生成了 {len(processed_df)} 条view相关数据")
            return processed_df, original_indices
        else:
            logging.warning("没有找到符合条件的数据")
            return pd.DataFrame(), []

    except Exception as e:
        logging.error(f"处理view数据过程中发生错误: {str(e)}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")
        return pd.DataFrame(), []


def filter_positive_view_data():
    """
    筛选SPL房型和标准房型都包含view，且匹配分数大于0.9的高分数据，
    在view前添加地名前缀，并通过模型预测匹配关系，
    返回评分小于0.9的困难正样本

    Returns:
        processed_df: 处理后的DataFrame（包含困难正样本）
        original_indices: 被过滤的原始数据的索引
    """
    logging.info("开始筛选view相关高分数据以生成困难正样本...")

    try:
        # 检查输入文件是否存在
        if not os.path.exists(CONFIG['view_input_file']):
            logging.error(f"视图数据输入文件不存在: {CONFIG['view_input_file']}")
            return pd.DataFrame(), []

        # 读取CSV文件
        logging.info(f"开始读取视图数据CSV文件: {CONFIG['view_input_file']}")
        df = pd.read_csv(CONFIG['view_input_file'])

        # 检查必要的列是否存在
        required_columns = ['SPL房型', '标准房型', '匹配分数']
        recommended_columns = ['SPL床型', '标准床型']  # 建议的列，非必须
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"以下列在CSV中不存在: {missing_columns}")
            return pd.DataFrame(), []

        # 记录可用的列
        available_columns = df.columns.tolist()
        has_spl_bed = 'SPL床型' in available_columns
        has_s_bed = '标准床型' in available_columns

        if not has_spl_bed or not has_s_bed:
            logging.warning(f"CSV中没有床型信息列，将使用空字符串代替")

        # 筛选条件：SPL房型和标准房型都包含view，且匹配分数大于0.9
        filter_mask = (
            (df['SPL房型'].str.contains('view', case=False, na=False)) &
            (df['标准房型'].str.contains('view', case=False, na=False)) &
            (df['api_score'] > 0.9)
        )

        # 获取符合条件的数据
        filtered_df = df[filter_mask].copy()

        # filtered_df = filtered_df.head(20000)
        # 获取被过滤数据的索引
        original_indices = filtered_df.index.tolist()

        filtered_count = len(filtered_df)
        logging.info(f"共找到 {filtered_count} 条符合条件的高分view数据用于生成困难正样本")

        if filtered_count == 0:
            return pd.DataFrame(), []

        # 随机选择10个地名用于替换
        if len(PLACE_NAMES) > 10:
            selected_places = random.sample(PLACE_NAMES, 10)
        else:
            selected_places = PLACE_NAMES

        logging.info(
            f"已随机选择{len(selected_places)}个地名用于景观替换（困难正样本）: {selected_places}")

        # 不再需要初始化SimCSEPredictor，替换为API请求
        logging.info(f"将使用API调用: {CONFIG['api_url']}")
        predictor = None  # 仍然传递None作为predictor参数，保持代码兼容性

        # 记录每条数据的原始信息
        data_records = []

        # 对每条原始数据进行处理，先记录原始数据
        for index, row in filtered_df.iterrows():
            spl_room_name = row['SPL房型']
            s_room_name = row['标准房型']
            original_score = row['匹配分数']

            # 获取床型信息（如果有）
            spl_bed_name = row['SPL床型'] if has_spl_bed else ""
            s_bed_name = row['标准床型'] if has_s_bed else ""

            # 记录原始信息
            data_records.append({
                'spl_room_text': spl_room_name,
                's_room_text': s_room_name,
                'original_score': original_score,
                'prefix': 'original',
                'api_prediction': 1,  # 原始数据匹配分数高，预测为匹配
                'api_score': original_score,
                'spl_bed_text': spl_bed_name,
                's_bed_text': s_bed_name
            })

        # 准备批量处理数据
        all_batches = []  # 存储所有批次的请求数据
        all_api_records = []  # 存储所有需要预测的记录

        # 对每条原始数据，生成地名修改后的数据
        logging.info("生成困难正样本数据批次...")
        for i, record in enumerate(data_records):
            if record['prefix'] != 'original':  # 跳过非原始数据
                continue

            spl_room_name = record['spl_room_text']
            s_room_name = record['s_room_text']
            original_score = record['original_score']
            spl_bed_name = record['spl_bed_text']
            s_bed_name = record['s_bed_text']

            # 当前批次的数据
            batch = {
                'data_records': []  # 存储这批数据的记录
            }

            # 遍历选择的地名列表，为每个地名创建一条新数据
            for place_name in selected_places:
                # 在SPL房型和标准房型的view前添加地名
                modified_spl = re.sub(
                    r'view', f' {place_name} view', spl_room_name, flags=re.IGNORECASE)
                modified_s = re.sub(
                    r'view', f' {place_name} view', s_room_name, flags=re.IGNORECASE)

                # 记录数据信息
                new_record = {
                    'spl_room_text': modified_spl,
                    's_room_text': modified_s,
                    'original_score': original_score,
                    'prefix': place_name,
                    'spl_bed_text': spl_bed_name,
                    's_bed_text': s_bed_name,
                    'api_prediction': 0,
                    'api_score': 0
                }

                all_api_records.append(new_record)
                batch['data_records'].append(new_record)

                # 如果当前批次数据达到批量上限，添加到请求列表并创建新批次
                if len(batch['data_records']) >= CONFIG['batch_size']:
                    all_batches.append(batch)
                    batch = {
                        'data_records': []
                    }

            # 如果当前批次还有数据，添加到请求列表
            if len(batch['data_records']) > 0:
                all_batches.append(batch)

        total_batches = len(all_batches)
        total_samples = sum(len(batch['data_records'])
                            for batch in all_batches)
        logging.info(
            f"共生成 {total_batches} 个批次，总共 {total_samples} 条困难正样本数据，每批次最多 {CONFIG['batch_size']} 条数据")

        # 使用单线程顺序处理所有批次
        logging.info(f"使用单线程顺序处理所有批次，生成困难正样本")

        # 处理所有批次
        all_results = []
        total_matches = 0
        start_time = time.time()

        # 创建进度条，用于显示处理进度
        progress_bar = tqdm(
            total=total_batches,
            desc=f"处理困难正样本数据批次",
            unit="批次",
            ncols=100,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # 单线程顺序处理
        sample_type = 'hard_positive'
        for batch_id, batch in enumerate(all_batches):
            result = process_batch_with_predictor(
                predictor, batch, batch_id, sample_type)
            if result['status'] == 'success':
                all_results.extend(result['results'])
                match_count = result['match_count']
                total_matches += match_count

                # 更新进度条
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "匹配数": total_matches,
                    "批次": f"{batch_id+1}/{total_batches}",
                    "成功率": f"{total_matches/((batch_id+1)*CONFIG['batch_size'])*100:.1f}%"
                })

        # 关闭进度条
        progress_bar.close()

        total_time = time.time() - start_time
        samples_per_second = total_samples / total_time if total_time > 0 else 0
        logging.info(
            f"困难正样本批次处理完成，总耗时: {total_time:.2f}秒，平均每秒处理 {samples_per_second:.2f} 个样本")

        # 释放模型
        logging.info("API模式无需释放模型")

        # 筛选符合条件的样本（困难正样本：评分小于0.9）
        hard_samples = all_results

        logging.info(f"找到 {len(hard_samples)} 条困难正样本")

        # 如果样本数量过多，可以限制数量
        max_samples = CONFIG['max_hard_samples']  # 使用配置中的最大样本数量
        if len(hard_samples) > max_samples:
            hard_samples = random.sample(hard_samples, max_samples)
            logging.info(f"随机选择 {max_samples} 条困难正样本")

        # 构建处理后的数据
        processed_rows = []
        for record in hard_samples:
            # 获取房型和床型信息
            spl_room = record['spl_room_text']
            s_room = record['s_room_text']
            spl_bed = record.get('spl_bed_text', '')
            s_bed = record.get('s_bed_text', '')

            # 拼接房型和床型信息，如果床型存在则添加
            spl_full_text = spl_room
            if spl_bed and not pd.isna(spl_bed) and spl_bed.strip():
                spl_full_text += ", " + spl_bed.strip()

            s_full_text = s_room
            if s_bed and not pd.isna(s_bed) and s_bed.strip():
                s_full_text += ", " + s_bed.strip()

            # 确保数据结构与其他函数一致
            row_data = {
                's_hotel_id': '',   # 空值，保持列结构一致
                'spl_room_type_id': '',  # 空值，保持列结构一致
                'spl_room_text': spl_full_text,  # 使用拼接后的文本
                's_room_text': s_full_text,  # 使用拼接后的文本
                'label': 1,  # 设置为正样本（困难正样本）
                'hard_sample_type': 'hard_positive',  # 标记样本类型
                'api_score': record['api_score']  # 保存模型评分以供参考
            }
            processed_rows.append(row_data)

        # 将结果保存到输出文件
        if hard_samples:
            # 保存完整的结果（包括所有字段）到对应的output_file
            result_df = pd.DataFrame(all_results)
            hard_positive_output = CONFIG['view_output_file'].replace(
                '.csv', '_hard_positive.csv')
            result_df.to_csv(hard_positive_output, index=False)
            logging.info(f"已将困难正样本详细结果保存到: {hard_positive_output}")

        # 创建处理后的DataFrame
        processed_df = pd.DataFrame(processed_rows)

        if not processed_df.empty:
            logging.info(f"处理完成，生成了 {len(processed_df)} 条困难正样本")
            return processed_df, original_indices
        else:
            logging.warning(f"没有找到符合条件的困难正样本")
            return pd.DataFrame(), []

    except Exception as e:
        logging.error(f"处理view困难正样本过程中发生错误: {str(e)}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")
        return pd.DataFrame(), []


def main():
    try:
        # 检查输入文件是否存在
        if not os.path.exists(CONFIG['input_file']):
            logging.error(f"输入文件不存在: {CONFIG['input_file']}")
            return

        # 读取CSV文件
        logging.info(f"开始读取CSV文件: {CONFIG['input_file']}")
        df = pd.read_csv(CONFIG['input_file'])

        # 检查目标列是否存在
        required_columns = CONFIG['target_columns'] + [CONFIG['label_column']]
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"以下列在CSV中不存在: {missing_columns}")
            return

        # 不再处理目标列，直接输出原始数据
        logging.info("已移除文本处理功能，将直接使用原始数据进行过滤")

        # 保存原始文件（不做处理）
        df.to_csv(CONFIG['output_file'], index=False)
        logging.info(f"原始数据已保存到: {CONFIG['output_file']}")

        # # 过滤并处理main相关数据
        main_processed_df, main_indices = filter_main_data(df)

        # # 过滤并处理side和view相关数据
        side_view_processed_df, side_view_indices = filter_side_view_data(df)

        # 过滤并处理floor相关数据
        floor_processed_df, floor_indices = filter_floor_data(df)

        # 过滤并处理western style相关数据
        # western_style_processed_df, western_style_indices = filter_western_style_data(
        #     df)

        # 调用函数处理view数据
        view_negative_processed_df, view_negative_indices = filter_negative_view_data()
        logging.info(f"困难负样本数据处理完成，共处理 {len(view_negative_processed_df)} 条记录")
        view_positive_processed_df, view_positive_indices = filter_positive_view_data()
        logging.info(f"困难正样本数据处理完成，共处理 {len(view_positive_processed_df)} 条记录")

        # 合并所有过滤后的DataFrame
        logging.info("开始合并所有过滤后的数据...")
        all_processed_dfs = []

        # 定义目标列结构
        target_columns = ['s_hotel_id', 'spl_room_type_id',
                          'spl_room_text', 's_room_text', 'label']

        # 标准化并添加main相关数据
        if not main_processed_df.empty:
            standardized_df = standardize_dataframe(
                main_processed_df, target_columns)
            all_processed_dfs.append(standardized_df)
            logging.info(f"包含 {len(standardized_df)} 条处理后的main相关数据")

        # 标准化并添加side和view相关数据
        if not side_view_processed_df.empty:
            standardized_df = standardize_dataframe(
                side_view_processed_df, target_columns)
            all_processed_dfs.append(standardized_df)
            logging.info(f"包含 {len(standardized_df)} 条处理后的side和view相关数据")

        # 标准化并添加floor相关数据
        if not floor_processed_df.empty:
            standardized_df = standardize_dataframe(
                floor_processed_df, target_columns)
            all_processed_dfs.append(standardized_df)
            logging.info(f"包含 {len(standardized_df)} 条处理后的floor相关数据")

        # 标准化并添加困难负样本(view negative)数据
        if not view_negative_processed_df.empty:
            # 添加hard_sample_type列到目标列结构中
            hard_sample_columns = target_columns + \
                ['hard_sample_type', 'api_score']
            standardized_df = standardize_dataframe(
                view_negative_processed_df, hard_sample_columns)
            all_processed_dfs.append(standardized_df)
            logging.info(f"包含 {len(standardized_df)} 条处理后的困难负样本数据")

        # 标准化并添加困难正样本(view positive)数据
        if not view_positive_processed_df.empty:
            # 添加hard_sample_type列到目标列结构中
            hard_sample_columns = target_columns + \
                ['hard_sample_type', 'api_score']
            standardized_df = standardize_dataframe(
                view_positive_processed_df, hard_sample_columns)
            all_processed_dfs.append(standardized_df)
            logging.info(f"包含 {len(standardized_df)} 条处理后的困难正样本数据")

        # 如果配置启用了困难样本处理
        if CONFIG['return_hard_samples']:
            # 困难样本数据已通过view_positive_processed_df和view_negative_processed_df添加
            logging.info("困难样本数据已通过视图处理函数添加")

        if all_processed_dfs:
            # 合并处理后的数据
            merged_processed_df = pd.concat(
                all_processed_dfs, ignore_index=True)
            # 去除可能的重复行
            merged_processed_df = merged_processed_df.drop_duplicates()

            # 输出合并后的数据
            merged_count = len(merged_processed_df)
            logging.info(f"处理后共有 {merged_count} 条数据")

            # 直接将处理后的数据保存到合并输出文件
            merged_processed_df.to_csv(
                CONFIG['merged_output_file'], index=False)
            logging.info(f"已将合并后的数据保存到: {CONFIG['merged_output_file']}")

            # 决定是保存到新文件还是替换原始文件
            if CONFIG['replace_original']:
                try:
                    # 合并所有被过滤的原始数据的索引
                    all_indices = []
                    if main_indices:
                        all_indices.extend(main_indices)
                    if floor_indices:
                        all_indices.extend(floor_indices)
                    if side_view_indices:
                        all_indices.extend(side_view_indices)

                    # 去重并确保只使用有效的索引
                    valid_indices = []
                    for idx in all_indices:
                        if idx in df.index:
                            valid_indices.append(idx)
                        else:
                            logging.warning(f"索引 {idx} 不存在于原始DataFrame中，已跳过")

                    if view_negative_indices:
                        all_indices.extend(view_negative_indices)
                    if view_positive_indices:
                        all_indices.extend(view_positive_indices)

                    # 只有在有效索引不为空时才进行删除操作
                    if valid_indices:
                        unique_valid_indices = list(set(valid_indices))
                        logging.info(
                            f"将替换原始文件中的 {len(unique_valid_indices)} 条记录")

                        # 删除原始DataFrame中被过滤的行
                        df_remaining = df.drop(
                            unique_valid_indices, errors='ignore').reset_index(drop=True)

                        # 将处理后的数据与剩余的原始数据合并
                        final_df = pd.concat(
                            [df_remaining, merged_processed_df], ignore_index=True)

                        # 将合并后的结果保存回原始文件
                        final_df.to_csv(CONFIG['output_file'], index=False)
                        logging.info(
                            f"已用处理后的数据替换原始文件中的过滤数据: {CONFIG['output_file']}")
                    else:
                        logging.warning("没有找到有效的索引，无法替换原始文件中的数据")
                except Exception as e:
                    logging.error(f"替换原始文件时发生错误: {str(e)}")
                    import traceback
                    logging.error(f"详细错误: {traceback.format_exc()}")
                    logging.warning("由于错误，将仅保存合并数据而不替换原始文件")
            else:
                logging.info("未开启替换模式，仅保存合并数据")
        else:
            logging.warning("没有找到符合条件的数据，无法合并")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":

    main()
