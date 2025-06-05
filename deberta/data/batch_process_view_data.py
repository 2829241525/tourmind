#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import pandas as pd
import requests
import time
import json
from tqdm import tqdm
import re

# 获取项目根目录并添加到系统路径
project_root = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
print(f"项目根目录: {project_root}")
if project_root not in sys.path:
    sys.path.append(project_root)
    print(f"已添加项目根目录到系统路径: {project_root}")
print(f"当前系统路径: {sys.path}")

# 配置信息
CONFIG = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_merged_cleaned.csv',  # 视图数据输入文件
    'output_file': '/home/maxon/disk2/roomMatch/room_match/deberta/data/view_similarity_results_merged_cleaned.csv',  # 输出文件
    'log_file': os.path.abspath('batch_process_view.log'),  # 日志文件路径
    'batch_size': 1000,  # 批处理大小

    # API 配置
    'api_url': 'http://10.0.0.110:8202/match/',  # API接口地址
    'model_name': 'cross_entropy_0414_ev1',  # 模型名称
    'threshold': 0.5,  # 匹配阈值
    'max_retries': 3,  # 最大重试次数
    'retry_delay': 2,  # 重试等待时间（秒）
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

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file'], encoding='utf-8'),
        logging.StreamHandler()
    ]
)


def filter_view_data(df):
    """
    筛选SPL房型和标准房型都包含view的数据

    Args:
        df: 原始DataFrame

    Returns:
        filtered_df: 筛选后的DataFrame
    """
    logging.info("开始筛选view相关数据...")

    # 过滤条件：SPL房型和标准房型都包含view,spl_room_text,s_room_text
    filter_mask = (
        # (df['SPL房型'].str.contains('view', case=False, na=False)) &
        # (df['标准房型'].str.contains('view', case=False, na=False)) &
        (df['spl_room_text'].str.contains('view', case=False, na=False)) &
        (df['s_room_text'].str.contains('view', case=False, na=False)) &
        (df['label']==1) 
    )

    # 获取符合条件的数据
    filtered_df = df[filter_mask].copy()

    filtered_count = len(filtered_df)
    logging.info(f"共找到 {filtered_count} 条符合条件的view相关数据")

    return filtered_df


def process_batch(batch_data, batch_id):
    """
    使用API服务处理单批数据

    Args:
        batch_data: 包含这批数据的字典
        batch_id: 批次ID

    Returns:
        处理结果字典，包含batch_id, results, status字段
    """
    try:
        start_time = time.time()
        # 获取文本对，并将房型与床型信息用逗号拼接
        spl_texts = []
        s_texts = []
        spl_bed_texts = []
        s_bed_texts = []

        for record in batch_data:
            # 获取房型和床型信息,spl_room_text,s_room_text
            # spl_room = record['SPL房型']
            # s_room = record['标准房型']           
            spl_room = record['spl_room_text']
            s_room = record['s_room_text']
            spl_bed = record.get('SPL床型', '')
            s_bed = record.get('标准床型', '')

            # 添加到列表中
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
                    timeout=300  # 设置超时时间为300秒，因为批量较大
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

                # 提取相似度结果和预测结果
                similarities = response_data['similarities']
                predictions = response_data['predictions']

                # 处理结果
                batch_results = []

                for i, (similarity, prediction) in enumerate(zip(similarities, predictions)):
                    # 从batch_data获取对应的记录
                    record = batch_data[i].copy()

                    # 添加预测结果
                    similarity_value = float(similarity)
                    prediction_value = int(prediction)

                    record['api_score'] = similarity_value
                    record['api_prediction'] = prediction_value
                    batch_results.append(record)

                # 记录API请求耗时
                end_time = time.time()
                elapsed = end_time - start_time
                logging.info(f"[批次 {batch_id}] API响应耗时: {elapsed:.2f}秒")

                # 返回处理结果
                return {
                    'batch_id': batch_id,
                    'results': batch_results,
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


def main():
    try:
        # 检查输入文件是否存在
        if not os.path.exists(CONFIG['input_file']):
            logging.error(f"输入文件不存在: {CONFIG['input_file']}")
            return

        # 读取CSV文件
        logging.info(f"开始读取CSV文件: {CONFIG['input_file']}")
        df = pd.read_csv(CONFIG['input_file'])

        # 检查必要的列是否存在,spl_room_text,s_room_text
        # required_columns = ['SPL房型', '标准房型']
        required_columns = ['spl_room_text', 's_room_text']
        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.error(f"以下列在CSV中不存在: {missing_columns}")
            return

        # 记录可用的列
        available_columns = df.columns.tolist()
        logging.info(f"CSV文件包含以下列: {available_columns}")

        # 筛选view相关数据
        filtered_df = filter_view_data(df)
        # filtered_df = filtered_df.head(2000)
        if filtered_df.empty:
            logging.warning("没有找到符合条件的view相关数据")
            return

        # 将DataFrame转换为字典列表
        records = filtered_df.to_dict('records')

        # 计算批次数量
        batch_size = CONFIG['batch_size']
        num_records = len(records)
        num_batches = (num_records + batch_size - 1) // batch_size

        logging.info(f"数据将被分为 {num_batches} 个批次进行处理，每批次最多 {batch_size} 条数据")

        # 创建进度条
        progress_bar = tqdm(
            total=num_batches,
            desc="处理数据批次",
            unit="批次",
            ncols=100,
            position=0,
            leave=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # 处理所有批次
        all_results = []
        for batch_id in range(num_batches):
            # 获取当前批次的数据
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, num_records)
            batch_data = records[start_idx:end_idx]

            # 处理当前批次
            result = process_batch(batch_data, batch_id)

            # 更新进度条
            progress_bar.update(1)
            progress_bar.set_postfix({
                "批次": f"{batch_id+1}/{num_batches}"
            })

            # 检查处理结果
            if result['status'] == 'success':
                all_results.extend(result['results'])
            else:
                logging.error(
                    f"批次 {batch_id} 处理失败: {result.get('error_message', '未知错误')}")
                # 对于失败的批次，保留原始数据，但不添加API结果
                for record in batch_data:
                    record['api_score'] = -1  # 使用-1表示API调用失败
                    record['api_prediction'] = -1
                    all_results.append(record)

        # 关闭进度条
        progress_bar.close()


        # 创建结果DataFrame
        result_df = pd.DataFrame(all_results)

        # 保存结果到CSV
        result_df.to_csv(CONFIG['output_file'], index=False)
        logging.info(f"处理结果已保存到: {CONFIG['output_file']}")

        # 输出统计信息
        total_processed = len(result_df)
        success_count = len(result_df[result_df['api_score'] >= 0])
        prediction_count = len(result_df[result_df['api_prediction'] == 1])

        logging.info(f"共处理 {total_processed} 条数据")
        logging.info(
            f"成功处理 {success_count} 条数据 ({success_count/total_processed*100:.2f}%)")
        logging.info(
            f"模型预测匹配的数据 {prediction_count} 条 ({prediction_count/total_processed*100:.2f}%)")

    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        import traceback
        logging.error(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
