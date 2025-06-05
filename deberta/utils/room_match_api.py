#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import time
import requests
import pandas as pd
from tqdm import tqdm
import logging
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,  # 改回INFO级别，减少日志量
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("room_match_api.log", mode='w'),  # 使用'w'模式覆盖旧日志
        logging.StreamHandler()
    ]
)

# 配置信息/home/maxon/disk2/roomMatch/room_match/data/room_match_orders.xlsx
config = {
    "csv_path": "/home/maxon/disk2/roomMatch/room_match/deberta/data/qs_match.csv",
    "output_csv_path": "/home/maxon/disk2/roomMatch/room_match/deberta/data/qs_match_result.csv",
    "api_url": "http://10.0.0.110:8205/match/",
    "token": "tourmind_room_match",
    "threshold": 0.9,
    "sleep_time": 0.01,  # API调用间隔时间(秒)
    "batch_size": 100,    # 批量请求大小，每次发送的数据条数，根据API性能调整
    "max_retry": 2       # API调用失败最大重试次数
}

# HTTP请求头
headers = {
    'Cookie': 'admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k9fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
    'Content-Type': 'application/json',
    'Accept': '*/*',
    'Host': '10.0.0.110:8205',
    'Connection': 'keep-alive'
}


def call_match_api_batch(spl_room_names, spl_bed_names, std_room_names, std_bed_names):
    """
    批量调用房型匹配API

    Args:
        spl_room_names (list): SPL房型名称列表
        spl_bed_names (list): SPL床型名称列表
        std_room_names (list): 标准房型名称列表
        std_bed_names (list): 标准床型名称列表

    Returns:
        list: API响应结果列表
    """
    # 参数校验
    if not all([isinstance(param, list) for param in [spl_room_names, spl_bed_names, std_room_names, std_bed_names]]):
        logging.error("批量调用API参数必须是列表类型")
        return None

    # 检查列表长度是否一致
    list_lengths = [len(x) for x in [spl_room_names,
                                     spl_bed_names, std_room_names, std_bed_names]]
    if len(set(list_lengths)) > 1:
        logging.error(f"批量调用API参数列表长度不一致: {list_lengths}")
        return None

    batch_size = len(spl_room_names)
    logging.info(f"准备批量调用API，数据量: {batch_size}条")

    # 处理空字符串
    spl_room_names = [name if name else "" for name in spl_room_names]
    spl_bed_names = [name if name else "" for name in spl_bed_names]
    std_room_names = [name if name else "" for name in std_room_names]
    std_bed_names = [name if name else "" for name in std_bed_names]

    # 构造请求数据
    payload = {
        "spl_room_names": spl_room_names,
        "spl_room_bed_names": spl_bed_names,
        "s_room_names": std_room_names,
        "s_room_bed_names": std_bed_names,
        "token": config["token"],
        "model": "all_0421",
        "threshold": config["threshold"]
    }

    retry_count = 0
    while retry_count < config["max_retry"]:
        try:
            start_time = time.time()
            logging.info(f"发送批量请求，数据量: {batch_size}条，第{retry_count+1}次尝试")

            response = requests.post(
                config["api_url"], headers=headers, json=payload, timeout=90)  # 增加超时时间

            end_time = time.time()
            elapsed = end_time - start_time
            logging.info(f"API响应耗时: {elapsed:.2f}秒")

            if response.status_code != 200:
                logging.error(f"API响应错误，状态码: {response.status_code}")
                retry_count += 1
                time.sleep(2)  # 出错重试前等待
                continue

            try:
                result = response.json()

                if 'similarities' in result and 'predictions' in result:
                    similarities = result['similarities']
                    predictions = result['predictions']

                    # 检查返回结果数量是否与请求数量一致
                    if len(similarities) != batch_size or len(predictions) != batch_size:
                        logging.warning(
                            f"API返回结果数量与请求不一致: 请求{batch_size}条，返回相似度{len(similarities)}条，预测{len(predictions)}条")

                    # 构造返回结果列表
                    formatted_results = []
                    for i in range(min(len(similarities), len(predictions))):
                        formatted_results.append({
                            'match_score': similarities[i] if i < len(similarities) else 0,
                            'match_result': bool(predictions[i]) if i < len(predictions) else False,
                            'raw_response': result  # 将原始响应放在每个结果中
                        })

                    # 如果返回的结果数量不足，补充空结果
                    while len(formatted_results) < batch_size:
                        formatted_results.append({
                            'match_score': 0,
                            'match_result': False,
                            'raw_response': {'error': '服务器返回结果不完整'}
                        })

                    logging.info(f"批量API调用成功，获取{len(formatted_results)}条结果")
                    return formatted_results

                logging.error(f"API返回格式异常: {result}")
                retry_count += 1
                time.sleep(2)
                continue

            except json.JSONDecodeError as e:
                logging.error(f"响应内容解析失败: {e}")
                retry_count += 1
                time.sleep(2)
                continue

        except requests.exceptions.RequestException as e:
            logging.error(f"请求异常: {e}")
            retry_count += 1
            time.sleep(2)
            continue
        except Exception as e:
            logging.error(f"未知异常: {e}")
            retry_count += 1
            time.sleep(2)
            continue

    logging.error(f"批量API调用失败，已重试{config['max_retry']}次")
    return None


def call_match_api(spl_room_name, spl_bed_name, std_room_name, std_bed_name):
    """
    调用单个房型匹配API (兼容旧代码)

    Args:
        spl_room_name (str): SPL房型名称
        spl_bed_name (str): SPL床型名称
        std_room_name (str): 标准房型名称
        std_bed_name (str): 标准床型名称

    Returns:
        dict: API响应结果
    """
    # 调用批量API，但只传递一个数据
    results = call_match_api_batch(
        [spl_room_name], [spl_bed_name], [std_room_name], [std_bed_name]
    )

    # 返回第一个结果
    if results and len(results) > 0:
        return results[0]
    return None


def write_results_to_csv():
    """
    从之前的日志中提取API调用结果并写入CSV
    """
    # 读取原始CSV
    df = pd.read_csv(config["csv_path"])

    # 创建新的结果DataFrame，包含原始数据和新列
    result_df = pd.DataFrame()

    # 复制原始列
    for col in df.columns:
        result_df[col] = df[col].copy()

    # 添加新列，明确数据类型
    result_df['匹配分数'] = [0.0] * len(df)  # 默认浮点数
    result_df['匹配结果'] = [""] * len(df)   # 字符串
    result_df['相似度值'] = [""] * len(df)   # 字符串

    # 保存CSV
    result_df.to_csv(config["output_csv_path"], index=False, encoding='utf-8')

    logging.info(f"创建了初始CSV文件: {config['output_csv_path']}")

    return result_df


def process_batch(start_idx, end_idx):
    """
    处理一批数据并将结果直接写入CSV

    Args:
        start_idx (int): 起始索引
        end_idx (int): 结束索引
    """
    # 读取CSV文件
    df = pd.read_csv(config["output_csv_path"])

    # 确保所有列都有正确的类型
    if '匹配分数' not in df.columns:
        df['匹配分数'] = 0.0
    if '匹配结果' not in df.columns:
        df['匹配结果'] = ""
    if '相似度值' not in df.columns:
        df['相似度值'] = ""

    # 计算真实的结束索引
    real_end_idx = min(end_idx, len(df))
    total_rows = real_end_idx - start_idx
    logging.info(f"处理数据范围: {start_idx} - {real_end_idx}，共{total_rows}条")

    # 批量处理数据
    success_count = 0
    fail_count = 0

    # 分成更小的批次进行处理
    batch_size = min(config["batch_size"], total_rows)  # 确保批次大小不超过总数据量
    num_batches = (total_rows + batch_size - 1) // batch_size

    logging.info(f"将分{num_batches}批进行处理，每批{batch_size}条")

    with tqdm(total=total_rows, desc="批量处理进度") as pbar:
        for batch_idx, batch_start in enumerate(range(start_idx, real_end_idx, batch_size)):
            batch_end = min(batch_start + batch_size, real_end_idx)
            batch_indices = list(range(batch_start, batch_end))
            batch_size_actual = len(batch_indices)

            logging.info(
                f"处理第{batch_idx+1}/{num_batches}批: {batch_start}-{batch_end}，共{batch_size_actual}条")

            # 收集该批次的所有数据
            batch_rows = df.iloc[batch_indices]

            spl_rooms = []
            spl_beds = []
            std_rooms = []
            std_beds = []

            # 收集数据
            for _, row in batch_rows.iterrows():
                spl_rooms.append(
                    row['SPL房型'] if pd.notna(row['SPL房型']) else "")
                spl_beds.append(row['SPL床型'] if pd.notna(row['SPL床型']) else "")
                std_rooms.append(row['标准房型'] if pd.notna(row['标准房型']) else "")
                std_beds.append(row['标准床型'] if pd.notna(row['标准床型']) else "")

            # 批量调用API
            batch_start_time = time.time()
            responses = call_match_api_batch(
                spl_rooms, spl_beds, std_rooms, std_beds)
            batch_elapsed = time.time() - batch_start_time

            batch_success = 0
            batch_fail = 0

            # 创建临时结果列表
            scores = []
            results = []
            sim_values = []

            if responses:
                # 处理每个响应结果
                for idx, response in enumerate(responses):
                    if response and isinstance(response, dict) and 'match_score' in response:
                        # 提取数据，稍后一次性设置
                        score = float(response.get('match_score', 0.0))
                        result = "匹配成功" if response.get(
                            'match_result') else "匹配失败"

                        # 处理相似度值
                        sim_value = ""
                        if 'raw_response' in response and isinstance(response['raw_response'], dict) and 'similarities' in response['raw_response']:
                            similarities = response['raw_response']['similarities']
                            if idx < len(similarities):
                                sim_value = f"{similarities[idx]:.6f}"
                            else:
                                sim_value = "0.000000"

                        scores.append(score)
                        results.append(result)
                        sim_values.append(sim_value)
                        batch_success += 1
                    else:
                        scores.append(0.0)
                        results.append("API调用失败")
                        sim_values.append("")
                        batch_fail += 1
            else:
                # API调用整体失败
                for _ in range(batch_size_actual):
                    scores.append(0.0)
                    results.append("API调用批量失败")
                    sim_values.append("")
                    batch_fail += 1

            # 创建新的DataFrame以避免类型警告
            new_data = {}
            for col in df.columns:
                # 保持原始列不变
                new_data[col] = df[col].tolist()

            # 更新处理后的列
            for i, idx in enumerate(batch_indices):
                if i < len(scores):
                    df.loc[idx, '匹配分数'] = scores[i]
                    df.loc[idx, '匹配结果'] = results[i]
                    df.loc[idx, '相似度值'] = sim_values[i]

            # 单独处理结果列，确保类型正确
            new_data['匹配分数'] = df['匹配分数'].astype(float).tolist()
            new_data['匹配结果'] = df['匹配结果'].astype(str).tolist()
            new_data['相似度值'] = df['相似度值'].astype(str).tolist()

            # 创建新的DataFrame并保存
            new_df = pd.DataFrame(new_data)
            new_df.to_csv(config["output_csv_path"],
                          index=False, encoding='utf-8')

            success_count += batch_success
            fail_count += batch_fail

            # 更新进度条
            pbar.update(batch_size_actual)

            # 每处理一个小批次保存一次
            df.to_csv(config["output_csv_path"], index=False, encoding='utf-8')
            logging.info(
                f"第{batch_idx+1}批处理完成，成功:{batch_success}，失败:{batch_fail}，耗时:{batch_elapsed:.2f}秒")

            # 休眠一段时间，避免请求过于频繁
            time.sleep(config["sleep_time"])

    # 最后再保存一次确保所有数据都已保存
    df.to_csv(config["output_csv_path"], index=False, encoding='utf-8')

    logging.info(
        f"批次处理完成: {start_idx}-{real_end_idx}，总计成功: {success_count}，总计失败: {fail_count}")

    return success_count, fail_count


def process_csv_in_batches(batch_size=100):
    """
    分批处理CSV文件，并在处理过程中保存结果

    Args:
        batch_size (int): 每批处理的数据数量
    """
    try:
        # 准备CSV文件
        df = write_results_to_csv()
        total_rows = len(df)

        logging.info(f"开始分批处理CSV文件，共{total_rows}行数据，批次大小:{batch_size}")

        # 计算批次数
        num_batches = (total_rows + batch_size - 1) // batch_size

        total_success = 0
        total_fail = 0

        # 分批处理
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, total_rows)

            logging.info(
                f"开始处理第{batch+1}/{num_batches}批: {start_idx}-{end_idx}")

            success, fail = process_batch(start_idx, end_idx)
            total_success += success
            total_fail += fail

        logging.info(
            f"所有批次处理完成，共处理{total_rows}条数据，成功: {total_success}，失败: {total_fail}")

    except Exception as e:
        logging.error(f"批次处理CSV时出错: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    logging.info("开始执行房型匹配API调用")

    # 分批处理CSV数据
    process_csv_in_batches(batch_size=100)

    logging.info("执行完成")
