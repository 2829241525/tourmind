import pandas as pd
import requests
import json
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any
import logging
import re
import os
import traceback
import time


###########################
#正样本信息，通过模型匹配过滤
###########################

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 配置信息
CONFIG = {
    'API_URL': 'http://10.0.0.110:8100/match',
    'HEADERS': {
        'User-Agent': 'Mozilla/5.0',
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'Connection': 'keep-alive'
    },
    'BATCH_SIZE': 100,  # 减小批处理大小
    'SIMILARITY_THRESHOLD': 0.65,  # 相似度阈值
    'INPUT_FILE': '../data/supply_country_info.csv',
    'OUTPUT_FILE': '../data/positive_country.csv',
    'UNMATCH_FILE': '../data/unmatch_country.csv',  # 新增未匹配数据输出文件
    # 'INPUT_FILE': 'room_match/data/supply_system_info.csv',
    # 'OUTPUT_FILE': 'room_match/data/positive_system.csv',
    'MAX_RETRIES': 3,  # 最大重试次数
    'RETRY_DELAY': 1,  # 重试延迟（秒）
}

def clean_float_for_json(value):
    """
    处理浮点数，确保其可以被JSON序列化
    
    Args:
        value: 任意值
        
    Returns:
        处理后的值，确保可以被JSON序列化
    """
    if pd.isna(value):
        return None
    if isinstance(value, (float, np.float64, np.float32)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)
    return str(value) if value is not None else None

def prepare_batch_data(batch_rows: pd.DataFrame) -> Dict[str, List[str]]:
    """
    准备批量请求的数据
    
    Args:
        batch_rows: 包含一批数据的DataFrame
        
    Returns:
        Dict: 包含处理后的数据的字典
        
    Raises:
        ValueError: 当数据格式不正确或缺少必需的列时
    """
    # 检查必需的列
    required_columns = ['supplier_name', 'spl_room_name', 'spl_room_bed_name', 
                       's_room_name', 's_room_bed_name']
    
    missing_columns = [col for col in required_columns if col not in batch_rows.columns]
    if missing_columns:
        raise ValueError(f"数据中缺少必需的列: {missing_columns}")
    
    # 初始化数据结构
    processed_data = {
        "supplier_names": [],
        "spl_room_names": [],
        "spl_room_bed_names": [],
        "s_room_names": [],
        "s_room_bed_names": [],
        "threshold": CONFIG['SIMILARITY_THRESHOLD']
    }
    
    # 处理每一行数据
    for _, row in batch_rows.iterrows():
        try:
            # 清理和验证每个字段
            supplier_name = clean_text(row['supplier_name'])
            spl_room_name = clean_text(row['spl_room_name'])
            spl_room_bed_name = clean_text(row['spl_room_bed_name'])
            s_room_name = clean_text(row['s_room_name'])
            s_room_bed_name = clean_text(row['s_room_bed_name'])
            
            # 验证必需字段不为空
            if not spl_room_name or not s_room_name:
                logging.warning(f"行数据包含空值: supplier_name={supplier_name}, spl_room_name={spl_room_name}, s_room_name={s_room_name}")
                continue
            
            # 添加到处理后的数据中
            processed_data["supplier_names"].append(supplier_name or "")
            processed_data["spl_room_names"].append(spl_room_name or "")
            processed_data["spl_room_bed_names"].append(spl_room_bed_name or "")
            processed_data["s_room_names"].append(s_room_name or "")
            processed_data["s_room_bed_names"].append(s_room_bed_name or "")
            
        except Exception as e:
            logging.error(f"处理行数据时出错: {str(e)}")
            continue
    
    # 验证处理后的数据
    if not processed_data["supplier_names"]:
        raise ValueError("处理后的数据为空")
    
    # 确保所有列表长度一致
    list_lengths = [len(v) for k, v in processed_data.items() if k != "threshold"]
    if len(set(list_lengths)) > 1:
        raise ValueError("处理后的数据列表长度不一致")
    
    return processed_data

def clean_text(text) -> str:
    """
    清理文本数据
    
    Args:
        text: 输入文本
        
    Returns:
        str: 清理后的文本
    """
    if pd.isna(text) or text is None:
        return ""
    
    # 转换为字符串
    text = str(text).strip()
    
    # 移除特殊字符
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # 替换多个空格为单个空格
    text = re.sub(r'\s+', ' ', text)
    
    return text

def process_batch(batch_rows: pd.DataFrame) -> List[pd.Series]:
    """
    处理一批数据
    
    Args:
        batch_rows: 包含一批数据的DataFrame
        
    Returns:
        List[pd.Series]: 匹配成功的数据行列表
    """
    retries = 0
    while retries < CONFIG['MAX_RETRIES']:
        try:
            # 数据预处理和验证
            if batch_rows.empty:
                logging.warning("批次数据为空")
                return []
                
            # 准备数据
            try:
                data = prepare_batch_data(batch_rows)
            except ValueError as e:
                logging.error(f"数据准备失败: {str(e)}")
                return []
            except Exception as e:
                logging.error(f"数据准备过程中发生未知错误: {str(e)}")
                return []
                
            # 检查数据是否为空或无效
            if not data or not any(data.values()):
                logging.warning("批次数据为空或无效")
                return []
                
            # 记录发送的数据
            logging.debug(f"发送到API的数据: {json.dumps(data, ensure_ascii=False)}")
            
            # 发送API请求
            try:
                response = requests.post(
                    CONFIG['API_URL'],
                    headers=CONFIG['HEADERS'],
                    json=data,
                    timeout=1000  # 设置超时时间
                )
                
                # 检查响应状态码
                if response.status_code != 200:
                    logging.error(f"API请求失败，状态码: {response.status_code}")
                    if response.status_code == 422:
                        logging.error(f"请求数据格式错误，请求数据: {json.dumps(data, ensure_ascii=False)}")
                        logging.error(f"响应内容: {response.text}")
                    retries += 1
                    if retries < CONFIG['MAX_RETRIES']:
                        time.sleep(CONFIG['RETRY_DELAY'])
                        continue
                    return []
                    
                # 解析响应数据
                try:
                    response_data = response.json()
                except json.JSONDecodeError as e:
                    logging.error(f"API响应解析失败: {str(e)}")
                    retries += 1
                    if retries < CONFIG['MAX_RETRIES']:
                        time.sleep(CONFIG['RETRY_DELAY'])
                        continue
                    return []
                    
                # 处理匹配结果
                matching_results = []
                unmatched_data = []  # 存储未匹配的数据
                
                if 'similarities' in response_data and 'predictions' in response_data:
                    similarities = response_data['similarities']
                    predictions = response_data['predictions']
                    
                    # 确保数据长度匹配
                    if len(similarities) != len(batch_rows) or len(predictions) != len(batch_rows):
                        logging.error("API返回的数据长度与请求不匹配")
                        retries += 1
                        if retries < CONFIG['MAX_RETRIES']:
                            time.sleep(CONFIG['RETRY_DELAY'])
                            continue
                        return []
                    
                    # 处理每一行数据
                    for idx, (similarity, prediction, row) in enumerate(zip(similarities, predictions, batch_rows.itertuples())):
                        if prediction == 1 and similarity >= CONFIG['SIMILARITY_THRESHOLD']:
                            matching_results.append(row)
                            logging.debug(f"找到匹配: similarity={similarity}, prediction={prediction}")
                        elif prediction == 0:
                            unmatched_data.append({
                                'supplier_name': row.supplier_name,
                                'spl_room_name': row.spl_room_name,
                                's_room_name': row.s_room_name,
                                'similarity': similarity
                            })
                    
                    # 打印未匹配数据的统计信息
                    if unmatched_data:
                        logging.info(f"\n本批次未匹配数据统计 (共{len(unmatched_data)}条):")
                        for idx, data in enumerate(unmatched_data, 1):
                            logging.info(f"{idx}. 供应商: {data['supplier_name']}")
                            logging.info(f"   供应商房型: {data['spl_room_name']}")
                            logging.info(f"   系统房型: {data['s_room_name']}")
                            logging.info(f"   相似度: {data['similarity']:.4f}")
                            logging.info("-" * 50)
                        
                        # 保存未匹配数据到CSV
                        unmatch_df = pd.DataFrame(unmatched_data)
                        mode = 'w' if not os.path.exists(CONFIG['UNMATCH_FILE']) else 'a'
                        header = not os.path.exists(CONFIG['UNMATCH_FILE'])
                        unmatch_df.to_csv(CONFIG['UNMATCH_FILE'], mode=mode, header=header, index=False)
                        logging.info(f"已保存 {len(unmatched_data)} 条未匹配数据到文件 {CONFIG['UNMATCH_FILE']}")
                    
                    return matching_results
                else:
                    logging.error("API响应数据格式不正确")
                    logging.error(f"响应数据: {json.dumps(response_data, ensure_ascii=False)}")
                    retries += 1
                    if retries < CONFIG['MAX_RETRIES']:
                        time.sleep(CONFIG['RETRY_DELAY'])
                        continue
                    return []
                
            except requests.exceptions.Timeout:
                logging.error("API请求超时")
                retries += 1
                if retries < CONFIG['MAX_RETRIES']:
                    time.sleep(CONFIG['RETRY_DELAY'])
                    continue
                return []
            except requests.exceptions.RequestException as e:
                logging.error(f"API请求异常: {str(e)}")
                retries += 1
                if retries < CONFIG['MAX_RETRIES']:
                    time.sleep(CONFIG['RETRY_DELAY'])
                    continue
                return []
            except Exception as e:
                logging.error(f"处理API响应时发生未知错误: {str(e)}")
                retries += 1
                if retries < CONFIG['MAX_RETRIES']:
                    time.sleep(CONFIG['RETRY_DELAY'])
                    continue
                return []
                
        except Exception as e:
            logging.error(f"批处理过程中发生未知错误: {str(e)}")
            retries += 1
            if retries < CONFIG['MAX_RETRIES']:
                time.sleep(CONFIG['RETRY_DELAY'])
                continue
            return []
            
    return []  # 所有重试都失败后返回空列表

def process_artificial_data():
    """
    处理人工数据的主函数
    """
    try:
        # 读取数据
        logging.info("开始读取数据...")
        df = pd.read_csv(CONFIG['INPUT_FILE'])

        # 去除spl_room_name和s_room_name为空的数据
        df = df[df['spl_room_name'].notna() & df['s_room_name'].notna()]

        total_rows = len(df)
        logging.info(f"总共读取 {total_rows} 条数据")
        
        # 初始化计数器和结果列表
        processed_rows = 0
        error_count = 0
        success_count = 0
        all_results = []
        
        # 创建进度条
        with tqdm(total=total_rows, desc=f"处理中 [成功: {success_count}, 错误: {error_count}]") as pbar:
            # 按批次处理数据
            for start_idx in range(0, total_rows, CONFIG['BATCH_SIZE']):
                try:
                    # 获取当前批次
                    end_idx = min(start_idx + CONFIG['BATCH_SIZE'], total_rows)
                    batch_df = df.iloc[start_idx:end_idx].copy()
                    
                    # 处理批次数据
                    batch_results = process_batch(batch_df)
                    
                    # 更新计数器
                    processed_count = len(batch_df)
                    if batch_results:
                        success_count += len(batch_results)
                        all_results.extend(batch_results)
                    else:
                        error_count += processed_count
                    
                    processed_rows += processed_count
                    
                    # 更新进度条
                    pbar.set_description(f"处理中 [成功: {success_count}, 错误: {error_count}]")
                    pbar.update(processed_count)
                    
                    # 定期保存结果
                    if len(all_results) >= 1000 or end_idx == total_rows:
                        if all_results:
                            result_df = pd.DataFrame(all_results)
                            mode = 'w' if not os.path.exists(CONFIG['OUTPUT_FILE']) else 'a'
                            header = not os.path.exists(CONFIG['OUTPUT_FILE'])
                            result_df.to_csv(CONFIG['OUTPUT_FILE'], mode=mode, header=header, index=False)
                            logging.info(f"已保存 {len(all_results)} 条结果到文件")
                            all_results = []
                    
                except Exception as e:
                    error_count += len(batch_df)
                    logging.error(f"处理批次数据时出错 [start_idx={start_idx}]: {str(e)}")
                    continue
        
        # 打印最终统计信息
        logging.info("\n处理完成:")
        logging.info(f"总数据量: {total_rows:,} 条")
        logging.info(f"成功处理: {success_count:,} 条")
        logging.info(f"处理失败: {error_count:,} 条")
        logging.info(f"匹配数据: {success_count:,} 条")
        
    except Exception as e:
        logging.error(f"处理过程中发生错误: {str(e)}")
        logging.error(traceback.format_exc())

if __name__ == "__main__":
    process_artificial_data() 