import os
import json
import pandas as pd
from openai import OpenAI
from typing import Dict, List, Any, Optional, Union, Tuple
from tqdm import tqdm
import time
from pathlib import Path
import threading
import queue
import math
import concurrent.futures
from functools import partial
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# 配置常量
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR)))

# 默认文件路径配置
DEFAULT_INPUT_FILE = os.path.join(
    PROJECT_ROOT, "room_match/process_data/room_group_process_40w.csv")
DEFAULT_OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "room_match/data/grouped_room_by_qwenmax.csv")
DEFAULT_LOG_FILE = os.path.join(
    PROJECT_ROOT, "room_match/data/different_rooms_log.txt")

# API限流配置
QPM_LIMIT = 1200  # 每分钟查询次数限制
TPM_LIMIT = 1000000  # 每分钟Token数限制
MAX_BATCH_SIZE = 50  # 最大批处理大小
OPTIMAL_BATCH_SIZE = 20  # 默认最优批处理大小
ESTIMATED_TOKENS_PER_DESC = 100  # 每个描述估计消耗的Token数
ESTIMATED_RESPONSE_TOKENS = 50  # 每个响应估计消耗的Token数
MAX_THREADS = 20  # 最大线程数

# 多线程配置
USE_THREADING = True  # 默认关闭多线程

# 环境变量配置
# 请替换为您的API密钥
DASHSCOPE_API_KEY = "sk-6d6b7b8eac764d08b90a817610f2f6ce"
# 如果环境变量中已存在API密钥，则优先使用环境变量中的值
if os.getenv("DASHSCOPE_API_KEY") is None:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    logger.info(f"已设置DASHSCOPE_API_KEY环境变量")
else:
    logger.info(f"使用系统环境变量中的DASHSCOPE_API_KEY")

# API请求控制
api_request_times = []
api_lock = threading.Lock()

# 用于收集包含不同组（值为1）的room_text
different_rooms_data = []

# 创建OpenAI客户端


def get_client():
    """创建并返回一个OpenAI客户端实例，便于多线程使用"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# 全局客户端，用于单线程情况
client = get_client()


def estimate_tokens(text: str) -> int:
    """
    估计文本的token数量

    Args:
        text: 输入文本

    Returns:
        int: 估计的token数量
    """
    # 简单估算：英文约为字符数/4，中文约为字符数/2
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    english_chars = len(text) - chinese_chars

    return math.ceil(chinese_chars / 2 + english_chars / 4)


def calculate_optimal_batch_size(avg_desc_len: float) -> int:
    """
    根据平均描述长度计算最优批处理大小

    Args:
        avg_desc_len: 平均描述长度

    Returns:
        int: 最优批处理大小
    """
    estimated_tokens_per_desc = max(
        ESTIMATED_TOKENS_PER_DESC, estimate_tokens("a" * int(avg_desc_len)))

    # 基于TPM计算
    batch_size_tpm = TPM_LIMIT / 60 / \
        (estimated_tokens_per_desc + ESTIMATED_RESPONSE_TOKENS)

    # 基于QPM计算
    batch_size_qpm = QPM_LIMIT / 60

    # 取较小的值，并确保在合理范围内
    optimal_size = min(batch_size_tpm, batch_size_qpm, MAX_BATCH_SIZE)
    return max(5, min(MAX_BATCH_SIZE, int(optimal_size)))


def dynamic_sleep(recent_requests: List[float]) -> float:
    """
    根据最近的请求时间动态计算需要休眠的时间

    Args:
        recent_requests: 最近的请求时间戳列表

    Returns:
        float: 建议休眠时间(秒)
    """
    with api_lock:
        # 清理超过60秒的旧请求记录
        current_time = time.time()
        recent_requests[:] = [
            t for t in recent_requests if current_time - t < 60]

        # 如果当前请求数超过限制的80%，增加休眠时间
        if len(recent_requests) > 0.8 * QPM_LIMIT:
            return max(0, (60 - (current_time - recent_requests[0])) / (QPM_LIMIT - len(recent_requests)))
        return 0.1  # 默认最小休眠时间


def record_api_request():
    """记录一次API请求"""
    with api_lock:
        current_time = time.time()
        api_request_times.append(current_time)


def evaluate_room_group_batch(room_data: List[Dict[str, str]], model: str = "qwen-max", group_id: str = None) -> List[int]:
    """
    批量评估同一组的房间数据是否真的属于同一组

    Args:
        room_data: 包含RoomName和BedTypeDesc的字典列表
        model: 要使用的模型名称，默认为qwen-max
        group_id: 当前处理的分组ID，用于记录日志

    Returns:
        List[int]: 表示每条数据是否属于同一组的列表（0表示同组，1表示不同组）
    """
    if not room_data:
        return []

    # 构建批量处理的prompt
    room_text = "\n".join(
        [f"[{i+1}]{data.get('RoomName', '')}, {data.get('BedTypeDesc', '')}"
         for i, data in enumerate(room_data)])

    prompt = f"""
你是一个专业的酒店房型分类专家，擅长判断不同房间描述是否属于同一类型。
请判断以下房间描述是否属于同一类型，返回一个数组，0表示与其他大多数房型相同，1表示与其他大多数房型不同。

以下是多条房间数据：
{room_text}

请根据这些房间的描述判断它们是否属于同一类型的房间。判断原则如下：
1. 相同的房型描述通常表示同一类型，描述上包括简写、缩写、加减号、括号等
2. 房型描述上如果有额外信息，如无障碍，阳台，海景，景观等，这些信息将判断是不同组别
3. 如果房型描述上有存在床型不一致或数量不一致，将判断是不同组别
4. 如果房型描述上有存在额外限制，如无烟房，禁烟房，无烟楼层，禁烟楼层等，这些信息将判断是不同组别
5. 如果房型描述上有存在或，or等方式连接，将判断是不同组别

请直接输出一个数组，表示每条房间数据是否与同组大多数房型是同一类型：
[0, 0, 0, ..., 1, 0]
其中0表示与同组大多数房型为同一类型，1表示与同组大多数房型不同。

示例:
输入:
[1]Deluxe Room, 1 King Bed, Non Smoking, 1 King Bed
[2]Deluxe Room, 1 King Bed, Non Smoking, Balcony, 1 King Bed

输出：
[0, 1]

原因：
-- 因为[2]比[1]多了个Balcony，所以属于不同组，返回1

输入:
[1]2 double mobility hearing accessible with tub non-smoking
[2]2dbl mob/hear acc w/tub ns, 2 double beds

输出：
[0, 0]

原因：
-- 虽然[2] 的描述为简写形式，但和[1]的描述完全相同，所以属于同一组，返回0

输入:
[1]LA Story - 1 Bed - 2 Free Cocktails Included!
[2]LA Story - 2 Beds - 2 Free Cocktails Included!

输出：
[0, 1]

原因：
-- 床型数量描述不一致，所以属于不同组，返回1

输入:
[1]2 Double Courtyard View, 2 Double
[2]Room, 2 Double Beds, Courtyard View, 2 Double Beds

输出：
[0, 0]

原因：
-- 虽然[2] 的描述比[1]多了个Room和beds，但描述的房型是相同的，所以属于同一组，返回0

请注意，数组长度应与输入的房间数据条数相同，并且数组元素顺序应与输入的房间数据顺序一致。
"""

    try:
        logger.info(f"正在评估{len(room_data)}条房间数据是否为同一组...")

        # 记录本次API请求
        record_api_request()

        # 动态休眠，避免触发限流
        sleep_time = dynamic_sleep(api_request_times)
        if sleep_time > 0.1:
            logger.info(f"为避免限流，休眠 {sleep_time:.2f}秒")
            time.sleep(sleep_time)

        # 调用大模型API
        local_client = client  # 使用全局客户端
        completion = local_client.chat.completions.create(
            model=model,
            messages=[
                {'role': 'system', 'content': '你是酒店房型分类专家，擅长判断不同房间描述是否属于同一类型。'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.1,
        )

        # 获取返回的内容
        response_text = completion.choices[0].message.content

        # 解析返回的数组
        import re
        # 找到可能的数组格式
        array_pattern = r'\[[\s\d,]+\]'
        array_match = re.search(array_pattern, response_text)

        if array_match:
            # 提取数组字符串并转换为实际数组
            array_str = array_match.group(0)
            try:
                result_array = json.loads(array_str)
                # 确保数组长度与输入数据长度一致
                if len(result_array) != len(room_data):
                    logger.warning(
                        f"返回的数组长度({len(result_array)})与输入数据长度({len(room_data)})不一致，将使用默认值")
                    result_array = [0] * len(room_data)
                logger.info(f"成功解析模型返回的分组数组: {result_array}")

                # 如果结果中包含1，将room_text和分组信息收集到不同房间数据列表中
                if 1 in result_array:
                    different_rooms_data.append({
                        "group_id": group_id,
                        "room_text": room_text,
                        "results": result_array,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    logger.info(f"发现不同组房型，已收集该组room_text")

                return result_array
            except json.JSONDecodeError:
                logger.error(f"无法解析返回的数组格式: {array_str}")
                return [0] * len(room_data)
        else:
            logger.warning("无法从响应中找到数组格式的结果，将使用默认值")
            return [0] * len(room_data)

    except Exception as e:
        logger.error(f"API调用错误: {e}")
        # 如果处理失败，所有元素默认为0（同组）
        return [0] * len(room_data)


def save_different_rooms_to_txt(output_file: str = DEFAULT_LOG_FILE) -> None:
    """
    将收集到的包含不同组(值为1)的房间数据保存到文本文件中

    Args:
        output_file: 输出文件路径
    """
    if not different_rooms_data:
        logger.info("没有发现包含不同组的房间数据")
        return

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(
                f"=== 包含不同组的房间数据日志 (生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n\n")
            f.write(f"总共发现 {len(different_rooms_data)} 组包含不同分类的房间数据\n\n")

            for i, data in enumerate(different_rooms_data):
                f.write(
                    f"--- 组 {i+1}/{len(different_rooms_data)} (ID: {data['group_id']}, 时间: {data['timestamp']}) ---\n")
                f.write(f"房间数据:\n{data['room_text']}\n\n")
                f.write(f"分组结果: {data['results']}\n")

                # 标记哪些是不同组
                room_lines = data['room_text'].split('\n')
                different_indices = [i for i, res in enumerate(
                    data['results']) if res == 1]
                for idx in different_indices:
                    if 0 <= idx < len(room_lines):
                        f.write(f"不同组: {room_lines[idx]}\n")

                f.write("\n" + "-" * 80 + "\n\n")

        logger.info(f"已将包含不同组的房间数据保存到文件: {output_file}")
    except Exception as e:
        logger.error(f"保存包含不同组的房间数据到文件时出错: {e}")


def process_batch_chunk(chunk_data: Tuple[str, List[Dict[str, str]], str]) -> Tuple[str, List[int]]:
    """
    处理一个数据块，用于并行处理

    Args:
        chunk_data: (分组标识, 房间数据列表, 模型名称)的元组

    Returns:
        Tuple[str, List[int]]: (分组标识, 结果列表)的元组
    """
    group_id, room_data, model = chunk_data
    try:
        # 为每个线程创建自己的客户端
        thread_client = get_client()
        # 保存原始客户端
        global client
        original_client = client
        # 设置线程本地客户端
        client = thread_client

        results = evaluate_room_group_batch(room_data, model, group_id)

        # 恢复原始客户端
        client = original_client
        return group_id, results
    except Exception as e:
        logger.error(f"处理数据块时出错: {e}")
        # 恢复原始客户端
        client = original_client
        # 返回全0数组（表示都是同组）
        return group_id, [0] * len(room_data)


def process_csv_file(input_file: str, output_file: str, model: str = "qwen-max", batch_size: int = None, use_threading: bool = USE_THREADING, log_file: str = DEFAULT_LOG_FILE) -> None:
    """
    处理CSV文件中的房间数据，按SHotelId、InitialGroupId和groupdf分组，判断各组内的数据是否为同一类型

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        model: 要使用的模型名称
        batch_size: 每批处理的数据量，如果为None则自动计算
        use_threading: 是否使用多线程处理
        log_file: 包含不同组的房间数据日志文件路径
    """
    # 确保输入文件存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")

    # 清空不同房间数据列表
    global different_rooms_data
    different_rooms_data = []

    start_time = time.time()
    logger.info(f"开始处理CSV文件: {input_file}")

    # 读取CSV文件
    df = pd.read_csv(input_file)
    # 仅在测试时处理部分数据
    if "QWEN_TEST_MODE" in os.environ:
        test_rows = int(os.environ.get("QWEN_TEST_ROWS", "100"))
        df = df.head(test_rows)
        logger.info(f"测试模式：仅处理前{test_rows}行数据")

    original_cols = len(df.columns)
    total_rows = len(df)

    logger.info(f"读取完成，共{total_rows}行数据")

    # 确保必要的列存在
    required_columns = ["SHotelId", "InitialGroupId",
                        "groupdf", "RoomName", "BedTypeDesc"]
    missing_columns = [
        col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"CSV文件缺少必要的列: {', '.join(missing_columns)}")

    # 按SHotelId、InitialGroupId和groupdf三级分组
    grouped_data = {}
    group_indices = {}

    for i, row in df.iterrows():
        # 跳过groupdf为1或空的行
        if pd.isna(row["groupdf"]) or row["groupdf"] == 1:
            continue

        hotel_id = row["SHotelId"]
        group_id = row["InitialGroupId"]
        group_df = row["groupdf"]
        key = f"{hotel_id}_{group_id}_{group_df}"

        if key not in grouped_data:
            grouped_data[key] = []
            group_indices[key] = []

        grouped_data[key].append({
            "RoomName": row["RoomName"] if not pd.isna(row["RoomName"]) else "",
            "BedTypeDesc": row["BedTypeDesc"] if not pd.isna(row["BedTypeDesc"]) else ""
        })
        group_indices[key].append(i)

    # 添加modeldf列，默认为空
    df["modeldf"] = pd.NA

    # 准备处理数据
    all_tasks = []
    for key, room_data in grouped_data.items():
        if len(room_data) <= 1:
            continue  # 跳过只有一条数据的组
        all_tasks.append((key, room_data, model))

    # 处理所有分组
    logger.info(f"开始处理{len(all_tasks)}个房型分组")

    # 处理方式：多线程或单线程
    if use_threading and len(all_tasks) > 1:
        logger.info(f"使用并行处理，最多{min(MAX_THREADS, len(all_tasks))}个线程")

        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(all_tasks))) as executor:
            # 提交所有任务
            future_to_group = {executor.submit(
                process_batch_chunk, task): task[0] for task in all_tasks}

            # 使用tqdm显示进度
            with tqdm(total=len(all_tasks), desc="处理房型分组") as pbar:
                for future in concurrent.futures.as_completed(future_to_group):
                    group_key = future_to_group[future]
                    try:
                        _, results = future.result()
                        # 将结果写入DataFrame
                        for idx, result in zip(group_indices[group_key], results):
                            df.at[idx, "modeldf"] = result
                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"处理组 {group_key} 时出错: {e}")
    else:
        logger.info(f"使用单线程处理")
        # 单线程处理
        with tqdm(total=len(all_tasks), desc="处理房型分组") as pbar:
            for task in all_tasks:
                group_key, room_data, model_name = task
                try:
                    results = evaluate_room_group_batch(
                        room_data, model_name, group_key)
                    # 将结果写入DataFrame
                    for idx, result in zip(group_indices[group_key], results):
                        df.at[idx, "modeldf"] = result
                except Exception as e:
                    logger.error(f"处理组 {group_key} 时出错: {e}")
                pbar.update(1)

    # 保存结果
    # 调整列顺序，确保modeldf列在groupdf列后面
    if "modeldf" in df.columns and "groupdf" in df.columns:
        columns = list(df.columns)
        # 删除modeldf列
        columns.remove("modeldf")
        # 找到groupdf列的索引位置
        groupdf_idx = columns.index("groupdf")
        # 在groupdf列后面插入modeldf列
        columns.insert(groupdf_idx + 1, "modeldf")
        # 重新排列DataFrame的列顺序
        df = df[columns]
        logger.info("已调整列顺序，将modeldf列放置在groupdf列后面")

    df.to_csv(output_file, index=False)

    # 保存包含不同组的房间数据
    save_different_rooms_to_txt(log_file)

    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f"\n处理完成！")
    logger.info(f"总处理时间: {processing_time:.2f}秒")
    logger.info(f"处理速度: {total_rows/processing_time:.2f}行/秒")
    logger.info(f"输入文件: {input_file} ({total_rows}行, {original_cols}列)")
    logger.info(f"输出文件: {output_file} ({total_rows}行, {len(df.columns)}列)")
    logger.info(f"新增列: modeldf")

    if different_rooms_data:
        logger.info(f"不同组房间数据日志文件: {log_file} (共{len(different_rooms_data)}组)")

    # 统计处理结果
    processed_rows = df["modeldf"].notna().sum()
    success_rate = processed_rows / total_rows * 100
    logger.info(f"处理率: {success_rate:.2f}% ({processed_rows}/{total_rows})")

    # 统计标记为不同组的比例
    if processed_rows > 0:
        different_group_rows = df[df["modeldf"] == 1].shape[0]
        different_group_rate = different_group_rows / processed_rows * 100
        logger.info(
            f"标记为非同组的比例: {different_group_rate:.2f}% ({different_group_rows}/{processed_rows})")


# 测试示例
if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用大模型批量评估CSV文件中的房型分组")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_FILE,
                        help=f"输入CSV文件路径 (默认: {DEFAULT_INPUT_FILE})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_FILE,
                        help=f"输出CSV文件路径 (默认: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--model", "-m", default="qwen-max",
                        help="使用的模型名称 (默认: qwen-max)")
    parser.add_argument("--batch", "-b", type=int, default=None,
                        help="每批处理的数据量 (默认: 自动计算)")
    parser.add_argument("--threads", "-th", action="store_true", default=True,
                        help="是否使用多线程处理 (默认: False)")
    parser.add_argument("--log", "-l", default=DEFAULT_LOG_FILE,
                        help=f"包含不同组的房间数据日志文件路径 (默认: {DEFAULT_LOG_FILE})")
    parser.add_argument("--test", "-t", action="store_true",
                        help="仅测试模式，处理几个样本")

    args = parser.parse_args()

    # 显示配置信息
    if not args.test:
        logger.info("===== 运行配置 =====")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")
        logger.info(f"日志文件: {args.log}")
        logger.info(f"使用模型: {args.model}")
        logger.info(f"批处理量: {'自动计算' if args.batch is None else args.batch}")
        logger.info(f"多线程: {'是' if args.threads else '否'}")
        logger.info("===================\n")

        # 完整处理模式
        process_csv_file(args.input, args.output, args.model,
                         args.batch, args.threads, args.log)
