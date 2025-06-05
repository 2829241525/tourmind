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
    PROJECT_ROOT, "room_match/data/bed_model.csv")
DEFAULT_OUTPUT_FILE = os.path.join(
    PROJECT_ROOT, "room_match/data/bed_all_data_process_by_qwen_max.csv")

# API限流配置
QPM_LIMIT = 1200  # 每分钟查询次数限制
TPM_LIMIT = 1000000  # 每分钟Token数限制
MAX_BATCH_SIZE = 50  # 最大批处理大小
OPTIMAL_BATCH_SIZE = 20  # 默认最优批处理大小
ESTIMATED_TOKENS_PER_DESC = 100  # 每个描述估计消耗的Token数
ESTIMATED_RESPONSE_TOKENS = 50  # 每个响应估计消耗的Token数
MAX_THREADS = 10  # 最大线程数

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


def extract_room_info_batch(room_descriptions: List[str], model: str = "qwen-max") -> List[Dict[str, str]]:
    """
    批量处理多个房间描述，一次API调用处理多条数据

    Args:
        room_descriptions: 多个房间描述文本列表
        model: 要使用的模型名称，默认为qwen-max

    Returns:
        List[Dict[str, str]]: 包含每个房间描述提取的床型信息和推断原因的字典列表
    """
    if not room_descriptions:
        return []

    # 过滤空值
    valid_descriptions = []
    empty_indices = []

    for i, desc in enumerate(room_descriptions):
        if pd.isna(desc) or not desc:
            empty_indices.append(i)
        else:
            valid_descriptions.append((i, desc))

    if not valid_descriptions:
        return [{"bed_info": "[unknown]", "reason": "输入为空"}] * len(room_descriptions)

    # 构建批量处理的prompt
    descriptions_text = "\n".join(
        [f"[{i+1}] {desc}" for i, (_, desc) in enumerate(valid_descriptions)])

    prompt = f"""
你是一个专业的酒店房型信息抽取专家，擅长从各种语言（包括英文、中文等）的酒店房间描述中提取床型信息。请从以下多条文本中提取标准化的床型信息：

请只提取每条描述中的床型信息（不需要房型或其他信息），结果格式要求为：
[索引号] 提取的床型描述 ||| 推断原因

请按照以下规则提取床型信息:

***抽取的床型，必须包含bed字符，否则直接输出[unknown]***

1. 输入文本必须包含输出文本的每个词汇，且前后顺序一致,并且提取的信息以bed或beds结尾。

2. 床型表达方式多样:
   - 连字符形式: "one-queen-size-bed", "king-size-bed"
   - 数字前缀: "2double beds", "1king bed"
   - 常规形式: "one queen bed", "two twin beds"
   - 逗号分隔: "one king, one sofa bed"
   - 存在连接词汇: "one king and one double bed", "one king or one double bed","one king with sofa bed"

3.数量词可包含："one", "two", "three", "four", "five", "six", "double", "single", "twin", "queen", "king", "full", "triple" 等

4.床型类型词可包含："twin", "queen", "king", "double", "single", "full", "triple", "bunk" 等

5. 注意事项:
   - 忽略包含"room"或"view"的描述部分，除非它是床型的一部分
   - 抽取的内容不包含"room"或"view"。
   - sofa bed和extra-bed属于附带的床型，一般不包含整个房间的床型信息，所以不提取
   - 如果无法提取信息，回答"[unknown]"


示例:
输入文本列表:
[1] royal queen suite with extra-bed
[2] Suite with 1 queen bed and 1 sofa bed
[3] king deluxe with sofa bed
[4] standard room, one or two beds
[5] double executive two bed executive room nonsmoking
[6] double sea-view (double-one double bed, premium room, sea-view)
[7] deluxe -double bed room
[8] superior suite,multiple beds
[9] double deluxe bed and breakfast  -double deluxe
[10] double bed room
[11] executive twin-room twinbed

输出:
[1] [unknown] ||| extra-bed属于附带，不符合
[2] 1 queen bed and 1 sofa bed ||| sofa bed属于附带，但有and连接，包含整体的床型信息，符合要求
[3] [unknown] ||| sofa bed属于附带，不符合
[4] one or two beds ||| or连接，符合要求
[5] two bed ||| 符合要求
[6] double bed ||| 括号包含的也符合要求
[7] double bed ||| 虽然double前有-，但也符合要求
[8] multiple beds ||| 符合要求
[9] double deluxe bed ||| 符合要求
[10] double bed ||| 符合要求
[11] twinbed ||| twinbed虽然相连，但依旧以bed结尾，符合要求

再次强调：输出的字符串，在传入的对应字符串中有对应的词，包括bed,beds，且顺序一致,并且提取的信息以bed或beds结尾。
请直接输出提取结果和推断原因，不要有任何额外的解释或说明。对于无效或空的输入，请回答"[unknown] ||| 无法从输入中提取有效床型信息"。

输入房型信息:
{descriptions_text}

"""

    try:
        logger.info(f"正在批量处理{len(valid_descriptions)}条房间描述...")

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
                {'role': 'system', 'content': '你是酒店房型信息抽取专家，擅长提取床型信息。'},
                {'role': 'user', 'content': prompt}
            ],
            temperature=0.1,
        )

        # 获取返回的内容
        response_text = completion.choices[0].message.content

        # 解析提取结果
        final_results = [{"bed_info": "[unknown]",
                          "reason": "处理失败"}] * len(room_descriptions)

        # 使用正则表达式提取索引、床型信息和推断原因
        import re
        info_pattern = r'\[(\d+)\]\s+(.+?)\s*\|\|\|\s*(.*?)(?=\n\[\d+\]|\Z)'
        matches = re.finditer(info_pattern, response_text, re.DOTALL)

        for match in matches:
            try:
                idx = int(match.group(1))
                bed_info = match.group(2).strip()
                reason = match.group(3).strip()

                if 1 <= idx <= len(valid_descriptions):
                    original_idx = valid_descriptions[idx-1][0]
                    final_results[original_idx] = {
                        "bed_info": bed_info, "reason": reason}
            except Exception as e:
                logger.error(f"解析结果时出错: {e}")
                continue

        logger.info(f"成功提取{len(valid_descriptions)}条床型信息及推断原因")
        return final_results

    except Exception as e:
        logger.error(f"批量API调用错误: {e}")
        # 如果批量处理失败，返回错误信息
        return [{"bed_info": "[unknown]", "reason": f"API调用错误: {str(e)}"}] * len(room_descriptions)


def process_batch_chunk(chunk_data: Tuple[int, List[str], str]) -> Tuple[int, List[Dict[str, str]]]:
    """
    处理一个数据块，用于并行处理

    Args:
        chunk_data: (起始索引, 房间描述列表, 模型名称)的元组

    Returns:
        Tuple[int, List[Dict[str, str]]]: (起始索引, 结果列表)的元组
    """
    chunk_start, chunk, model = chunk_data
    try:
        # 为每个线程创建自己的客户端
        thread_client = get_client()
        # 保存原始客户端
        global client
        original_client = client
        # 设置线程本地客户端
        client = thread_client

        results = extract_room_info_batch(chunk, model)

        # 恢复原始客户端
        client = original_client
        return chunk_start, results
    except Exception as e:
        logger.error(f"处理数据块时出错: {e}")
        # 恢复原始客户端
        client = original_client
        # 返回空结果
        return chunk_start, [{"bed_info": "[unknown]", "reason": f"处理错误: {str(e)}"}] * len(chunk)


def process_batch(room_descriptions: List[str], model: str = "qwen-max", batch_size: int = None, use_threading: bool = True) -> List[Dict[str, str]]:
    """
    批量处理多个房间描述，使用批量API调用来提高效率

    Args:
        room_descriptions: 房间描述文本列表
        model: 要使用的模型名称
        batch_size: 每次处理的批次大小，如果为None则自动计算
        use_threading: 是否使用多线程处理

    Returns:
        List[Dict[str, str]]: 包含提取的床型信息和推断原因的字典列表
    """
    num_descriptions = len(room_descriptions)

    # 计算平均描述长度
    avg_desc_len = sum(len(str(desc)) for desc in room_descriptions if desc) / \
        max(1, sum(1 for desc in room_descriptions if desc))
    logger.info(f"平均描述长度: {avg_desc_len:.2f}字符")

    # 自动计算最优批处理大小
    if batch_size is None:
        batch_size = calculate_optimal_batch_size(avg_desc_len)
        logger.info(f"自动计算的最优批处理大小: {batch_size}")

    # 如果描述数量少于批处理大小，直接处理
    if num_descriptions <= batch_size:
        logger.info(f"一次性处理所有{num_descriptions}条房间描述...")
        return extract_room_info_batch(room_descriptions, model)

    # 准备结果容器和进度条
    final_results = [None] * num_descriptions

    # 将数据分块
    chunks = []
    for i in range(0, num_descriptions, batch_size):
        chunk = room_descriptions[i:i+batch_size]
        chunks.append((i, chunk, model))

    # 处理方式：多线程或单线程
    if use_threading and len(chunks) > 1:
        logger.info(
            f"使用并行处理，{len(chunks)}个数据块，最多{min(MAX_THREADS, len(chunks))}个线程")

        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(MAX_THREADS, len(chunks))) as executor:
            # 提交所有任务
            future_to_chunk = {executor.submit(
                process_batch_chunk, chunk_data): chunk_data[0] for chunk_data in chunks}

            # 使用tqdm显示进度
            with tqdm(total=num_descriptions, desc="批量处理房间描述") as pbar:
                for future in concurrent.futures.as_completed(future_to_chunk):
                    try:
                        chunk_start, chunk_results = future.result()
                        # 将结果放入正确的位置
                        for i, result in enumerate(chunk_results):
                            if chunk_start + i < num_descriptions:
                                final_results[chunk_start + i] = result
                        pbar.update(len(chunk_results))
                    except Exception as e:
                        logger.error(f"处理结果时出错: {e}")
    else:
        logger.info(f"使用单线程处理，{len(chunks)}个数据块")
        # 单线程处理
        with tqdm(total=num_descriptions, desc="批量处理房间描述") as pbar:
            for chunk_data in chunks:
                chunk_start, chunk_results = process_batch_chunk(chunk_data)
                # 将结果放入正确的位置
                for i, result in enumerate(chunk_results):
                    if chunk_start + i < num_descriptions:
                        final_results[chunk_start + i] = result
                pbar.update(len(chunk_results))

    # 确保没有None值
    for i in range(len(final_results)):
        if final_results[i] is None:
            final_results[i] = {"bed_info": "[unknown]", "reason": "处理失败"}

    return final_results


def process_csv_file(input_file: str, output_file: str, room_desc_column: str, model: str = "qwen-max", batch_size: int = None, use_threading: bool = True) -> None:
    """
    处理CSV文件中的房间描述并添加模型提取的床型信息

    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        room_desc_column: 包含房间描述的列名
        model: 要使用的模型名称
        batch_size: 每批处理的数据量，如果为None则自动计算
        use_threading: 是否使用多线程处理
    """
    # 确保输入文件存在
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"输入文件不存在: {input_file}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"创建输出目录: {output_dir}")

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

    # 获取房间描述列
    if room_desc_column not in df.columns:
        raise ValueError(f"列 '{room_desc_column}' 不存在于CSV文件中")

    room_descriptions = df[room_desc_column].tolist()

    # 批量处理房间描述
    logger.info(
        f"开始批量处理房间描述，{'自动计算' if batch_size is None else batch_size}批次大小")
    results = process_batch(room_descriptions, model,
                            batch_size, use_threading)

    # 添加提取的床型信息和推断原因列
    df["model_extract_bed"] = [result["bed_info"] for result in results]
    df["model_extract_reason"] = [result["reason"] for result in results]

    # 保存结果
    df.to_csv(output_file, index=False)

    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f"\n处理完成！")
    logger.info(f"总处理时间: {processing_time:.2f}秒")
    logger.info(f"处理速度: {total_rows/processing_time:.2f}行/秒")
    logger.info(f"输入文件: {input_file} ({total_rows}行, {original_cols}列)")
    logger.info(f"输出文件: {output_file} ({total_rows}行, {len(df.columns)}列)")
    logger.info(f"新增列: model_extract_bed, model_extract_reason")

    # 统计处理结果
    unknown_count = sum(
        1 for result in results if result["bed_info"] == "[unknown]")
    success_rate = (total_rows - unknown_count) / \
        total_rows * 100 if total_rows > 0 else 0
    logger.info(
        f"成功率: {success_rate:.2f}% ({total_rows - unknown_count}/{total_rows})")


# 测试示例
if __name__ == "__main__":
    import argparse

    # 命令行参数解析
    parser = argparse.ArgumentParser(description="使用大模型批量提取CSV文件中的床型信息")
    parser.add_argument("--input", "-i", default=DEFAULT_INPUT_FILE,
                        help=f"输入CSV文件路径 (默认: {DEFAULT_INPUT_FILE})")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT_FILE,
                        help=f"输出CSV文件路径 (默认: {DEFAULT_OUTPUT_FILE})")
    parser.add_argument("--column", "-c", default="spl_cleaned_room",
                        help="包含房间描述的列名 (默认: spl_cleaned_room)")
    parser.add_argument("--model", "-m", default="qwen-max",
                        help="使用的模型名称 (默认: qwen-max)")
    parser.add_argument("--batch", "-b", type=int, default=None,
                        help="每批处理的数据量 (默认: 自动计算)")
    parser.add_argument("--threads", "-th", action="store_true", default=True,
                        help="是否使用多线程处理 (默认: True)")
    parser.add_argument("--test", "-t", action="store_true",
                        help="仅测试模式，处理几个样本")

    args = parser.parse_args()

    # 显示配置信息
    if not args.test:
        logger.info("===== 运行配置 =====")
        logger.info(f"输入文件: {args.input}")
        logger.info(f"输出文件: {args.output}")
        logger.info(f"描述列名: {args.column}")
        logger.info(f"使用模型: {args.model}")
        logger.info(f"批处理量: {'自动计算' if args.batch is None else args.batch}")
        logger.info(f"多线程: {'是' if args.threads else '否'}")
        logger.info("===================\n")

    if args.test:
        # 测试模式，只处理几个样本
        test_descriptions = [
            "Deluxe Room with one king-size bed and City View",
            "Standard two queen beds mountain view",
            "Suite with 1 king-bed and sofa-bed",
            "superior double-room with river-view (two twin beds or one-double-size-bed)",
            "superior room with two-queen-size-beds",
            "suite, one-king-size-bed with sofa bed, balcony, city-view, one-king-size-bed and one double sofa bed",
            "suite, one-king-size-bed with sofa bed, nonsmoking (one-king-size-bed and one double sofa bed) - package rate",
            "room  one-king-size-bed with sofa bed  balcony  city-view (balcony)",
            "family double-room, one-king-size-bed with sofa bed (one-king-size-bed and one double sofa bed)",
            "family double-room, one-king-size-bed with sofa bed(one-king-size-bed and one double sofa bed)",
            "junior suite double-room - with breakfast (one-double-size-bed)",
            "executive suite, one-king-size-bed with sofa bed, nonsmoking, balcony (balcony) one-king-size-bed and one double sofa bed",
            "deluxe room with one-king-size-bed, pool-view ( beachs ide)",
            "royal queen suite with extra-bed",
            "king deluxe with sofa bed",
            "standard room, one or two beds",
            "double executive two bed executive room nonsmoking",
            "double sea-view (double-one double bed, premium room, sea-view)"
        ]

        logger.info("===== 批量测试模式 =====")
        batch_size_test = args.batch or calculate_optimal_batch_size(
            sum(len(desc)
                for desc in test_descriptions) / len(test_descriptions)
        )
        logger.info(f"测试批量处理，批次大小: {batch_size_test}")

        start_time = time.time()
        results = process_batch(
            test_descriptions, args.model, batch_size_test, args.threads)
        end_time = time.time()

        logger.info(f"\n批量处理耗时: {end_time - start_time:.2f}秒")
        logger.info(
            f"平均每条处理耗时: {(end_time - start_time)/len(test_descriptions):.2f}秒")

        for i, result in enumerate(results):
            logger.info(f"\n示例 {i+1}: {test_descriptions[i]}")
            logger.info(f"提取的床型: {result['bed_info']}")
            logger.info(f"推断原因: {result['reason']}")
    else:
        # 完整处理模式
        process_csv_file(args.input, args.output,
                         args.column, args.model, args.batch, args.threads)
