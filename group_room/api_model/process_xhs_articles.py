import os
import json
import time
from openai import OpenAI
from typing import Dict, List, Any, Optional, Tuple, Set
from tqdm import tqdm
import threading
import logging
import concurrent.futures
import math

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
DEFAULT_INPUT_FILE = os.path.join(CURRENT_DIR, "xhs_artical.json")
DEFAULT_OUTPUT_FILE = os.path.join(CURRENT_DIR, "processed_xhs_articles.jsonl")

# API限流配置
QPM_LIMIT = 1200  # 每分钟查询次数限制
TPM_LIMIT = 1000000  # 每分钟Token数限制
MAX_BATCH_SIZE = 10  # 最大批处理大小
MAX_THREADS = 10  # 最大线程数

# 多线程配置
USE_THREADING = True  # 默认开启多线程

# 环境变量配置
DASHSCOPE_API_KEY = "sk-6d6b7b8eac764d08b90a817610f2f6ce"
if os.getenv("DASHSCOPE_API_KEY") is None:
    os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY
    logger.info(f"已设置DASHSCOPE_API_KEY环境变量")
else:
    logger.info(f"使用系统环境变量中的DASHSCOPE_API_KEY")

# API请求控制
api_request_times = []
api_lock = threading.Lock()


def get_client():
    """创建并返回一个OpenAI客户端实例"""
    return OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )


# 全局客户端
client = get_client()


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


def clean_article_text_ai(title: str, content: str, model: str = "qwen-max", article_id: str = None) -> str:
    """
    使用大模型清理和整理小红书文章文本

    Args:
        title: 文章标题
        content: 文章内容
        model: 要使用的模型名称，默认为qwen-max
        article_id: 文章ID，用于日志记录

    Returns:
        str: 清理后的文本
    """
    prompt = f"""
你是一个专业的文本处理专家，请帮我清理和整理小红书文章内容。

请按照以下要求处理文章：
1. 删除所有表情符号，但保留其表达的含义，用合适的文字描述替换
2. 保持文章的结构和逻辑清晰
3. 规范标点符号的使用
4. 保留重要的信息和行程安排
5. 将整个内容整理成一段连贯流畅的文本
6. 删除无关的广告信息和推广内容
7. 保留地点名称、时间、价格等关键信息
8. 使文本更易读，语言更自然

文章标题：{title}
文章内容：{content}

请直接返回清理后的文本内容，不要添加任何解释或说明：
"""

    try:
        # 动态睡眠控制请求频率
        sleep_time = dynamic_sleep(api_request_times)
        if sleep_time > 0:
            time.sleep(sleep_time)

        record_api_request()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=2000
        )

        cleaned_text = response.choices[0].message.content.strip()

        # 验证是否还包含表情符号
        if any(ord(char) > 0x1F300 for char in cleaned_text):  # 检查Unicode表情符号
            logger.warning(
                f"AI处理后仍包含表情符号，尝试二次处理: {article_id or title[:30]}...")
            # 二次处理
            prompt_retry = f"""
请严格检查并删除以下文本中的所有表情符号，确保输出的是纯文本，不含任何表情符号：

{cleaned_text}

直接返回处理后的纯文本："""

            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt_retry}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            cleaned_text = response.choices[0].message.content.strip()

        logger.info(f"AI清理文章成功: {article_id or title[:30]}...")
        return cleaned_text

    except Exception as e:
        logger.error(f"AI清理文章失败 - {article_id or title[:30]}...: {str(e)}")
        # 如果AI处理失败，尝试简单的表情符号清理
        try:
            # 使用Unicode范围过滤表情符号
            filtered_text = ''.join(char for char in f"{title} {content}".strip()
                                    if not (0x1F300 <= ord(char) <= 0x1F9FF or  # 表情符号范围
                                            # 杂项符号
                                            0x2600 <= ord(char) <= 0x26FF or
                                            # 装饰符号
                                            0x2700 <= ord(char) <= 0x27BF or
                                            0xFE00 <= ord(char) <= 0xFE0F))     # 变体选择器
            return filtered_text
        except:
            # 如果所有处理都失败，返回原始文本并记录警告
            logger.warning(f"所有清理方法都失败，返回原始文本: {article_id or title[:30]}...")
            return f"{title} {content}".strip()


def process_single_article(article_data: Tuple[int, Dict[str, Any], str]) -> Tuple[int, Dict[str, str]]:
    """
    处理单篇文章，用于并行处理

    Args:
        article_data: (索引, 文章数据, 模型名称)的元组

    Returns:
        Tuple[int, Dict[str, str]]: (索引, 处理结果)的元组
    """
    index, article, model = article_data
    try:
        # 为每个线程创建自己的客户端
        thread_client = get_client()
        # 保存原始客户端
        global client
        original_client = client
        # 设置线程本地客户端
        client = thread_client

        title = article.get('strategyTitle', '').strip()
        content = article.get('strategyContent', '').strip()
        article_id = article.get('strategyID', f'article_{index}')

        if not title and not content:
            logger.warning(f"跳过空文章: {article_id}")
            # 恢复原始客户端
            client = original_client
            return index, None

        # 使用AI清理文章文本
        cleaned_text = clean_article_text_ai(title, content, model, article_id)

        # 构造输出格式 - 只包含text字段，其中包含题目和内容
        processed_article = {
            "text": cleaned_text
        }

        # 恢复原始客户端
        client = original_client
        return index, processed_article

    except Exception as e:
        logger.error(f"处理文章失败 - 索引 {index}: {str(e)}")
        # 恢复原始客户端
        client = original_client
        return index, None


def calculate_optimal_thread_count(total_articles: int) -> int:
    """
    根据文章数量计算最优线程数

    Args:
        total_articles: 文章总数

    Returns:
        int: 最优线程数
    """
    # 基于QPM限制计算理论最大线程数
    max_threads_by_qpm = QPM_LIMIT // 60  # 每秒最大请求数

    # 根据文章数量调整
    if total_articles <= 10:
        return min(total_articles, 5)
    elif total_articles <= 50:
        return min(total_articles, 10)
    else:
        return min(total_articles, MAX_THREADS, max_threads_by_qpm)


def load_xhs_articles(input_file: str) -> List[Dict[str, Any]]:
    """
    从JSON文件加载小红书文章数据

    Args:
        input_file: 输入文件路径

    Returns:
        List[Dict]: 文章数据列表
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"成功加载 {len(articles)} 篇文章")
        return articles
    except Exception as e:
        logger.error(f"加载文章失败: {str(e)}")
        return []


def deduplicate_articles_by_title(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    根据标题去重文章

    Args:
        articles: 文章列表

    Returns:
        List[Dict[str, Any]]: 去重后的文章列表
    """
    seen_titles: Set[str] = set()
    unique_articles = []
    duplicates_count = 0

    for article in articles:
        title = article.get('strategyTitle', '').strip()

        # 空标题的文章保留
        if not title:
            unique_articles.append(article)
            continue

        # 检查标题是否已存在
        if title not in seen_titles:
            seen_titles.add(title)
            unique_articles.append(article)
        else:
            duplicates_count += 1
            logger.debug(f"发现重复标题，跳过: {title[:50]}...")

    logger.info(
        f"标题去重完成: 原始 {len(articles)} 篇 -> 去重后 {len(unique_articles)} 篇 (跳过 {duplicates_count} 篇重复)")
    return unique_articles


def save_processed_articles(articles: List[Dict[str, str]], output_file: str):
    """
    将处理后的文章保存为JSONL格式

    Args:
        articles: 处理后的文章列表
        output_file: 输出文件路径
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for article in articles:
                json.dump(article, f, ensure_ascii=False)
                f.write('\n')
        logger.info(f"成功保存 {len(articles)} 篇处理后的文章到 {output_file}")
    except Exception as e:
        logger.error(f"保存文章失败: {str(e)}")


def process_xhs_articles(input_file: str = DEFAULT_INPUT_FILE,
                         output_file: str = DEFAULT_OUTPUT_FILE,
                         model: str = "qwen-max",
                         limit: int = None,
                         use_threading: bool = USE_THREADING):
    """
    处理小红书文章的主函数

    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSONL文件路径
        model: 使用的模型名称
        limit: 限制处理的文章数量（None为全部）
        use_threading: 是否使用多线程处理
    """
    start_time = time.time()
    logger.info(f"开始处理小红书文章...")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"使用AI清理模式")
    logger.info(f"使用模型: {model}")
    logger.info(f"多线程处理: {use_threading}")

    # 加载文章数据
    articles = load_xhs_articles(input_file)
    if not articles:
        logger.error("没有找到文章数据，退出处理")
        return

    # 根据标题去重
    articles = deduplicate_articles_by_title(articles)

    # 限制处理数量
    if limit:
        articles = articles[:limit]
        logger.info(f"限制处理前 {limit} 篇文章")

    # 准备处理任务
    all_tasks = []
    for i, article in enumerate(articles):
        all_tasks.append((i, article, model))

    # 计算最优线程数
    optimal_threads = calculate_optimal_thread_count(len(all_tasks))
    logger.info(f"准备处理 {len(all_tasks)} 篇文章")

    # 存储处理结果
    processed_articles = [None] * len(all_tasks)

    # 处理方式：多线程或单线程
    if use_threading and len(all_tasks) > 1:
        logger.info(f"使用多线程处理，线程数: {optimal_threads}")

        # 创建线程池
        with concurrent.futures.ThreadPoolExecutor(max_workers=optimal_threads) as executor:
            # 提交所有任务
            future_to_index = {executor.submit(
                process_single_article, task): task[0] for task in all_tasks}

            # 使用tqdm显示进度
            with tqdm(total=len(all_tasks), desc="处理文章") as pbar:
                completed_count = 0
                for future in concurrent.futures.as_completed(future_to_index):
                    index = future_to_index[future]
                    try:
                        _, result = future.result()
                        if result is not None:
                            processed_articles[index] = result
                        completed_count += 1

                        # 每处理10篇文章保存一次（增量保存）
                        if completed_count % 10 == 0:
                            # 过滤掉None值
                            valid_articles = [
                                article for article in processed_articles if article is not None]
                            save_processed_articles(
                                valid_articles, output_file)
                            logger.info(f"已保存前 {len(valid_articles)} 篇文章")

                        pbar.update(1)
                    except Exception as e:
                        logger.error(f"处理文章索引 {index} 时出错: {e}")
                        pbar.update(1)
    else:
        logger.info(f"使用单线程处理")
        # 单线程处理
        with tqdm(total=len(all_tasks), desc="处理文章") as pbar:
            for i, task in enumerate(all_tasks):
                try:
                    _, result = process_single_article(task)
                    if result is not None:
                        processed_articles[i] = result

                    # 每处理10篇文章保存一次（增量保存）
                    if (i + 1) % 10 == 0:
                        valid_articles = [
                            article for article in processed_articles if article is not None]
                        save_processed_articles(valid_articles, output_file)
                        logger.info(f"已保存前 {len(valid_articles)} 篇文章")

                except Exception as e:
                    logger.error(f"处理文章索引 {i} 时出错: {e}")
                finally:
                    pbar.update(1)

    # 最终保存所有文章（过滤掉None值）
    final_articles = [
        article for article in processed_articles if article is not None]
    save_processed_articles(final_articles, output_file)

    # 统计信息
    end_time = time.time()
    processing_time = end_time - start_time

    logger.info(f"文章处理完成！")
    logger.info(f"总计处理: {len(final_articles)}/{len(articles)} 篇文章")
    logger.info(f"处理耗时: {processing_time:.2f} 秒")
    logger.info(f"平均速度: {len(final_articles)/processing_time:.2f} 篇/秒")


if __name__ == "__main__":
    # 运行文章处理 - 使用AI模式，限制处理前10篇
    process_xhs_articles(limit=10)  # 只处理前10篇文章
