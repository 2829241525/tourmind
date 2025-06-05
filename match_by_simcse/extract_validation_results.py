#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import glob
import logging
import argparse
from datetime import datetime

# 默认配置
DEFAULT_CONFIG = {
    "log_dir": "/home/maxon/disk2/roomMatch/room_match/match_by_simcse/logs",
    "output_file": "/home/maxon/disk2/roomMatch/room_match/match_by_simcse/deberta-v3-base.txt",
    "log_pattern": "*.log",
    "max_file_size_mb": 100,  # 超过此大小的文件将使用逐行处理
    "default_files": [
        # "/home/maxon/disk2/roomMatch/room_match/match_by_simcse/logs/train_0512_roberta.log",
        # "/home/maxon/disk2/roomMatch/room_match/match_by_simcse/logs/train_0513_bert.log"
        "/home/maxon/disk2/roomMatch/room_match/deberta/logs/train_20250512_011831.log"
    ]
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='从日志文件中提取验证结果')

    parser.add_argument('--log-dir', type=str, default=DEFAULT_CONFIG["log_dir"],
                        help='日志文件目录路径')
    parser.add_argument('--output-txt', type=str, default=DEFAULT_CONFIG["output_file"],
                        help='TXT输出文件路径')
    parser.add_argument('--pattern', type=str, default=DEFAULT_CONFIG["log_pattern"],
                        help='日志文件匹配模式 (默认: "*.log")')
    parser.add_argument('--max-file-size', type=int, default=DEFAULT_CONFIG["max_file_size_mb"],
                        help='使用逐行处理的文件大小阈值(MB)')
    parser.add_argument('--files', type=str, nargs='+', default=DEFAULT_CONFIG["default_files"],
                        help='指定要处理的日志文件列表，默认处理train.log等主要日志文件')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='显示详细日志信息')

    args = parser.parse_args()

    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 更新配置
    config = DEFAULT_CONFIG.copy()
    config["log_dir"] = args.log_dir
    config["output_file"] = args.output_txt
    config["log_pattern"] = args.pattern
    config["max_file_size_mb"] = args.max_file_size
    config["specific_files"] = args.files

    return config


def get_file_size_mb(file_path):
    """获取文件大小（MB）"""
    return os.path.getsize(file_path) / (1024 * 1024)


def extract_validation_results_whole_file(file_path):
    """从整个日志文件中提取验证结果（适用于较小的文件）"""
    results = []
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # 查找所有验证结果块，以"【验证集】评估结果"开头
        validation_blocks = re.finditer(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} - INFO - 【验证集】评估结果[\s\S]*?'
            r'(F1: [\d\.]+)(?:[\s\S]*?Validation - Loss: [\d\.]+)?)',
            content
        )

        for match in validation_blocks:
            full_text = match.group(1)

            # 提取epoch，用于结果排序
            epoch_match = re.search(r'评估结果 \(Epoch (\d+)\):', full_text)
            epoch = epoch_match.group(1) if epoch_match else "0"

            # 提取日期和时间
            date_time_match = re.match(
                r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', full_text)
            date_time = date_time_match.group(1) if date_time_match else "未知时间"

            # 构造结果字典
            result = {
                "file": os.path.basename(file_path),
                "datetime": date_time,
                "epoch": epoch,
                "full_text": full_text
            }
            results.append(result)

        return results

    except Exception as e:
        logger.error(f"处理文件 {file_path} 时出错: {str(e)}")
        return []


def extract_validation_results_line_by_line(file_path):
    """逐行读取处理大文件中的验证结果"""
    results = []
    try:
        collecting = False
        current_block = []
        has_f1 = False

        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 检查是否是验证结果开始
                if '【验证集】评估结果' in line:
                    # 如果之前有正在收集的块，则处理并添加到结果
                    if collecting and current_block and has_f1:
                        process_validation_block(
                            current_block, file_path, results)

                    # 开始新的收集
                    collecting = True
                    has_f1 = False
                    current_block = [line]
                elif collecting:
                    current_block.append(line)

                    # 检查是否包含F1值
                    if 'F1:' in line:
                        has_f1 = True

                    # 检查是否包含结束标记或者在F1值后面有新的日志开始
                    if ('Validation - Loss:' in line or
                            (has_f1 and re.match(r'\d{4}-\d{2}-\d{2}', line.strip()))):
                        process_validation_block(
                            current_block, file_path, results)
                        collecting = False
                        has_f1 = False
                        current_block = []

        # 处理最后一个块（如果有）
        if collecting and current_block and has_f1:
            process_validation_block(current_block, file_path, results)

        return results

    except Exception as e:
        logger.error(f"逐行处理文件 {file_path} 时出错: {str(e)}")
        return []


def process_validation_block(lines, file_path, results):
    """处理收集到的验证块"""
    try:
        full_text = ''.join(lines)

        # 提取epoch用于排序
        epoch_match = re.search(r'评估结果 \(Epoch (\d+)\):', full_text)
        epoch = epoch_match.group(1) if epoch_match else "0"

        # 提取日期和时间
        datetime_match = re.search(
            r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', full_text)
        date_time = datetime_match.group(1) if datetime_match else "未知时间"

        # 确保文本块包含验证结果和F1值
        if "【验证集】评估结果" in full_text and "F1:" in full_text:
            result = {
                "file": os.path.basename(file_path),
                "datetime": date_time,
                "epoch": epoch,
                "full_text": full_text
            }
            results.append(result)
    except Exception as e:
        logger.error(f"处理验证块时出错: {str(e)}")


def extract_validation_results(file_path, config):
    """根据文件大小选择适当的提取方法"""
    file_size_mb = get_file_size_mb(file_path)

    if file_size_mb > config["max_file_size_mb"]:
        logger.info(f"文件 {file_path} 大小为 {file_size_mb:.2f}MB, 使用逐行处理")
        results = extract_validation_results_line_by_line(file_path)
    else:
        logger.info(f"文件 {file_path} 大小为 {file_size_mb:.2f}MB, 一次性处理")
        results = extract_validation_results_whole_file(file_path)

    logger.info(f"从 {file_path} 提取了 {len(results)} 条验证结果")
    return results


def save_results_to_txt(all_results, file_count, error_count, config):
    """将结果保存为TXT格式"""
    with open(config["output_file"], 'w', encoding='utf-8') as f:
        f.write(
            f"# 验证结果摘要 - 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(
            f"# 总共从 {file_count} 个文件中提取了 {len(all_results)} 条验证结果 (失败: {error_count})\n\n")

        current_file = None
        for result in all_results:
            if current_file != result["file"]:
                current_file = result["file"]
                f.write(f"\n## 文件: {current_file}\n\n")

            # 直接输出原始日志中的验证结果文本块
            f.write(f"{result['full_text']}\n\n")
            f.write("-" * 80 + "\n\n")  # 分隔线，方便阅读

    logger.info(f"验证结果已保存到TXT文件: {config['output_file']}")


def main():
    # 解析命令行参数
    config = parse_args()

    all_results = []
    file_count = 0
    error_count = 0

    # 确定要处理的文件列表
    if config.get("specific_files"):
        # 如果指定了特定文件，则直接使用
        logger.info(f"使用指定的 {len(config['specific_files'])} 个文件")
        log_files = config["specific_files"]
    else:
        # 否则，根据模式查找文件
        log_files = glob.glob(os.path.join(
            config["log_dir"], config["log_pattern"]))
        logger.info(f"找到 {len(log_files)} 个匹配的日志文件")

    # 处理每个日志文件
    for log_file in log_files:
        try:
            file_count += 1
            logger.info(f"处理文件 ({file_count}/{len(log_files)}): {log_file}")
            results = extract_validation_results(log_file, config)
            all_results.extend(results)
        except Exception as e:
            error_count += 1
            logger.error(f"处理文件 {log_file} 失败: {str(e)}")

    if not all_results:
        logger.warning("未找到任何验证结果，请检查日志文件格式是否正确")
        return

    # 按文件名和epoch排序结果
    all_results.sort(key=lambda x: (x["file"], int(x["epoch"])))

    # 保存结果
    save_results_to_txt(all_results, file_count, error_count, config)

    logger.info(f"处理完成，共提取 {len(all_results)} 条验证结果")
    logger.info(f"成功处理 {file_count - error_count} 个文件，失败 {error_count} 个文件")


if __name__ == "__main__":
    main()
