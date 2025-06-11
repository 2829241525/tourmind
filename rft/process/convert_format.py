#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 配置信息
from pathlib import Path
import logging
import json
INPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/rft/process/train_0611.jsonl"
OUTPUT_FILE = "/home/maxon/disk2/roomMatch/room_match/rft/process/train_0611_converted.jsonl"
LOG_LEVEL = "INFO"


# 配置日志
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            '/home/maxon/disk2/roomMatch/room_match/rft/process/convert_format.log')
    ]
)


def convert_structure(input_line):
    """
    将train_0611.jsonl的结构转换为evaluate.jsonl的结构

    输入格式: {"question": "...", "answer": "..."}
    输出格式: {"messages": [{"role": "user", "content": "..."}], "explanation": "..."}
    """
    try:
        data = json.loads(input_line.strip())

        converted_data = {
            "messages": [
                {
                    "role": "user",
                    "content": data["question"]
                }
            ],
            "explanation": data["answer"]
        }

        return converted_data
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"转换数据时出错: {e}, 原始数据: {input_line[:100]}...")
        return None


def main():
    """主函数：执行文件格式转换"""
    input_path = Path(INPUT_FILE)
    output_path = Path(OUTPUT_FILE)

    # 检查输入文件是否存在
    if not input_path.exists():
        logging.error(f"输入文件不存在: {INPUT_FILE}")
        return

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0
    error_count = 0

    logging.info(f"开始转换文件: {INPUT_FILE} -> {OUTPUT_FILE}")

    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
                open(output_path, 'w', encoding='utf-8') as outfile:

            for line_num, line in enumerate(infile, 1):
                if line.strip():  # 跳过空行
                    converted_data = convert_structure(line)

                    if converted_data:
                        json.dump(converted_data, outfile, ensure_ascii=False)
                        outfile.write('\n')
                        converted_count += 1
                    else:
                        error_count += 1
                        logging.warning(f"第 {line_num} 行转换失败")

    except Exception as e:
        logging.error(f"文件处理出错: {e}")
        return

    logging.info(f"转换完成! 成功转换 {converted_count} 条记录，{error_count} 条记录出错")
    logging.info(f"输出文件: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
