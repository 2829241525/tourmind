#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import numpy as np
from collections import Counter
from transformers import AutoTokenizer
import os
from pathlib import Path
from typing import Dict, List, Tuple

# 配置参数
CONFIG = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/room_type_word_frequency/raw_room_data.json',
    'model_name': '/home/maxon/disk2/roomMatch/room_match/chinese_match/pretrained_models/mdeberta-v3-base',
    'output_dir': '/home/maxon/disk2/roomMatch/room_match/room_type_word_frequency/word_frequency_results',
    'batch_size': 1000,
    'min_frequency': 1,  # 最小词频过滤
}


def load_room_data(input_file: str) -> List[Dict]:
    """
    加载房型数据
    """
    print(f"正在读取文件: {input_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"成功加载 {len(data)} 条房型数据")
        return data
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        raise


def extract_room_texts(data: List[Dict]) -> Tuple[List[str], List[str]]:
    """
    提取房型名称文本
    返回 (room_name列表, room_name_cn列表)
    """
    room_names = []
    room_names_cn = []

    for item in data:
        room_name = item.get('room_name', '')
        room_name_cn = item.get('room_name_cn', '')

        if room_name and room_name.strip():
            room_names.append(room_name.strip())

        if room_name_cn and room_name_cn.strip():
            room_names_cn.append(room_name_cn.strip())

    print(f"提取到 {len(room_names)} 个room_name，{len(room_names_cn)} 个room_name_cn")
    return room_names, room_names_cn


def tokenize_texts(texts: List[str], tokenizer, batch_size: int = 1000) -> List[List[str]]:
    """
    使用tokenizer对文本进行分词
    """
    print(f"正在对 {len(texts)} 条文本进行分词...")
    all_tokens = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_encoded = tokenizer(
            batch_texts, add_special_tokens=False, padding=False, truncation=False)

        # 将token ids转换回tokens
        for token_ids in batch_encoded['input_ids']:
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            # 过滤掉特殊token和空token
            filtered_tokens = [token for token in tokens if token and not token.startswith(
                '[') and not token.startswith('<')]
            all_tokens.append(filtered_tokens)

        if i % 10000 == 0:
            print(f"已处理 {i}/{len(texts)} 条文本")

    print("分词完成")
    return all_tokens


def count_word_frequency(tokenized_texts: List[List[str]]) -> Counter:
    """
    统计词频
    """
    print("正在统计词频...")
    word_counter = Counter()

    for tokens in tokenized_texts:
        word_counter.update(tokens)

    print(f"统计到 {len(word_counter)} 个不同的词")
    return word_counter


def save_frequency_to_csv(word_counter: Counter, output_file: str, min_frequency: int = 1):
    """
    将词频保存为CSV文件，按频率从低到高排序
    """
    print(f"正在保存词频结果到: {output_file}")

    # 过滤最小频率
    filtered_items = [(word, freq) for word,
                      freq in word_counter.items() if freq >= min_frequency]

    # 按频率从低到高排序
    sorted_items = sorted(filtered_items, key=lambda x: x[1])

    # 创建DataFrame
    df = pd.DataFrame(sorted_items, columns=['词', '频率'])

    # 添加排名列
    df['排名'] = range(1, len(df) + 1)

    # 重新排列列顺序
    df = df[['排名', '词', '频率']]

    # 保存为CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"已保存 {len(df)} 个词到 {output_file}")

    # 打印一些统计信息
    print(f"\n词频统计信息:")
    print(f"总词数: {len(df)}")
    print(f"最低频率: {df['频率'].min()}")
    print(f"最高频率: {df['频率'].max()}")
    print(f"平均频率: {df['频率'].mean():.2f}")
    print(f"中位数频率: {df['频率'].median():.2f}")


def analyze_room_name_frequency():
    """
    分析房型名称词频的主函数
    """
    print("开始分析房型名称词频...")

    # 加载数据
    data = load_room_data(CONFIG['input_file'])

    # 提取文本
    room_names, room_names_cn = extract_room_texts(data)

    # 初始化tokenizer
    print(f"加载tokenizer: {CONFIG['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])

    # 处理room_name
    if room_names:
        print("\n--- 处理 room_name ---")
        tokenized_room_names = tokenize_texts(
            room_names, tokenizer, CONFIG['batch_size'])
        room_name_counter = count_word_frequency(tokenized_room_names)

        output_file = os.path.join(
            CONFIG['output_dir'], 'room_name_frequency.csv')
        save_frequency_to_csv(
            room_name_counter, output_file, CONFIG['min_frequency'])

    # 处理room_name_cn
    if room_names_cn:
        print("\n--- 处理 room_name_cn ---")
        tokenized_room_names_cn = tokenize_texts(
            room_names_cn, tokenizer, CONFIG['batch_size'])
        room_name_cn_counter = count_word_frequency(tokenized_room_names_cn)

        output_file = os.path.join(
            CONFIG['output_dir'], 'room_name_cn_frequency.csv')
        save_frequency_to_csv(room_name_cn_counter,
                              output_file, CONFIG['min_frequency'])

    # 处理合并的文本
    if room_names or room_names_cn:
        print("\n--- 处理合并文本 ---")
        all_texts = room_names + room_names_cn
        tokenized_all_texts = tokenize_texts(
            all_texts, tokenizer, CONFIG['batch_size'])
        all_counter = count_word_frequency(tokenized_all_texts)

        output_file = os.path.join(
            CONFIG['output_dir'], 'all_room_names_frequency.csv')
        save_frequency_to_csv(all_counter, output_file,
                              CONFIG['min_frequency'])

    print("\n词频分析完成!")


def main():
    """
    主函数
    """
    try:
        analyze_room_name_frequency()
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
