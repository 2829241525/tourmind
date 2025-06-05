#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import requests
import json
import time
import logging
from typing import List, Dict
import os

# 配置信息
CONFIG = {
    'CSV_FILE_PATH': '/home/maxon/disk2/roomMatch/room_match/room_type_word_frequency/word_frequency_results/room_name_frequency.csv',
    'OUTPUT_PATH': '/home/maxon/disk2/roomMatch/room_match/room_type_word_frequency/word_frequency_results',
    'API_URL': 'http://8.134.51.15:13001/match/',
    'FREQUENCY_THRESHOLD': 10,
    'SIMILARITY_THRESHOLD': 0.5,
    'BATCH_SIZE': 1000,
    'REQUEST_DELAY': 0.1,  # 请求间延迟（秒）
    'TOKEN': 'tourmind_room_match',
    'THRESHOLD': 0.9,
    'BASE_ROOM_NAMES': {
        'spl': 'Deluxe Room Double Or Twin',
        's': 'Deluxe with Double Bed 1 Double Bed'
    }
}

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(
            CONFIG['OUTPUT_PATH'], 'process_log.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class WordFrequencyProcessor:
    def __init__(self):
        self.headers = {
            'Cookie': 'admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k5fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN',
            'cookie': 'admintools_user_session=MTc0MjIwMjI3OHw5bzlBTUp3VHBUNjk3OTJlU0xfSnVwanVfM05haTZZaDhWUUxMeDRMNXNIQ1ZPcmdYVW41ZzZ2d0J0VTdXSzZVRUVnY0RSZ09VN0k5fDI5N-SfNq50abl6wb9cLn0lzTqhq9Fq8l_L-jkQ4OKN',
            'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'Host': '8.134.51.15:13001',
            'Connection': 'keep-alive'
        }
        self.high_similarity_words = []

    def read_low_frequency_words(self) -> List[str]:
        """读取词频低于阈值的词"""
        low_frequency_words = []

        try:
            with open(CONFIG['CSV_FILE_PATH'], 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    frequency = int(row['频率'])
                    if frequency < CONFIG['FREQUENCY_THRESHOLD']:
                        word = row['词'].strip()
                        if word:  # 确保词不为空
                            low_frequency_words.append(word)

            logger.info(
                f"读取到 {len(low_frequency_words)} 个低频词（频率 < {CONFIG['FREQUENCY_THRESHOLD']}）")
            return low_frequency_words

        except Exception as e:
            logger.error(f"读取CSV文件失败: {e}")
            return []

    def create_request_payload(self, words: List[str]) -> Dict:
        """创建API请求负载"""
        # 为每个词创建组合的房间名称，如果词以下划线或▁符号开头则去除
        processed_words = []
        for word in words:
            if word.startswith('_'):
                processed_word = word.lstrip('_')
            elif word.startswith('▁'):
                processed_word = word.lstrip('▁')
            else:
                processed_word = word
            processed_words.append(processed_word)

        spl_room_names = [
            f"{processed_word} {CONFIG['BASE_ROOM_NAMES']['spl']}" for processed_word in processed_words]
        s_room_names = [
            f"{processed_word} {CONFIG['BASE_ROOM_NAMES']['s']}" for processed_word in processed_words]

        payload = {
            "spl_room_names": spl_room_names,
            "spl_room_bed_names": [""] * len(words),
            "s_room_names": s_room_names,
            "s_room_bed_names": [""] * len(words),
            "token": CONFIG['TOKEN'],
            "threshold": CONFIG['THRESHOLD']
        }

        return payload

    def call_similarity_api(self, words: List[str]) -> List[float]:
        """调用相似度API"""
        payload = self.create_request_payload(words)

        try:
            response = requests.post(
                CONFIG['API_URL'],
                headers=self.headers,
                data=json.dumps(payload),
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                similarities = result.get('similarities', [])
                logger.info(f"成功获取 {len(similarities)} 个相似度值")
                return similarities
            else:
                logger.error(
                    f"API请求失败，状态码: {response.status_code}, 响应: {response.text}")
                return []

        except Exception as e:
            logger.error(f"API请求异常: {e}")
            return []

    def process_words_in_batches(self, words: List[str]):
        """分批处理词语"""
        total_words = len(words)
        processed = 0

        for i in range(0, total_words, CONFIG['BATCH_SIZE']):
            batch = words[i:i + CONFIG['BATCH_SIZE']]
            batch_size = len(batch)

            logger.info(
                f"处理批次 {i//CONFIG['BATCH_SIZE'] + 1}, 词数: {batch_size}")

            similarities = self.call_similarity_api(batch)

            if similarities:
                # 检查相似度并记录高相似度的词
                for j, similarity in enumerate(similarities):
                    if similarity > CONFIG['SIMILARITY_THRESHOLD']:
                        original_word = batch[j]
                        # 如果词以下划线或▁符号开头则去除
                        if original_word.startswith('_'):
                            processed_word = original_word.lstrip('_')
                        elif original_word.startswith('▁'):
                            processed_word = original_word.lstrip('▁')
                        else:
                            processed_word = original_word

                        self.high_similarity_words.append({
                            'word': processed_word,  # 保存处理后的词（去除下划线/▁符号）
                            'processed_word': processed_word,  # 处理后的词
                            'similarity': similarity,
                            'spl_room_name': f"{processed_word} {CONFIG['BASE_ROOM_NAMES']['spl']}",
                            's_room_name': f"{processed_word} {CONFIG['BASE_ROOM_NAMES']['s']}"
                        })
                        logger.info(
                            f"发现高相似度词: {original_word} -> {processed_word}, 相似度: {similarity:.4f}")

            processed += batch_size
            logger.info(f"已处理: {processed}/{total_words} 个词")

            # 请求间延迟
            time.sleep(CONFIG['REQUEST_DELAY'])

    def save_results(self):
        """保存结果到CSV文件"""
        if not self.high_similarity_words:
            logger.info("没有找到高相似度的词")
            return

        output_file = os.path.join(
            CONFIG['OUTPUT_PATH'], 'high_similarity_words.csv')

        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as file:
                fieldnames = ['word', 'processed_word', 'similarity',
                              'spl_room_name', 's_room_name']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                writer.writeheader()
                writer.writerows(self.high_similarity_words)

            logger.info(f"结果已保存到: {output_file}")
            logger.info(f"共找到 {len(self.high_similarity_words)} 个高相似度词")

        except Exception as e:
            logger.error(f"保存结果失败: {e}")

    def run(self):
        """运行主程序"""
        logger.info("开始处理低频词相似度检测")

        # 读取低频词
        words = self.read_low_frequency_words()
        if not words:
            logger.error("没有读取到任何低频词")
            return

        # 分批处理
        self.process_words_in_batches(words)

        # 保存结果
        self.save_results()

        logger.info("处理完成")


def main():
    processor = WordFrequencyProcessor()
    processor.run()


if __name__ == "__main__":
    main()
