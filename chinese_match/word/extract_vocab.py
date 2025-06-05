#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
提取mdeberta-v3-base模型的词表
"""

import os
import logging
from pathlib import Path
from transformers import AutoTokenizer
import sentencepiece as spm

# 配置信息
MODEL_PATH = "/home/maxon/disk2/roomMatch/room_match/chinese_match/pretrained_models/mdeberta-v3-base"
OUTPUT_DIR = "/home/maxon/disk2/roomMatch/room_match/chinese_match/word"
VOCAB_TXT_PATH = os.path.join(OUTPUT_DIR, "vocab.txt")
VOCAB_JSON_PATH = os.path.join(OUTPUT_DIR, "vocab.json")
VOCAB_SPM_PATH = os.path.join(OUTPUT_DIR, "vocab_pieces.txt")

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_vocab():
    """提取词表"""
    try:
        # 确保输出目录存在
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        logger.info(f"开始提取模型词表: {MODEL_PATH}")

        # 方法1: 使用transformers加载tokenizer
        logger.info("方法1: 使用transformers提取词表...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        # 获取词表大小
        vocab_size = tokenizer.vocab_size
        logger.info(f"词表大小: {vocab_size}")

        # 保存词表到txt文件
        with open(VOCAB_TXT_PATH, 'w', encoding='utf-8') as f:
            # 获取所有token
            vocab = tokenizer.get_vocab()
            # 按照token id排序
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

            for token, token_id in sorted_vocab:
                f.write(f"{token_id}\t{token}\n")

        logger.info(f"词表已保存至: {VOCAB_TXT_PATH}")

        # 保存为json格式
        import json
        with open(VOCAB_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)

        logger.info(f"词表JSON格式已保存至: {VOCAB_JSON_PATH}")

        # 方法2: 直接从SentencePiece模型提取
        spm_model_path = os.path.join(MODEL_PATH, "spm.model")
        if os.path.exists(spm_model_path):
            logger.info("方法2: 使用SentencePiece直接提取...")
            sp = spm.SentencePieceProcessor()
            sp.load(spm_model_path)

            # 获取vocab大小
            spm_vocab_size = sp.get_piece_size()
            logger.info(f"SentencePiece词表大小: {spm_vocab_size}")

            # 保存SentencePiece词表
            with open(VOCAB_SPM_PATH, 'w', encoding='utf-8') as f:
                for i in range(spm_vocab_size):
                    piece = sp.id_to_piece(i)
                    f.write(f"{i}\t{piece}\n")

            logger.info(f"SentencePiece词表已保存至: {VOCAB_SPM_PATH}")

        # 输出统计信息
        logger.info("词表提取完成!")
        logger.info(f"Transformers词表大小: {vocab_size}")
        if os.path.exists(spm_model_path):
            logger.info(f"SentencePiece词表大小: {spm_vocab_size}")

        # 显示一些示例token
        logger.info("词表示例 (前20个token):")
        for i, (token, token_id) in enumerate(sorted(vocab.items(), key=lambda x: x[1])[:20]):
            logger.info(f"  {token_id}: {repr(token)}")

    except Exception as e:
        logger.error(f"提取词表时出错: {e}")
        raise


if __name__ == "__main__":
    extract_vocab()
