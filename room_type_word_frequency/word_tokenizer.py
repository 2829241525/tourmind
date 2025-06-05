#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import AutoTokenizer
import os

# 配置参数
CONFIG = {
    'model_name': '/home/maxon/disk2/roomMatch/room_match/chinese_match/pretrained_models/mdeberta-v3-base',
    'test_phrases': [
        # 中文房型名
        # "豪华双人房",
        # "经济型单人房间",
        # "商务标准间-双床",
        # "大床房 (1张床)",

        # # 英文房型名
        # "Standard Double Room",
        # "Suite with Sea View",
        # "Family Room, 2 Bedrooms",
        # "Deluxe King Room with Balcony",

        # # 混合语言
        # "豪华 Suite 套房",
        # "Standard 标准双人房",

        # # 包含数字和特殊符号
        # "Room 101 - 海景房",
        # "2-Bedroom Apartment",
        # "三人间（3张单人床）",

        # # 短语测试
        # "房",
        # "Room",
        # "豪华",
        # "Standard",
        "Wharney Deluxe Room Double Or Twin",
        "Wharney Deluxe with Double Bed 1 Double Bed"
    ]
}


def load_tokenizer():
    """
    加载tokenizer
    """
    print(f"正在加载tokenizer: {CONFIG['model_name']}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG['model_name'])
        print("tokenizer加载成功")
        return tokenizer
    except Exception as e:
        print(f"tokenizer加载失败: {str(e)}")
        raise


def tokenize_phrase(text: str, tokenizer) -> list:
    """
    对单个词组进行分词
    """
    # 使用tokenizer进行编码
    encoded = tokenizer(text, add_special_tokens=False,
                        padding=False, truncation=False)

    # 将token ids转换回tokens
    token_ids = encoded['input_ids']
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    # 过滤掉特殊token
    filtered_tokens = [token for token in tokens if token and not token.startswith(
        '[') and not token.startswith('<')]

    return filtered_tokens


def analyze_tokenization():
    """
    分析配置词组的分词结果
    """
    print("开始分词分析...")
    print("=" * 60)

    # 加载tokenizer
    tokenizer = load_tokenizer()

    print(f"\n配置的测试词组共 {len(CONFIG['test_phrases'])} 个:")
    print("-" * 60)

    for i, phrase in enumerate(CONFIG['test_phrases'], 1):
        print(f"\n{i}. 原文: {phrase}")

        # 进行分词
        tokens = tokenize_phrase(phrase, tokenizer)

        print(f"   分词结果: {tokens}")
        print(f"   token数量: {len(tokens)}")

        # 显示每个token的详细信息
        print("   详细分解:")
        for j, token in enumerate(tokens):
            print(f"     [{j+1}] '{token}'")

    print("\n" + "=" * 60)
    print("分词分析完成!")


def main():
    """
    主函数
    """
    try:
        analyze_tokenization()
    except Exception as e:
        print(f"程序执行失败: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
