#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import sentencepiece as spm
import numpy as np

# 配置信息
config = {
    "model_path": "/home/maxon/disk2/roomMatch/room_match/deberta/pretrained_models/deberta-v3-base",
    "output_dir": "/home/maxon/disk2/roomMatch/room_match/deberta/model_info",
    "spm_model_file": "spm.model",
    "config_file": "config.json",
    "vocab_output_file": "vocab.txt",
    "vocab_with_index_file": "vocab_with_index.txt",
    "model_info_file": "model_info.json"
}


def main():
    """从DeBERTa-v3-base模型中提取词表信息并保存"""
    # 确保输出目录存在
    os.makedirs(config["output_dir"], exist_ok=True)

    # 加载SentencePiece模型
    spm_model_path = os.path.join(
        config["model_path"], config["spm_model_file"])
    print(f"正在加载SentencePiece模型: {spm_model_path}")

    try:
        sp_model = spm.SentencePieceProcessor()
        sp_model.Load(spm_model_path)
        print(f"成功加载模型，词表大小: {sp_model.GetPieceSize()}")
    except Exception as e:
        print(f"加载SentencePiece模型失败: {e}")
        return

    # 加载模型配置
    config_path = os.path.join(config["model_path"], config["config_file"])
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            model_config = json.load(f)
        print(f"成功加载模型配置，词表大小: {model_config.get('vocab_size', 'N/A')}")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        model_config = {}

    # 导出词表
    vocab_size = sp_model.GetPieceSize()
    vocab_output_path = os.path.join(
        config["output_dir"], config["vocab_output_file"])

    print(f"正在导出词表到: {vocab_output_path}")
    with open(vocab_output_path, 'w', encoding='utf-8') as f:
        for i in range(vocab_size):
            piece = sp_model.IdToPiece(i)
            f.write(f"{piece}\n")

    # 导出带下标的词表
    vocab_with_index_path = os.path.join(
        config["output_dir"], config["vocab_with_index_file"])

    print(f"正在导出带下标的词表到: {vocab_with_index_path}")
    with open(vocab_with_index_path, 'w', encoding='utf-8') as f:
        for i in range(vocab_size):
            piece = sp_model.IdToPiece(i)
            f.write(f"{i}\t{piece}\n")

    # 保存模型基本信息
    model_info = {
        "model_name": "deberta-v3-base",
        "vocab_size": vocab_size,
        "model_config": model_config
    }

    model_info_path = os.path.join(
        config["output_dir"], config["model_info_file"])
    with open(model_info_path, 'w', encoding='utf-8') as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    print(f"模型信息已保存到: {model_info_path}")
    print(f"词表导出完成，共 {vocab_size} 个词元")


if __name__ == "__main__":
    main()
