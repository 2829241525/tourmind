#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
房间名称中的床位信息提取 - 预测脚本
基于预训练的DeBERTa模型预测未知床位信息并输出到CSV文件
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import argparse
from tqdm import tqdm
from datetime import datetime
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 默认配置参数
CONFIG = {
    "model_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/room_bed_extraction/checkpoint-9626"),
    "input_file": os.path.join(os.path.dirname(os.path.abspath(__file__)), "bed_all_data_process_model.csv"),
    "output_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "../results"),
    "room_name_column": "spl_room_text",
    "bed_name_column": None,
    "max_length": 64,
    "batch_size": 64,
}

def load_model_and_tokenizer(model_dir, device):
    """加载预训练模型和分词器"""
    logger.info(f"从 {model_dir} 加载模型和分词器")
    
    try:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
        model = DebertaV2ForTokenClassification.from_pretrained(model_dir).to(device)
        model.eval()  # 设置为评估模式
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise

def load_data(input_file):
    """加载需要预测的数据，只保留床位为unknown的房间"""
    logger.info(f"从 {input_file} 加载数据")
    
    try:
        df = pd.read_csv(input_file)
        # if room_name_column not in df.columns:
        #     logger.error(f"输入文件中没有找到列 '{room_name_column}'")
        #     raise ValueError(f"输入文件中没有找到列 '{room_name_column}'")
        
        # # 保留非空房间名
        # valid_data = df[df[room_name_column].notna()].copy()
        
        # # 预处理房间名称，从spl_room_text中提取[room]和[bed]标记的内容
        # if room_name_column == "spl_room_text":
        #     logger.info("检测到spl_room_text列，正在提取房间名和unknown床位信息...")
            
        #     # 初始化列表存储提取的数据
        #     room_names = []
        #     bed_names = []
        #     unknown_bed_indices = []  # 存储未知床位的索引
        #     original_indices = []     # 存储原始数据的索引
            
        #     for idx, text in enumerate(valid_data[room_name_column]):
        #         # 检查是否有[bed]标记，且床位为unknown
        #         if "[bed]" in text and "[unknown]" in text.split("[bed]")[1].lower():
        #             # 提取房间名称
        #             room_part = text.split("[bed]")[0].replace("[room]", "").strip()
                    
        #             # 添加到待处理列表
        #             room_names.append(room_part)
        #             bed_names.append("[unknown]")
        #             unknown_bed_indices.append(len(room_names) - 1)  # 记录未知床位的索引
        #             original_indices.append(idx)  # 记录原始数据索引
            
        #     # 创建一个新的DataFrame，只包含未知床位的记录
        #     unknown_bed_data = valid_data.iloc[original_indices].copy()
            
        #     # 添加提取的列
        #     unknown_bed_data["extracted_room_name"] = room_names
        #     unknown_bed_data["extracted_bed_name"] = bed_names
            
        #     # 更新处理后的列名
        #     room_name_column = "extracted_room_name"
        #     bed_name_column = "extracted_bed_name"
            
        #     logger.info(f"成功提取了未知床位的房间，有效数据行数: {len(unknown_bed_data)}")
            
        #     # 替换有效数据为只包含未知床位的数据
        #     valid_data = unknown_bed_data
        # else:
        #     # 对于其他类型的输入，如果指定了床位列，只保留床位为unknown的记录
        #     if bed_name_column and bed_name_column in valid_data.columns:
        #         valid_data = valid_data[
        #             valid_data[bed_name_column].notna() & 
        #             valid_data[bed_name_column].str.lower().str.contains("unknown")
        #         ].copy()
        #         logger.info(f"筛选出未知床位的记录: {len(valid_data)}")
        #     else:
        #         logger.warning(f"未指定床位列名，将使用所有有效数据: {len(valid_data)}")
        
        return df
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise

def predict_batch(model, tokenizer, room_names, device, max_length):
    """批量预测房间名称中的床位信息"""
    # 分词
    tokenized = tokenizer(
        room_names, 
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    
    # 将输入移动到模型所在设备
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    
    # 获取预测
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = torch.argmax(outputs.logits, dim=2)
        confidence_scores = torch.softmax(outputs.logits, dim=2)
    
    # 初始化结果列表
    bed_infos = []
    label_sequences = []
    confidence_values = []
    
    # 定义标签映射
    id2label = {0: "O", 1: "B-BED", 2: "I-BED", 3: "E-BED"}
    
    # 处理每个预测结果
    for i, (room_name, pred, conf) in enumerate(zip(room_names, predictions, confidence_scores)):
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][i])
        labels = [id2label[p.item()] for p in pred]
        
        # 提取床位信息
        bed_tokens = []
        label_sequence = []
        confidence = []
        
        for token, label, conf_scores in zip(tokens, pred, conf):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
                
            pred_label = id2label[label.item()]
            label_sequence.append(pred_label)
            score = conf_scores[label.item()].item()
            confidence.append(score)
            
            if label.item() in [1, 2, 3]:  # B-BED, I-BED, E-BED
                bed_tokens.append(token)
        
        # 合并token为文本
        bed_info = tokenizer.convert_tokens_to_string(bed_tokens).strip()
        bed_infos.append(bed_info)
        
        # 保存标签序列和平均置信度
        label_sequences.append(" ".join(label_sequence))
        avg_confidence = np.mean(confidence) if confidence else 0
        confidence_values.append(avg_confidence)
    
    return bed_infos, label_sequences, confidence_values

def predict_and_export(input_file, model_dir, output_dir, room_name_column, bed_name_column, max_length, batch_size):
    """预测未知床位信息并导出到CSV文件"""
    # 检查GPU可用性
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_dir, device)
    
    # 加载数据（只包含未知床位的记录）
    data = load_data(input_file)
    
    if len(data) == 0:
        logger.warning("没有找到未知床位的记录，处理终止")
        return None
    
    room_names = data["spl_cleaned_room"].tolist()
    
    # 初始化结果列表
    all_bed_infos = []
    all_label_sequences = []
    all_confidence_values = []
    
    # 批量处理数据
    logger.info(f"开始批量预测 {len(room_names)} 条未知床位数据")
    for i in tqdm(range(0, len(room_names), batch_size)):
        batch = room_names[i:i+batch_size]
        bed_infos, label_sequences, confidence_values = predict_batch(
            model, tokenizer, batch, device, max_length
        )
        all_bed_infos.extend(bed_infos)
        all_label_sequences.extend(label_sequences)
        all_confidence_values.extend(confidence_values)
    
    # 创建结果DataFrame
    results = data.copy()
    results["predicted_bed_info"] = all_bed_infos
    results["label_sequence"] = all_label_sequences
    results["confidence_score"] = all_confidence_values
    
    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"unknown_bed_prediction_{timestamp}.csv")
    
    # 保存结果
    results.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"预测结果已保存到 {output_file}")
    
    # 打印一些统计信息
    total_rooms = len(results)
    rooms_with_bed = len(results[results["predicted_bed_info"] != ""])
    logger.info(f"总共处理未知床位房间数: {total_rooms}")
    logger.info(f"成功预测出床位信息的房间数: {rooms_with_bed} ({rooms_with_bed/total_rooms*100:.2f}%)")
    
    # 输出前10个预测示例
    logger.info("\n预测示例:")
    sample_df = results[results["predicted_bed_info"] != ""].head(10)
    for _, row in sample_df.iterrows():
        logger.info(f"原始文本: {row['spl_room_text'] if 'spl_room_text' in row else 'N/A'}")
        logger.info(f"预测床位: {row['predicted_bed_info']}")
        logger.info(f"置信度: {row['confidence_score']:.4f}")
        logger.info("-----")
    
    return output_file

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="未知床位信息提取预测")
    parser.add_argument("--model_dir", type=str, default=CONFIG["model_dir"], 
                        help="预训练模型目录")
    parser.add_argument("--input_file", type=str, default=CONFIG["input_file"], 
                        help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"], 
                        help="输出目录")
    parser.add_argument("--room_name_column", type=str, default=CONFIG["room_name_column"], 
                        help="房间名称列名")
    parser.add_argument("--bed_name_column", type=str, default=CONFIG["bed_name_column"], 
                        help="床位名称列名")
    parser.add_argument("--max_length", type=int, default=CONFIG["max_length"], 
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"], 
                        help="批处理大小")
    
    args = parser.parse_args()
    
    try:
        output_file = predict_and_export(
            args.input_file,
            args.model_dir,
            args.output_dir,
            args.room_name_column,
            args.bed_name_column,
            args.max_length,
            args.batch_size
        )
        if output_file:
            logger.info(f"预测完成。结果保存在: {output_file}")
        else:
            logger.warning("预测过程未生成结果文件")
    except Exception as e:
        logger.error(f"预测过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 