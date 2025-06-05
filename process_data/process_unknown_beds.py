#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
处理未知床位信息脚本
根据DeBERTa模型预测结果替换CSV文件中的[unknown]标记
"""

import os
import logging
import pandas as pd
import numpy as np
import torch
import argparse
import re
from tqdm import tqdm
from datetime import datetime
from transformers import DebertaV2TokenizerFast, DebertaV2ForTokenClassification

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logs/process_unknown_beds.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 默认配置参数
CONFIG = {
    "model_dir": "/home/maxon/disk2/roomMatch/room_match/sequence_annotation/models/room_bed_extraction/checkpoint-14439",
    "input_file": "/home/maxon/disk2/roomMatch/room_match/deberta/data/processed_country.csv",
    "output_dir": "/home/maxon/disk2/roomMatch/room_match/results",
    "max_length": 64,
    "batch_size": 512,
}


def load_model_and_tokenizer(model_dir, device):
    """加载预训练模型和分词器"""
    logger.info(f"从 {model_dir} 加载模型和分词器")

    try:
        tokenizer = DebertaV2TokenizerFast.from_pretrained(model_dir)
        model = DebertaV2ForTokenClassification.from_pretrained(
            model_dir).to(device)
        model.eval()  # 设置为评估模式
        return model, tokenizer
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}")
        raise


def load_data(input_file):
    """加载需要处理的数据"""
    logger.info(f"从 {input_file} 加载数据")

    try:
        df = pd.read_csv(input_file)
        # 检查必要的列是否存在
        required_columns = ["spl_room_text", "s_room_text"]
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"输入文件中没有找到列 '{col}'")
                raise ValueError(f"输入文件中没有找到列 '{col}'")

        # 保留非空房间名
        valid_data = df[df["spl_room_text"].notna()].copy()

        logger.info(f"加载数据完成，共 {len(valid_data)} 行有效数据")
        return valid_data
    except Exception as e:
        logger.error(f"加载数据失败: {str(e)}")
        raise


def extract_room_text(text):
    """从标记文本中提取房间名称部分"""
    if "[room]" in text and "[bed]" in text:
        return text.split("[bed]")[0].replace("[room]", "").strip()
    return ""


def convert_numbers_to_words(text):
    """将文本中的数字转换为英文单词"""
    # 将7以内的阿拉伯数字转换为英文单词
    number_mapping = {
        '1': 'one',
        '2': 'two',
        '3': 'three',
        '4': 'four',
        '5': 'five',
        '6': 'six',
        '7': 'seven'
    }

    # 测试示例
    original_text = text

    # 使用正则表达式查找独立的数字并替换
    for num, word in number_mapping.items():
        # 确保匹配的是独立的数字，而不是数字的一部分
        pattern = re.compile(
            rf'(?:^|(?<=[^a-zA-Z0-9]))'  # 确保前面是非字母数字
            rf'{num}'  # 需要匹配的数字
            rf'(?:$|(?=[^a-zA-Z0-9]))',  # 确保后面是非字母数字
            re.IGNORECASE
        )
        text = pattern.sub(word, text)

    # 如果文本发生了变化，记录转换情况
    if original_text != text:
        logger.debug(f"数字转换: '{original_text}' -> '{text}'")

    return text


def test_number_conversion():
    """测试数字转换功能"""
    test_cases = [
        "2 beds room",            # 开头数字
        "room with 2 beds",       # 中间数字
        "twin room 2",            # 结尾数字
        "room2",                  # 数字作为单词一部分
        "2rooms",                 # 数字作为单词一部分
        "12 beds",                # 多位数字
        "1bed 2bath",             # 多个数字
        "book 1-2 rooms"          # 带连字符的数字
    ]

    results = []
    for case in test_cases:
        converted = convert_numbers_to_words(case)
        results.append((case, converted))

    for original, converted in results:
        logger.info(f"测试数字转换: '{original}' -> '{converted}'")

    return results


def predict_batch(model, tokenizer, room_names, device, max_length):
    """批量预测房间名称中的床位信息"""
    # 对房间名进行数字转换处理
    processed_room_names = [
        convert_numbers_to_words(name) for name in room_names]

    # if len(room_names) > 0:
    #     # logger.info(
    #     #     f"数字转换示例：原始房间名：'{room_names[0]}'，转换后：'{processed_room_names[0]}'")

    # 分词
    tokenized = tokenizer(
        processed_room_names,
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
    confidence_values = []

    # 定义标签映射
    id2label = {0: "O", 1: "B-BED", 2: "I-BED", 3: "E-BED"}

    # 处理每个预测结果
    for i, (room_name, pred, conf) in enumerate(zip(room_names, predictions, confidence_scores)):
        tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"][i])

        # 提取床位信息
        bed_tokens = []
        confidence = []

        for token, label, conf_scores in zip(tokens, pred, conf):
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue

            score = conf_scores[label.item()].item()
            confidence.append(score)

            if label.item() in [1, 2, 3]:  # B-BED, I-BED, E-BED
                bed_tokens.append(token)

        # 合并token为文本
        bed_info = tokenizer.convert_tokens_to_string(bed_tokens).strip()
        bed_infos.append(bed_info)

        # 计算平均置信度
        avg_confidence = np.mean(confidence) if confidence else 0
        confidence_values.append(avg_confidence)

    return bed_infos, confidence_values


def process_unknown_beds(input_file, model_dir, output_dir, max_length, batch_size):
    """处理未知床位信息并导出到CSV文件"""
    # 检查GPU可用性
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(model_dir, device)

    # 加载数据
    data = load_data(input_file)

    # 筛选出含有[unknown]床位且房间名称包含"bed"的记录
    unknown_beds = []
    room_texts = []
    indices = []

    for idx, row in data.iterrows():
        spl_room_text = row["spl_room_text"]
        if (isinstance(spl_room_text, str) and
            "[bed]" in spl_room_text and
            "[unknown]" in spl_room_text and
                "bed" in spl_room_text.lower()):

            room_text = extract_room_text(spl_room_text)
            if room_text:
                unknown_beds.append(row)
                room_texts.append(room_text)  # 只将房间名称部分提取出来进行处理
                indices.append(idx)

    logger.info(f"发现 {len(unknown_beds)} 条未知床位且房间名含'bed'的记录")

    if len(unknown_beds) == 0:
        logger.warning("没有找到需要处理的记录，处理终止")
        return None

    # 批量处理数据
    all_bed_infos = []
    all_confidence_values = []

    # 改进进度条显示
    total_items = len(room_texts)
    logger.info(f"开始批量预测 {total_items} 条未知床位数据")

    # 显示更详细的信息
    print(f"\n使用 {device} 处理未知床位数据...")
    print(f"总数据量: {total_items:,}条")
    print(f"每批处理: {batch_size:,}条")

    # 创建更详细的进度条
    with tqdm(total=total_items, desc="未知床位处理进度", unit='条',
              bar_format='{desc}: {n_fmt}/{total_fmt}条 [{elapsed}<{remaining}] {rate_fmt}',
              mininterval=0.1) as pbar:

        for i in range(0, len(room_texts), batch_size):
            batch = room_texts[i:i+batch_size]
            bed_infos, confidence_values = predict_batch(
                model, tokenizer, batch, device, max_length
            )
            all_bed_infos.extend(bed_infos)
            all_confidence_values.extend(confidence_values)

            # 更新进度条
            pbar.update(len(batch))

    # 替换原始数据中的[unknown]标记
    unknown_data = data.iloc[indices].copy()
    unknown_data["predicted_bed_info"] = all_bed_infos
    unknown_data["confidence_score"] = all_confidence_values

    # 更新原始数据
    result_data = data.copy()

    # 显示替换进度
    print(f"\n开始替换原始数据中的未知床位信息...")
    replaced_count = 0

    with tqdm(total=len(indices), desc="替换进度", unit='条',
              bar_format='{desc}: {n_fmt}/{total_fmt}条 [{elapsed}<{remaining}] {rate_fmt}',
              mininterval=0.1) as pbar:

        for idx, (original_idx, row) in enumerate(zip(indices, unknown_data.iterrows())):
            predicted_bed = all_bed_infos[idx]
            if predicted_bed and all_confidence_values[idx] > 0.7:  # 设置一个置信度阈值
                # 替换spl_room_text中的[unknown]
                original_text = result_data.at[original_idx, "spl_room_text"]

                # 提取房间名部分和床位部分
                if "[room]" in original_text and "[bed]" in original_text:
                    room_part = original_text.split("[bed]")[0]
                    bed_part = "[bed]" + original_text.split("[bed]")[1]

                    # 转换房间名部分中的数字为英文
                    room_part = convert_numbers_to_words(room_part)

                    # 替换[unknown]为预测的床位（也转换其中的数字）
                    bed_part = bed_part.replace(
                        "[unknown]", convert_numbers_to_words(predicted_bed))

                    # 合并更新后的文本
                    updated_text = room_part + bed_part
                else:
                    # 如果无法分割，直接替换
                    updated_text = original_text.replace(
                        "[unknown]", convert_numbers_to_words(predicted_bed))

                result_data.at[original_idx, "spl_room_text"] = updated_text

                # 替换s_room_text中的[unknown]
                if "s_room_text" in result_data.columns and isinstance(result_data.at[original_idx, "s_room_text"], str):
                    s_text = result_data.at[original_idx, "s_room_text"]
                    if "[unknown]" in s_text:
                        updated_s_text = s_text.replace(
                            "[unknown]", convert_numbers_to_words(predicted_bed))
                        result_data.at[original_idx,
                                       "s_room_text"] = updated_s_text

                replaced_count += 1

            pbar.update(1)

    # 对整个数据集中其他字段也进行数字转换
    logger.info("对CSV中的所有房间名进行数字转换...")

    # 对所有spl_room_text进行处理，确保数字转换格式一致
    for idx, row in result_data.iterrows():
        if isinstance(row["spl_room_text"], str) and "[room]" in row["spl_room_text"] and "[bed]" in row["spl_room_text"]:
            # 分离房间名和床位部分
            parts = row["spl_room_text"].split("[bed]")
            room_part = parts[0]
            bed_part = "[bed]" + parts[1] if len(parts) > 1 else ""

            # 只转换房间名部分的数字
            converted_room = convert_numbers_to_words(room_part)

            # 更新文本
            result_data.at[idx, "spl_room_text"] = converted_room + bed_part

    # 生成输出文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        output_dir, f"processed_country_with_predictions_{timestamp}.csv")

    # 保存结果
    print(f"\n正在保存处理结果...")
    result_data.to_csv(output_file, index=False, encoding="utf-8")
    logger.info(f"处理结果已保存到 {output_file}")

    # 打印一些统计信息
    successful_predictions = sum([1 for idx, conf in enumerate(
        all_confidence_values) if all_bed_infos[idx] and conf > 0.7])
    logger.info(f"总共处理未知床位房间数: {len(unknown_beds)}")
    logger.info(
        f"成功预测并替换的记录数: {successful_predictions} ({successful_predictions/len(unknown_beds)*100:.2f}%)")
    logger.info(f"数字转换格式保存到CSV: {output_file}")

    # 输出前5个预测示例
    logger.info("\n预测示例:")
    for i in range(min(5, len(unknown_beds))):
        original_idx = indices[i]
        original_text = data.at[original_idx, "spl_room_text"]
        updated_text = result_data.at[original_idx, "spl_room_text"]
        logger.info(f"原始文本: {original_text}")
        logger.info(f"更新后文本: {updated_text}")
        logger.info(f"预测床位: {all_bed_infos[i]}")
        logger.info(f"置信度: {all_confidence_values[i]:.4f}")
        logger.info("-----")

    print(
        f"\n处理完成！共处理 {len(unknown_beds)} 条数据，成功替换 {replaced_count} 条 ({replaced_count/len(unknown_beds)*100:.2f}%)")
    return output_file


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="处理未知床位信息")
    parser.add_argument("--model_dir", type=str, default=CONFIG["model_dir"],
                        help="预训练模型目录")
    parser.add_argument("--input_file", type=str, default=CONFIG["input_file"],
                        help="输入CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=CONFIG["output_dir"],
                        help="输出目录")
    parser.add_argument("--max_length", type=int, default=CONFIG["max_length"],
                        help="最大序列长度")
    parser.add_argument("--batch_size", type=int, default=CONFIG["batch_size"],
                        help="批处理大小")
    parser.add_argument("--test_number_conversion", action="store_true",
                        help="测试数字转换功能")

    args = parser.parse_args()

    try:
        # 确保logs目录存在
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)

        # 如果指定了测试数字转换
        if args.test_number_conversion:
            logger.info("执行数字转换测试")
            test_number_conversion()
            return

        # 记录开始时间
        start_time = datetime.now()
        print(f"开始处理未知床位信息: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

        output_file = process_unknown_beds(
            args.input_file,
            args.model_dir,
            args.output_dir,
            args.max_length,
            args.batch_size
        )

        # 记录结束时间及总耗时
        end_time = datetime.now()
        duration = end_time - start_time
        hours, remainder = divmod(duration.total_seconds(), 3600)
        minutes, seconds = divmod(remainder, 60)

        print(f"总耗时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

        if output_file:
            logger.info(f"处理完成。结果保存在: {output_file}")
            print(f"处理完成。结果保存在: {output_file}")
        else:
            logger.warning("处理过程未生成结果文件")
            print("处理过程未生成结果文件")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        print(f"处理过程中发生错误: {str(e)}")
        raise


if __name__ == "__main__":
    main()
