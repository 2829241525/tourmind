import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from transformers import AutoTokenizer
import os


def analyze_text_length(csv_path, model_name="../pretrained_models/mdeberta-v3-base"):
    print(f"正在读取文件: {csv_path}")
    # 初始化tokenizer
    print(f"加载tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    if 'spl_room_text' not in df.columns:
        print("错误：CSV文件中没有找到'spl_room_text'列")
        return
    
    # 计算token长度
    print("正在计算token长度...")
    lengths = []
    texts = []  # 存储原始文本
    batch_size = 1000  # 批处理以提高效率
    
    for i in range(0, len(df), batch_size):
        batch_texts = df['spl_room_text'].iloc[i:i+batch_size].tolist()
        batch_tokens = tokenizer(batch_texts, add_special_tokens=True)
        batch_lengths = [len(tokens) for tokens in batch_tokens['input_ids']]
        lengths.extend(batch_lengths)
        texts.extend(batch_texts)
        if i % 10000 == 0:
            print(f"已处理 {i}/{len(df)} 条数据")
    
    lengths = np.array(lengths)
    
    # 计算基本统计信息
    stats = {
        '最小token长度': lengths.min(),
        '最大token长度': lengths.max(),
        '平均token长度': lengths.mean(),
        '中位数token长度': np.median(lengths),
        '标准差': lengths.std(),
        '90%分位数': np.percentile(lengths, 90),
        '95%分位数': np.percentile(lengths, 95),
        '99%分位数': np.percentile(lengths, 99),
    }
    
    print("\nToken长度统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}")
    
    # 打印长度分布的直方图
    print("\nToken长度分布:")
    hist = np.histogram(lengths, bins=10)
    for i in range(len(hist[0])):
        bar = "#" * int(hist[0][i] / max(hist[0]) * 50)
        print(f"{int(hist[1][i]):5d}-{int(hist[1][i+1]):5d}: {bar} ({hist[0][i]})")
    
    # 打印最长的10个文本
    print("\n最长的10个文本:")
    print("-" * 100)
    indices = np.argsort(lengths)[-10:][::-1]  # 获取最长的10个文本的索引
    for idx in indices:
        print(f"长度: {lengths[idx]} tokens")
        print(f"文本: {texts[idx]}")
        print("-" * 100)

def main():
    # 获取当前文件的目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(current_dir)
    # 构建CSV文件的完整路径
    csv_path = os.path.join(current_dir, 'processed_cleaned_spaces.csv')
    model_name = os.path.join(
        current_dir, '../pretrained_models/mdeberta-v3-base')

    # 使用默认模型
    parser = argparse.ArgumentParser(description='分析CSV文件中source_text的token长度分布')
    parser.add_argument('csv_path', type=str, help='CSV文件的路径', nargs='?', default=csv_path)
    parser.add_argument('--model', type=str, default=model_name, 
                        help='要使用的预训练模型名称')
    args = parser.parse_args()
    
    if not Path(args.csv_path).exists():
        print(f"错误：文件 {args.csv_path} 不存在")
        return
        
    analyze_text_length(args.csv_path, args.model)

if __name__ == "__main__":
    main() 