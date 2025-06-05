import pandas as pd
import numpy as np

def sample_by_country(input_file, output_file, total_samples=10000):
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 计算每个国家的数量和比例
    country_counts = df['country_code'].value_counts()
    country_proportions = country_counts / len(df)
    
    # 计算每个国家需要抽取的样本数（至少1个样本）
    sample_sizes = (country_proportions * total_samples).round().astype(int)
    sample_sizes = sample_sizes.clip(lower=1)  # 确保每个国家至少有1个样本
    
    # 调整总样本数以匹配目标数量
    while sample_sizes.sum() > total_samples:
        # 从样本数最多的国家中减少1个样本
        max_country = sample_sizes.idxmax()
        if sample_sizes[max_country] > 1:  # 确保不会减到0
            sample_sizes[max_country] -= 1
    
    # 按国家抽样
    sampled_data = []
    for country in sample_sizes.index:
        country_data = df[df['country_code'] == country]
        n_samples = min(sample_sizes[country], len(country_data))
        sampled = country_data.sample(n=n_samples, random_state=42)
        sampled_data.append(sampled)
    
    # 合并所有抽样数据
    final_df = pd.concat(sampled_data, ignore_index=True)
    
    # 打印抽样统计信息
    print("\n抽样统计信息:")
    print(f"总样本数: {len(final_df)}")
    print("\n各国家样本数:")
    print(final_df['country_code'].value_counts())
    
    # 保存结果
    final_df.to_csv(output_file, index=False)
    print(f"\n结果已保存至: {output_file}")

if __name__ == "__main__":
    input_file = "匹配测试酒店.csv"
    output_file = "10000sampled_hotels.csv"
    sample_by_country(input_file, output_file) 