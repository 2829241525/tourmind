import pandas as pd
import numpy as np

# 配置
INPUT_FILE = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/processed_cleaned_spaces_side_access_fix.csv'
OUTPUT_FILE = 'sampled_hotel_data_20percent.csv'
SAMPLE_RATE = 0.2
RANDOM_SEED = 42


def main():
    # 设置随机种子
    np.random.seed(RANDOM_SEED)

    # 读取CSV文件
    print("正在读取CSV文件...")
    df = pd.read_csv(INPUT_FILE)

    print(f"原始数据行数: {len(df)}")
    print(f"唯一s_hotel_id数量: {df['s_hotel_id'].nunique()}")

    # 获取所有唯一的s_hotel_id
    unique_hotel_ids = df['s_hotel_id'].unique()

    # 随机选取20%的s_hotel_id
    sample_size = int(len(unique_hotel_ids) * SAMPLE_RATE)
    sampled_hotel_ids = np.random.choice(
        unique_hotel_ids, size=sample_size, replace=False)

    print(f"选取的s_hotel_id数量: {len(sampled_hotel_ids)}")

    # 根据选取的s_hotel_id筛选数据
    sampled_df = df[df['s_hotel_id'].isin(sampled_hotel_ids)]

    print(f"筛选后数据行数: {len(sampled_df)}")

    # 保存到新的CSV文件
    sampled_df.to_csv(OUTPUT_FILE, index=False)
    print(f"数据已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
