import pandas as pd
import numpy as np
from pathlib import Path

# 配置
INPUT_CSV = '/home/maxon/disk2/roomMatch/room_match/process_data/room_match_arti_data.csv'
OUTPUT_CSV = '/home/maxon/disk2/roomMatch/room_match/process_data/room_ratio_top1000.csv'


def calculate_room_ratio():
    """计算每个酒店的B/A比值并输出前1000名，包含原始数据"""

    print("开始读取CSV文件...")
    # 读取CSV文件
    df = pd.read_csv(INPUT_CSV)
    print(f"数据读取完成，共 {len(df)} 条记录")

    # 按s_hotel_id分组处理
    results = []

    # 获取所有唯一的hotel_id
    hotel_ids = df['s_hotel_id'].unique()
    total_hotels = len(hotel_ids)

    print(f"开始处理 {total_hotels} 个酒店的数据...")

    for i, hotel_id in enumerate(hotel_ids):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{total_hotels}")

        # 获取当前酒店的数据
        hotel_data = df[df['s_hotel_id'] == hotel_id]

        # 计算A值：spl_room_name和spl_room_bed_name组合后去重的数量
        spl_combinations = hotel_data[[
            'spl_room_name', 'spl_room_bed_name']].dropna()
        if len(spl_combinations) > 0:
            spl_combinations['combined'] = spl_combinations['spl_room_name'].astype(
                str) + '|' + spl_combinations['spl_room_bed_name'].astype(str)
            A = len(spl_combinations['combined'].unique())
        else:
            A = 0

        # 计算B值：s_room_name和s_room_bed_name组合后去重的数量
        room_combinations = hotel_data[[
            's_room_name', 's_room_bed_name']].dropna()
        if len(room_combinations) > 0:
            room_combinations['combined'] = room_combinations['s_room_name'].astype(
                str) + '|' + room_combinations['s_room_bed_name'].astype(str)
            B = len(room_combinations['combined'].unique())
        else:
            B = 0

        # 计算比值B/A
        if A > 0:
            ratio = B / A
        else:
            ratio = 0

        results.append({
            's_hotel_id': hotel_id,
            'A_count': A,
            'B_count': B,
            'ratio_B_A': ratio
        })

    print("数据处理完成，开始排序...")

    # 转换为DataFrame并排序
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ratio_B_A', ascending=False)

    # 获取前1000名
    top_1000 = results_df.head(1000)

    print(
        f"前1000名的比值范围: {top_1000['ratio_B_A'].min():.4f} - {top_1000['ratio_B_A'].max():.4f}")
    print(f"平均比值: {top_1000['ratio_B_A'].mean():.4f}")

    # 获取前1000名酒店的原始数据
    top_hotel_ids = top_1000['s_hotel_id'].tolist()
    original_data = df[df['s_hotel_id'].isin(top_hotel_ids)].copy()

    # 将统计结果合并到原始数据中
    final_data = original_data.merge(top_1000[['s_hotel_id', 'A_count', 'B_count', 'ratio_B_A']],
                                     on='s_hotel_id', how='left')

    # 按ratio_B_A降序和s_hotel_id排序
    final_data = final_data.sort_values(
        ['ratio_B_A', 's_hotel_id'], ascending=[False, True])

    # 保存到CSV文件
    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"结果已保存到: {OUTPUT_CSV}")
    print(f"最终输出包含 {len(final_data)} 条原始记录")

    return final_data


if __name__ == "__main__":
    result = calculate_room_ratio()
    print("处理完成！")
