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
        # 处理A值计算：当spl_room_bed_name为空时，只使用spl_room_name
        # 先获取spl_room_name非空的数据
        valid_room_data = hotel_data[hotel_data['spl_room_name'].notna()].copy(
        )

        if len(valid_room_data) > 0:
            # 创建组合：如果spl_room_bed_name为空，则只使用spl_room_name
            def create_combination(row):
                room_name = str(row['spl_room_name'])
                bed_name = str(row['spl_room_bed_name']) if pd.notna(
                    row['spl_room_bed_name']) else ''
                if bed_name and bed_name != 'nan':
                    return f"{room_name}|{bed_name}"
                else:
                    return room_name

            valid_room_data['combined'] = valid_room_data.apply(
                create_combination, axis=1)
            spl_combinations = valid_room_data
        else:
            spl_combinations = pd.DataFrame()

        if len(spl_combinations) > 0:
            # 对组合进行去重
            unique_combinations = spl_combinations['combined'].unique()
            A = len(unique_combinations)
        else:
            A = 0

        # 计算B值：s_room_name和s_room_bed_name组合后去重的数量
        # 处理B值计算：当s_room_bed_name为空时，只使用s_room_name
        # 先获取s_room_name非空的数据
        valid_room_b_data = hotel_data[hotel_data['s_room_name'].notna()].copy(
        )

        if len(valid_room_b_data) > 0:
            # 创建组合：如果s_room_bed_name为空，则只使用s_room_name
            def create_combination_b(row):
                room_name = str(row['s_room_name'])
                bed_name = str(row['s_room_bed_name']) if pd.notna(
                    row['s_room_bed_name']) else ''
                if bed_name and bed_name != 'nan':
                    return f"{room_name}|{bed_name}"
                else:
                    return room_name

            valid_room_b_data['combined'] = valid_room_b_data.apply(
                create_combination_b, axis=1)
            room_combinations = valid_room_b_data
        else:
            room_combinations = pd.DataFrame()

        if len(room_combinations) > 0:
            # 对组合进行去重
            unique_room_combinations = room_combinations['combined'].unique()
            B = len(unique_room_combinations)
        else:
            B = 0

        # 计算比值B/A
        if A > 0:
            ratio = B / A
        else:
            ratio = 0

        # 计算该酒店的原始记录数量
        hotel_record_count = len(hotel_data)

        results.append({
            's_hotel_id': hotel_id,
            'A_count': A,
            'B_count': B,
            'ratio_B_A': ratio,
            'record_count': hotel_record_count
        })

    print("数据处理完成，开始排序...")

    # 转换为DataFrame
    results_df = pd.DataFrame(results)

    # 过滤：只保留组内数量大于10的酒店
    filtered_df = results_df[results_df['record_count'] > 10]
    print(f"组内数量大于10的酒店数量: {len(filtered_df)}")

    # 按ratio从高到低排序
    filtered_df = filtered_df.sort_values('ratio_B_A', ascending=False)

    # 获取前1000名
    top_1000 = filtered_df.head(1000)

    print(
        f"前1000名的比值范围: {top_1000['ratio_B_A'].min():.4f} - {top_1000['ratio_B_A'].max():.4f}")
    print(f"平均比值: {top_1000['ratio_B_A'].mean():.4f}")

    # 获取前1000名酒店的原始数据
    top_hotel_ids = top_1000['s_hotel_id'].tolist()
    original_data = df[df['s_hotel_id'].isin(top_hotel_ids)].copy()

    # 将统计结果合并到原始数据中
    final_data = original_data.merge(top_1000[['s_hotel_id', 'A_count', 'B_count', 'ratio_B_A', 'record_count']],
                                     on='s_hotel_id', how='left')

    # 按ratio_B_A降序和s_hotel_id排序
    final_data = final_data.sort_values(
        ['ratio_B_A', 's_hotel_id'], ascending=[False, True])

    # 处理字段：如果包含|字符，则只保留左侧内容
    fields_to_process = ['spl_room_name',
                         'spl_room_bed_name', 's_room_name', 's_room_bed_name']

    for field in fields_to_process:
        if field in final_data.columns:
            # 对非空值进行处理
            mask = final_data[field].notna()
            final_data.loc[mask, field] = final_data.loc[mask, field].astype(str).apply(
                lambda x: x.split('|')[0] if '|' in x else x
            )

    print("字段处理完成：已移除|字符右侧的内容")

    # 保存到CSV文件
    final_data.to_csv(OUTPUT_CSV, index=False)
    print(f"结果已保存到: {OUTPUT_CSV}")
    print(f"最终输出包含 {len(final_data)} 条原始记录")

    return final_data


if __name__ == "__main__":
    result = calculate_room_ratio()
    print("处理完成！")
