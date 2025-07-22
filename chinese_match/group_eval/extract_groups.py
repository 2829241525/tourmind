import pandas as pd
import os


def extract_groups(input_csv_path, output_csv_path):
    """提取并保存2个及以上的分组数据"""
    # 读取CSV文件
    df = pd.read_csv(input_csv_path, on_bad_lines='skip', escapechar='\\')

    # 计算每个酒店每个标记的房间数量
    group_sizes = df.groupby(['s_hotel_id', '标记列']
                             ).size().reset_index(name='group_size')

    # 找出size >= 2的组
    large_groups = group_sizes[group_sizes['group_size'] >= 2]

    # 按group_size降序排序
    large_groups = large_groups.sort_values('group_size', ascending=False)

    # 获取这些组的详细数据
    result_rows = []
    for _, group in large_groups.iterrows():
        hotel_id = group['s_hotel_id']
        mark = group['标记列']

        # 获取该组的所有房间数据
        group_data = df[
            (df['s_hotel_id'] == hotel_id) &
            (df['标记列'] == mark)
        ]

        result_rows.extend(group_data.to_dict('records'))

    # 转换为DataFrame并保存
    if result_rows:
        result_df = pd.DataFrame(result_rows)
        result_df.to_csv(output_csv_path, index=False, encoding='utf-8')
        print(f"已保存分组数据到: {output_csv_path}")
        print(f"共有 {len(large_groups)} 个大于等于2的分组")
        print(f"总房间数: {len(result_df)} 个")
    else:
        print("没有找到符合条件的分组")


if __name__ == '__main__':
    # 示例用法
    input_csv = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/s_room_eval.csv"
    output_csv = "/home/maxon/disk2/roomMatch/room_match/chinese_match/group_eval/s_room_eval_large_groups.csv"
    extract_groups(input_csv, output_csv)
