import pandas as pd
import os


def filter_label_1_data():
    """
    读取room_group_train_fix.csv文件，筛选出label为1的数据，
    保存到新的CSV文件中，保持原有格式
    """
    # 输入文件路径
    input_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix.csv'

    # 输出文件路径
    output_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_label1.csv'

    print(f"正在读取文件: {input_file}")

    try:
        # 读取CSV文件
        df = pd.read_csv(input_file)

        print(f"原始数据总行数: {len(df)}")
        print(f"数据列名: {list(df.columns)}")

        # 查看label列的分布
        print(f"Label分布:")
        print(df['label'].value_counts())

        # 筛选label为1的数据
        df_filtered = df[df['label'] == 0]

        print(f"筛选后的数据行数 (label=1): {len(df_filtered)}")

        # 保存到新的CSV文件，保持原有格式
        df_filtered.to_csv(output_file, index=False, encoding='utf-8')

        print(f"筛选后的数据已保存到: {output_file}")

        # 显示前几行样本数据
        print("\n前5行筛选后的数据:")
        print(df_filtered.head())

    except Exception as e:
        print(f"处理过程中出现错误: {str(e)}")


if __name__ == "__main__":
    filter_label_1_data()
