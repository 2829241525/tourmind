import pandas as pd
import glob
import os
import numpy as np
import re


def merge_and_remove_conflicts(input_files, output_file):
    # 合并所有CSV文件
    print("开始合并CSV文件...")
    dfs = []
    for file in input_files:
        print(f"正在读取: {os.path.basename(file)}")
        df = pd.read_csv(file)
        dfs.append(df)

    # 合并所有数据框
    df = pd.concat(dfs, ignore_index=True)
    # 找出'spl_room_text'和's_room_text'相同的组
    # grouped = df.groupby(['spl_room_text', 's_room_text'])
    grouped = df.groupby(['text1', 'text2'])

    # 找出标签不一致的组
    conflicts = []
    conflict_indices = set()  # 用于存储需要删除的行的索引

    for name, group in grouped:
        if len(group['label'].unique()) > 1:
            conflicts.append(group)
            conflict_indices.update(group.index)

    if conflicts:
        # 将有冲突的数据合并到一个DataFrame
        conflicts_df = pd.concat(conflicts)
        print(f'发现 {len(conflicts)} 组文本相同但标签不同的数据')
        print(f'冲突数据总行数: {len(conflicts_df)}')

        # 删除所有冲突数据
        df_clean = df.drop(index=conflict_indices)
    else:
        print('没有发现文本相同但标签不同的数据')
        df_clean = df

    # 基于指定列删除重复行
    df_unique = df_clean.drop_duplicates(
        subset=['text1', 'text2', 'label'], keep='first')

    # 根据spl_room_text分组处理数据平衡问题
    print("开始处理数据平衡...")

    # 根据spl_room_text分组
    spl_groups = df_unique.groupby('text1')

    # 存储要保留的索引
    indices_to_keep = []

    # 遍历每个分组
    for spl_text, group in spl_groups:
        # 检查该组中是否存在label为1的数据
        if 1 not in group['label'].values:
            print(f"分组 '{spl_text}' 中不存在label=1的数据，将删除整个分组")
            continue

        # 计算label为0和1的数量
        label_0_count = (group['label'] == 0).sum()
        label_1_count = (group['label'] == 1).sum()

        # 如果label为0的比例过高（超过6:1）
        if label_0_count / label_1_count > 10:
            # 计算需要保留的label为0的样本数
            target_0_count = label_1_count * 10

            # 随机选择要保留的label为0的样本
            label_0_indices = group[group['label'] == 0].index
            np.random.seed(42)  # 设置随机种子以确保结果可复现
            indices_to_keep_0 = np.random.choice(
                label_0_indices,
                size=int(target_0_count),
                replace=False
            )

            # 保留所有label为1的样本
            label_1_indices = group[group['label'] == 1].index

            # 合并要保留的索引
            indices_to_keep.extend(list(indices_to_keep_0))
            indices_to_keep.extend(list(label_1_indices))

            print(
                f"分组 '{spl_text}' 中label比例不平衡：{label_0_count}:{label_1_count}，调整为 {target_0_count}:{label_1_count}")
        else:
            # 如果比例正常，保留所有数据
            indices_to_keep.extend(list(group.index))

    # 根据索引筛选数据
    balanced_df = df_unique.loc[indices_to_keep]

    # 更新df_unique为平衡后的数据
    df_unique = balanced_df.reset_index(drop=True)

    # 处理text1和text2中的空格问题
    print("处理text1和text2中的空格问题...")

    # 定义空格处理函数：去除首尾空格并将连续多个空格替换为一个空格
    def clean_spaces(text):
        if isinstance(text, str):
            # 首先去除首尾空格，然后将连续的多个空格替换为一个空格
            return re.sub(r'\s+', ' ', text.strip())
        return text

    # 应用空格处理函数
    df_unique['text1'] = df_unique['text1'].apply(clean_spaces)
    df_unique['text2'] = df_unique['text2'].apply(clean_spaces)

    print(f"空格处理完成")
    print(f"数据平衡处理完成，处理后的行数: {len(df_unique)}")

    # 保存去重后的结果到新的CSV文件
    df_unique.to_csv(output_file, index=False)

    # 打印统计信息
    total_rows = len(df)
    conflicts_removed = len(conflict_indices)
    clean_rows = len(df_clean)
    final_rows = len(df_unique)

    print(f'合并后总行数: {total_rows}')
    print(f'删除的冲突数据行数: {conflicts_removed}')
    print(f'删除冲突后的行数: {clean_rows}')
    print(f'最终去重后的行数: {final_rows}')

    # 统计标签数量和比例
    label_counts = df_unique['label'].value_counts()
    print('\n标签分布统计:')
    for label, count in label_counts.items():
        percentage = (count / final_rows) * 100
        print(f'标签 {label}: {count} 条数据 ({percentage:.2f}%)')


if __name__ == '__main__':
    # 指定包含所有输入CSV文件的目录/home/maxon/disk2/roomMatch/room_match/process_data/room_samples.csv
    input_dir = '/home/maxon/disk2/roomMatch/room_match/deberta/data'
    # 获取所有需要处理的CSV文件
    input_files = [
        os.path.join(input_dir, 'room_samples.csv')
    ]
    output_file = os.path.join(input_dir, 'room_samples_clean.csv')
    merge_and_remove_conflicts(input_files, output_file)
