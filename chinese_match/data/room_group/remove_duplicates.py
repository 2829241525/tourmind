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
        df = pd.read_csv(file, encoding='utf-8')

        # 验证列名和数据类型
        expected_columns = ['SHotelId',
                            'spl_room_text', 's_room_text', 'label']
        if list(df.columns) != expected_columns:
            print(f"警告：文件 {file} 的列名不匹配")
            print(f"期望的列名: {expected_columns}")
            print(f"实际的列名: {list(df.columns)}")

        # 先检查并处理缺失值
        print(f"文件 {os.path.basename(file)} 原始行数: {len(df)}")
        print(f"各列缺失值情况:")
        print(df.isnull().sum())

        # 删除任何包含缺失值的行
        df = df.dropna()
        print(f"删除缺失值后行数: {len(df)}")

        # 确保数据类型正确（现在没有NaN值了）
        try:
            df['SHotelId'] = df['SHotelId'].astype('int64')
            df['label'] = df['label'].astype('int64')
            df['spl_room_text'] = df['spl_room_text'].astype('str')
            df['s_room_text'] = df['s_room_text'].astype('str')
        except Exception as e:
            print(f"数据类型转换出错: {e}")
            print(f"SHotelId unique values: {df['SHotelId'].unique()[:10]}")
            print(f"label unique values: {df['label'].unique()[:10]}")

        dfs.append(df)

    # 合并所有数据框
    df = pd.concat(dfs, ignore_index=True)

    # 数据清理：检查和过滤数据
    print("开始最终数据清理...")
    initial_rows = len(df)

    # 确认列数
    if len(df.columns) != 4:
        print(f"警告：数据框应有4列，实际有{len(df.columns)}列")

    # 删除label不为0或1的数据
    print(f"清理前总行数: {len(df)}")

    # 检查label列的唯一值
    unique_labels = df['label'].unique()
    print(f"发现的label值: {unique_labels}")

    # 过滤label值，只保留0和1
    df = df[df['label'].isin([0, 1])]
    print(f"删除label不为0或1的数据后行数: {len(df)}")

    # 再次检查是否有缺失值（理论上应该没有）
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"发现 {missing_values} 个缺失值，继续删除...")
        df = df.dropna()
        print(f"删除缺失值后行数: {len(df)}")

    cleaned_rows = len(df)
    print(f"数据清理完成，从 {initial_rows} 行清理到 {cleaned_rows} 行")

    # 步骤1：合并和清理完成
    print(f"步骤1完成：数据合并和清理，处理后行数: {len(df)}")

    # 找出'spl_room_text'和's_room_text'相同的组
    grouped = df.groupby(['spl_room_text', 's_room_text'])

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

    # 步骤2：删除冲突数据完成
    print(f"步骤2完成：删除冲突数据，处理后行数: {len(df_clean)}")

    # 基于指定列删除重复行
    df_unique = df_clean.drop_duplicates(
        subset=['spl_room_text', 's_room_text', 'label'], keep='first')

    # 步骤3：删除重复行完成
    print(f"步骤3完成：删除重复行，处理后行数: {len(df_unique)}")

    # 步骤4：数据平衡处理（已注释掉）
    # print("开始处理数据平衡...")
    # 根据spl_room_text分组处理数据平衡问题
    # spl_groups = df_unique.groupby('spl_room_text')
    # 存储要保留的数据行
    # balanced_rows = []
    # 遍历每个分组
    # for spl_text, group in spl_groups:
    #     # 计算label为0和1的数量
    #     label_0_count = (group['label'] == 0).sum()
    #     label_1_count = (group['label'] == 1).sum()
    #     # 如果没有label=1的数据，保留所有数据
    #     if label_1_count == 0:
    #         print(f"分组 '{spl_text}' 中不存在label=1的数据，保留所有数据")
    #         balanced_rows.append(group.copy())
    #         continue
    #     # 如果label为0的比例过高（超过10:1）
    #     if label_0_count / label_1_count > 10:
    #         # 计算需要保留的label为0的样本数
    #         target_0_count = label_1_count * 10
    #         # 获取label为0和1的数据
    #         label_0_data = group[group['label'] == 0].copy()
    #         label_1_data = group[group['label'] == 1].copy()
    #         # 随机选择要保留的label为0的样本
    #         np.random.seed(42)  # 设置随机种子以确保结果可复现
    #         if len(label_0_data) > target_0_count:
    #             label_0_selected = label_0_data.sample(
    #                 n=int(target_0_count), random_state=42)
    #         else:
    #             label_0_selected = label_0_data
    #         # 合并要保留的数据
    #         group_balanced = pd.concat(
    #             [label_0_selected, label_1_data], ignore_index=False)
    #         balanced_rows.append(group_balanced)
    #         print(
    #             f"分组 '{spl_text}' 中label比例不平衡：{label_0_count}:{label_1_count}，调整为 {len(label_0_selected)}:{label_1_count}")
    #     else:
    #         # 如果比例正常，保留所有数据
    #         balanced_rows.append(group.copy())
    # # 合并所有平衡后的数据
    # if balanced_rows:
    #     df_unique = pd.concat(balanced_rows, ignore_index=True)
    # else:
    #     # 如果没有数据，保持原来的数据框但清空
    #     df_unique = df_unique.iloc[0:0].copy()
    # # 确保数据类型正确
    # if len(df_unique) > 0:
    #     df_unique['SHotelId'] = df_unique['SHotelId'].astype('int64')
    #     df_unique['label'] = df_unique['label'].astype('int64')
    #     df_unique['spl_room_text'] = df_unique['spl_room_text'].astype('str')
    #     df_unique['s_room_text'] = df_unique['s_room_text'].astype('str')

    # 步骤4：数据平衡处理（已跳过）
    print(f"步骤4完成：跳过数据平衡处理，保持行数: {len(df_unique)}")

    # 处理spl_room_text和s_room_text中的空格问题
    print("处理spl_room_text和s_room_text中的空格问题...")

    # 修改空格处理函数：只处理普通空格，保留其他空白字符的原有格式
    def clean_spaces(text):
        if isinstance(text, str):
            # 只去除首尾空格，将连续的普通空格替换为一个空格
            # 保留制表符、换行符等其他空白字符
            text = text.strip()  # 去除首尾空格
            text = re.sub(r' +', ' ', text)  # 只替换连续的普通空格
            return text
        return text

    # 应用空格处理函数
    df_unique['spl_room_text'] = df_unique['spl_room_text'].apply(clean_spaces)
    df_unique['s_room_text'] = df_unique['s_room_text'].apply(clean_spaces)

    print(f"空格处理完成")
    print(f"数据平衡处理完成，处理后的行数: {len(df_unique)}")

    # 步骤5：空格处理完成
    print(f"步骤5完成：空格处理，处理后行数: {len(df_unique)}")

    # 确保最终输出格式与原始文件一致
    print("确保输出格式一致...")

    # 确保列顺序正确
    expected_columns = ['SHotelId', 'spl_room_text', 's_room_text', 'label']
    df_unique = df_unique[expected_columns]

    # 确保数据类型正确
    df_unique['SHotelId'] = df_unique['SHotelId'].astype('int64')
    df_unique['label'] = df_unique['label'].astype('int64')
    df_unique['spl_room_text'] = df_unique['spl_room_text'].astype('str')
    df_unique['s_room_text'] = df_unique['s_room_text'].astype('str')

    print(f"最终输出格式:")
    print(f"列名: {list(df_unique.columns)}")
    print(f"数据类型: {df_unique.dtypes.to_dict()}")
    print(f"数据形状: {df_unique.shape}")

    # 最终过滤：删除spl_room_text和s_room_text完全一致且label为0的数据
    print("执行最终过滤...")
    before_filter_rows = len(df_unique)

    # 创建条件：spl_room_text和s_room_text完全一致且label为0
    identical_text_label0 = (df_unique['spl_room_text'] == df_unique['s_room_text']) & (
        df_unique['label'] == 0)

    # 统计要删除的数据
    rows_to_remove = identical_text_label0.sum()
    print(f"发现 {rows_to_remove} 条spl_room_text和s_room_text完全一致且label为0的数据，将被删除")

    # 删除这些数据
    df_unique = df_unique[~identical_text_label0]

    after_filter_rows = len(df_unique)
    print(f"最终过滤完成，从 {before_filter_rows} 行过滤到 {after_filter_rows} 行")

    # 步骤6：最终过滤完成
    print(f"步骤6完成：最终过滤，处理后行数: {len(df_unique)}")

    # 保存去重后的结果到新的CSV文件
    df_unique.to_csv(output_file, index=False, encoding='utf-8')

    # 打印统计信息
    conflicts_removed = len(conflict_indices) if conflict_indices else 0
    clean_rows = len(df_clean)
    final_rows = len(df_unique)

    print(f'\n=== 数据处理统计信息 ===')
    print(f'原始合并后总行数: {initial_rows}')
    print(f'数据清理后行数: {cleaned_rows}')
    print(f'删除的冲突数据行数: {conflicts_removed}')
    print(f'删除冲突后的行数: {clean_rows}')
    print(f'数据平衡处理后行数: {before_filter_rows}')
    print(f'删除文本一致且label=0的数据行数: {rows_to_remove}')
    print(f'最终输出行数: {final_rows}')
    print(f'总共删除行数: {initial_rows - final_rows}')

    # 统计标签数量和比例
    label_counts = df_unique['label'].value_counts()
    print('\n标签分布统计:')
    for label, count in label_counts.items():
        percentage = (count / final_rows) * 100
        print(f'标签 {label}: {count} 条数据 ({percentage:.2f}%)')

    print(f'\n=== 输出文件 ===')
    print(f'最终输出文件: {output_file}')


if __name__ == '__main__':
    # 指定包含输入CSV文件的目录
    input_dir = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group'
    # 获取需要处理的CSV文件
    input_files = [
        os.path.join(input_dir, 'room_group_train_fix.csv')
    ]
    output_file = os.path.join(input_dir, 'room_group_train_fix_clean.csv')
    merge_and_remove_conflicts(input_files, output_file)
