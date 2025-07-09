import pandas as pd
import re
import os


def contains_view_as_word(text):
    """
    检查文本是否包含独立的'view'单词
    view前后不能有字母数字，可以有空格或标点符号
    """
    if not isinstance(text, str):
        return False

    # 使用正则表达式匹配独立的view单词
    # \b匹配单词边界，但我们需要更精确的控制
    # 使用(?<![a-zA-Z0-9])和(?![a-zA-Z0-9])来确保前后不是字母数字
    pattern = r'(?<![a-zA-Z0-9])view(?![a-zA-Z0-9])'
    return bool(re.search(pattern, text, re.IGNORECASE))


def filter_view_mismatched_data(input_file, output_file):
    """
    过滤掉label=1且view不匹配的数据
    """
    print("开始读取CSV文件...")
    df = pd.read_csv(input_file, encoding='utf-8')

    print(f"原始数据行数: {len(df)}")
    print(f"各标签数量统计:")
    print(df['label'].value_counts())

    # 只处理label=1的数据
    label_1_data = df[df['label'] == 1].copy()
    print(f"\nlabel=1的数据行数: {len(label_1_data)}")

    # 检查每行数据是否需要过滤
    to_remove_indices = []

    for idx, row in label_1_data.iterrows():
        spl_has_view = contains_view_as_word(row['spl_room_text'])
        s_has_view = contains_view_as_word(row['s_room_text'])

        # 如果一个包含view而另一个不包含，则标记删除
        if spl_has_view != s_has_view:
            to_remove_indices.append(idx)
            print(f"将删除行 {idx}:")
            print(f"  spl_room_text: {row['spl_room_text']}")
            print(f"  s_room_text: {row['s_room_text']}")
            print(f"  spl_has_view: {spl_has_view}, s_has_view: {s_has_view}")
            print()

    print(f"发现 {len(to_remove_indices)} 条label=1且view不匹配的数据，将被删除")

    # 从原始数据中删除这些行
    df_filtered = df.drop(index=to_remove_indices)

    print(f"过滤后数据行数: {len(df_filtered)}")
    print(f"删除的数据行数: {len(to_remove_indices)}")

    # 统计过滤后的标签分布
    print(f"\n过滤后各标签数量统计:")
    label_counts = df_filtered['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df_filtered)) * 100
        print(f'标签 {label}: {count} 条数据 ({percentage:.2f}%)')

    # 保存过滤后的数据
    df_filtered.to_csv(output_file, index=False, encoding='utf-8')
    print(f"\n过滤后的数据已保存到: {output_file}")

    return df_filtered


def test_view_detection():
    """
    测试view检测函数
    """
    test_cases = [
        ("ocean view room", True),
        ("sea view", True),
        ("view of the city", True),
        ("overview room", False),  # view前面有字母
        ("review room", False),   # view前面有字母
        ("view123", False),       # view后面有数字
        ("abc view def", True),
        ("view,room", True),      # view后面是标点符号
        ("view-room", True),      # view后面是连字符
        ("view ", True),          # view后面是空格
        (" view", True),          # view前面是空格
        ("preview", False),       # view前面有字母
        ("viewer", False),        # view后面有字母
        ("View Room", True),      # 大写的VIEW
        ("VIEW OF SEA", True),    # 全大写
        ("room with view.", True),  # view后面是句号
    ]

    print("测试view检测函数:")
    for text, expected in test_cases:
        result = contains_view_as_word(text)
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> {result} (期望: {expected})")


if __name__ == '__main__':
    # 测试view检测函数
    test_view_detection()
    print("\n" + "="*50 + "\n")

    # 处理数据
    input_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean.csv'
    output_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean_filtered.csv'

    filter_view_mismatched_data(input_file, output_file)
