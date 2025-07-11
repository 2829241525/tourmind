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


def contains_side_or_access(text, word):
    """
    检查文本是否包含指定词汇（side或access）
    前面可以有字母、特殊符号、空格等
    后面不能有字母，但可以有特殊符号、空格、数字等
    """
    if not isinstance(text, str):
        return False

    # 使用正则表达式：前面可以是任何字符，后面不能是字母
    # (?![a-zA-Z]) 是负向前瞻，确保后面不是字母
    pattern = f'{word.lower()}(?![a-zA-Z])'
    return bool(re.search(pattern, text.lower()))


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

    # 继续处理side和access词汇的筛选
    print(f"\n开始处理side和access词汇的筛选...")

    # 对side进行筛选
    label_1_data_side = df_filtered[df_filtered['label'] == 1].copy()
    to_remove_indices_side = []

    for idx, row in label_1_data_side.iterrows():
        spl_has_side = contains_side_or_access(row['spl_room_text'], 'side')
        s_has_side = contains_side_or_access(row['s_room_text'], 'side')

        if spl_has_side != s_has_side:
            to_remove_indices_side.append(idx)
            print(f"将删除行 {idx} (side不匹配):")
            print(f"  spl_room_text: {row['spl_room_text']}")
            print(f"  s_room_text: {row['s_room_text']}")
            print(f"  spl_has_side: {spl_has_side}, s_has_side: {s_has_side}")
            print()

    print(f"发现 {len(to_remove_indices_side)} 条label=1且side不匹配的数据，将被删除")

    # 删除side不匹配的行
    df_filtered = df_filtered.drop(index=to_remove_indices_side)

    # 对access进行筛选
    label_1_data_access = df_filtered[df_filtered['label'] == 1].copy()
    to_remove_indices_access = []

    for idx, row in label_1_data_access.iterrows():
        spl_has_access = contains_side_or_access(
            row['spl_room_text'], 'access')
        s_has_access = contains_side_or_access(row['s_room_text'], 'access')

        if spl_has_access != s_has_access:
            to_remove_indices_access.append(idx)
            print(f"将删除行 {idx} (access不匹配):")
            print(f"  spl_room_text: {row['spl_room_text']}")
            print(f"  s_room_text: {row['s_room_text']}")
            print(
                f"  spl_has_access: {spl_has_access}, s_has_access: {s_has_access}")
            print()

    print(f"发现 {len(to_remove_indices_access)} 条label=1且access不匹配的数据，将被删除")

    # 删除access不匹配的行
    df_filtered = df_filtered.drop(index=to_remove_indices_access)

    print(f"\n最终过滤后数据行数: {len(df_filtered)}")
    print(
        f"总共删除的数据行数: {len(to_remove_indices) + len(to_remove_indices_side) + len(to_remove_indices_access)}")
    print(f"  其中view不匹配删除: {len(to_remove_indices)} 条")
    print(f"  其中side不匹配删除: {len(to_remove_indices_side)} 条")
    print(f"  其中access不匹配删除: {len(to_remove_indices_access)} 条")

    # 统计过滤后的标签分布
    print(f"\n过滤后各标签数量统计:")
    label_counts = df_filtered['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df_filtered)) * 100
        print(f'标签 {label}: {count} 条数据 ({percentage:.2f}%)')

    # 统计label=1数据中spl_room_text和s_room_text完全一致的占比
    label_1_filtered = df_filtered[df_filtered['label'] == 1]
    if len(label_1_filtered) > 0:
        identical_count = (
            label_1_filtered['spl_room_text'] == label_1_filtered['s_room_text']).sum()
        identical_percentage = (identical_count / len(label_1_filtered)) * 100
        print(f"\nlabel=1数据中文本完全一致的统计:")
        print(f"总数: {len(label_1_filtered)} 条")
        print(f"完全一致: {identical_count} 条")
        print(f"一致占比: {identical_percentage:.2f}%")

        # 统计分词后set处理的一致性
        set_identical_count = 0
        for idx, row in label_1_filtered.iterrows():
            # 分词并转换为set
            spl_words = set(str(row['spl_room_text']).lower().split())
            s_words = set(str(row['s_room_text']).lower().split())

            # 比较两个set是否相等
            if spl_words == s_words:
                set_identical_count += 1

        set_identical_percentage = (
            set_identical_count / len(label_1_filtered)) * 100
        print(f"\nlabel=1数据中分词后set处理一致的统计:")
        print(f"总数: {len(label_1_filtered)} 条")
        print(f"分词set一致: {set_identical_count} 条")
        print(f"分词set一致占比: {set_identical_percentage:.2f}%")

        # 将label=1数据按set是否一致分类
        set_identical_indices = []
        set_different_indices = []

        for idx, row in label_1_filtered.iterrows():
            # 分词并转换为set
            spl_words = set(str(row['spl_room_text']).lower().split())
            s_words = set(str(row['s_room_text']).lower().split())

            # 根据set是否相等分类
            if spl_words == s_words:
                set_identical_indices.append(idx)
            else:
                set_different_indices.append(idx)

        # 输出set一致的数据
        if set_identical_indices:
            set_identical_data = df_filtered.loc[set_identical_indices]
            set_identical_file = output_file.replace(
                '.csv', '_set_identical.csv')
            set_identical_data.to_csv(
                set_identical_file, index=False, encoding='utf-8')
            print(f"\nset一致的数据已保存到: {set_identical_file}")
            print(f"set一致数据行数: {len(set_identical_data)}")

        # 输出set不一致的数据
        if set_different_indices:
            set_different_data = df_filtered.loc[set_different_indices]
            set_different_file = output_file.replace(
                '.csv', '_set_different.csv')
            set_different_data.to_csv(
                set_different_file, index=False, encoding='utf-8')
            print(f"\nset不一致的数据已保存到: {set_different_file}")
            print(f"set不一致数据行数: {len(set_different_data)}")
    else:
        print(f"\n没有label=1的数据")

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


def test_side_access_detection():
    """
    测试side和access检测函数
    """
    side_test_cases = [
        ("sea side room", True),       # side后面是空格
        ("seaside room", True),        # side后面是空格
        ("side123", True),             # side后面是数字
        ("room-side", True),           # side后面没有字符（字符串结尾）
        ("side-view", True),           # side后面是连字符
        ("side.", True),               # side后面是句号
        ("SIDE ROOM", True),           # 大写，side后面是空格
        ("room without side", True),   # side后面没有字符（字符串结尾）
        ("no side here", True),        # side后面是空格
        ("inside room", False),        # side后面是字母e
        ("outside view", False),       # side后面是字母空格前的字母
        ("sided room", False),         # side后面是字母d
        ("bedroom", False),            # 不包含side
    ]

    access_test_cases = [
        ("beach access", True),        # access后面没有字符（字符串结尾）
        ("beach access ", True),       # access后面是空格
        ("access123", True),           # access后面是数字
        ("room-access", True),         # access后面没有字符（字符串结尾）
        ("access-road", True),         # access后面是连字符
        ("easy access room", True),    # access后面是空格
        ("ACCESS ROAD", True),         # 大写，access后面是空格
        ("no access", True),           # access后面没有字符（字符串结尾）
        ("room access.", True),        # access后面是句号
        ("accessible room", False),    # access后面是字母i
        ("beachaccess", True),         # access后面没有字符（字符串结尾）
        ("accessory", False),          # access后面是字母o
        ("bedroom", False),            # 不包含access
    ]

    print("\n测试side检测函数:")
    for text, expected in side_test_cases:
        result = contains_side_or_access(text, 'side')
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> {result} (期望: {expected})")

    print("\n测试access检测函数:")
    for text, expected in access_test_cases:
        result = contains_side_or_access(text, 'access')
        status = "✓" if result == expected else "✗"
        print(f"{status} '{text}' -> {result} (期望: {expected})")


if __name__ == '__main__':
    # 测试view检测函数
    test_view_detection()

    # 测试side和access检测函数
    test_side_access_detection()
    print("\n" + "="*50 + "\n")

    # 处理数据
    input_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean.csv'
    output_file = '/home/maxon/disk2/roomMatch/room_match/chinese_match/data/room_group/room_group_train_fix_clean_filtered.csv'

    filter_view_mismatched_data(input_file, output_file)
