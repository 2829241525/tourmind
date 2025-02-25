import pandas as pd
import re
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import time

# 配置常量
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 文件路径配置
INPUT_FILE = os.path.join(PROJECT_ROOT, "room_match/data/test.csv")
OUTPUT_MATCH_FILE = os.path.join(PROJECT_ROOT, "room_match/data/processed_country.csv")
OUTPUT_TRIPLET_FILE = os.path.join(PROJECT_ROOT, "room_match/data/processed_country_triplet.csv")
MAPPING_FILE = os.path.join(CURRENT_DIR, "../data/mapping.xlsx")

# 文本处理配置
UNKNOWN_TEXT = "[unknown]"
SUPPLY_PREFIX = "[supply]"
ROOM_PREFIX = "[room]"
BED_PREFIX = "[bed]"

# 正则表达式配置
CHINESE_PATTERN = re.compile(r'^[\u4e00-\u9fff\s]+$')
ENGLISH_CLEAN_PATTERN = re.compile(r'[^a-zA-Z\s]')
CHINESE_REMOVE_PATTERN = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+')

class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.room_rules, self.bed_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> Tuple[dict, List[Tuple[str, str, int]]]:
        # 定义映射替换模式
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        room_rules = {}
        bed_rules = []

        # 按照sheet顺序构建字典
        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)
            if not {'key', 'value'}.issubset(sheet_data.columns):
                continue

            # 处理每一行数据
            current_sheet_rules = []
            for _, row in sheet_data.iterrows():
                key = str(row['key']).strip().lower()
                values = [v.strip().strip('"') for v in str(row['value']).split(',')]
                rules = [(value.lower(), key, len(value)) for value in values if value]
                current_sheet_rules.extend(rules)
            
            # 按原文本长度降序排序当前sheet的规则
            current_sheet_rules = sorted(current_sheet_rules, key=lambda x: x[2], reverse=True)
            
            # 如果是bed工作表，单独处理床型
            if sheet_name.lower() == 'bed':
                bed_rules = current_sheet_rules
            else:
                # 其他工作表作为房型规则，使用sheet_name作为key
                room_rules[sheet_name.lower()] = current_sheet_rules

        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        # 遍历每个sheet的规则
        for sheet_name, rules in self.room_rules.items():
            # 遍历当前sheet中的所有规则
            for value, key, _ in rules:
                pattern = re.compile(
                    rf'(?:^|(?<=[^a-zA-Z0-9-]))'  # 开始边界：确保前面是非字母数字和横杠
                    rf'{re.escape(value)}'  # 需要匹配的文本
                    rf'(?:$|(?=[^a-zA-Z0-9-]))',  # 结束边界：确保后面是非字母数字和横杠
                    re.IGNORECASE
                )
                if pattern.search(text_lower):  # 如果找到匹配
                    text_lower = pattern.sub(key, text_lower)
                    break  # 找到匹配后就停止当前sheet的搜索
            
            # 如果在当前sheet中找到了匹配，继续处理下一个sheet的规则
            # 这样可以让不同类型的规则（不同sheet）都能应用到文本上

        return text_lower

    def clean_bed_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        # 只使用bed工作表的规则
        for value, key, _ in self.bed_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'
                rf'{re.escape(value)}'
                rf'(?:$|(?=[^a-zA-Z0-9-]))',
                re.IGNORECASE
            )
            text_lower = pattern.sub(key, text_lower)

        return text_lower

# 创建全局的DataCleaner实例
cleaner = DataCleaner(Path(MAPPING_FILE))

def extract_english(text: str) -> str:
    if pd.isna(text):
        return UNKNOWN_TEXT
    # 首先按|分割，并去除分隔符
    parts = [part.strip() for part in text.split('|')]
    parts = [p for p in parts if p]  # 移除空字符串
    
    if not parts:  # 如果没有有效内容
        return UNKNOWN_TEXT
    
    # 检查是否所有部分都只包含中文
    def contains_only_chinese(text):
        text = text.strip()
        if not text:
            return False
        pattern = re.compile(r'^[\u4e00-\u9fff\s]+$')
        return bool(pattern.match(text))
    
    # 检查是否只包含英文和特殊字符
    def contains_only_english(text):
        text = text.strip()
        if not text:
            return False
        # 移除所有特殊字符和空格后，检查是否只包含英文
        cleaned_text = re.sub(r'[^a-zA-Z]', '', text)
        return bool(cleaned_text) and not re.search(r'[\u4e00-\u9fff]', text)
    
    # 获取文本中英文内容的数量
    def get_english_content_length(text):
        # 移除中文字符
        english_only = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', text)
        # 只保留字母和空格
        english_only = re.sub(r'[^a-zA-Z\s]', '', english_only)
        return len(english_only.strip())
    
    # 如果所有部分都只包含中文，返回所有中文内容
    if all(contains_only_chinese(part) for part in parts):
        return ' '.join(parts)
    
    # 如果所有部分都是英文，检查是否内容一致
    if all(contains_only_english(part) for part in parts):
        # 清理并比较内容
        cleaned_parts = [re.sub(r'\s+', ' ', re.sub(r'[^a-zA-Z\s]', '', part)).lower().strip() for part in parts]
        if len(set(cleaned_parts)) == 1:  # 所有内容都相同
            return parts[0].strip()
    
    # 处理混合内容，选择英文内容最多的部分
    english_lengths = [(part, get_english_content_length(part)) for part in parts]
    max_english_length = max(length for _, length in english_lengths)
    
    if max_english_length > 0:  # 如果有英文内容
        # 选择英文内容最多的部分
        selected_parts = [part for part, length in english_lengths if length == max_english_length]
        return selected_parts[0].strip()
    
    # 如果没有找到英文内容，但有中文内容，返回所有内容
    has_chinese = any(contains_only_chinese(part) for part in parts)
    if has_chinese:
        return ' '.join(parts)
    
    # 如果既没有英文也没有中文，尝试保留所有非空内容
    if parts:
        return ' '.join(parts)
    
    return UNKNOWN_TEXT

def process_text_with_prefix(prefix: str, room_name: str, bed_name: str, is_supplier: bool = False) -> str:
    # 提取英文
    room = extract_english(room_name)
    bed = extract_english(bed_name)
    prefix_text = prefix if prefix else UNKNOWN_TEXT
    
    # 只对供应商端数据进行清洗
    if is_supplier:
        if room:
            room = cleaner.clean_room_text(room)
        if bed:
            bed = cleaner.clean_bed_text(bed)
    
    # 组合成最终格式
    parts = []
    # if prefix:
    #     parts.append(f"{SUPPLY_PREFIX} {prefix_text}" if prefix_text else "")
    if room:
        parts.append(f"{ROOM_PREFIX} {room}")
    if bed:
        parts.append(f"{BED_PREFIX} {bed}")
    #小写
    parts = [part.lower() for part in parts]
    return " ".join(filter(None, parts))

def process_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_rows = len(df)
    print("\n开始处理数据...")
    start_time = time.time()
    
    # 处理supplier端数据
    print("\n处理supplier端数据...")
    tqdm.pandas(desc="supplier处理进度")
    df['spl_room_text'] = df.progress_apply(
        lambda x: process_text_with_prefix(x['supplier_name'], 
                                         x['spl_room_name'], 
                                         x['spl_room_bed_name'],
                                         is_supplier=True), 
        axis=1
    )
    
    # 处理source端数据
    print("\n处理source端数据...")
    tqdm.pandas(desc="source处理进度")
    df['s_room_text'] = df.progress_apply(
        lambda x: process_text_with_prefix(None,
                                         x['s_room_name'], 
                                         x['s_room_bed_name'],
                                         is_supplier=False), 
        axis=1
    )

    # 处理negative数据
    print("\n处理negative数据...")
    tqdm.pandas(desc="negative处理进度")
    df['negative'] = df.progress_apply(
        lambda x: process_text_with_prefix(None,
                                         x['neg_room_name'], 
                                         x['neg_bed_name'],
                                         is_supplier=False), 
        axis=1
    )
    
    # 创建匹配数据框
    df_match = pd.DataFrame({
        's_hotel_id': df['s_hotel_id'],
        'spl_room_type_id': df['spl_room_type_id'],
        'spl_room_text': df['spl_room_text'],
        's_room_text': df['s_room_text'],
        'label': df['label']
    })
    
    # 创建三元组数据框
    df_triplet = (df[df['label'] == 1]
                 [['spl_room_type_id', 'spl_room_text', 's_room_text', 'negative']]
                 .copy())
    
    # 过滤无效的negative数据
    df_triplet = df_triplet[
        (df_triplet['spl_room_text'] != df_triplet['negative']) & 
        (df_triplet['s_room_text'] != df_triplet['negative'])
    ]
    
    end_time = time.time()
    print(f"\n数据处理完成！耗时: {end_time - start_time:.2f}秒")
    print(f"匹配数据集大小: {len(df_match)}")
    print(f"三元组数据集大小: {len(df_triplet)}")
    
    return df_match, df_triplet

if __name__ == "__main__":
    # 读取数据
    print("读取数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据大小: {len(df)}")
    
    # 处理数据
    df_match, df_triplet = process_data(df)
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    df_match.to_csv(OUTPUT_MATCH_FILE, index=False)
    df_triplet.to_csv(OUTPUT_TRIPLET_FILE, index=False)
    print("数据保存完成！") 