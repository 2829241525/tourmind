import pandas as pd
import re
import os
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
import time

class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.room_rules, self.bed_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
        # 定义映射替换模式
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        room_rules = []
        bed_rules = []

        # 按照sheet顺序构建字典
        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)
            if not {'key', 'value'}.issubset(sheet_data.columns):
                continue

            # 处理每一行数据
            for _, row in sheet_data.iterrows():
                key = str(row['key']).strip().lower()
                values = [v.strip().strip('"') for v in str(row['value']).split(',')]
                rules = [(value.lower(), key, len(value)) for value in values if value]
                
                # 如果是bed工作表，只处理床型
                room_rules.extend(rules)
                if sheet_name.lower() == 'bed':
                    bed_rules.extend(rules)
                # 其他工作表都处理房型



        # 按原文本长度降序排序，确保优先匹配最长的文本
        room_rules = sorted(room_rules, key=lambda x: x[2], reverse=True)
        bed_rules = sorted(bed_rules, key=lambda x: x[2], reverse=True)
        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()
        # 遍历所有房型规则
        for value, key, _ in self.room_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'  # 开始边界：确保前面是非字母数字和横杠
                rf'{re.escape(value)}'  # 需要匹配的文本
                rf'(?:$|(?=[^a-zA-Z0-9-]))',  # 结束边界：确保后面是非字母数字和横杠
                re.IGNORECASE
            )
            if pattern.search(text_lower):  # 如果找到匹配
                text_lower = pattern.sub(key, text_lower)
                break  # 找到匹配后就停止搜索

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

# 创建全局的DataCleaner实例（只创建一次）
cleaner = DataCleaner(Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/mapping.xlsx")))

def extract_english(text):
    if pd.isna(text):
        return "[unknown]"
    # 首先按|分割，并去除分隔符
    parts = [part.strip() for part in text.split('|')]
    parts = [p for p in parts if p]  # 移除空字符串
    
    if not parts:  # 如果没有有效内容
        return "[unknown]"
    
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
    
    return "[unknown]"

def process_text_with_prefix(prefix, room_name, bed_name, is_supplier=False):
    # 提取英文
    room = extract_english(room_name)
    bed = extract_english(bed_name)
    prefix_text = prefix if prefix else "[unknown]"
    
    # 只对供应商端数据进行清洗
    if is_supplier:
        if room:
            room = cleaner.clean_room_text(room)
        if bed:
            bed = cleaner.clean_bed_text(bed)
    
    # 组合成最终格式
    parts = []
    if prefix:
        parts.append(f"[supply] {prefix_text}" if prefix_text else "")
    if room:
        parts.append(f"[room] {room}")
    if bed:
        parts.append(f"[bed] {bed}")
    return " ".join(filter(None, parts))

def process_data(file):
    # 读取CSV文件
    df = file
    total_rows = len(df)
    
    print("\n开始处理数据...")
    start_time = time.time()
    
    # 处理supplier端数据
    print("\n处理supplier端数据...")
    tqdm.pandas(desc="supplier处理进度")
    df['spl_room_text'] = df.progress_apply(lambda x: process_text_with_prefix(
        x['supplier_name'], 
        x['spl_room_name'], 
        x['spl_room_bed_name'],
        is_supplier=True  # 标记为供应商数据
    ), axis=1)
    
    # 处理source端数据
    print("\n处理source端数据...")
    tqdm.pandas(desc="source处理进度")
    df['s_room_text'] = df.progress_apply(lambda x: process_text_with_prefix(
        None,  # 不需要supplier前缀
        x['s_room_name'], 
        x['s_room_bed_name'],
        is_supplier=False  # 标记为非供应商数据
    ), axis=1)

    # 处理negative数据
    print("\n处理negative数据...")
    tqdm.pandas(desc="negative处理进度")
    df['negative'] = df.progress_apply(lambda x: process_text_with_prefix(
        None,
        x['neg_room_name'], 
        x['neg_bed_name'],
        is_supplier=False  # 标记为非供应商数据
    ), axis=1)
    
    # 创建第一个输出数据框，保持原始顺序
    print("\n创建输出数据框...")
    df_match = pd.DataFrame()
    df_match['spl_room_type_id'] = df['spl_room_type_id']
    df_match['spl_room_text'] = df['spl_room_text']
    df_match['s_room_text'] = df.apply(lambda x: x['s_room_text'] if x['label'] == 1 else x['negative'], axis=1)
    df_match['label'] = df['label']
    
    # 创建第二个输出数据框（只包含label为1的数据）
    df_triplet = df[df['label'] == 1][['spl_room_type_id', 'spl_room_text', 's_room_text', 'negative']].copy()
    # 添加过滤条件：如果negative为空或者包含两个"unknown"字符串，则将这条数据过滤掉
    df_triplet = df_triplet[~df_triplet['negative'].isnull() & ~(df_triplet['negative'].str.count('unknown') == 2)]
    df_triplet.columns = ['spl_room_type_id', 'spl_room_text', 'positive', 'negative']


    end_time = time.time()
    processing_time = end_time - start_time
    rows_per_second = total_rows / processing_time
    
    print(f"\n处理完成:")
    print(f"总数据量: {total_rows:,} 条")
    print(f"处理时间: {processing_time:.2f} 秒")
    print(f"处理速度: {rows_per_second:.2f} 条/秒")
    
    return df_match, df_triplet

if __name__ == "__main__":
    # 设置当前工作目录为脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    # 获取项目根目录
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # 输入和输出文件路径
    file_path = os.path.join(project_root, "room_match/data/all_room_match_system_data.csv")
    output_match_path = os.path.join(project_root, "room_match/data/processed_all_system_room_match.csv")
    output_triplet_path = os.path.join(project_root, "room_match/data/processed_all_system_room_triplet.csv")

    print(f"\n读取数据文件: {file_path}")
    df = pd.read_csv(file_path)
    # #取出前20000条数据进行测试
    # df = df.head(20000)
    print(f"数据形状: {df.shape}")
    
    # 处理数据
    df_match, df_triplet = process_data(df)
    
    # 打印处理后的数据样例
    print("\n处理后的匹配数据样例:")
    print(df_match.head())
    print("\n处理后的三元组数据样例:")
    print(df_triplet.head())
    
    # 保存处理后的数据
    print("\n保存数据...")
    df_match.to_csv(output_match_path, index=False)
    df_triplet.to_csv(output_triplet_path, index=False)
    
    print(f"\n数据已保存到:\n{output_match_path}\n{output_triplet_path}")
