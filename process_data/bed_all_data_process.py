import pandas as pd
import re
import os
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import time
import multiprocessing as mp
from functools import partial
from threading import Lock
import numpy as np

"""
床型抽取数据构建
"""
# 配置常量
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))

# 文件路径配置
INPUT_FILE = os.path.join(PROJECT_ROOT, "room_match/data/all_country_data.csv")
# INPUT_FILE = os.path.join(PROJECT_ROOT, "room_match/data/bed_unmatch.csv")
OUTPUT_MATCH_FILE = os.path.join(PROJECT_ROOT, "room_match/data/bed_all_data_process.csv")
MAPPING_FILE = os.path.join(CURRENT_DIR, "../data/mapping.xlsx")

# 文本处理配置 - 保留常量定义，但不再使用前缀
UNKNOWN_TEXT = "[unknown]"

# 正则表达式配置
CHINESE_PATTERN = re.compile(r'^[\u4e00-\u9fff\s]+$')
ENGLISH_CLEAN_PATTERN = re.compile(r'[^a-zA-Z\s]')
CHINESE_REMOVE_PATTERN = re.compile(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+')

# 正则表达式匹配ID常量
REGEX_NO_MATCH = 0           # 未匹配到任何正则
REGEX_HYPHEN = 1             # 连字符形式 (如 one-queen-size-bed)
REGEX_BRACKET_HYPHEN = 2     # 括号内连字符形式 (如 (one-queen-bed))
REGEX_NUMERIC_PREFIX = 3     # 数字前缀形式 (如 2double beds)
REGEX_NORMAL_BED = 4         # 常规床型形式 (如 one queen bed)
REGEX_BASIC_SIZE_BED = 5     # 基本大小床模式 (如 king bed)
REGEX_GENERAL_BED = 6        # 通用床型模式 (如 queen size bed)
REGEX_COMMA_BED = 7          # 逗号分隔床型模式 (如 one king, sofa bed)

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
                room_rules[sheet_name.lower()] = current_sheet_rules
            else:
                # 其他工作表作为房型规则，使用sheet_name作为key
                room_rules[sheet_name.lower()] = current_sheet_rules

        return room_rules, bed_rules

    def clean_room_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()

        # 将7以内的阿拉伯数字转换为英文单词
        number_mapping = {
            '1': 'one',
            '2': 'two',
            '3': 'three',
            '4': 'four',
            '5': 'five',
            '6': 'six'       
        }
        
        # 使用正则表达式查找独立的数字并替换
        for num, word in number_mapping.items():
            # 确保匹配的是独立的数字，而不是数字的一部分
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9]))'  # 确保前面是非字母数字
                rf'{num}'  # 需要匹配的数字
                rf'(?:$|(?=[^a-zA-Z0-9]))',  # 确保后面是非字母数字
                re.IGNORECASE
            )
            text_lower = pattern.sub(word, text_lower)

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
        if '[unknown]' in text:
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

def process_text_with_prefix(supplier: str, country: str, room_name: str, bed_name: str, is_supplier: bool = False) -> Dict:
    # 提取英文
    room = extract_english(room_name)
    bed = extract_english(bed_name)
    supplier_text = supplier if supplier else ""
    country_text = country if country else ""
    
    # 保存原始提取数据
    original_room = room
    original_bed = bed
    
    # 清洗后的房型和床型
    cleaned_room = ""
    cleaned_bed = ""
    
    # 只对供应商端数据进行清洗
    if is_supplier:
        if room and isinstance(room, str):
            cleaned_room = cleaner.clean_room_text(room)
            room = cleaned_room
        if bed and isinstance(bed, str):
            cleaned_bed = cleaner.clean_bed_text(bed)
            bed = cleaned_bed
    
    # 组合成最终格式，不再添加前缀标签
    parts = []
    # if supplier and isinstance(supplier, str):
    #     parts.append(supplier_text)
    # if country and isinstance(country, str):
    #     parts.append(country_text)
    if room and isinstance(room, str):
        parts.append(room)
    if bed and isinstance(bed, str):
        parts.append(bed)
    
    # 转为小写，确保只对字符串使用lower()
    parts = [part.lower() if isinstance(part, str) else str(part).lower() for part in parts if part]
    combined_text = " ".join(parts)
    
    # 安全地处理字符串转换
    def safe_lower(text):
        if isinstance(text, str):
            return text.lower()
        elif text:
            return str(text).lower()
        return ""
    
    # 从cleaned_room中提取床型数据
    pc_bed = ""
    # 添加匹配到的正则表达式编号，默认为0表示未匹配到任何正则
    regex_match_id = REGEX_NO_MATCH
    
    if cleaned_room:
        # 床型关键词和数量词列表
        bed_keywords = ["bed", "beds"]
        size_keywords = ["twin", "queen", "king", "double", "single", "full", "triple", "size"]
        quantity_words = ["one", "two", "three", "four", "five", "six", "double", "single", "twin", "queen", "king", "full", "triple"]
        quantity_words_mini = ["one", "two", "three", "four", "five", "six"]
        
        cleaned_text = cleaned_room.lower()
        
        # 1. 连字符形式优先处理（如 "one-queen-size-bed"）
        hyphen_bed_pattern = re.compile(
            r'\b((?:' + '|'.join(quantity_words) + r')(?:-[\w-]+)*-(?:bed|beds)s?)\b',
            re.IGNORECASE
        )
        
        # 2. 处理括号内的连字符形式
        bracket_hyphen_pattern = re.compile(r'\(([^()]*?(?:' + '|'.join(quantity_words) + r')(?:-[\w-]+)*-(?:bed|beds)[^()]*?)\)', re.IGNORECASE)
        
        # 3. 处理数字前缀情况，如 "2double beds"
        numeric_bed_pattern = re.compile(
            r'\b(\d+(?:' + '|'.join(size_keywords) + r')(?:\s+|\-)?(?:bed|beds)s?)\b',
            re.IGNORECASE
        )
        
        # 4. 直接寻找形如 "xxx bed" 或 "xxx beds" 的形式，确保bed在最后
        # 更精确地匹配从数量词开始的部分，避免包含前面的修饰词
        bed_at_end_pattern = re.compile(
            r'\b((?:' + '|'.join(quantity_words) + r')(?:\s+[\w-]+)*\s+(?:bed|beds))\b',
            re.IGNORECASE
        )
        
        # 匹配所有可能的模式
        all_matches = []
        
        # 首先匹配连字符形式 - 优先级最高
        hyphen_matches = hyphen_bed_pattern.findall(cleaned_text)
        if hyphen_matches:
            all_matches.extend([(match, cleaned_text.find(match), REGEX_HYPHEN) for match in hyphen_matches])
        
        # 匹配括号内的连字符形式
        bracket_matches = bracket_hyphen_pattern.findall(cleaned_text)
        if bracket_matches:
            for match in bracket_matches:
                if "room" not in match.lower() and "view" not in match.lower():
                    all_matches.append((match, cleaned_text.find(match), REGEX_BRACKET_HYPHEN))
        
        # 匹配数字前缀形式
        numeric_matches = numeric_bed_pattern.findall(cleaned_text)
        if numeric_matches:
            all_matches.extend([(match, cleaned_text.find(match), REGEX_NUMERIC_PREFIX) for match in numeric_matches])
        
        # 获取所有常规床型匹配组合
        # 全新方法：找出所有可能的床型组合
        end_matches = []
        
        # 预处理文本，将特殊字符替换为空格
        preprocessed_text = cleaned_text
        # 将括号、连字符、逗号等特殊字符替换为空格
        for char in ['(', ')', '-', ',', ';', '/', '\\']:
            preprocessed_text = preprocessed_text.replace(char, ' ')
        
        # 分割处理后的文本
        words = preprocessed_text.split()
        
        for i, word in enumerate(words):
            # 找出所有以量词开头的位置
            if word.lower() in quantity_words_mini:
                # 从量词开始向后查找床型关键词
                for j in range(i, len(words)):
                    if any(keyword in words[j].lower() for keyword in bed_keywords):
                        # 找到一个有效的床型组合
                        match_phrase = " ".join(words[i:j+1])
                        if "room" not in match_phrase.lower() and "view" not in match_phrase.lower():
                            # 由于我们处理了特殊字符，需要在原文中找到对应位置
                            # 构建正则表达式模式，允许特殊字符存在于词之间
                            pattern_parts = []
                            for word_idx, word in enumerate(words[i:j+1]):
                                if word_idx == 0:
                                    pattern_parts.append(r'\b' + re.escape(word))
                                else:
                                    pattern_parts.append(re.escape(word))
                            
                            # 构建允许特殊字符和空格作为分隔符的正则表达式
                            flexible_pattern = r'[\s\(\)\-,;/\\]*'.join(pattern_parts) + r'\b'
                            matches = list(re.finditer(flexible_pattern, cleaned_text, re.IGNORECASE))
                            
                            if matches:
                                # 使用原文中找到的匹配
                                original_match = matches[0].group()
                                match_pos = matches[0].start()
                                if "room" not in original_match.lower() and "view" not in original_match.lower():
                                    end_matches.append((original_match, match_pos))
        
        # 将找到的常规床型匹配添加到所有匹配中
        if all_matches is None and end_matches:
            all_matches.extend([(match, pos, REGEX_NORMAL_BED) for match, pos in end_matches])
        
        # 处理逗号后的床型，优先检查连字符形式
        if all_matches is None and "," in cleaned_text:
            parts = cleaned_text.split(",")
            for part_idx, part in enumerate(parts):
                part = part.strip()
                
                # 在逗号分隔的部分中再次尝试所有匹配方式
                # 优先匹配连字符形式
                comma_hyphen_matches = hyphen_bed_pattern.findall(part)
                if comma_hyphen_matches:
                    # 计算在原文中的实际位置
                    real_pos = sum(len(p) + 1 for p in parts[:part_idx]) + part.find(comma_hyphen_matches[0])
                    all_matches.append((comma_hyphen_matches[0], real_pos, REGEX_HYPHEN))
                    continue
                
                # 其次匹配常规床型
                # 预处理文本，将特殊字符替换为空格
                preprocessed_part = part
                for char in ['(', ')', '-', ';', '/', '\\']:
                    preprocessed_part = preprocessed_part.replace(char, ' ')
                
                part_words = preprocessed_part.split()
                for i, word in enumerate(part_words):
                    if word.lower() in quantity_words_mini:
                        for j in range(i, len(part_words)):
                            if any(keyword in part_words[j].lower() for keyword in bed_keywords):
                                # 构建正则表达式模式，允许特殊字符存在于词之间
                                pattern_parts = []
                                for word_idx, word in enumerate(part_words[i:j+1]):
                                    if word_idx == 0:
                                        pattern_parts.append(r'\b' + re.escape(word))
                                    else:
                                        pattern_parts.append(re.escape(word))
                                
                                # 构建允许特殊字符和空格作为分隔符的正则表达式
                                flexible_pattern = r'[\s\(\)\-,;/\\]*'.join(pattern_parts) + r'\b'
                                matches = list(re.finditer(flexible_pattern, part, re.IGNORECASE))
                                
                                if matches:
                                    original_match = matches[0].group()
                                    if "room" not in original_match.lower() and "view" not in original_match.lower():
                                        # 计算在原文中的实际位置
                                        real_pos = sum(len(p) + 1 for p in parts[:part_idx]) + matches[0].start()
                                        if real_pos >= 0:
                                            all_matches.append((original_match, real_pos, REGEX_NORMAL_BED))
        
        # 选择最佳匹配
        if all_matches:
            # 按照优先级和出现顺序排序所有匹配项
            all_matches.sort(key=lambda x: (x[2], x[1]))
            
            # 获取所有以量词开头以床型结尾的匹配
            normal_bed_matches = []
            for match, pos, match_type in all_matches:
                if match_type == REGEX_NORMAL_BED and "room" not in match.lower() and "view" not in match.lower():
                    # 确保以床型关键词结尾
                    match_words = match.split()
                    if match_words and any(keyword in match_words[-1].lower() for keyword in bed_keywords):
                        normal_bed_matches.append((match, pos))
            
            # 如果找到了匹配，处理最短匹配和扩展逻辑
            if normal_bed_matches:
                # 按照长度排序
                normal_bed_matches.sort(key=lambda x: len(x[0]))
                
                # 获取最短的匹配
                shortest_match, shortest_pos = normal_bed_matches[0]
                
                # 尝试扩展匹配
                can_expand = False
                expanded_match = ""
                
                # 从最短匹配开始，尝试扩展匹配
                for match, pos in normal_bed_matches:
                    if len(match) > len(shortest_match):
                        # 检查最短匹配是否在较长匹配的结尾部分
                        if match.endswith(shortest_match) or match.endswith(shortest_match + " "):
                            # 提取扩展部分（前缀部分）
                            expansion = match[:match.find(shortest_match)].strip()
                            
                            # 检查扩展部分是否有连接词
                            has_connector = any(connector in expansion.lower() for connector in ["and", "or", ",", "with","-"])
                            
                            # 检查是否包含排除词
                            has_excluded = any(excluded in match.lower() for excluded in ["room", "view", "breakfast"])
                            
                            # 如果有连接词且没有排除词
                            if (has_connector and not has_excluded):
                                can_expand = True
                                expanded_match = match
                                break
                
                # 确定最终使用的匹配结果
                if can_expand and expanded_match:
                    pc_bed = expanded_match
                else:
                    pc_bed = shortest_match
                    
                regex_match_id = REGEX_NORMAL_BED
            else:
                # 继续处理其他类型的匹配
                for match, _, match_type in all_matches:
                    # 检查是否包含"room"
                    if "room" not in match.lower() and "view" not in match.lower():
                        # 确保床型关键词在末尾
                        words = match.split()
                        last_word = words[-1] if words else ""
                        if any(keyword in last_word.lower() for keyword in bed_keywords):
                            pc_bed = match
                            regex_match_id = match_type
                            break
        
        # 如果没有找到匹配，尝试直接从文本中识别常见床型模式
        if "bed" in cleaned_room and "bedroom" not in cleaned_room and not pc_bed:
            # 定义常见床型与大小的组合
            bed_sizes = ["king", "queen", "single", "double", "twin", "full","bunk"]
            
            # 1. 首先尝试匹配"size bed"模式
            for size in bed_sizes:
                bed_pattern = f"{size} bed"
                if bed_pattern in cleaned_text:
                    bed_pos = cleaned_text.find(bed_pattern)
                    prefix_text = cleaned_text[:bed_pos].strip()
                    words = prefix_text.split()
                    
                    # 检查前面是否有数词
                    if words and words[-1] in quantity_words:
                        pc_bed = f"{words[-1]} {bed_pattern}"
                    else:
                        pc_bed = bed_pattern
                    regex_match_id = REGEX_BASIC_SIZE_BED  # 基本大小床模式
                    break
        
            
            # 3. 处理数量词和bed之间的内容（处理逗号分隔的情况）
            if not pc_bed:
                # 先检查是否包含特定的床型描述模式
                if "one king, sofa bed" in cleaned_text:
                    pc_bed = "one king, sofa bed"
                    regex_match_id = REGEX_COMMA_BED
                elif "king-size bed" in cleaned_text:
                    pc_bed = "king-size bed"
                    regex_match_id = REGEX_COMMA_BED
                # 如果不包含上述特定模式，则不再处理其他逗号分隔的情况
                # 旧的复杂逻辑分隔情况分析代码已被移除
        
        # 清理提取的结果
        if pc_bed:
            # 去除开头的逗号和空格
            pc_bed = pc_bed.lstrip(' ,/-')
            # 去除结尾的标点和空格
            pc_bed = pc_bed.rstrip(' ,.;:!?-')
            # 如果包含"room"，不提取
            if "room" in pc_bed.lower():
                pc_bed = ""
            # 确保以"bed"或"beds"结尾
            words = pc_bed.split()
            if words and not any(keyword in words[-1].lower() for keyword in bed_keywords):
                # 找到最后一个包含"bed"的词
                last_bed_idx = -1
                for i, word in enumerate(words):
                    if any(keyword in word.lower() for keyword in bed_keywords):
                        last_bed_idx = i
                
                if last_bed_idx >= 0:
                    pc_bed = " ".join(words[:last_bed_idx+1])
    
    # 返回字典包含所有信息
    return {
        "combined_text": combined_text,
        "cleaned_room": safe_lower(cleaned_room),
        "cleaned_bed": safe_lower(cleaned_bed),
        "pc_bed": pc_bed,
        "regex_match_id": regex_match_id  # 添加匹配到的正则表达式编号
    }

def process_chunk(args):
    chunk, func, is_supplier = args
    try:
        result = chunk.apply(func, axis=1)
        return result
    except Exception as e:
        print(f"处理chunk时出错: {str(e)}")
        raise e

def process_data_parallel(df: pd.DataFrame, func, is_supplier: bool, desc: str) -> pd.DataFrame:
    # 获取CPU核心数，预留一个核心给系统
    num_cores = max(1, mp.cpu_count() - 1)
    total_rows = len(df)
    
    # 使用更小的chunk size来让进度条更新更平滑
    # 设置一个合理的最小chunk大小，比如每个chunk处理1000条数据
    chunk_size = min(5000, max(1, total_rows // (num_cores * 10)))
    chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
    
    print(f"\n使用{num_cores}个进程处理{desc}...")
    print(f"总数据量: {total_rows:,}条")
    print(f"每批处理: {chunk_size:,}条")
    
    results = []
    
    # 使用进程池处理数据
    with mp.Pool(num_cores) as pool:
        with tqdm(total=total_rows, desc=desc, unit='条', 
                 bar_format='{desc}: {n_fmt}/{total_fmt}条 [{elapsed}<{remaining}] {rate_fmt}',
                 mininterval=0.1) as pbar:
            # 使用imap_unordered可以让进度条更新更及时
            for result in pool.imap_unordered(process_chunk, [(chunk, func, is_supplier) for chunk in chunks]):
                results.append(result)
                pbar.update(len(result))
    
    # 合并结果并确保顺序正确
    final_result = pd.concat(results)
    return final_result.sort_index()

def process_supplier_text(row):
    result = process_text_with_prefix(row['supplier_name'], 
                                  row['source_file'],
                                  row['spl_room_name'], 
                                  row['spl_room_bed_name'],
                                  is_supplier=True)
    return pd.Series({
        'spl_room_text': result['combined_text'],
        'spl_cleaned_room': result['cleaned_room'],
        'spl_cleaned_bed': result['cleaned_bed'],
        'spl_pc_bed': result['pc_bed'],
        'spl_regex_match_id': result['regex_match_id']  # 添加匹配到的正则表达式编号
    })

def process_source_text(row):
    result = process_text_with_prefix(None,
                                    None,
                                  row['s_room_name'], 
                                  row['s_room_bed_name'],
                                  is_supplier=True)
    return pd.Series({
        's_room_text': result['combined_text'],
        's_original_room': result['original_room'],
        's_original_bed': result['original_bed'],
        's_cleaned_room': result['cleaned_room'],
        's_cleaned_bed': result['cleaned_bed'],
        's_pc_bed': result['pc_bed'],
        's_regex_match_id': result['regex_match_id']  # 添加匹配到的正则表达式编号
    })

# 正则表达式匹配ID解释
def get_regex_match_description(match_id):
    """
    根据匹配ID返回对应的正则表达式描述
    
    Args:
        match_id: 匹配到的正则表达式ID
        
    Returns:
        str: 正则表达式的描述
    """
    descriptions = {
        REGEX_NO_MATCH: "未匹配到任何正则",
        REGEX_HYPHEN: "连字符形式 (如 one-queen-size-bed)",
        REGEX_BRACKET_HYPHEN: "括号内连字符形式 (如 (one-queen-bed))",
        REGEX_NUMERIC_PREFIX: "数字前缀形式 (如 2double beds)",
        REGEX_NORMAL_BED: "常规床型形式 (如 one queen bed)",
        REGEX_BASIC_SIZE_BED: "基本大小床模式 (如 king bed)",
        REGEX_GENERAL_BED: "通用床型模式 (如 queen size bed)",
        REGEX_COMMA_BED: "逗号分隔床型模式 (如 one king, sofa bed)"
    }
    return descriptions.get(match_id, f"未知匹配ID: {match_id}")

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    total_rows = len(df)
    print("\n开始处理数据...")
    start_time = time.time()
    
    # 筛选出spl_room_bed_name为空的记录
    print(f"过滤前数据大小: {len(df)}")
    df = df[df['spl_room_bed_name'].isna()]
    print(f"过滤后数据大小: {len(df)}")

    # 去除重复的spl_cleaned_room
    df = df.drop_duplicates(subset=['spl_room_name'])
    print(f"去重后数据集大小: {len(df)}")
    
    # 处理supplier端数据
    print("\n处理supplier端数据...")
    supplier_results = process_data_parallel(df, process_supplier_text, True, "supplier处理进度")
    
    # 直接使用supplier_results代替原来的df合并
    df = supplier_results
    
    # 创建匹配数据框
    df_match = pd.DataFrame({
        'spl_cleaned_room': df['spl_cleaned_room'],
        'spl_pc_bed': df['spl_pc_bed'],
        'spl_regex_match_id': df['spl_regex_match_id']  # 添加匹配到的正则表达式编号列
    })
    
    # 输出正则表达式匹配统计信息
    print("\n正则表达式匹配统计：")
    regex_counts = df_match['spl_regex_match_id'].value_counts().sort_index()
    for match_id, count in regex_counts.items():
        percentage = count / len(df_match) * 100
        print(f"  匹配ID {match_id} ({get_regex_match_description(match_id)}): {count}条 ({percentage:.2f}%)")
    
    # 有效匹配率（排除ID为0的情况）
    valid_matches = df_match[df_match['spl_regex_match_id'] > 0]
    valid_match_rate = len(valid_matches) / len(df_match) * 100 if len(df_match) > 0 else 0
    print(f"\n有效匹配率: {valid_match_rate:.2f}% ({len(valid_matches)}/{len(df_match)})")
    
    end_time = time.time()
    print(f"\n数据处理完成！耗时: {end_time - start_time:.2f}秒")
    print(f"原始匹配数据集大小: {len(df_match)}")
    
    return df_match

if __name__ == "__main__":
    # 读取数据
    print("读取数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"原始数据大小: {len(df)}")
    
    # # 获取前1000条数据
    # df = df.head(1000000)
    
    # 处理数据
    df_match = process_data(df)
    
    # 保存处理后的数据
    print("\n保存处理后的数据...")
    df_match.to_csv(OUTPUT_MATCH_FILE, index=False)
    print("数据保存完成！") 