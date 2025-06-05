#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import orjson
import os
from collections import Counter
from typing import Dict, List, Tuple
from multiprocessing import Pool, cpu_count

def load_processed_data(input_file: str = "processed_room_names.json") -> List[Dict]:
    """
    加载处理后的房型名称数据 using orjson
    """
    try:
        with open(input_file, 'rb') as f: # orjson reads bytes
            data = orjson.loads(f.read())
        print(f"成功加载 {len(data)} 条处理后的房型数据 (using orjson)")
        return data
    except Exception as e:
        print(f"加载数据失败: {str(e)}")
        raise

def analyze_tokens_for_item(item_data: Dict) -> Tuple[Counter, Counter, Counter, Counter]:
    """
    Analyzes tokens for a single item from processed_data.
    Returns (original_chinese_counter, original_english_counter, cn_chinese_counter, cn_english_counter)
    """
    chinese_range = range(ord('\u4e00'), ord('\u9fff') + 1)
    
    original_tokens = item_data.get('processed_tokens', [])
    cn_tokens = item_data.get('processed_tokens_cn', [])

    item_original_chinese_counter = Counter()
    item_original_english_counter = Counter()
    item_cn_chinese_counter = Counter()
    item_cn_english_counter = Counter()

    for token in original_tokens:
        if any(ord(char) in chinese_range for char in token):
            item_original_chinese_counter[token] += 1
        elif token.strip():
            item_original_english_counter[token] += 1
            
    for token in cn_tokens:
        if any(ord(char) in chinese_range for char in token):
            item_cn_chinese_counter[token] += 1
        elif token.strip():
            item_cn_english_counter[token] += 1
            
    return item_original_chinese_counter, item_original_english_counter, item_cn_chinese_counter, item_cn_english_counter

def analyze_word_frequency(processed_data: List[Dict]) -> Tuple[Counter, Counter, Counter, Counter, Counter, Counter]:
    """
    分析词频, 分别处理 original 和 cn 的词元。
    Uses multiprocessing to parallelize token analysis.
    Returns:
        original_chinese_counter, original_english_counter,
        cn_chinese_counter, cn_english_counter,
        combined_chinese_counter, combined_english_counter
    """
    
    num_processes = cpu_count()
    print(f"Starting token analysis with {num_processes} processes...")

    # Initialize aggregate counters
    original_chinese_counter_agg = Counter()
    original_english_counter_agg = Counter()
    cn_chinese_counter_agg = Counter()
    cn_english_counter_agg = Counter()

    with Pool(processes=num_processes) as pool:
        # Process data in chunks or map directly if memory allows per process
        # For simplicity, mapping directly. Adjust if individual items are excessively large.
        results = pool.map(analyze_tokens_for_item, processed_data)

    print("Aggregating results from parallel processing...")
    for res_orig_ch, res_orig_en, res_cn_ch, res_cn_en in results:
        original_chinese_counter_agg.update(res_orig_ch)
        original_english_counter_agg.update(res_orig_en)
        cn_chinese_counter_agg.update(res_cn_ch)
        cn_english_counter_agg.update(res_cn_en)
    
    print("Token analysis complete.")

    # For combined, we sum the respective counters
    combined_chinese_counter_agg = original_chinese_counter_agg + cn_chinese_counter_agg
    combined_english_counter_agg = original_english_counter_agg + cn_english_counter_agg
    
    return (
        original_chinese_counter_agg, original_english_counter_agg, 
        cn_chinese_counter_agg, cn_english_counter_agg,
        combined_chinese_counter_agg, combined_english_counter_agg
    )

def save_frequency_set(counters: Tuple[Counter, Counter], base_filename: str, 
                         output_dir: str = ".", top_n: int = 100):
    """
    Saves a set of Chinese and English word frequency counts using orjson.
    counters: A tuple (chinese_counter, english_counter)
    base_filename: e.g., "original_room_name" or "cn_room_name"
    """
    os.makedirs(output_dir, exist_ok=True)
    chinese_counter, english_counter = counters

    # Save Chinese word frequency
    chinese_output_filename = os.path.join(output_dir, f"{base_filename}_chinese_frequency.json")
    with open(chinese_output_filename, 'wb') as f: # orjson writes bytes
        # Convert to list of tuples to preserve order
        top_chinese = list(chinese_counter.most_common(top_n))
        f.write(orjson.dumps(top_chinese, option=orjson.OPT_INDENT_2))
    print(f"Saved: {chinese_output_filename}")

    # Save English word frequency
    english_output_filename = os.path.join(output_dir, f"{base_filename}_english_frequency.json")
    with open(english_output_filename, 'wb') as f: # orjson writes bytes
        # Convert to list of tuples to preserve order
        top_english = list(english_counter.most_common(top_n))
        f.write(orjson.dumps(top_english, option=orjson.OPT_INDENT_2))
    print(f"Saved: {english_output_filename}")
    
    # Save combined frequency for this set (Chinese + English for this specific source)
    all_counter_for_set = chinese_counter + english_counter
    all_output_filename = os.path.join(output_dir, f"{base_filename}_all_frequency.json")
    with open(all_output_filename, 'wb') as f: # orjson writes bytes
        # Convert to list of tuples to preserve order
        all_items_in_set = list(all_counter_for_set.most_common()) # Save all items
        f.write(orjson.dumps(all_items_in_set, option=orjson.OPT_INDENT_2))
    print(f"Saved: {all_output_filename}")

    # 新增：保存为 Go 代码可用的 map 格式
    all_output_map_filename = os.path.join(output_dir, f"{base_filename}_all_frequency_map.json")
    with open(all_output_map_filename, 'wb') as f:
        freq_map = dict(all_counter_for_set)
        f.write(orjson.dumps(freq_map, option=orjson.OPT_INDENT_2))
    print(f"Saved: {all_output_map_filename} (Go map format)")


def main():
    try:
        processed_data = load_processed_data()
        
        print("开始分析词频...")
        (
            original_chinese_counter, original_english_counter,
            cn_chinese_counter, cn_english_counter,
            combined_chinese_counter, combined_english_counter
        ) = analyze_word_frequency(processed_data)
        
        output_dir = "word_frequency_results"

        print("\n--- Original Room Name Tokens Analysis ---")
        print(f"Distinct Chinese words: {len(original_chinese_counter)}")
        print(f"Distinct English words: {len(original_english_counter)}")
        save_frequency_set((original_chinese_counter, original_english_counter), 
                             "original_room_name", output_dir)

        print("\n--- Chinese Room Name (room_name_cn) Tokens Analysis ---")
        print(f"Distinct Chinese words: {len(cn_chinese_counter)}")
        # English words from room_name_cn might be noise or errors, but we report if any
        print(f"Distinct English words (from CN names): {len(cn_english_counter)}") 
        save_frequency_set((cn_chinese_counter, cn_english_counter), 
                             "cn_room_name", output_dir)

        print("\n--- Combined Tokens Analysis (Overall) ---")
        print(f"Distinct Chinese words (overall): {len(combined_chinese_counter)}")
        print(f"Distinct English words (overall): {len(combined_english_counter)}")
        save_frequency_set((combined_chinese_counter, combined_english_counter), 
                             "combined_all_sources", output_dir)
        
        print("\n词频分析完成")
        
    except Exception as e:
        print(f"词频分析失败: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main() 