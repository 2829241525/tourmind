#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import jieba
import nltk
from typing import List, Set, Optional, Dict
import json

# Download NLTK resources (uncomment these lines if you need to download them first time)
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem import PorterStemmer  # Changed from WordNetLemmatizer to PorterStemmer
from nltk.corpus import stopwords as nltk_stopwords

def preprocess_room_names(room_names: list[str], chinese_stopwords: set = None, english_stopwords: set = None) -> list[list[str]]:
    """
    Preprocess a list of room type names which can be in Chinese, English, or mixed.
    
    Args:
        room_names: A list of strings, where each string is a room type name.
        chinese_stopwords: An optional set of Chinese stopwords.
        english_stopwords: An optional set of English stopwords.
        
    Returns:
        A list of lists, where each inner list contains the processed tokens
        for the corresponding input room type name.
    """
    if not room_names:
        return []
    
    # Initialize stopword sets if None
    if chinese_stopwords is None:
        chinese_stopwords = set()
    
    if english_stopwords is None:
        english_stopwords = set(nltk_stopwords.words('english'))
    
    # Initialize stemmer for English words
    stemmer = PorterStemmer()  # Changed from WordNetLemmatizer to PorterStemmer
    
    # Regex for identifying Chinese characters
    chinese_char_pattern = re.compile(r'[\u4e00-\u9fff]+')
    
    # Regex for cleaning punctuation and special characters
    punctuation_pattern = re.compile(r'[^\w\s\u4e00-\u9fff]')
    
    result = []
    
    for room_name in room_names:
        if not room_name.strip():
            result.append([])
            continue
        
        # Clean the room name by removing punctuation
        cleaned_name = punctuation_pattern.sub(' ', room_name)
        
        # Split the text into Chinese and non-Chinese segments
        current_pos = 0
        segments = []
        
        for match in chinese_char_pattern.finditer(cleaned_name):
            # Add non-Chinese segment before the match (if exists)
            if match.start() > current_pos:
                non_chinese = cleaned_name[current_pos:match.start()].strip()
                if non_chinese:
                    segments.append(('en', non_chinese))
            
            # Add Chinese segment
            segments.append(('zh', match.group()))
            current_pos = match.end()
        
        # Add any remaining non-Chinese segment
        if current_pos < len(cleaned_name):
            non_chinese = cleaned_name[current_pos:].strip()
            if non_chinese:
                segments.append(('en', non_chinese))
        
        # Process each segment according to its language
        tokens = []
        
        for lang, text in segments:
            if lang == 'zh':
                # Process Chinese segment with jieba
                chinese_tokens = jieba.cut(text)
                tokens.extend([token for token in chinese_tokens if token not in chinese_stopwords])
            else:
                # Process English segment
                english_words = text.lower().split()
                for word in english_words:
                    if word and word not in english_stopwords:
                        # Stem the word instead of lemmatizing
                        stemmed = stemmer.stem(word)
                        tokens.append(stemmed)
        
        # Add the processed tokens to the result
        result.append(tokens)
    
    return result

if __name__ == "__main__":
    input_file = "raw_room_data.json"
    output_file = "processed_room_names.json"

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            raw_data_list = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found. Please run the extraction script first.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{input_file}'.")
        exit(1)

    # Define custom stopwords (optional) - can be moved to a config or arguments
    custom_chinese_stopwords = {"的", "房"} 
    custom_english_stopwords = {"a", "with", "is"}

    processed_data_list = []
    
    print(f"Starting processing of {len(raw_data_list)} entries from {input_file}...")

    # Prepare lists for batch processing if preprocess_room_names is efficient with it
    original_room_names = []
    original_room_names_cn = []
    # Keep track of entries that have valid room_name_cn to map results back
    valid_cn_indices = [] 

    for i, raw_item in enumerate(raw_data_list):
        original_room_names.append(raw_item.get("room_name", "")) # Use .get for safety, provide default
        room_cn = raw_item.get("room_name_cn")
        if room_cn and room_cn.strip():
            original_room_names_cn.append(room_cn)
            valid_cn_indices.append(i)
        else:
            # if no valid room_name_cn, it will get an empty list of tokens later
            pass 

    # Process room_name (original)
    print(f"Processing {len(original_room_names)} original room names...")
    all_processed_tokens = preprocess_room_names(
        original_room_names,
        chinese_stopwords=custom_chinese_stopwords,
        english_stopwords=custom_english_stopwords
    )
    print("Finished processing original room names.")

    # Process room_name_cn
    print(f"Processing {len(original_room_names_cn)} Chinese room names (room_name_cn)...")
    processed_tokens_for_valid_cn = []
    if original_room_names_cn:
        processed_tokens_for_valid_cn = preprocess_room_names(
            original_room_names_cn,
            chinese_stopwords=custom_chinese_stopwords,
            english_stopwords=custom_english_stopwords # English stopwords might not be relevant here
        )
    print("Finished processing Chinese room names.")
    
    # Reconstruct the full list for processed_tokens_cn, inserting empty lists for entries without valid room_name_cn
    all_processed_tokens_cn = [[] for _ in range(len(raw_data_list))]
    for i, tokens_cn in enumerate(processed_tokens_for_valid_cn):
        original_idx = valid_cn_indices[i]
        all_processed_tokens_cn[original_idx] = tokens_cn

    # Combine into the final structure
    for i, raw_item in enumerate(raw_data_list):
        processed_data_list.append({
            "original": raw_item.get("room_name", ""),
            "room_name_cn": raw_item.get("room_name_cn"),
            "processed_tokens": all_processed_tokens[i],
            "processed_tokens_cn": all_processed_tokens_cn[i]
        })

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data_list, f, ensure_ascii=False, indent=2)
        print(f"Processed data saved to {output_file}")
    except Exception as e:
        print(f"Error saving processed data to '{output_file}': {e}")
        exit(1)

    print("Processing complete.") 
