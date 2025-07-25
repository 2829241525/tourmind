#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import pandas as pd
import csv
from pathlib import Path

# 配置文件路径
JSON_FILE = "/home/maxon/disk2/roomMatch/room_match/hotel_match/eval_model_compare/deberta_matching_results_34_20250724_184237.json"
EXCEL_FILE = "/home/maxon/disk2/roomMatch/room_match/hotel_match/eval_model_compare/51.Confident.xlsx"
OUTPUT_CSV = "/home/maxon/disk2/roomMatch/room_match/hotel_match/eval_model_compare/hotel_match_results_fixed.csv"


def load_json_data(json_file):
    """加载JSON文件数据"""
    print(f"正在读取JSON文件: {json_file}")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_excel_data(excel_file):
    """加载Excel文件数据"""
    print(f"正在读取Excel文件: {excel_file}")
    df = pd.read_excel(excel_file)
    print(f"Excel文件包含 {len(df)} 行数据")
    print(f"Excel文件列名: {list(df.columns)}")
    return df


def determine_match_status(first_hotel_matched, tourmind_shotel, standard_hotel_id, found_in_excel=True):
    """
    根据规则确定匹配状态:
    - all_match: 第一条Matched为true，且xlsx中tourmind_shotel包含对应的StandardHotelID
    - hotel_no_match_simlow: 第一条Matched为false，但xlsx中tourmind_shotel包含StandardHotelID
    - hotel_no_match_nodata: 第一条Matched为false，且xlsx中tourmind_shotel不包含StandardHotelID但不为空
    - hotel_match: 第一条Matched为true，但xlsx中tourmind_shotel为空
    - all_no_match: 第一条Matched为false，且xlsx中tourmind_shotel为空
    - not_found_in_excel_hotel_true: 在Excel中未找到，但JSON中第一条Matched为true
    - not_found_in_excel_hotel_false: 在Excel中未找到，且JSON中第一条Matched为false
    """
    if not found_in_excel:
        # 如果在Excel中未找到
        if first_hotel_matched:
            return "not_found_in_excel_hotel_true"
        else:
            return "not_found_in_excel_hotel_false"

    # 检查tourmind_shotel是否为空
    tourmind_empty = pd.isna(tourmind_shotel) or str(
        tourmind_shotel).strip() == ''

    if first_hotel_matched:
        if not tourmind_empty and str(standard_hotel_id) in str(tourmind_shotel):
            return "all_match"
        elif tourmind_empty:
            return "hotel_match"
        else:
            # 如果JSON匹配成功，但Excel中有其他标准酒店ID（不匹配当前StandardHotelID）
            return "hotel_match_other_id"
    else:
        if not tourmind_empty:
            # 进一步判断是否包含StandardHotelID
            if str(standard_hotel_id) in str(tourmind_shotel):
                return "hotel_no_match_simlow"
            else:
                return "hotel_no_match_nodata"
        else:
            return "all_no_match"


def process_data():
    """处理数据并生成CSV"""
    # 加载数据
    json_data = load_json_data(JSON_FILE)
    excel_df = load_excel_data(EXCEL_FILE)

    # 准备结果列表
    results = []

    # 处理JSON中的每个匹配项
    matches = json_data.get('matches', [])
    print(f"JSON文件包含 {len(matches)} 个匹配项")

    for i, match in enumerate(matches):
        if i % 1000 == 0:
            print(f"处理进度: {i}/{len(matches)}")

        supplier_hotel_id = match.get('supplier_hotel_id')
        standard_hotels = match.get('standard_hotels', [])

        # 获取第一条标准酒店信息
        if standard_hotels:
            first_hotel = standard_hotels[0]
            first_hotel_matched = first_hotel.get('Matched', False)
            standard_hotel_id = first_hotel.get('StandardHotelID')
            similarity = first_hotel.get('Similarity', 0.0)
        else:
            first_hotel_matched = False
            standard_hotel_id = None
            similarity = 0.0

        # 在Excel中查找对应的SourceKey
        excel_row = excel_df[excel_df['SourceKey'] == supplier_hotel_id]

        if not excel_row.empty:
            # 获取Excel中的数据
            excel_data = excel_row.iloc[0]
            source_key = excel_data.get('SourceKey')
            hotel_name = excel_data.get('HotelName', '')
            address = excel_data.get('Address', '')
            tourmind_shotel = excel_data.get('tourmind_shotel', '')

            # 确定匹配状态
            match_status = determine_match_status(
                first_hotel_matched, tourmind_shotel, standard_hotel_id, found_in_excel=True)

            # 添加到结果中
            result_row = {
                'SourceKey': source_key,
                'HotelName': hotel_name,
                'Address': address,
                'tourmind_shotel': tourmind_shotel,
                'supplier_hotel_id': supplier_hotel_id,
                'match_status': match_status,
                'StandardHotelID': standard_hotel_id,
                'Similarity': similarity,
                'first_hotel_matched': first_hotel_matched,
                'supplier_hotel_name': match.get('supplier_hotel_name', ''),
                'supplier_hotel_name_cn': match.get('supplier_hotel_name_cn', ''),
                'supplier_hotel_address': match.get('supplier_hotel_address', ''),
                'supplier_hotel_address_cn': match.get('supplier_hotel_address_cn', '')
            }
            results.append(result_row)
        else:
            # 如果在Excel中找不到对应的SourceKey，根据JSON匹配状态分类
            match_status = determine_match_status(
                first_hotel_matched, None, standard_hotel_id, found_in_excel=False)

            result_row = {
                'SourceKey': None,
                'HotelName': '',
                'Address': '',
                'tourmind_shotel': '',
                'supplier_hotel_id': supplier_hotel_id,
                'match_status': match_status,
                'StandardHotelID': standard_hotel_id,
                'Similarity': similarity,
                'first_hotel_matched': first_hotel_matched,
                'supplier_hotel_name': match.get('supplier_hotel_name', ''),
                'supplier_hotel_name_cn': match.get('supplier_hotel_name_cn', ''),
                'supplier_hotel_address': match.get('supplier_hotel_address', ''),
                'supplier_hotel_address_cn': match.get('supplier_hotel_address_cn', '')
            }
            results.append(result_row)

    # 输出CSV文件
    print(f"正在保存结果到: {OUTPUT_CSV}")
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 统计结果
    print("\n=== 处理结果统计 ===")
    print(f"总共处理: {len(results)} 条记录")

    status_counts = result_df['match_status'].value_counts()
    for status, count in status_counts.items():
        print(f"{status}: {count} 条")

    print(f"\n结果已保存到: {OUTPUT_CSV}")
    return result_df


if __name__ == "__main__":
    result_df = process_data()
