import unittest
import pandas as pd
import os
import shutil
import asyncio
import logging
from datetime import datetime

from hotel_by_srid import HotelProcessor

class TestHotelProcessor(unittest.TestCase):
    def setUp(self):
        # 设置测试文件路径
        self.original_csv = "room_match/data/room_match_folks.csv"
        self.test_csv = "room_match/data/test_room_match_folks.csv"
        self.test_api_data = f"{self.test_csv}.api_data.json"
        
        # 读取原始CSV文件的前1000行创建测试文件
        df = pd.read_csv(self.original_csv)
        # 确保CSV文件包含所需的列
        required_columns = ['s_hotel_id', 'hotel_name', 'country_code', 'group_code']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        test_df = df.head(1000)
        test_df.to_csv(self.test_csv, index=False)
        
        # 初始化HotelProcessor
        self.processor = HotelProcessor(self.test_csv, self.test_api_data)
        
    def tearDown(self):
        # 只清理API数据文件，保留CSV文件以便检查
        if os.path.exists(self.test_api_data):
            os.remove(self.test_api_data)
            
    def test_csv_update(self):
        # 记录更新前的状态
        df_before = pd.read_csv(self.test_csv)
        null_before = {col: df_before[col].isna().sum() for col in ['s_hotel_id', 'hotel_name', 'country_code', 'group_code']}
        print(f"更新前空值统计: {null_before}")
        
        # 运行异步数据获取和更新
        asyncio.run(self.processor.fetch_all_api_data())
        self.processor.update_csv()
        
        # 验证更新后的状态
        df_after = pd.read_csv(self.test_csv)
        null_after = {col: df_after[col].isna().sum() for col in ['s_hotel_id', 'hotel_name', 'country_code', 'group_code']}
        print(f"更新后空值统计: {null_after}")
        
        # 确保至少有一些数据被更新
        self.assertNotEqual(null_before, null_after, "没有数据被更新")
        
        # 验证更新的数据内容
        print("\n验证更新后的数据:")
        # 检查行数是否保持不变
        self.assertEqual(len(df_before), len(df_after), "更新前后行数不一致")
        
        # 检查必填字段是否都已更新
        required_fields = ['s_hotel_id', 'hotel_name', 'country_code']
        for field in required_fields:
            non_null_count = df_after[field].notna().sum()
            print(f"{field} 非空值数量: {non_null_count}")
            self.assertGreater(non_null_count, df_before[field].notna().sum(), f"{field}字段未被正确更新")
        
        # 检查数据一致性
        print("\n检查数据一致性:")
        # 对于相同的s_room_id，验证其hotel信息是否一致
        sample_rows = df_after.groupby('s_room_id').first().head()
        for _, row in sample_rows.iterrows():
            if pd.notna(row['s_hotel_id']):
                matching_rows = df_after[df_after['s_room_id'] == row.name]
                for field in ['s_hotel_id', 'hotel_name', 'country_code', 'group_code']:
                    unique_values = matching_rows[field].unique()
                    self.assertEqual(len(unique_values), 1, 
                                  f"s_room_id {row.name} 的 {field} 存在不一致的值")
                print(f"s_room_id {row.name} 的数据一致性验证通过")
        
        print("\n更新验证完成")
        
if __name__ == '__main__':
    unittest.main() 