import pandas as pd

# 文件路径配置
ROOM_FILE = '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/s_room_2025-07-21_095100.csv'
HOTEL_COUNTRY_FILE = '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/s_hotel_country.csv'
OUTPUT_FILE = '/home/maxon/disk2/roomMatch/room_match/Langchain/room_group/data/s_room_with_country_2025-07-21_095100.csv'

# 读取CSV文件，处理可能的格式问题
room_df = pd.read_csv(ROOM_FILE, on_bad_lines='skip', encoding='utf-8')
hotel_country_df = pd.read_csv(HOTEL_COUNTRY_FILE, encoding='utf-8')

# 合并数据框
result_df = pd.merge(
    room_df,
    hotel_country_df[['s_hotel_id', 'country_code']],
    on='s_hotel_id',
    how='left'
)

# 保存结果
result_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8')
print(f"处理完成，输出文件保存在: {OUTPUT_FILE}")
print(f"总行数: {len(result_df)}")
