import pandas as pd
import json
import os
from datetime import datetime

# 配置信息
CONFIG = {
    'input_csv_path': '/home/maxon/disk2/roomMatch/room_match/rft/data/processed_cleaned_spaces.csv',
    'output_dir': '/home/maxon/disk2/roomMatch/room_match/rft/process',
    'sample_size': 5000,
    'random_seed': 42
}


def generate_json_data(row):
    """生成单条JSON数据"""
    return {
        "messages": [
            {
                "role": "user",
                "content": f"你是一个房型匹配专家。请判断以下两个房型是否匹配：\n\n供应商房型：{row['spl_room_text']}\n标准房型：{row['s_room_text']}，如果匹配返回1，不匹配返回0，不需要额外的输出，只需要返回1或0"
            }
        ],
        "solution": f"解析：通过分析两个房型描述的的房型，床型，景观，床型数量等诸多角度的特征后，{'可以判定这两个房型是匹配的' if str(row['label']) == '1' else '可以判定这两个房型是不匹配的'}。",
        "final_answer": str(row['label'])
    }


def main():
    # 创建输出文件名（包含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        CONFIG['output_dir'], f'training_data_{timestamp}.txt')
    output_csv_file = os.path.join(
        CONFIG['output_dir'], f'training_data_{timestamp}.csv')

    print(f"开始读取CSV文件: {CONFIG['input_csv_path']}")
    try:
        # 读取CSV文件
        df = pd.read_csv(CONFIG['input_csv_path'])
        print(f"成功读取CSV文件，总行数: {len(df)}")

        # 随机抽样
        sampled_df = df.sample(n=min(CONFIG['sample_size'], len(
            df)), random_state=CONFIG['random_seed'])
        print(f"已随机抽样 {len(sampled_df)} 条数据")

        # 生成并写入JSON数据
        with open(output_file, 'w', encoding='utf-8') as f:
            for _, row in sampled_df.iterrows():
                json_data = generate_json_data(row)
                f.write(json.dumps(json_data, ensure_ascii=False) + '\n')

        print(f"JSON训练数据已成功写入到文件: {output_file}")

        # 输出CSV文件（只包含三列）
        output_df = sampled_df[['spl_room_text',
                                's_room_text', 'label', 'spl_room_type_id']].copy()
        output_df.to_csv(output_csv_file, index=False, encoding='utf-8')
        print(f"CSV数据已成功写入到文件: {output_csv_file}")

    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")


if __name__ == "__main__":
    main()
