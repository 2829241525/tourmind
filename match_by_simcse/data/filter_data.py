import pandas as pd
import argparse
import os

# 配置项
DEFAULT_CONFIG = {
    'input_file': 'country.csv',
    'output_file': 'filter.csv',
    'check_nonsmok': True,  # 是否检查nonsmok关键词
    'case_sensitive': False,  # 是否区分大小写
}

def filter_data(input_file=DEFAULT_CONFIG['input_file'], 
               output_file=DEFAULT_CONFIG['output_file']):
    """
    过滤数据：
    对于label为1的数据，如果spl_room_text和s_room_text中有一方包含"nonsmok"而另一方不包含，则去除该行
    
    Args:
        input_file (str): 输入CSV文件路径，默认为country.csv
        output_file (str): 输出CSV文件路径，默认为filter.csv
    """
    # 读取CSV文件
    df = pd.read_csv(input_file)
    
    # 初始化要删除的行索引
    rows_to_drop = []
    
    # 遍历label为1的数据
    for idx, row in df[df['label'] == 1].iterrows():
        if DEFAULT_CONFIG['case_sensitive']:
            spl_text = str(row['spl_room_text'])
            s_text = str(row['s_room_text'])
        else:
            spl_text = str(row['spl_room_text']).lower()
            s_text = str(row['s_room_text']).lower()
        
        # 检查nonsmok（双向检查）
        if DEFAULT_CONFIG['check_nonsmok']:
            spl_has_nonsmok = "nonsmok" in spl_text
            s_has_nonsmok = "nonsmok" in s_text
            if spl_has_nonsmok != s_has_nonsmok:  # 如果一方有而另一方没有
                rows_to_drop.append(idx)
    
    # 删除不符合条件的行
    df_filtered = df.drop(rows_to_drop)
    
    # 保存结果
    df_filtered.to_csv(output_file, index=False)
    print(f"原始数据行数: {len(df)}")
    print(f"过滤后数据行数: {len(df_filtered)}")
    print(f"删除的行数: {len(rows_to_drop)}")
    print(f"删除比例: {len(rows_to_drop)/len(df)*100:.2f}%")

def main():
    parser = argparse.ArgumentParser(description='过滤CSV数据')
    parser.add_argument('--input', type=str, default=DEFAULT_CONFIG['input_file'],
                      help=f'输入CSV文件路径 (默认: {DEFAULT_CONFIG["input_file"]})')
    parser.add_argument('--output', type=str, default=DEFAULT_CONFIG['output_file'],
                      help=f'输出CSV文件路径 (默认: {DEFAULT_CONFIG["output_file"]})')
    
    args = parser.parse_args()
    
    filter_data(args.input, args.output)

if __name__ == '__main__':
    main() 