import pandas as pd
import argparse
import os
#吸烟相关信息数据清洗
# 配置项
DEFAULT_CONFIG = {
    'input_file': 'processed_jp.csv',
    'output_file': 'processed_jp_filtered.csv',
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
    #设定基础路径
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    # 读取CSV文件
    df = pd.read_csv(os.path.join(CURRENT_DIR, input_file))
    
    # 初始化要删除的行索引
    rows_to_drop = []
    
    # 遍历label为1的数据
    for idx, row in df[df['label'] == 1].iterrows():
        spl_text = str(row['spl_room_text']).lower()
        s_text = str(row['s_room_text']).lower()
        
        
        spl_has_nonsmok = "non-smoking" in spl_text
        s_has_nonsmok = "non-smoking" in s_text
        if spl_has_nonsmok != s_has_nonsmok:  # 如果一方有而另一方没有
            rows_to_drop.append(idx)

        spl_has_smok = "smoking" in spl_text
        s_has_smok = "smoking" in s_text
        if spl_has_smok != s_has_smok and spl_has_nonsmok is False and s_has_nonsmok is False:  # 如果一方有而另一方没有
            rows_to_drop.append(idx)
    
    # 保存被删除的行到新的CSV文件
    dropped_file = output_file.replace('.csv', '_dropped.csv')
    df.loc[rows_to_drop].to_csv(os.path.join(CURRENT_DIR, dropped_file), index=False)

    # 删除不符合条件的行
    df_filtered = df.drop(rows_to_drop)
    
    # 保存结果
    df_filtered.to_csv(os.path.join(CURRENT_DIR, output_file), index=False)
    print(f"原始数据行数: {len(df)}")
    print(f"过滤后数据行数: {len(df_filtered)}")
    print(f"删除的行数: {len(rows_to_drop)}")
    print(f"删除比例: {len(rows_to_drop)/len(df)*100:.2f}%")
    print(f"被删除的行已保存到: {dropped_file}")

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