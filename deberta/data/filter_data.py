import pandas as pd
import argparse
import os
#吸烟相关信息数据清洗
# 配置项
DEFAULT_CONFIG = {
    'input_file': 'processed_country_data_splct_clean.csv',
    'output_file': 'processed_country_data_splct_filtered.csv',
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

    # 处理label为1的情况下的bed后缀逻辑
    def check_bed_suffix_conflict(row):
        if row['label'] != 1:
            return False
            
        # 获取[bed]后面的文本
        def get_bed_suffix(text):
            if '[bed]' not in text.lower():
                return ''
            return text.lower().split('[bed]')[1].strip()
            
        spl_suffix = get_bed_suffix(row['spl_room_text'])
        s_suffix = get_bed_suffix(row['s_room_text'])
        
        # 检查是否存在冲突（s中有and，spl中有or）
        if 'and' in s_suffix and 'or' in spl_suffix:
            return True
        
            
        return False

    # 找出需要删除的行
    # rows_to_drop_bed = df_filtered[df_filtered.apply(check_bed_suffix_conflict, axis=1)].index
    
    # if len(rows_to_drop_bed) > 0:
    #     print(f"发现 {len(rows_to_drop_bed)} 行在label为1时存在bed后缀的and/or冲突，将被删除")
        
    # # 删除这些行
    # df_filtered = df_filtered.drop(rows_to_drop_bed)

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