import pandas as pd
import os

def analyze_text_length():
    # 读取数据文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(current_dir, 'processed_all_room_match.csv')
    
    # 读取CSV文件
    df = pd.read_csv(data_file)
    
    # 计算spl_room_text的长度
    df['spl_text_length'] = df['spl_room_text'].str.len()
    # 计算s_room_text的长度
    df['target_text_length'] = df['s_room_text'].str.len()
    
    # 找到最长的spl_room_text
    max_spl_idx = df['spl_text_length'].idxmax()
    max_spl_text = df.loc[max_spl_idx, 'spl_room_text']
    max_spl_length = df.loc[max_spl_idx, 'spl_text_length']
    
    # 找到最长的s_room_text
    max_target_idx = df['target_text_length'].idxmax()
    max_target_text = df.loc[max_target_idx, 's_room_text']
    max_target_length = df.loc[max_target_idx, 'target_text_length']
    
    print("\n源文本最长的数据:")
    print(f"长度: {max_spl_length}")
    print(f"文本: {max_spl_text}")
    print("\n目标文本最长的数据:")
    print(f"长度: {max_target_length}")
    print(f"文本: {max_target_text}")
    
    # 打印一些统计信息
    print("\n统计信息:")
    print("源文本长度统计:")
    print(df['spl_text_length'].describe())
    print("\n目标文本长度统计:")
    print(df['target_text_length'].describe())

if __name__ == '__main__':
    analyze_text_length() 