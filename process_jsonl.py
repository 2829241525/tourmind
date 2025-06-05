import json
import os
from tqdm import tqdm

# 配置信息
CONFIG = {
    'input_file': '/home/maxon/disk2/roomMatch/room_match/rft/process/train.jsonl',
    'output_file': '/home/maxon/disk2/roomMatch/room_match/rft/process/extracted_content.txt',
    'batch_size': 1000,  # 每次处理的行数
}


def process_jsonl():
    if not os.path.exists(CONFIG['input_file']):
        print(f"输入文件不存在: {CONFIG['input_file']}")
        return

    total_lines = sum(1 for _ in open(
        CONFIG['input_file'], 'r', encoding='utf-8'))
    print(f"总行数: {total_lines}")

    with open(CONFIG['output_file'], 'w', encoding='utf-8') as out_file:
        with open(CONFIG['input_file'], 'r', encoding='utf-8') as in_file:
            for line_num, line in tqdm(enumerate(in_file, 1), total=total_lines, desc="处理进度"):
                try:
                    data = json.loads(line.strip())
                    # 检查messages字段中的content
                    messages = data.get('messages', [])
                    content = ""
                    for msg in messages:
                        if msg.get('role') == 'user':
                            content = msg.get('content', '')
                            break

                    final_answer = data.get('final_answer', '')

                    # 写入格式化的内容
                    out_file.write(f"=== 条目 {line_num} ===\n")
                    out_file.write("Content:\n")
                    out_file.write(f"{content}\n")
                    out_file.write("Final Answer:\n")
                    out_file.write(f"{final_answer}\n")
                    out_file.write("\n" + "="*50 + "\n\n")

                except json.JSONDecodeError as e:
                    print(f"第 {line_num} 行解析失败: {e}")
                except Exception as e:
                    print(f"处理第 {line_num} 行时发生错误: {e}")


if __name__ == "__main__":
    print(f"开始处理文件: {CONFIG['input_file']}")
    process_jsonl()
    print(f"处理完成，结果已保存到: {CONFIG['output_file']}")
