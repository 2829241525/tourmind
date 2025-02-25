import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime


class DataCleaner:
    def __init__(self, mapping_file: Path):
        self.mapping_file = mapping_file
        self.replacement_rules = self._load_replacement_rules()

    def _load_replacement_rules(self) -> List[Tuple[str, str, int]]:
        # 定义映射替换模式
        xlsx = pd.ExcelFile(self.mapping_file)
        sheet_names = xlsx.sheet_names
        rules = []

        # 按照sheet顺序构建字典
        for sheet_idx, sheet_name in enumerate(sheet_names, 1):
            sheet_data = pd.read_excel(xlsx, sheet_name=sheet_name)

            if {'key', 'value'}.issubset(sheet_data.columns):
                for row_idx, row in sheet_data.iterrows():
                    key = str(row['key']).strip().lower()
                    values = [v.strip().strip('"') for v in str(row['value']).split(',')]
                    rules.extend([
                        (value.lower(), key, len(value))
                        for value in values if value
                    ])

        # 按原文本长度降序排序，确保优先匹配最长的文本
        sorted_rules = sorted(rules, key=lambda x: x[2], reverse=True)
        return sorted_rules

    @staticmethod
    def _remove_after_pipe(text: Optional[str]) -> Optional[str]:
        # 删除第一个竖线(|)及其后面的所有内容
        if pd.isna(text):
            return text
        return re.sub(r'\|.*', '', str(text))

    @staticmethod
    def _concatenate_columns(row: pd.Series) -> str:
        # 将房间名称和床型信息拼接成特定格式
        return f"[room] {row['s_room_name']} [bed] {row['s_room_bed_name']}"

    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return text

        text_lower = str(text).lower()

        # 依次应用所有替换规则
        for value, key, _ in self.replacement_rules:
            pattern = re.compile(
                rf'(?:^|(?<=[^a-zA-Z0-9-]))'  # 开始边界：确保前面是非字母数字和横杠
                rf'{re.escape(value)}'  # 需要匹配的文本
                rf'(?:$|(?=[^a-zA-Z0-9-]))',  # 结束边界：确保后面是非字母数字和横杠
                re.IGNORECASE
            )
            text_lower = pattern.sub(key, text_lower)

        return text_lower

    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        processed_df = df.copy()

        # 步骤一：删除竖线后的内容并转小写
        for col in ['s_room_name', 's_room_bed_name']:
            processed_df[col] = processed_df[col].apply(self._remove_after_pipe).str.lower()

        # 步骤二：拼接房型和床型信息
        processed_df['concatenated'] = processed_df.apply(self._concatenate_columns, axis=1)

        # 步骤三：应用文本清洗规则，并显示进度条
        tqdm.pandas(desc="正在处理数据")
        processed_df['cleaned_text'] = processed_df['concatenated'].progress_apply(self.clean_text)

        return processed_df


def main():
    # 配置文件路径
    INPUT_FILE = Path("sample_data.csv")
    MAPPING_FILE = Path("mapping.xlsx")
    OUTPUT_FILE = Path(f"cleaned_data.csv")

    # 数据清洗
    cleaner = DataCleaner(MAPPING_FILE)
    processed_df = cleaner.process_dataframe(INPUT_FILE)
    processed_df.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
