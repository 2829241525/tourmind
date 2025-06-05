# import re
# import unicodedata

# # 床型正则表达式常量
# # double or twin
# DBL_OR_TWIN = r'((?P<num1>\d|one|two)?\s?(?P<type1>large twin|twin|single|queen|king|super king|double|round|semi.double))\s?(?:bed)?s?\s?(?:[|/]|or)\s?((?P<num2>\d?|one|two)?\s?(?P<type2>large twin|twin|single|queen|king|super king|double|semi.double))\s?(?:bed)?\b'
# # double and twin
# DBL_AND_TWIN = r'((?P<num1>\d)?\s?(?P<type1>twin|single|queen|king|double|bunk|sofa|semi.double))\s?(?:bed)?s?\s?(?:[&|]|and|with)\s?((?P<num2>\d)*\s?(?P<type2>twin|single|queen|king|double|bunk|sofa|tatami|semi.double))\s?\w*\s?(?:bed)?\s?\b'
# # king,queen,single,twin
# KNG_QUEEN = r'\b((?P<num>\d|one|two|three)?\s?(?P<type>king|queen|round|kang))\s?(?:size)?\s?(?:bed)?s?\b'
# # twin
# TWN = r'(?P<num>\d|one|two|three|four)?\s?(twin)\s?(bed)?s?\b'
# # double
# DBL = r'(?P<name1>(?P<num1>\d|one|two|three)?\s?(?:double)\s?(?P<bed1>bed)?s?)\b|(?P<name2>(?P<num2>\d|one|two|three)?\s?(?:double)\s(?P<bed2>bed)s?)\b'
# # single
# SGL = r'(?P<name1>(?P<num1>\d|one|two|three)?\s?(?:single)\s?(?P<bed1>bed)?s?)\b|(?P<name2>(?P<num2>\d|one|two|three)?\s?(?:single)\s(?P<bed2>bed)s?)\b'
# # 双层床 Room BUNK BED
# BUNK = r'(?P<num>\d)?\s?(twin)?\s?(bunk)\s?(bed)?'
# # 多张床
# MUL = r'\b(?P<num>\d|one|two|three|four|multiple)\s?(bed)s?\b'
# # 不处理床型
# PASS_STR = r'((or).+\s(and))|(\s(and).+(or))|(bunk)s?'
# # 半双人床
# SEMI = r'(?i:(?P<num>\d??).??semi.?double)'

# # 预编译正则表达式
# dbl_or_twin_reg = re.compile(DBL_OR_TWIN)
# dbl_and_twin_reg = re.compile(DBL_AND_TWIN)
# kng_queen_reg = re.compile(KNG_QUEEN)
# twin_reg = re.compile(TWN)
# single_reg = re.compile(SGL)
# dbl_reg = re.compile(DBL)
# bunk_reg = re.compile(BUNK)
# multiple_reg = re.compile(MUL)
# pass_reg = re.compile(PASS_STR)
# semi_reg = re.compile(SEMI)

# # 英文数字转换
# n_replace = {
#     "one": "1",
#     "two": "2",
#     "three": "3",
#     "four": "4",
#     "five": "5",
#     "six": "6",
#     "beds": "bed",
#     "bunks": "bunk",
# }

# # 中文床型ID映射表
# cn_bed_map = {
#     "大床": 2,
#     "双床": 3,
#     "单人床": 1,
#     "上下铺": 31,
#     "三人间": 4,
#     "大床房": 2,
#     "双床房": 3,
#     "单人间": 1,
#     "双人间": 2,
#     "圆床": 54,
#     "榻榻米": 59
# }


# def get_bed_id(spl_room_type, spl_bed_type=""):
#     """获取床型ID"""
#     if not spl_room_type and not spl_bed_type:
#         return 0
    
#     bed_str = spl_bed_type if spl_bed_type else spl_room_type
    
#     if is_chinese_char(bed_str):
#         # 中文处理逻辑
#         return get_type_bed_id_cn(bed_str)
    
#     bed_str = num_replace(bed_str.lower())
#     if bypass(bed_str):
#         return 9
    
#     _, _, bed_id = get_bed(bed_str)
#     return bed_id

# def with_beds(s):
#     """判断字符串是否包含床型描述"""
#     if not s:
#         return False
    
#     if not is_chinese_char(s):
#         s = num_replace(s.lower())
#         _, bed_type, _ = get_bed(s)
#         if bed_type == "unknown":
#             return False
#     else:
#         bed_id = get_type_bed_id_cn(s)
#         if bed_id == 0:
#             return False
    
#     return True

# def get_bed(s):
#     """获取床型信息"""
#     if dbl_or_twin_reg.search(s):
#         return double_or_twin(s)
#     elif dbl_and_twin_reg.search(s):
#         return double_and_twin(s)
#     elif kng_queen_reg.search(s):
#         return king_queen(s)
#     elif twin_reg.search(s):
#         return twin(s)
#     elif multiple_reg.search(s):
#         return multiple(s)
#     elif semi_reg.search(s):
#         return semi_double(s)
#     elif dbl_reg.search(s):
#         return double(s)
#     elif bunk_reg.search(s):
#         return bunk_bed(s)
#     elif single_reg.search(s):
#         return single(s)
#     else:
#         return s, "unknown", 9

# def bunk_bed(s):
#     """处理上下铺床型"""
#     room = bunk_reg.sub(" ", s)
#     matches = bunk_reg.findall(s)
    
#     # 提取匹配结果
#     match = bunk_reg.search(s)
#     if match:
#         result = match.groupdict()
        
#         if result["num"] == "2":
#             bed = "2bunk"
#             bed_id = 32
#         elif result["num"] == "3":
#             bed = "3bunk"
#             bed_id = 35
#         elif result["num"] == "4":
#             bed = "4bunk"
#             bed_id = 36
#         else:
#             bed = "bunk"
#             bed_id = 31
#     else:
#         bed = "unknown"
#         bed_id = 9
        
#     return room, bed, bed_id

# def multiple(s):
#     """处理多床类型"""
#     room = multiple_reg.sub(" ", s)
#     match = multiple_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         if result["num"] == "2":
#             bed = "twin"
#             bed_id = 3
#         else:
#             bed = "Multiple bed"
#             bed_id = 56
#     else:
#         bed = "unknown"
#         bed_id = 9
        
#     return room, bed, bed_id

# def double_or_twin(s):
#     """处理大床或双床类型"""
#     room = dbl_or_twin_reg.sub(" ", s)
#     match = dbl_or_twin_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         bed_type = (result["type1"] or "") + (result["type2"] or "")
#         num1 = int(result["num1"]) if result["num1"] and result["num1"].isdigit() else 0
#         num2 = int(result["num2"]) if result["num2"] and result["num2"].isdigit() else 0
#         num = num1 + num2
        
#         if num == 0 or num == 3:
#             bed = "doubleortwin"
#             bed_id = 9
#         elif num1 == 2 and num2 == 0 and bed_type == "twinsingle":
#             bed = "2 twin bed"
#             bed_id = 3
#         elif num1 == 1 and num2 == 1:
#             if bed_type in ["kingdouble", "doubleking"]:
#                 bed = "1 double bed"
#                 bed_id = 2
#             elif bed_type in ["twindouble", "doubletwin", "singledouble", "doublesingle"]:
#                 bed = "1 single bed"
#                 bed_id = 1
#             elif bed_type in ["queenking", "kingqueen"]:
#                 bed = "1 queen bed"
#                 bed_id = 2
#             else:
#                 bed = "doubleortwin"
#                 bed_id = 9
#         else:
#             bed = "doubleortwin"
#             bed_id = 9
#     else:
#         bed = "unknown"
#         bed_id = 9
        
#     return room, bed, bed_id

# def double_and_twin(s):
#     """处理大床和双床类型"""
#     room = dbl_and_twin_reg.sub(" ", s)
#     match = dbl_and_twin_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         num1, num2 = result["num1"] or "", result["num2"] or ""
#         type1, type2 = result["type1"] or "", result["type2"] or ""
        
#         if (num1 == "1" or num1 == "") and (num1 == num2):
#             bed_type = type1 + type2
            
#             if re.search(r'(double|queen|king)(single|twin)|(single|twin)(double|queen|king)', bed_type):
#                 bed = "doubleandtwin"
#                 bed_id = 5
#                 return room, bed, bed_id
            
#             if re.search(r'(double|queen|king|sofa)(double|queen|king|sofa)', bed_type):
#                 if "sofa" in s:
#                     bed = "1double1sofa"
#                     bed_id = 10
#                 elif type1 == "double" and type2 == "double" and s.endswith("double bed"):
#                     bed = "1 double bed"
#                     bed_id = 2
#                 else:
#                     bed = "2 double bed"
#                     bed_id = 6
#                 return room, bed, bed_id
            
#             if re.search(r'(semi.doublesingle)|(singlesemi.double)', bed_type) or ("semi" in bed_type and num1+num2 == "11"):
#                 bed = "1 Double and 1 Single"
#                 bed_id = 5
#                 return room, bed, bed_id
            
#             if bed_type == "bunkqueen":
#                 bed = "1 double 1 bunk"
#                 bed_id = 28
#                 return room, bed, bed_id
            
#             if bed_type == "queentatami":
#                 bed = "1 double 1 tatami"
#                 bed_id = 59
#                 return room, bed, bed_id
            
#             bed = "2bed"
#             bed_id = 6
#             return room, bed, bed_id
        
#         if num1+num2 == "12" or num1+num2 == "21":
#             if re.search(r'1(single|twin)', num1+type1+num2+type2):
#                 bed_id = 44
#             else:
#                 bed_id = 7
#             bed = f"{num1} {type1} and {num2} {type2}"
#             return room, bed, bed_id
        
#         if re.search(r'1(double|queen|king)', num1+type1+num2+type2) and (num1 == "" or num2 == ""):
#             bed = "1double1single"
#             bed_id = 10
#             return room, bed, bed_id
        
#         bed = "Multiple Bed"
#         bed_id = 56
#     else:
#         bed = "unknown"
#         bed_id = 9
    
#     return room, bed, bed_id

# def king_queen(s):
#     """处理大床类型"""
#     room = kng_queen_reg.sub(" ", s)
#     match = kng_queen_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         if result["num"] == "" or result["num"] == "1":
#             if result["type"] == "round":
#                 bed = result["type"]
#                 bed_id = 54
#             else:
#                 bed = result["type"]
#                 bed_id = 2
#         elif result["num"] == "2":
#             bed = result["num"] + " " + result["type"]
#             bed_id = 6
#         elif result["num"] == "3":
#             bed = "3" + result["type"]
#             bed_id = 48
#         elif result["num"] == "4":
#             bed = "4" + result["type"]
#             bed_id = 49
#         else:
#             bed = "multiplebeds"
#             bed_id = 56
#     else:
#         bed = "unknown"
#         bed_id = 9
    
#     return room, bed, bed_id

# def twin(s):
#     """处理双床类型"""
#     s = s.lower()
#     if s == "1 large twin bed":
#         return "", "1 large twin bed", 2
    
#     room = twin_reg.sub(" ", s)
#     match = twin_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         num = result["num"]
#         if num == "":
#             bed = "twin"
#             bed_id = 3
#         else:
#             if num == "1":
#                 bed = "single"
#                 bed_id = 1
#             elif num == "2":
#                 bed = "twin"
#                 bed_id = 3
#             elif num == "3":
#                 bed = "three-single"
#                 bed_id = 4
#             elif num == "4":
#                 bed = "four-single"
#                 bed_id = 8
#             else:
#                 bed = "multiplebeds"
#                 bed_id = 9
#     else:
#         bed = "unknown"
#         bed_id = 9
    
#     return room, bed, bed_id

# def single(s):
#     """处理单人床类型"""
#     room = single_reg.sub(" ", s)
#     match = single_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         num = result.get("num1") or ""
#         if num in ["1", ""]:
#             bed = "single"
#             bed_id = 1
#         elif num == "2":
#             bed = "twin"
#             bed_id = 3
#         elif num == "3":
#             bed = "three-single"
#             bed_id = 4
#         elif num == "4":
#             bed = "four-single"
#             bed_id = 8
#         else:
#             bed = "multiplebeds"
#             bed_id = 56
#     else:
#         bed = "unknown" 
#         bed_id = 9
    
#     return room, bed, bed_id

# def double(s):
#     """处理双人床类型"""
#     room = dbl_reg.sub(" ", s)
#     match = dbl_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         num = result.get("num1") or ""
#         if num:
#             if num == "1":
#                 bed = "double"
#                 bed_id = 2
#             elif num == "2":
#                 bed = "two-double"
#                 bed_id = 6
#             elif num == "3":
#                 bed = "3double"
#                 bed_id = 48
#             elif num == "4":
#                 bed = "4double"
#                 bed_id = 49
#             else:
#                 bed = "multiplebeds"
#                 bed_id = 0
#             return room, bed, bed_id
        
#         bed = "double"
#         bed_id = 2
#     else:
#         bed = "unknown"
#         bed_id = 9
    
#     return room, bed, bed_id

# def semi_double(s):
#     """处理半双人床类型"""
#     match = semi_reg.search(s)
    
#     if match:
#         result = match.groupdict()
        
#         if result["num"] in ["", "1"]:
#             bed_id = 55
#         elif result["num"] == "2":
#             bed_id = 3
#         else:
#             bed_id = 9
        
#         bed = "semi-double"
#     else:
#         bed = "unknown"
#         bed_id = 9
    
#     return "", bed, bed_id

# def is_chinese_char(s):
#     """检测是否包含中文字符"""
#     for char in s:
#         if 'CJK' in unicodedata.name(char, ''):
#             return True
#     return False

# def bypass(s):
#     """检查是否跳过床型处理"""
#     return bool(pass_reg.search(s))

# def num_replace(s):
#     """英文数字转阿拉伯数字"""
#     words = re.split(r'[, -]', s)
    
#     for i, word in enumerate(words):
#         if word in n_replace:
#             words[i] = n_replace[word]
    
#     s = ' '.join(words)
    
#     # 替换 'single use' 为空
#     s = re.sub(r'(single)\s(use)', '', s)
    
#     return s

# def get_type_bed_id_cn(s):
#     """中文床型处理"""
#     for bed_type, bed_id in cn_bed_map.items():
#         if bed_type in s:
#             return bed_id
#     return 0

# def extract_bed_type(room_type_str, bed_type_str=""):
#     """
#     提取床型信息的主函数
    
#     参数:
#         room_type_str: 房型描述字符串
#         bed_type_str: 床型描述字符串(可选)
        
#     返回:
#         tuple: (床型ID, 床型描述)
#     """
#     bed_id = get_bed_id(room_type_str, bed_type_str)
    
#     bed_str = bed_type_str if bed_type_str else room_type_str
#     _, bed_desc, _ = get_bed(bed_str.lower() if not is_chinese_char(bed_str) else bed_str)
    
#     return bed_id, bed_desc

# # 测试函数
# def test_bed_conversion():
#     """测试床型转换功能"""
#     test_cases = [
#         # 英文床型测试
#         "Deluxe Room with 1 King Bed",
#         "Standard Room with 2 Twin Beds",
#         "Suite with 1 Queen Bed and 1 Sofa Bed",
#         "Family Room with 1 Double Bed and 2 Single Beds",
#         "Room with Bunk Bed",
#         "Semi-Double Room",
#         # 中文床型测试
#         "豪华大床房",
#         "标准双床房",
#         "家庭房带上下铺"
#     ]
    
#     print("床型转换测试结果:")
#     print("-" * 50)
#     for case in test_cases:
#         bed_id, bed_desc = extract_bed_type(case)
#         print(f"原始描述: {case}")
#         print(f"提取结果: ID={bed_id}, 描述={bed_desc}")
#         print("-" * 50)

# if __name__ == "__main__":
#     # 运行测试
#     test_bed_conversion() 