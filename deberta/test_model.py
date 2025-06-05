from transformers import AutoModel, AutoTokenizer
import os

def test_model():
    # 设置为离线模式
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    
    # 设置模型路径
    model_path = os.path.join(os.path.dirname(__file__), "pretrained_models", "sup-simcse-bert-base-uncased")
    
    print(f"正在从以下路径加载模型：{model_path}")
    
    try:
        # 加载分词器
        print("加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True  # 强制只使用本地文件
        )
        
        # 加载模型
        print("加载模型...")
        model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True  # 强制只使用本地文件
        )
        
        # 测试简单的文本编码
        text = "This is a test sentence."
        print(f"\n测试文本编码：'{text}'")
        
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        
        print("\n模型输出形状：")
        print(f"last_hidden_state: {outputs.last_hidden_state.shape}")
        print("\n测试成功！模型可以正常使用。")
        
    except Exception as e:
        print(f"测试失败：{str(e)}")
        print("\n请检查模型文件是否完整，目录中应该包含：")
        print("- config.json")
        print("- pytorch_model.bin")
        print("- tokenizer_config.json")
        print("- vocab.txt")
        print("- special_tokens_map.json")

if __name__ == "__main__":
    test_model() 