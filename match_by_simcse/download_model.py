from transformers import AutoModel, AutoTokenizer
import os

def download_model():
    # 设置使用镜像站点
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 设置模型名称
    model_name = "princeton-nlp/sup-simcse-roberta-base"
    # 设置保存路径
    save_path = os.path.join(os.path.dirname(__file__), "pretrained_models", "sup-simcse-roberta-base")
    
    print(f"开始下载模型 {model_name}")
    print(f"使用镜像站点: https://hf-mirror.com")
    print(f"模型将保存到: {save_path}")
    
    try:
        # 下载模型和分词器
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)
        
        # 保存模型和分词器
        tokenizer.save_pretrained(save_path)
        model.save_pretrained(save_path)
        
        print("模型下载完成！")
        print(f"模型已保存到: {save_path}")
        
        # 更新配置文件中的模型路径
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.json")
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config['pretrained_model'] = save_path
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            print("配置文件已更新！")
            
    except Exception as e:
        print(f"下载失败: {str(e)}")
        print("如果下载仍然失败，可以尝试使用 huggingface-cli 命令行工具下载：")
        print("1. pip install -U huggingface_hub")
        print("2. export HF_ENDPOINT=https://hf-mirror.com")
        print(f"3. huggingface-cli download --resume-download {model_name} --local-dir {save_path}")

if __name__ == "__main__":
    download_model() 