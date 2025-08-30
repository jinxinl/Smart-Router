from pathlib import Path
from typing import Dict
import shutil
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from transformers import pipeline
        

def download_model(model_name: str, cache_dir: Path) -> bool:
    """下载单个模型"""
    try:        
        print(f"正在下载: {model_name}")        
        # 创建本地目录
        local_dir = cache_dir / model_name.replace("/", "_")
        
        if local_dir.exists():
            print(f"模型已存在，跳过: {model_name}")
            return True
        
        if "sentence-transformers" in model_name: # embedding
            model = SentenceTransformer(model_name)
            local_dir.mkdir(parents=True, exist_ok=True)
            model.save(str(local_dir))
            print(f"下载完成: {model_name}")
            return True
        elif "codegen" in model_name.lower(): # code generation
            tokenizer = AutoTokenizer.from_pretrained(model_name,resume_download=True)
            model = AutoModelForCausalLM.from_pretrained(model_name,resume_download=True)
        elif "sentiment" in model_name.lower(): # sentiment analysis
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        elif "squad" in model_name.lower(): # qa
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        else: # cunstomed
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
        
        local_dir.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
        
        print(f"下载完成: {model_name}")
        return True
        
    except Exception as e:
        print(f"下载失败: {model_name}, 错误: {e}")
        return False


def get_hf_models() -> Dict[str, Dict]:
    return {
        "Salesforce/codegen-350M-mono": {
            "task": "code_generation",
            "description": "小型代码生成模型",
            "size": "350M"
        },
        "cardiffnlp/twitter-roberta-base-sentiment-latest": {
            "task": "sentiment_analysis", 
            "description": "Twitter情感分析模型",
            "size": "125M"
        },
        "deepset/roberta-base-squad2": {
            "task": "question_answering",
            "description": "英文问答模型",
            "size": "125M"
        },
        "sentence-transformers/all-MiniLM-L6-v2": {
            "task": "text_embedding",
            "description": "多语言文本嵌入模型",
            "size": "23M"
        }
    }


def check_disk_space(required_gb: float = 2.0) -> bool:
    """检查磁盘空间"""
    try:
        free_bytes = shutil.disk_usage('.').free
        free_gb = free_bytes / (1024**3)
        
        if free_gb < required_gb:
            print(f"磁盘空间不足: 需要 {required_gb}GB, 可用 {free_gb:.1f}GB")
            return False
        
        print(f"磁盘空间充足: 可用 {free_gb:.1f}GB")
        return True
        
    except Exception as e:
        print(f"无法检查磁盘空间: {e}")
        return True



def test_model_loading(model_name: str, cache_dir: Path) -> bool:
    """测试模型加载"""
    try:
        local_dir = cache_dir / model_name.replace("/", "_")
        if not local_dir.exists():
            print(f"模型目录不存在: {local_dir}")
            return False
        
        if "sentence-transformers" in model_name: # embedding test
            model = SentenceTransformer(str(local_dir))
            embedding = model.encode("This is a test sentence.")
            print(f"文本嵌入测试通过: 维度 {len(embedding)}")
            
        elif "codegen" in model_name: # code generation test
            pipe = pipeline("text-generation", model=str(local_dir))
            result = pipe("def hello():", max_length=20, num_return_sequences=1)
            print(f"代码生成测试通过: {result[0]['generated_text']}")
            
        elif "sentiment" in model_name: # sentiment analysis test
            pipe = pipeline("sentiment-analysis", model=str(local_dir))
            result = pipe("I love this!")
            print(f"情感分析测试通过: {result[0]['label']}")
            
        elif "squad" in model_name: # qa test
            pipe = pipeline("question-answering", model=str(local_dir))
            result = pipe(question="What is AI?", context="AI is artificial intelligence.")
            print(f"问答测试通过: {result['answer']}")
            
        else: # customed test
            pipe = pipeline("customed", model=str(local_dir))
            result = pipe("This is a test.")
            print(f"自定义测试通过")
        
        return True
        
    except Exception as e:
        print(f"模型加载测试失败: {e}")
        return False


def main():
    cache_dir = Path("./models_cache")
    cache_dir.mkdir(exist_ok=True)
    
    if not check_disk_space(): # 检查磁盘空间
        return

    models = get_hf_models() # 获取模型列表
    
    print(f"下载 {len(models)} 个模型:")
    for model_name, info in models.items():
        print(f"{model_name} ({info['size']}) - {info['description']}")
    
    success_count = 0
    for model_name, info in models.items():
        if download_model(model_name, cache_dir):
            success_count += 1
            print(f"下载完成: {success_count}/{len(models)} 个模型成功下载")
    

    # 测试模型加载
    if success_count > 0:
        print("测试模型加载...")
        for model_name in models.keys():
            local_dir = cache_dir / model_name.replace("/", "_")
            if local_dir.exists():
                print(f"测试: {model_name}")
                test_model_loading(model_name, cache_dir)
    
    print("下载完成")


if __name__ == "__main__":
    main()
