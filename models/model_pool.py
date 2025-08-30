from typing import Dict, List, Any, Optional
import threading
from .model_interface import BaseModel, OpenAIModel, AnthropicModel, HuggingFaceModel, DeepSeekModel

def create_model(config: Dict[str, Any]) -> BaseModel:
    """根据配置创建模型实例"""
    provider = config.get("provider", "").lower()
    
    if provider == "openai":
        return OpenAIModel(config)
    elif provider == "anthropic":
        return AnthropicModel(config)
    elif provider == "huggingface":
        return HuggingFaceModel(config)
    elif provider == "deepseek":
        return DeepSeekModel(config)
    else:
        raise ValueError(f"不支持的模型提供商: {provider}")


class ModelPool:
    def __init__(self):
        self._models: Dict[str, BaseModel] = {}
        self._model_configs = self._get_default_configs()
        self._lock = threading.Lock()
        self._initialize_models()
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        return {
            # Large Models
            "anthropic:claude-sonnet-4": {
                "provider": "anthropic",
                "model_name": "claude-sonnet-4-20250514",
                "max_tokens": 64000,
                "max_context_tokens": 64000,
                "cost_per_token": 0.03,
                "latency_class": "medium",
                "category": "large",
                "capabilities": ["reasoning", "coding", "analysis", "creative", "long_context", "chinese"]
            },
            "deepseek:deepseek-chat": {
                "provider": "deepseek",
                "model_name": "deepseek-chat",
                "max_tokens": 32768,
                "max_context_tokens": 32768,
                "cost_per_token": 0.001,  
                "latency_class": "medium",
                "category": "large",
                "capabilities": ["reasoning", "coding", "analysis", "creative", "chinese", "mathematics"]
            },
            
            # Medium Models
            "anthropic:claude-3-haiku": {
                "provider": "anthropic",
                "model_name": "claude-3-5-haiku-20241022",
                "max_tokens": 200000,
                "max_context_tokens": 200000,
                "cost_per_token": 0.25,
                "latency_class": "low",
                "category": "medium",
                "capabilities": ["chat", "qa", "summarization", "fast_response", "chinese"]
            },
            
            # Small Models
            "huggingface:codegen-350m": { # coding
                "provider": "huggingface",
                "model_name": "Salesforce/codegen-350M-mono",
                "local_path": "./models_cache/Salesforce_codegen-350M-mono",
                "max_tokens": 1024,
                "max_context_tokens": 1024,
                "cost_per_token": 0.0,
                "latency_class": "very_fast",
                "category": "small",
                "capabilities": ["coding", "code_completion", "programming"]
            },
            "huggingface:sentiment-analysis": { # sentiment analysis
                "provider": "huggingface",
                "model_name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "local_path": "./models_cache/cardiffnlp_twitter-roberta-base-sentiment-latest",
                "max_tokens": 512,
                "max_context_tokens": 512,
                "cost_per_token": 0.0,
                "latency_class": "very_fast",
                "category": "small",
                "capabilities": ["sentiment", "emotion", "classification"]
            },
            "huggingface:bert-qa": { # qa
                "provider": "huggingface",
                "model_name": "deepset/roberta-base-squad2",
                "local_path": "./models_cache/deepset_roberta-base-squad2",
                "max_tokens": 512,
                "max_context_tokens": 512,
                "cost_per_token": 0.0,
                "latency_class": "very_fast",
                "category": "small",
                "capabilities": ["qa", "education", "reading_comprehension"]
            },
            "huggingface:text-embedding": { # text embedding and similarity
                "provider": "huggingface",
                "model_name": "sentence-transformers/all-MiniLM-L6-v2",
                "local_path": "./models_cache/sentence-transformers_all-MiniLM-L6-v2",
                "max_tokens": 512,
                "max_context_tokens": 512,
                "cost_per_token": 0.0,
                "latency_class": "very_fast",
                "category": "small",
                "capabilities": ["text_embedding", "similarity", "semantic_search", "multilingual"]
            }
        }
    

    def _initialize_models(self):
        for model_id, config in self._model_configs.items():
            try:
                model = create_model(config)
                self._models[model_id] = model
                print(f"成功初始化模型: {model_id}")
            except Exception as e:
                print(f"初始化模型 {model_id} 失败: {e}")
    

    def get_available_models(self) -> List[BaseModel]:
        """获取所有可用模型"""
        available = []
        for model in self._models.values():
            if model.is_available():
                available.append(model)
        return available
    

    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """获取模型详细信息"""
        model = self._models.get(model_id)
        if not model:
            return None
            
        config = self._model_configs[model_id]
        info = model.get_model_info()
        info.update({
            "category": config.get("category"),
            "capabilities": config.get("capabilities", []),
            "performance": model.get_performance_stats()
        })
        return info
    

    def add_model(self, model_id: str, config: Dict[str, Any]):
        """动态添加新模型"""
        with self._lock:
            try:
                model = create_model(config)
                self._models[model_id] = model
                self._model_configs[model_id] = config
                print(f"成功添加模型: {model_id}")
                return True
            except Exception as e:
                print(f"添加模型 {model_id} 失败: {e}")
                return False
    

    def remove_model(self, model_id: str):
        """移除模型"""
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                del self._model_configs[model_id]
                print(f"成功移除模型: {model_id}")
                return True
            return False
    

    @property
    def large_models(self) -> Dict[str, BaseModel]:
        """获取大模型"""
        return {k: v for k, v in self._models.items() if self._model_configs[k].get("category") == "large"}
    

    @property  
    def medium_models(self) -> Dict[str, BaseModel]:
        """获取中等模型"""
        return {k: v for k, v in self._models.items() if self._model_configs[k].get("category") == "medium"}

    
    @property
    def small_models(self) -> Dict[str, BaseModel]:
        """获取小模型"""
        return {k: v for k, v in self._models.items() if self._model_configs[k].get("category") == "small"}
    
    
    def get_model_by_task_type(self, task_type: str) -> Optional[BaseModel]:
        """根据任务类型获取small模型"""
        task_model_mapping = { # 任务类型到模型ID的映射
            "code-generation": "huggingface:codegen-350m",
            "sentiment-analysis": "huggingface:sentiment-analysis", 
            "qa": "huggingface:bert-qa",
            "embedding": "huggingface:text-embedding"
        }
        
        model_id = task_model_mapping.get(task_type)
        if model_id and model_id in self._models:
            model = self._models[model_id]
            if model.is_available():
                return model
        
        print(f"未找到任务类型 {task_type} 对应的可用模型")
        return None
    
    
    def get_available_task_types(self) -> List[str]:
        """获取所有可用的任务类型"""
        task_model_mapping = {
            "code-generation": "huggingface:codegen-350m",
            "sentiment-analysis": "huggingface:sentiment-analysis", 
            "qa": "huggingface:bert-qa",
            "embedding": "huggingface:text-embedding"
        }
        
        available_tasks = []
        for task_type, model_id in task_model_mapping.items():
            if model_id in self._models and self._models[model_id].is_available():
                available_tasks.append(task_type)
        
        return available_tasks
    