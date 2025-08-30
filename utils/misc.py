import os
import json
from typing import Dict, Any
from datetime import datetime
import time
import numpy as np
import uuid


def load_data(file_path: str):
    """加载训练数据"""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    full_path = os.path.join(base_dir, file_path)
    
    with open(full_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"加载了 {len(data)} 条训练数据")
    return data


def make_json_serializable(obj):
    """将类型转换为可JSON序列化的格式"""
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__') and hasattr(obj, 'model_name'):
        # 模型对象
        return {
            'model_name': getattr(obj, 'model_name', 'unknown'),
            'provider': getattr(obj, 'provider', 'unknown'),
            'model_type': type(obj).__name__
        }
    elif hasattr(obj, '__dict__'):
        # 其他复杂对象转换为字符串表示
        return str(obj)
    else:
        return obj


def aggregate_response(model_response: Dict[str, Any], routing_info: Dict[str, Any], query: str) -> Dict[str, Any]:
        """聚合模型响应"""
        aggregated = { # 基础信息
            "response": model_response.get("response", ""),
            "success": model_response.get("success", False),
            "model_used": model_response.get("model", "unknown"),
            "tokens_used": model_response.get("tokens_used", 0),
            "latency": model_response.get("latency", 0.0),
            "timestamp": time.time()
        }
        
        aggregated["routing"] = { # 添加路由信息
            "category": routing_info.get("category", "unknown"),
            "confidence": routing_info.get("confidence", 0.0),
            "reasoning": routing_info.get("reasoning", ""),
            "alternatives": routing_info.get("alternatives", [])
        }
        
        return aggregated
    

class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations = {}
    
    def create_conversation(self, title: str = "新对话") -> str:
        """创建新对话"""
        conv_id = str(uuid.uuid4())
        self.conversations[conv_id] = {
            "id": conv_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "messages": [],
            "last_updated": datetime.now().isoformat()
        }
        return conv_id
    
    def add_message(self, conv_id: str, user_message: str, ai_response: dict):
        """添加消息到对话"""
        if conv_id not in self.conversations:
            conv_id = self.create_conversation()
        
        clean_ai_response = {k: v for k, v in ai_response.items() if k != 'model'} # 创建一个不包含模型对象的副本
        
        message = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "ai_response": clean_ai_response,
            "routing_info": ai_response.get("routing_info", {}),
            "features_used": ai_response.get("features_used", {}),
            "processing_time": ai_response.get("processing_time", 0)
        }
        
        self.conversations[conv_id]["messages"].append(message)
        self.conversations[conv_id]["last_updated"] = datetime.now().isoformat()
        
        # 如果是第一条消息，更新对话标题
        if len(self.conversations[conv_id]["messages"]) == 1:
            title = user_message[:30] + "..." if len(user_message) > 30 else user_message
            self.conversations[conv_id]["title"] = title
        
        return conv_id
    

    def get_conversation(self, conv_id: str) -> dict:
        """获取对话"""
        return self.conversations.get(conv_id)
    
    
    def get_all_conversations(self) -> list:
        """获取所有对话，按最后更新时间排序"""
        convs = list(self.conversations.values())
        return sorted(convs, key=lambda x: x["last_updated"], reverse=True)
    

    def delete_conversation(self, conv_id: str) -> bool:
        """删除对话"""
        if conv_id in self.conversations:
            del self.conversations[conv_id]
            return True
        return False

