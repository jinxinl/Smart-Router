from flask import Flask, render_template, request, jsonify
from datetime import datetime
from utils import feature_extractor
from utils.processor import Processor
from utils.smart_router import SmartRouter
from utils.feature_extractor import FeatureExtractor
from utils.misc import ConversationManager, make_json_serializable
from models.model_pool import ModelPool


app = Flask(__name__)
app.secret_key = 'smart-router-secret-key-2024'


print("正在初始化Smart Router系统...")
model_pool = ModelPool()
smart_router = SmartRouter(model_pool)
feature_extractor = FeatureExtractor()
processor = Processor(model_pool, smart_router, feature_extractor)
print("Smart Router系统初始化完成")

conversations = {} # 会话存储
conv_manager = ConversationManager() # 对话管理器


@app.route('/')
def index():
    """主页"""
    return render_template('chat.html')


@app.route('/models')
def models_page():
    """模型池管理页面"""
    return render_template('models.html')


@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """获取所有对话列表"""
    conversations = conv_manager.get_all_conversations()
    return jsonify(make_json_serializable({
        "success": True,
        "conversations": conversations
    }))


@app.route('/api/conversations', methods=['POST'])
def create_conversation():
    """创建新对话"""
    conv_id = conv_manager.create_conversation()
    return jsonify({
        "success": True,
        "conversation_id": conv_id
    })


@app.route('/api/conversations/<conv_id>', methods=['GET'])
def get_conversation(conv_id):
    """获取特定对话"""
    conversation = conv_manager.get_conversation(conv_id)
    if conversation:
        return jsonify(make_json_serializable({
            "success": True,
            "conversation": conversation
        }))
    else:
        return jsonify({
            "success": False,
            "error": "对话不存在"
        }), 404


@app.route('/api/conversations/<conv_id>', methods=['DELETE'])
def delete_conversation(conv_id):
    """删除对话"""
    success = conv_manager.delete_conversation(conv_id)
    return jsonify({
        "success": success
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """处理聊天请求"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        conv_id = data.get('conversation_id')
        
        if not message:
            return jsonify({
                "success": False,
                "error": "消息不能为空"
            }), 400
        
        print(f"[WEB] 处理查询: {message}")
        result = processor.process_single_query(message)
        
        if not conv_id or conv_id == 'null': # 保存到对话历史
            conv_id = conv_manager.create_conversation()

        final_conv_id = conv_manager.add_message(conv_id, message, result)
        
        routing_info = result.get("routing_info", {}).copy() # 获取路由信息
        if "model" in result and result["model"]:
            model_obj = result["model"]
            routing_info["selected_model"] = model_obj.model_name
            routing_info["selected_provider"] = getattr(model_obj, "provider", "unknown")
            print(f"[WEB] 添加模型信息: {model_obj.model_name} ({routing_info['selected_provider']})")
        
        
        return jsonify(make_json_serializable({
            "success": True,
            "conversation_id": final_conv_id,
            "response": result.get("response", "抱歉，处理出现问题"),
            "routing_info": routing_info,
            "features_used": result.get("features_used", {}),
            "processing_time": result.get("processing_time", 0),
            "timestamp": datetime.now().isoformat()
        }))
        

    except Exception as e:
        print(f"[ERROR] 聊天处理错误: {e}")
        return jsonify({
            "success": False,
            "error": f"处理出错: {str(e)}"
        }), 500


@app.route('/api/system/status', methods=['GET'])
def system_status():
    """获取系统状态"""
    try:
        all_models = [] # 获取所有模型信息
        for category in ['large', 'medium', 'small']:
            if category == 'large':
                models = model_pool.large_models
            elif category == 'medium':
                models = model_pool.medium_models
            else:
                models = model_pool.small_models
            
            for model_id, model in models.items():
                model_info = model.get_model_info()
                performance_stats = model.get_performance_stats()
                
                all_models.append({
                    "id": model_id,
                    "name": model.model_name,
                    "category": category,
                    "provider": model.provider,
                    "available": model.is_available(),
                    "max_tokens": model_info.get("max_tokens", 0),
                    "max_context_tokens": model_info.get("max_context_tokens", 0),
                    "cost_per_token": model_info.get("cost_per_token", 0),
                    "latency_class": model_info.get("latency_class", "unknown"),
                    "performance": {
                        "total_requests": performance_stats.get("total_requests", 0),
                        "total_tokens": performance_stats.get("total_tokens", 0),
                        "average_latency": performance_stats.get("average_latency", 0),
                        "error_rate": performance_stats.get("error_rate", 0)
                    }
                })
        
        available_models = [m for m in all_models if m["available"]] # 获取可用模型
        available_tasks = model_pool.get_available_task_types() # 获取任务类型
        router_stats = smart_router.get_performance_stats() # 获取路由器统计信息
        
        return jsonify(make_json_serializable({
            "success": True,
            "system_status": {
                "all_models": all_models,
                "available_models": available_models,  
                "available_task_types": available_tasks,
                "router_statistics": router_stats,
                "total_conversations": len(conv_manager.conversations)
            }
        }))
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500



if __name__ == '__main__':
    print("访问地址: http://localhost:5000")
    app.run(debug=False, host='0.0.0.0', port=5000)
