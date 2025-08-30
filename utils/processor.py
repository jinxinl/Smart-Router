from models.model_pool import ModelPool
from utils.feature_extractor import FeatureExtractor
from utils.smart_router import SmartRouter
from utils.misc import aggregate_response
import time

class Processor:
    def __init__(self, model_pool: ModelPool, smart_router: SmartRouter, feature_extractor: FeatureExtractor):
        self.model_pool = model_pool
        self.smart_router = smart_router
        self.feature_extractor = feature_extractor

    def process_single_query(self, query: str, context: dict = None) -> dict:
        start_time = time.time()
        try:
            print(f"正在处理查询: {query}")

            features = self.feature_extractor.extract_features(query, context) # 特征提取
            route_result = self.smart_router.route(features) # 智能路由算法选择模型
            
            model = route_result["model"]
            if model is None:
                return {
                    "response": "抱歉，当前没有可用的模型来处理您的请求。",
                    "success": False,
                    "error": "No available model",
                    "processing_time": time.time() - start_time
                }
            
            category = route_result.get('category', 'unknown')
            task_type = route_result.get('task_type')
            
            if task_type:
                print(f"选择模型: {model.model_name} (类别: {category}, 任务类型: {task_type})")
            else:
                print(f"选择模型: {model.model_name} (类别: {category})")

            print(f"选择理由: {route_result.get('reasoning', 'N/A')}")
            
            model_response = model.invoke(query) # 模型调用
            
            aggregated = aggregate_response(model_response, route_result, query) # 结果聚合 
            aggregated.update({ # 添加额外信息
                "query": query,
                "model": model,  
                "processing_time": time.time() - start_time,
                "routing_info": {
                    "selected_category": route_result.get('category'),
                    "selected_task_type": route_result.get('task_type'),
                    "confidence": route_result.get('confidence'),
                    "task_confidence": route_result.get('task_confidence'),
                    "reasoning": route_result.get('reasoning')
                },
                "features_used": {
                    "domain": features.get("domain"),
                    "task_type": features.get("task_type"),
                    "complexity": features.get("complexity"),
                    "token_estimate": features.get("token_est")
                }
            })
            
            print(f"处理完成，模型回复: {aggregated['response']}\n用时: {aggregated['processing_time']:.2f}秒")
            return aggregated
            
        except Exception as e:
            error_response = {
                "response": f"处理请求时发生错误: {str(e)}",
                "success": False,
                "error": str(e),
                "query": query,
                "processing_time": time.time() - start_time
            }
            
            print(f"处理失败: {str(e)}")
            return error_response

        
if __name__ == "__main__":
    processor = Processor()
    test_query = "write 'hello'"
    result = processor.process_single_query(test_query)
    print(f"测试完成: {result.get('success', False)}")

    # large: Create a detailed implementation plan for transitioning a large enterprise from monolithic architecture to microservices, including technical migration strategy, team restructuring, DevOps pipeline adjustments, and risk mitigation approaches.
    # medium: What are the benefits of intermittent fasting?
    # small: 
    #  - code-generation: 很少，一般会被判断为medium，可以用use c++ to write hello
    #  - sentiment-analysis: "What's the sentiment of this: 'This product completely failed to meet my expectations. Terrible purchase."
    #  - qa: 效果不好，需要上下文
    #  - embedding：很少，一般会被判断为medium，可以用 This is a test sentence.