import json
import time
from typing import List, Dict, Any
import os
from uu import Error
import anthropic


class ClaudeDataGenerator:
    def __init__(self, api_key: str = None):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))        
        if api_key is None:
            raise Error("ANTHROPIC_API_KEY not found")
    
        self.client = anthropic.Anthropic(api_key=api_key)
        self.generation_prompt = self._create_generation_prompt() 
    
    def _create_generation_prompt(self) -> str:
        return """
You are a training data generation expert for an intelligent AI model routing system. I need you to generate training data for training an XGBoost classifier to decide which type of AI model should handle user queries.

## Target Category Descriptions:

**large** - Large Models (like GPT-4):
- Complex analysis, deep reasoning, professional consulting, long text analysis, deep reasoning tasks
- Code generation, system design, academic writing
- Business analysis, creative content, multi-step problem solving
- Example: "Please analyze the impact of AI on the financial industry and provide investment advice"

**medium** - Medium Models (like GPT-3.5):
- General Q&A, concept explanations, simple summaries
- Translation, recommendations, tutorial writing
- Regular conversations, information queries
- Example: "Please explain what machine learning is"

**small** - Small Models (like BERT):
- Simple greetings, factual queries, keyword extraction
- Sentiment analysis, classification tasks, simple Q&A
- Short text processing
- Example: "Hello", "What is an API?"

## Output Format Requirements:

Please generate data in JSON format, wrap the returned data in json blocks, each sample should include:
```json
{{
  "query": "user query text",
  "best_model_category": "large/medium/small",
  "context": {{
    "has_history": true/false,
    "user_type": "beginner/intermediate/expert/professional",
    "session_duration": number (seconds),
    "domain_preference": "technology/business/education/general"
  }}
}}
```

Notes:
1. The context field has a 30% probability of being null
2. Queries should cover different domains: technology, business, science, education, health, entertainment, lifestyle, travel, general, etc. Domain types should be as diverse as possible
3. Query lengths should vary: from simple words to complex paragraphs
4. Ensure accurate labels: complexity should match model categories

Please generate {batch_size} training samples, ensuring reasonable category distribution. Only return the JSON format array, no other explanations.
"""    


    def generate_batch(self, batch_size: int = 20, category_focus: str = None) -> List[Dict[str, Any]]:
        """生成一批训练数据"""
        prompt = self.generation_prompt.format(batch_size=batch_size)
        
        if category_focus:
            prompt += f"\n\nSpecial Requirement: For this batch of data, please focus on generating samples for the {category_focus} category (accounting for more than 60% of the samples)."
        
        try:
            print(f"正在生成 {batch_size} 个样本...")
            
            response = self.client.messages.create(
                model="claude-3-7-sonnet-20250219", 
                max_tokens=4000,
                temperature=0.7,
                messages=[{
                    "role": "user", 
                    "content": prompt
                }]
            )
            
            response_text = response.content[0].text.strip() 
            try:
                if "```json" in response_text: # 如果返回的是```json包装的，提取
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.rfind("```")
                    response_text = response_text[start:end].strip()
                
                data = json.loads(response_text) # 转换成json格式
                
                # 验证数据格式
                valid_samples = []
                for i,sample in enumerate(data):
                    if self._validate_sample(sample):
                        valid_samples.append(sample)
                        print(f"sample {i}\n{sample}")
                    else:
                        print(f"警告：跳过无效样本: {sample}")
                
                print(f"成功生成 {len(valid_samples)} 个有效样本")
                return valid_samples
                
            except json.JSONDecodeError as e:
                print(f"JSON解析错误: {e}")
                return []
                
        except Exception as e:
            print(f"API调用失败: {e}")
            return []
    
    def _validate_sample(self, sample: Dict[str, Any]) -> bool:
        """验证样本格式是否正确"""
        required_keys = ["query", "best_model_category"]
        
        for key in required_keys: # 检查必需字段
            if key not in sample:
                return False
        
        if sample["best_model_category"] not in ["large", "medium", "small"]: # 检查类别
            return False
    
        if not sample["query"] or not isinstance(sample["query"], str): # 检查查询是否为空
            return False
        
        if "context" in sample and sample["context"] is not None: # 检查上下文格式
            if not isinstance(sample["context"], dict):
                return False
        
        return True
    
    def generate_dataset(self, total_samples: int = 1000, batch_size: int = 20) -> List[Dict[str, Any]]:
        """生成完整数据集"""
        all_samples = []
        batches_needed = (total_samples + batch_size - 1) // batch_size
        
        print(f"开始生成 {total_samples} 个样本，分 {batches_needed} 批...")
        
        categories = ["large", "medium", "small", None]  # 生成类别平衡的数据，None表示混合
        
        for i in range(batches_needed):
            category_focus = categories[i % len(categories)]
        
            remaining = total_samples - len(all_samples) 
            current_batch_size = min(batch_size, remaining)
            if current_batch_size <= 0:
                break
            print(f"\n第 {i+1}/{batches_needed} 批 (重点: {category_focus or '混合'})")
            
            batch_samples = self.generate_batch(current_batch_size, category_focus) # 分批生成数据
    
            if batch_samples:
                all_samples.extend(batch_samples)
                print(f"累计样本数: {len(all_samples)}")
            else:
                print("这一批生成失败，继续下一批...")
            
            if i < batches_needed - 1:
                print("等待2秒...")
                time.sleep(2)
        
        print(f"\n总共生成 {len(all_samples)} 个样本")
        return all_samples
    

    def analyze_dataset(self, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析数据集统计信息"""
        if not dataset:
            return {"error": "数据集为空"}
        
        stats = {
            "total_samples": len(dataset),
            "category_distribution": {},
            "context_coverage": 0,
            "avg_query_length": 0,
            "query_length_distribution": {},
            "domain_distribution": {},
        }
        
        for sample in dataset: # 统计类别分布
            category = sample["best_model_category"]
            stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1
        
        with_context = sum(1 for sample in dataset if sample.get("context") is not None) # 统计上下文覆盖率
        stats["context_coverage"] = with_context / len(dataset) 
        
        query_lengths = [len(sample["query"]) for sample in dataset] # 统计查询长度
        stats["avg_query_length"] = sum(query_lengths) / len(query_lengths)
        
        for length in query_lengths: # 查询长度分布
            if length < 20:
                bucket = "短查询(<20字符)"
            elif length < 100:
                bucket = "中等查询(20-100字符)"
            else:
                bucket = "长查询(>100字符)"
            stats["query_length_distribution"][bucket] = stats["query_length_distribution"].get(bucket, 0) + 1
        
        for sample in dataset: # 基于上下文统计领域分布
            context = sample.get("context")
            if context and "domain_preference" in context:
                domain = context["domain_preference"]
                stats["domain_distribution"][domain] = stats["domain_distribution"].get(domain, 0) + 1
        
        return stats

        
    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """保存数据集为json文件"""
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到: {file_path}")
    

    def generate_and_save(self, total_samples: int = 1000, batch_size: int = 20):
        """生成并保存完整的训练和测试数据集"""
        
        print("开始生成训练数据...")
        dataset = self.generate_dataset(total_samples=total_samples, batch_size=batch_size) 
        self.save_dataset(dataset, "dataset.json") 

        if not dataset:
            print("数据生成失败")
            return

        stats = self.analyze_dataset(dataset) # 分析数据集
        print("\n数据集统计:")
        print(f"总样本数: {stats['total_samples']}")
        print(f"类别分布: {stats['category_distribution']}")
        print(f"上下文覆盖率: {stats['context_coverage']:.2%}")
        print(f"平均查询长度: {stats['avg_query_length']:.1f}字符")
        print(f"查询长度分布: {stats['query_length_distribution']}")
        if stats['domain_distribution']:
            print(f"领域分布: {stats['domain_distribution']}")
        
       
def main():
    """主函数"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not found")
    
    try:
        generator = ClaudeDataGenerator(api_key=api_key)
        total_samples = 1000 # 生成数据总量
        batch_size = 20 # 批大小
        generator.generate_and_save(total_samples=total_samples, batch_size=batch_size) # 生成并保存数据
        
    except Exception as e:
        print(f"生成失败: {e}")


if __name__ == "__main__":
    main()