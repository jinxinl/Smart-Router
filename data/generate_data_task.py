import json
import time
from typing import List, Dict, Any
import os
from uu import Error
import anthropic


class ClaudeTaskDataGenerator:
    def __init__(self, api_key: str = None):
        self.data_dir = os.path.dirname(os.path.abspath(__file__))        
        if api_key is None:
            raise Error("ANTHROPIC_API_KEY not found")
    
        self.client = anthropic.Anthropic(api_key=api_key)
        self.generation_prompt = self._create_generation_prompt()
    

    def _create_generation_prompt(self) -> str:
        return """
You are a training data generation expert for an intelligent AI task type classification system. I need you to generate training data for training an XGBoost classifier to decide which type of AI task should handle user queries.

## Target Task Type Descriptions:

**code-generation** - Code Generation Tasks:
- Simple function writing, basic scripts, code snippets
- Code completion, simple debugging, basic refactoring  
- Algorithm implementation for common problems
- Configuration file generation, simple API usage examples
- Focus on: Python, JavaScript, basic data structures, simple logic
- Example: "Write a Python function to reverse a string"

**sentiment-analysis** - Sentiment Analysis Tasks:
- Analyze text sentiment (positive, negative, neutral)
- Social media post sentiment, product review analysis
- Customer feedback classification, simple emotion detection
- Short text sentiment evaluation (tweets, comments, reviews)
- Focus on: Clear emotional language, straightforward sentiment expressions
- Example: "What is the sentiment of this tweet: 'I love this new phone!'"

**qa** - Question Answering Tasks:
- Factual questions with clear answers
- Reading comprehension based on provided context
- Simple explanations of concepts, definitions
- Questions that can be answered from given passages
- Focus on: Context-based QA, extractive answers, factual queries
- Example: "Based on this paragraph, when was the company founded?"

**embedding** - Text Embedding Tasks:
- Text similarity comparison, semantic search queries
- Document clustering, text classification preprocessing
- Finding similar sentences or phrases
- Text vectorization for downstream tasks
- Focus on: Similarity tasks, semantic search, text comparison
- Example: "Find the most similar sentence to: 'The weather is nice today'"

## Output Format Requirements:

Please generate data in JSON format, wrap the returned data in json blocks, each sample should include:
```json
{{
  "query": "user query text",
  "task_type": "code-generation/sentiment-analysis/qa/embedding",
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
2. Keep queries within small model capabilities - simple functions, clear sentiments, extractive QA, basic similarity tasks
3. Query lengths should vary from short phrases to medium descriptions, avoid overly complex requests
4. Domains should be diverse: programming, business, science, education, social_media, technology, general, entertainment
5. Ensure accurate labels and clear task type distinctions for effective classifier training
6. Focus on practical, real-world use cases that match the specified model limitations

Please generate {batch_size} training samples with balanced task type distribution. Return ONLY the JSON format array, no additional explanations.
"""


    def generate_batch(self, batch_size: int = 20, task_focus: str = None) -> List[Dict[str, Any]]:
        """生成一批训练数据"""
        prompt = self.generation_prompt.format(batch_size=batch_size)
        if task_focus:
            prompt += f"\n\nSpecial Requirement: For this batch of data, please focus on generating samples for the {task_focus} task type (accounting for more than 60% of the samples)."
        
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
                if "```json" in response_text:
                    start = response_text.find("```json") + 7
                    end = response_text.find("```", start)
                    response_text = response_text[start:end].strip()
                elif "```" in response_text:
                    start = response_text.find("```") + 3
                    end = response_text.rfind("```")
                    response_text = response_text[start:end].strip()
                
                data = json.loads(response_text)
                
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
        required_keys = ["query", "task_type"]
        
        for key in required_keys: # 检查必需字段
            if key not in sample:
                return False
        
        if sample["task_type"] not in ["code-generation", "sentiment-analysis", "qa", "embedding"]: # 检查任务类型是否有效
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
        
        task_types = ["code-generation", "sentiment-analysis", "qa", "embedding", None]  # None表示混合
        
        for i in range(batches_needed):
            task_focus = task_types[i % len(task_types)]
        
            remaining = total_samples - len(all_samples)
            current_batch_size = min(batch_size, remaining)    
            if current_batch_size <= 0:
                break
            print(f"\n第 {i+1}/{batches_needed} 批 (重点: {task_focus or '混合'})")
            
            batch_samples = self.generate_batch(current_batch_size, task_focus) # 分批生成数据
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
            "task_type_distribution": {},
            "context_coverage": 0,
            "avg_query_length": 0,
            "query_length_distribution": {},
            "domain_distribution": {},
            "expertise_distribution": {},
            "complexity_distribution": {}
        }
        
        for sample in dataset: # 统计任务类型分布
            task_type = sample["task_type"]
            stats["task_type_distribution"][task_type] = stats["task_type_distribution"].get(task_type, 0) + 1
        
        with_context = sum(1 for sample in dataset if sample.get("context") is not None) # 统计上下文覆盖率
        stats["context_coverage"] = with_context / len(dataset) 
        
        query_lengths = [len(sample["query"]) for sample in dataset] # 统计查询长度
        stats["avg_query_length"] = sum(query_lengths) / len(query_lengths)
        
        for length in query_lengths: # 查询长度分布
            if length < 30:
                bucket = "短查询(<30字符)"
            elif length < 100:
                bucket = "中等查询(30-100字符)"
            else:
                bucket = "长查询(>100字符)"
            stats["query_length_distribution"][bucket] = stats["query_length_distribution"].get(bucket, 0) + 1
        
        for sample in dataset: # 基于上下文统计领域分布
            context = sample.get("context")
            if context and "domain_preference" in context:
                domain = context["domain_preference"]
                stats["domain_distribution"][domain] = stats["domain_distribution"].get(domain, 0) + 1
        
        for sample in dataset: # 统计专业程度分布
            context = sample.get("context")
            if context and "user_type" in context:
                expertise = context["user_type"]
                stats["expertise_distribution"][expertise] = stats["expertise_distribution"].get(expertise, 0) + 1
    
        return stats

    

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """保存数据集为json文件"""
        file_path = os.path.join(self.data_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"数据集已保存到: {file_path}")
    

    def generate_and_save(self, total_samples: int = 1000, batch_size: int = 20):
        """生成并保存完整的训练和测试数据集"""
        
        print("开始生成任务分类训练数据...")
        dataset = self.generate_dataset(total_samples=total_samples, batch_size=batch_size)
        self.save_dataset(dataset, "task_dataset.json")

        if not dataset:
            print("数据生成失败")
            return
        
        stats = self.analyze_dataset(dataset)
        print("\n数据集统计:")
        print(f"总样本数: {stats['total_samples']}")
        print(f"任务类型分布: {stats['task_type_distribution']}")
        print(f"上下文覆盖率: {stats['context_coverage']:.2%}")
        print(f"平均查询长度: {stats['avg_query_length']:.1f}字符")
        print(f"查询长度分布: {stats['query_length_distribution']}")
        if stats['domain_distribution']:
            print(f"领域分布: {stats['domain_distribution']}")
        if stats['expertise_distribution']:
            print(f"专业程度分布: {stats['expertise_distribution']}")
        if stats['complexity_distribution']:
            print(f"复杂度分布: {stats['complexity_distribution']}")
        
       
def main():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not found")
    
    try:
        generator = ClaudeTaskDataGenerator(api_key=api_key)    
        total_samples = 1000
        batch_size = 20
        generator.generate_and_save(total_samples=total_samples, batch_size=batch_size)
        
    except Exception as e:
        print(f"生成失败: {e}")


if __name__ == "__main__":
    main()
