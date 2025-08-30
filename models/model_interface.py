import os
from abc import ABC, abstractmethod
from typing import Any, Dict
import time
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
import openai
import anthropic
import requests
            


class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.provider = config.get("provider", "unknown") 
        self.model_name = config.get("model_name", "unknown")
        self.max_tokens = config.get("max_tokens", 4096)
        self.max_context_tokens = config.get("max_context_tokens", 8192)
        self.cost_per_token = config.get("cost_per_token", 0.0)
        self.latency_class = config.get("latency_class", "medium")  # fast, medium, slow
        
        # 性能统计
        self.total_requests = 0
        self.total_tokens = 0
        self.total_latency = 0.0
        self.error_count = 0
    
    @abstractmethod
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        调用模型生成响应

        Args:
            query: 用户查询
            **kwargs: 其他参数

        Returns:
            包含模型响应的字典
        """
        pass
    

    @abstractmethod 
    def is_available(self) -> bool:
        """检查模型是否可用"""
        pass


    def get_model_info(self) -> Dict[str, Any]:
        """获取模型基本信息"""
        return {
            "provider": self.provider,
            "model_name": self.model_name,
            "max_tokens": self.max_tokens,
            "max_context_tokens": self.max_context_tokens,
            "cost_per_token": self.cost_per_token,
            "latency_class": self.latency_class
        }
    

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        avg_latency = self.total_latency / max(self.total_requests, 1)
        error_rate = self.error_count / max(self.total_requests, 1)
        
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "average_latency": avg_latency,
            "error_rate": error_rate,
            "availability": self.is_available()
        }
    

    def _record_request(self, tokens_used: int, latency: float, success: bool = True):
        """记录请求统计信息"""
        self.total_requests += 1
        self.total_tokens += tokens_used
        self.total_latency += latency
        if not success:
            self.error_count += 1


class OpenAIModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "gpt-3.5-turbo")

        self.api_key = os.getenv("OPENAI_API_KEY","")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found")
    
        try:
            self.client = openai.OpenAI(api_key=self.api_key)
        except Exception as e:
            print(f"OpenAI客户端初始化失败: {e}")
            self.client = None


    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if self.client is None:
                raise Exception("OpenAI客户端未正确初始化")
            messages = [{"role": "user", "content": query}]
            
            system_prompt = kwargs.get("system_prompt") # 如果有提供system_prompt，则添加到messages中
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                frequency_penalty=kwargs.get("frequency_penalty", 0.0),
                presence_penalty=kwargs.get("presence_penalty", 0.0)
            )
            
            response_text = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
            latency = time.time() - start_time
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "latency": latency,
                "success": True,
                "provider": "openai",
                "cost": self._calculate_cost(tokens_used)
            }

        except Exception as e:
            latency = time.time() - start_time
            self._record_request(0, latency, False)
            
            return {
                "response": f"OpenAI API Error: {str(e)}",
                "provider": "openai",
                "model_name": self.model_name,
                "tokens_used": 0,
                "latency": latency,
                "success": False,
                "error": str(e),
            }
    

    def _calculate_cost(self, tokens: int) -> float:
        return tokens * self.cost_per_token / 1000  # 按 1K tokens计算
    
    def is_available(self) -> bool:
        return self.api_key is not None and self.client is not None



class AnthropicModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY","")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not found")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            messages = [{"role": "user", "content": query}]
            system_prompt = kwargs.get("system_prompt", "You are a helpful AI assistant.")
            
            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=kwargs.get("max_tokens", min(self.max_tokens, 4096)),
                messages=messages,
                system=system_prompt,
                temperature=kwargs.get("temperature", 0.7),
                top_p=kwargs.get("top_p", 1.0),
                top_k=kwargs.get("top_k", 5),
                timeout=60.0  
            )
            
            response_text = response.content[0].text
            tokens_used = response.usage.input_tokens + response.usage.output_tokens
            
            latency = time.time() - start_time
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "latency": latency,
                "success": True,
                "provider": "anthropic",
                "cost": self._calculate_cost(tokens_used)
            }

                        
        except Exception as e:
            latency = time.time() - start_time
            self._record_request(0, latency, False)
            
            return {
                "response": f"Anthropic API调用错误: {str(e)}",
                "model_name": self.model_name,
                "tokens_used": 0,
                "latency": latency,
                "success": False,
                "error": str(e),
                "provider": "anthropic"
            }


    def _calculate_cost(self, tokens: int) -> float:
        """计算API调用成本"""
        return tokens * self.cost_per_token / 1000  # 成本通常按1K tokens计算


    def is_available(self) -> bool:
        """检查Anthropic API是否可用"""
        return self.api_key is not None


class DeepSeekModel(BaseModel):
    """DeepSeek模型实现"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found")
        
        self.base_url = "https://api.deepseek.com/v1"
        
        
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        """调用DeepSeek模型"""
        start_time = time.time()
        try:
            messages = [{"role": "user", "content": query}]
            system_prompt = kwargs.get("system_prompt", "You are a helpful AI assistant.")
            if system_prompt:
                messages.insert(0, {"role": "system", "content": system_prompt})
            
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", min(self.max_tokens, 4096)),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 1.0),
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=60.0
            )
            
            if response.status_code != 200:
                raise Exception(f"DeepSeek API Request Error: {response.status_code} - {response.text}")
            
            response_data = response.json()
            
            if "error" in response_data:
                raise Exception(f"DeepSeek API Error: {response_data['error']['message']}")
            
            response_text = response_data["choices"][0]["message"]["content"]
            usage = response_data.get("usage", {})
            tokens_used = usage.get("total_tokens", 0)
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            
            latency = time.time() - start_time
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "latency": latency,
                "success": True,
                "provider": "deepseek",
                "cost": self._calculate_cost(tokens_used)
            }
            
        except Exception as e:
            latency = time.time() - start_time
            self._record_request(0, latency, False)
            
            return {
                "response": f"DeepSeek API Error: {str(e)}",
                "model_name": self.model_name,
                "tokens_used": 0,
                "latency": latency,
                "success": False,
                "error": str(e),
                "provider": "deepseek"
            }


    def _calculate_cost(self, tokens: int) -> float:
        """计算API调用成本"""
        return tokens * self.cost_per_token / 1000  # DeepSeek价格很便宜


    def is_available(self) -> bool:
        """检查DeepSeek API是否可用"""
        return self.api_key is not None


class HuggingFaceModel(BaseModel):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.model_name = config.get("model_name", "")
        self.local_path = config.get("local_path", "")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        self.use_local = config.get("use_local", True)
        
        self.model_path = self.local_path if os.path.exists(self.local_path) else self.model_name # use local path if exists, otherwise use model name
        
    def invoke(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            if "codegen" in self.model_name: # code-generation
                return self._call_codegen_model(query, **kwargs)
            elif "sentiment" in self.model_name: # sentiment-analysis
                return self._call_sentiment_model(query, **kwargs)
            elif "roberta" in self.model_name: # qa
                return self._call_qa_model(query, **kwargs)
            elif "all-MiniLM" in self.model_name: # text-embedding
                return self._call_text_embedding_model(query, **kwargs)
            
            return None
                
        except Exception as e:
            latency = time.time() - start_time
            self._record_request(0, latency, False)
            
            return {
                "response": f"HuggingFace Model Error: {str(e)}",
                "model_name": self.model_name,
                "tokens_used": 0,
                "latency": latency,
                "success": False,
                "error": str(e),
                "provider": "huggingface"
            }
    
    
    def _call_codegen_model(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            model = AutoModelForCausalLM.from_pretrained(self.model_path)

            inputs = tokenizer.encode(query, return_tensors="pt") # 编码输入
            
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_length=kwargs.get("max_length", 200),
                    temperature=kwargs.get("temperature", 0.7),
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response_text = generated_code[len(query):].strip() # 解析输出
            
            latency = time.time() - start_time
            tokens_used = len(outputs[0])
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": f"```python\n{response_text}\n```",
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "latency": latency,
                "success": True,
                "provider": "huggingface",
                "task_type": "code_generation"
            }
            
        except Exception as e:
            print(f"代码生成失败: {str(e)}")
            return None

    
    def _call_sentiment_model(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            classifier = pipeline("sentiment-analysis", model=self.model_path)
            result = classifier(query)            
            
            label = result[0]['label']
            score = result[0]['score']
            
            # 格式化标签显示
            label_display = label.lower()
            
            #response_text = f"Sentiment Analysis Result:\n"
            #response_text += f"Text: \"{query}\"\n"
            response_text = f"Sentiment: {label_display}\n"
            response_text += f"Confidence: {score:.3f} ({score*100:.1f}%)"
            
            latency = time.time() - start_time
            tokens_used = len(query.split()) + 10
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "latency": latency,
                "success": True,
                "provider": "huggingface",
                "task_type": "sentiment_analysis",
                "sentiment_label": label,
                "confidence": score
            }

        except Exception as e:
            print(f"情感分析失败: {str(e)}")
            return None
    

    def _call_qa_model(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            context = kwargs.get("context", "This is a general question without context, try tto use your internal knowledge to solve it.")
            qa_pipeline = pipeline("question-answering", model=self.model_path)
            
            result = qa_pipeline(question=query, context=context)
            response_text = f"answer: {result['answer']}\nconfidence: {result['score']:.3f}"
            
            latency = time.time() - start_time
            tokens_used = len(query.split()) + len(result['answer'].split())
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "latency": latency,
                "success": True,
                "provider": "huggingface",
                "task_type": "question_answering",
                "answer": result['answer'],
                "confidence": result['score']
            }
            
        except Exception as e:
            print(f"答案生成失败: {str(e)}")
            return None
    
    
    def _call_text_embedding_model(self, query: str, **kwargs) -> Dict[str, Any]:
        start_time = time.time()
        try:
            model = SentenceTransformer(self.model_path)
            embedding = model.encode(query) # 生成文本嵌入
            
            compare_text = kwargs.get("compare_text") # 得到比较文本
            if compare_text:
                # 计算相似度
                compare_embedding = model.encode(compare_text)
                similarity = np.dot(embedding, compare_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(compare_embedding)
                )
                
                response_text = f"Text Similarity Analysis:\n"
                response_text += f"Text 1: {query}\n"
                response_text += f"Text 2: {compare_text}\n"
                response_text += f"Similarity Score: {similarity:.4f}\n"
                response_text += f"Embedding Dimension: {len(embedding)}"
                
                task_type = "text_similarity"
                extra_data = {
                    "similarity_score": float(similarity),
                    "compare_text": compare_text
                }
            else:
                # 生成嵌入向量信息
                response_text = f"Text Embedding Analysis:\n"
                response_text += f"Input Text: {query}\n"
                response_text += f"Embedding Dimension: {len(embedding)}\n"
                response_text += f"Embedding Preview: [{', '.join([f'{x:.4f}' for x in embedding[:5]])}...]\n"
                response_text += f"Text Length: {len(query)} characters"
                
                task_type = "text_embedding"
                extra_data = {
                    "embedding_dim": len(embedding),
                    "text_length": len(query)
                }
            
            latency = time.time() - start_time
            tokens_used = len(query.split()) + 10
            self._record_request(tokens_used, latency, True)
            
            return {
                "response": response_text,
                "model_name": self.model_name,
                "tokens_used": tokens_used,
                "latency": latency,
                "success": True,
                "provider": "huggingface",
                "task_type": task_type,
                "embedding": embedding.tolist(), 
                **extra_data
            }
            
        except Exception as e:
            print(f"文本嵌入失败: {str(e)}")
            return None

    def is_available(self) -> bool:
        return True
