import re
import math
from typing import Dict, Any, List, Optional
from collections import Counter
import numpy as np


class FeatureExtractor:
    def __init__(self):
        self.domain_keywords = { 
            "technology": [
                # Programming & Development
                "python", "javascript", "code", "coding", "script", "program", "function", "algorithm", 
                "programming", "software", "development", "neural", "network", "machine learning", 
                "AI", "artificial intelligence", "blockchain", "API", "database", "microservice",
                "architecture", "cybersecurity", "quantum computing", "encryption", "HTML", "SEO",
                # Tech concepts
                "scalable", "framework", "implementation", "optimization", "infrastructure", 
                "data structures", "binary search", "recommendation system", "sentiment analysis",
                "IoT", "predictive maintenance", "Industry 4.0", "cryptographic", "semiconductor",
                "supply chain", "tech startup", "fintech", "SaaS", "healthcare solutions"
            ],
            "business": [
                # Business strategy & operations
                "business plan", "marketing strategy", "market analysis", "financial projections",
                "competitive landscape", "revenue", "ROI", "KPI", "budget", "investment",
                "startup", "company", "corporation", "strategy", "management", "operations",
                # Marketing & sales
                "marketing", "campaign", "branding", "advertising", "social media", "influencer",
                "digital marketing", "content calendar", "target audience", "positioning",
                "Gen Z", "millennials", "consumer", "customer", "client", "partnership",
                # Finance & economics
                "economic", "financial", "macroeconomic", "inflation", "currency", "investment thesis",
                "emerging markets", "foreign direct investment", "central bank", "monetary policy",
                "CFO", "risk assessment", "ESG", "sustainable fashion", "luxury brand"
            ],
            "science": [
                "research", "analysis", "study", "experiment", "methodology", "data", "statistical",
                "environmental", "climate", "ecosystem", "biodiversity", "carbon footprint",
                "sustainability", "renewable energy", "hydroelectric", "offshore wind farm",
                "lithium mining", "environmental impact", "assessment", "mitigation", "EPA guidelines",
                "ocean warming", "climate change", "scientific", "marine", "ecological"
            ],
            "education": [
                "curriculum", "teaching", "learning", "student", "undergraduate", "graduate",
                "course", "syllabus", "assessment", "education", "lesson plan", "academic",
                "university", "school", "training", "knowledge", "study guide", "pedagogy",
                "learning objectives", "educational", "instruction", "classroom", "professor"
            ],
            "health": [
                "health", "medical", "healthcare", "clinical", "patient", "treatment", "therapy",
                "diagnosis", "symptoms", "medicine", "pharmaceutical", "drug", "vaccine", "mRNA",
                "immunotherapy", "cancer", "melanoma", "diabetes", "flu", "cold", "pain",
                "vitamin", "deficiency", "HIPAA", "telehealth", "medical devices", "research",
                "neurological", "depression", "placebo", "clinical trials"
            ],
            "travel": [
                "travel", "tourism", "tourist", "attractions", "destinations", "itinerary",
                "vacation", "trip", "visit", "places", "Barcelona", "Paris", "Japan", "Tokyo",
                "hotel", "flight", "restaurant", "sightseeing", "guide", "journey"
            ],
            "entertainment": [
                "movie", "film", "book", "story", "entertainment", "sports", "Super Bowl",
                "World Cup", "Lakers", "game", "sci-fi", "Harry Potter", "Romeo and Juliet",
                "literature", "plot", "character", "music", "celebrity", "show"
            ],
            "lifestyle": [
                "lifestyle", "food", "cooking", "recipe", "pasta", "cookies", "Italian",
                "chocolate chip", "fashion", "shopping", "home", "living", "sustainable living",
                "fitness", "exercise", "workout", "wellness", "beauty", "personal care"
            ],
            "general": [
                "hello", "hi", "thanks", "thank you", "weather", "time", "today", "tomorrow",
                "help", "question", "answer", "basic", "simple", "general", "common", "everyday"
            ]
        }

        self.task_keywords = {
            "qa": [
                "what", "how", "why", "where", "who", "when", "which", "is", "are", "does",
                "explain", "define", "difference", "compare", "main", "best", "good",
                "symptoms", "causes", "features", "capital", "meaning", "concept"
            ],
            "generation": [
                "write", "create", "develop", "design", "generate", "draft", "build", "make",
                "compose", "produce", "construct", "formulate", "prepare", "plan", "strategy"
            ],
            "analysis": [
                "analyze", "analysis", "evaluate", "assess", "compare", "contrast", "review",
                "examine", "study", "investigate", "research", "survey", "impact", "implications",
                "comprehensive", "detailed", "in-depth"
            ],
            "translation": [
                "translate", "translation", "convert", "transform", "language", "Spanish",
                "paragraph", "sentence", "text"
            ],
            "coding": [
                "python", "script", "function", "program", "code", "algorithm", "implement",
                "programming", "software", "development", "debugging", "neural network",
                "machine learning", "data structures", "binary search", "microservice"
            ],
            "summarization": [
                "summarize", "summary", "brief", "overview", "outline", "key points",
                "main points", "plot", "extract", "keywords"
            ],
            "chat": [
                "hello", "hi", "hey", "thanks", "thank you", "chat", "talk", "discuss",
                "conversation", "feeling", "how are you"
            ]
        }

        self.complexity_indicators = {
            "high": [
                "comprehensive", "detailed", "advanced", "complex", "sophisticated", "in-depth",
                "strategic", "professional", "expert", "research", "thorough", "extensive",
                "multi-faceted", "holistic", "systematic", "rigorous", "technical", "specialized",
                # Specific complex topics
                "quantum computing", "neural network", "microservice architecture", "environmental impact assessment",
                "business plan", "financial projections", "risk assessment", "literature review",
                "curriculum design", "clinical trials", "investment thesis", "cybersecurity strategy"
            ],
            "medium": [
                "moderate", "standard", "regular", "typical", "normal", "intermediate",
                "blog post", "short", "guide", "tips", "basics", "introduction", "overview",
                "explain", "understand", "learn", "tutorial", "comparison"
            ],
            "low": [
                "simple", "basic", "easy", "quick", "brief", "short", "direct", "straightforward",
                "define", "what is", "hello", "thanks", "weather", "time", "capital", "translate",
                "extract", "check", "positive", "negative", "keywords"
            ]
        }

        self.urgency_indicators = {
            "high": [
                "urgent", "immediately", "asap", "quickly", "fast", "rapid", "emergency",
                "critical", "priority", "deadline", "time-sensitive"
            ],
            "medium": [
                "soon", "regular", "standard", "normal", "typical", "moderate"
            ],
            "low": [
                "whenever", "no rush", "when possible", "eventually", "casual", "leisure"
            ]
        }

    
    def extract_features(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """从查询中提取特征"""
        features = {}
        
        features.update(self._extract_text_features(query)) # 基础文本特征
        features.update(self._extract_semantic_features(query)) # 语义特征
        features.update(self._extract_domain_features(query)) # 领域特征
        features.update(self._extract_task_features(query)) # 任务类型特征        
        features.update(self._extract_complexity_features(query)) # 复杂度特征
        features.update(self._extract_urgency_features(query)) # 紧急度特征
        if context:  # 上下文特征
            features.update(self._extract_context_features(context))
        
        return features
    
    
    def _extract_text_features(self, query: str) -> Dict[str, Any]:
        """提取基础文本特征"""
        words = re.findall(r'[\u4e00-\u9fff]|[a-zA-Z]+|\d+', query) # 分离中文字符、英文单词、数字

        sentences = re.split(r'[。！？；]', query) # 中文句子分割
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return {
            "text_length": len(query),
            "word_count": len(words),
            "sentence_count": max(len(sentences), 1),
            "avg_word_length": np.mean([len(word) for word in words]) if words else 0,
            "has_question_mark": "？" in query or "?" in query,
            "has_exclamation": "！" in query or "!" in query,
            "punctuation_count": len(re.findall(r'[,.!?;:，。！？；：]', query)),
            "digit_count": len(re.findall(r'\d', query)),
            "uppercase_ratio": sum(1 for c in query if c.isupper()) / len(query) if query else 0
        }
    

    def _extract_semantic_features(self, query: str) -> Dict[str, Any]:
        """提取语义特征"""
        words = query.split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0 # 词汇丰富度
        
        # 检测情感倾向
        positive_words = [
            "good", "great", "excellent", "amazing", "wonderful", "fantastic", "perfect",
            "love", "like", "enjoy", "appreciate", "satisfied", "happy", "pleased",
            "awesome", "brilliant", "outstanding", "impressive", "helpful", "useful"
        ]

        negative_words = [
            "bad", "terrible", "awful", "horrible", "disappointing", "frustrating", "difficult",
            "hate", "dislike", "annoying", "problematic", "issue", "problem", "wrong",
            "failed", "broken", "useless", "poor", "unsatisfied", "unhappy", "sad"
        ]
        
        positive_count = sum(1 for word in positive_words if word in query)
        negative_count = sum(1 for word in negative_words if word in query)
        sentiment_score = (positive_count - negative_count) / len(words) if words else 0
        
        return {
            "lexical_diversity": lexical_diversity,
            "sentiment_score": sentiment_score,
            "semantic_complexity": self._calculate_semantic_complexity(query)
        }
    

    def _extract_domain_features(self, query: str) -> Dict[str, Any]:
        """提取领域特征"""
        domain_scores = {}
        query_lower = query.lower()
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            domain_scores[f"domain_{domain}"] = score
        
        best_domain = max(domain_scores.items(), key=lambda x: x[1]) 
        primary_domain = best_domain[0].replace("domain_", "") if best_domain[1] > 0 else "general"
        
        return {
            **domain_scores,
            "primary_domain": primary_domain,
            "domain_confidence": best_domain[1] / len(query.split()) if query.split() else 0
        }
    

    def _extract_task_features(self, query: str) -> Dict[str, Any]:
        """提取任务类型特征"""
        task_scores = {}
        query_lower = query.lower()
        
        for task_type, keywords in self.task_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            task_scores[f"task_{task_type}"] = score
        
        best_task = max(task_scores.items(), key=lambda x: x[1])
        primary_task = best_task[0].replace("task_", "") if best_task[1] > 0 else "general"
        
        return {
            **task_scores,
            "primary_task": primary_task,
            "task_confidence": best_task[1] / len(query.split()) if query.split() else 0
        }
    

    def _extract_complexity_features(self, query: str) -> Dict[str, Any]:
        """提取复杂度特征"""
        query_lower = query.lower()
        
        # 基于关键词的复杂度评估
        complexity_scores = {}
        for level, keywords in self.complexity_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            complexity_scores[f"complexity_{level}"] = score
        
        # 文本长度
        text_length = len(query)
        if text_length < 20:
            length_complexity = "low"
        elif text_length < 100:
            length_complexity = "medium"
        else:
            length_complexity = "high"
        
        # 词汇复杂度
        words = query.split()
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        if avg_word_length < 3:
            vocab_complexity = "low"
        elif avg_word_length < 5:
            vocab_complexity = "medium"
        else:
            vocab_complexity = "high"
        
        # 综合复杂度
        overall_complexity = self._calculate_overall_complexity(complexity_scores, length_complexity, vocab_complexity)
        
        return {
            **complexity_scores,
            "length_complexity": length_complexity,
            "vocab_complexity": vocab_complexity,
            "overall_complexity": overall_complexity,
            "complexity_score": self._complexity_to_score(overall_complexity)
        }
    

    def _extract_urgency_features(self, query: str) -> Dict[str, Any]:
        """提取紧急度特征"""
        query_lower = query.lower()
        
        urgency_scores = {}
        for level, keywords in self.urgency_indicators.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            urgency_scores[f"urgency_{level}"] = score
        
        # 标点符号
        exclamation_count = query.count("!") + query.count("！")
        question_count = query.count("?") + query.count("？")
        
        if exclamation_count > 1:
            punctuation_urgency = "high"
        elif exclamation_count > 0 or question_count > 1:
            punctuation_urgency = "medium"
        else:
            punctuation_urgency = "low"
        
        # 综合紧急度
        overall_urgency = self._calculate_overall_urgency(urgency_scores, punctuation_urgency)
        
        return {
            **urgency_scores,
            "punctuation_urgency": punctuation_urgency,
            "overall_urgency": overall_urgency,
            "urgency_score": self._urgency_to_score(overall_urgency)
        }
    

    def _extract_context_features(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """提取上下文特征"""
        return {
            "has_context": True,
            "context_length": len(str(context)),
            "user_history_length": len(context.get("user_history", [])),
            "session_duration": context.get("session_duration", 0),
            "previous_model_used": context.get("previous_model", "none")
        }
    

    def _calculate_semantic_complexity(self, query: str) -> float:
        """计算语义复杂度"""
        words = query.split()
        if not words:
            return 0.0
        
        # 基于词汇多样性和平均词长
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words)
        avg_word_length = np.mean([len(word) for word in words])
        
        # 句子结构复杂度
        sentence_complexity = len(re.findall(r'[,;:]', query)) / len(words)
        
        return (lexical_diversity + avg_word_length / 10 + sentence_complexity) / 3
    

    def _calculate_overall_complexity(self, complexity_scores: Dict[str, int], length_complexity: str, vocab_complexity: str) -> str:
        """计算总体复杂度"""
        high_score = complexity_scores.get("complexity_high", 0)
        medium_score = complexity_scores.get("complexity_medium", 0)
        low_score = complexity_scores.get("complexity_low", 0)
        
        # 权重计算
        complexity_levels = ["low", "medium", "high"]
        scores = [low_score, medium_score, high_score]
        
        # 长度和词汇复杂度
        length_weight = complexity_levels.index(length_complexity)
        vocab_weight = complexity_levels.index(vocab_complexity)
        
        total_score = sum(scores) + length_weight + vocab_weight # 综合
        
        if total_score <= 2:
            return "low"
        elif total_score <= 4:
            return "medium"
        else:
            return "high"
    

    def _calculate_overall_urgency(self, urgency_scores: Dict[str, int], punctuation_urgency: str) -> str:
        """计算总体紧急度"""
        high_score = urgency_scores.get("urgency_high", 0)
        medium_score = urgency_scores.get("urgency_medium", 0)
        low_score = urgency_scores.get("urgency_low", 0)
        
        urgency_levels = ["low", "medium", "high"]
        punct_weight = urgency_levels.index(punctuation_urgency)
        
        if high_score > 0 or punct_weight == 2:
            return "high"
        elif medium_score > 0 or punct_weight == 1:
            return "medium"
        else:
            return "low"
    
    
    def _complexity_to_score(self, complexity: str) -> float:
        """复杂度转换为数值分数"""
        mapping = {"low": 0.3, "medium": 0.6, "high": 0.9}
        return mapping.get(complexity, 0.5)
    

    def _urgency_to_score(self, urgency: str) -> float:
        """紧急度转换为数值分数"""
        mapping = {"low": 0.2, "medium": 0.5, "high": 0.8}
        return mapping.get(urgency, 0.3)
