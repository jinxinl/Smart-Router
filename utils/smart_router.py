import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib

from models.model_pool import ModelPool
from utils.feature_extractor import FeatureExtractor
from sklearn.preprocessing import LabelEncoder


class SmartRouter:
    def __init__(self, model_pool: ModelPool):
        self.model_pool = model_pool
        self.feature_extractor = FeatureExtractor()
        
        # 第一层路由模型：模型大小分类 (large/medium/small)
        self.xgb_model = None
        self.feature_columns = []
        self.label_encoder = LabelEncoder()
        
        # 第二层路由模型：任务类型分类 (仅用于small模型)
        self.xgb_task_model = None
        self.task_feature_columns = []
        self.task_label_encoder = LabelEncoder()

        # 模型路径
        self.model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
        self.xgb_model_path = os.path.join(self.model_dir, "xgboost_router.pkl")
        self.feature_columns_path = os.path.join(self.model_dir, "feature_columns.pkl")
        self.label_encoder_path = os.path.join(self.model_dir, "label_encoder.pkl")

        self.xgb_task_model_path = os.path.join(self.model_dir, "xgboost_task.pkl")
        self.task_feature_columns_path = os.path.join(self.model_dir, "task_feature_columns.pkl")
        self.task_label_encoder_path = os.path.join(self.model_dir, "label_encoder_task.pkl")
        
        os.makedirs(self.model_dir, exist_ok=True) # 确保模型目录存在
        self._load_models() # 加载已训练的模型
        
        # 性能统计
        self.routing_history = []
        self.performance_stats = {
            "total_requests": 0,
            "successful_routes": 0,
            "model_usage": {"large": 0, "medium": 0, "small": 0}
        }
    

    def route(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """智能路由算法，双层分类机制"""
        # 第一层：模型大小分类
        size_prediction = self._xgboost_route(features)        
        task_type = None
        task_confidence = None
        
        if size_prediction["category"] in ["large", "medium"]:
            selected_model = self._select_specific_model(size_prediction["category"], features)
            reasoning = f"选择{size_prediction['category']}模型，{size_prediction['reasoning']}"
        else:
            task_prediction = self._xgboost_task_route(features)
            print(f"任务类型预测结果: {task_prediction}")
            
            if task_prediction:
                task_type = task_prediction["task_type"]
                task_confidence = task_prediction["confidence"]
                selected_model = self._select_small_model_by_task(task_type, features)
                reasoning = f"选择small模型，因为任务类型是{task_type}，置信度为{task_confidence:.3f}"
            else:
                # 任务分类失败，回退到原有逻辑
                selected_model = self._select_specific_model(size_prediction["category"], features)
                reasoning = f"任务分类失败，回退到{size_prediction['category']}模型选择"
        
        # 记录路由历史
        result = {
            "model": selected_model,
            "category": size_prediction["category"],
            "task_type": task_type,
            "confidence": size_prediction["confidence"],
            "task_confidence": task_confidence,
            "reasoning": reasoning,
            "probabilities": size_prediction["probabilities"]
        }
        
        self._record_routing(features, result, selected_model)
        return result

    
    def _xgboost_route(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """第一层分类模型"""
        try:
            feature_vector = self._prepare_feature_vector(features) # 准备特征向量
            if feature_vector is None:
                return None
        
            # XGBoost预测
            proba = self.xgb_model.predict_proba([feature_vector])[0]
            pred_idx = self.xgb_model.predict([feature_vector])[0]
        
            pred = self.label_encoder.inverse_transform([pred_idx])[0] # 预测结果
            confidence = np.max(proba) # 置信度，即预测概率
            reasoning = f"预测为{pred}，置信度为{confidence}" # 选择理由

            return {
                "category": pred,
                "confidence": confidence,
                "probabilities": {
                    "large": proba[0],
                    "medium": proba[1],
                    "small": proba[2]
                },
                "reasoning": reasoning
            }
            
        except Exception as e:
            print(f"XGBoost路由失败: {e}")
            return None
    
    
    def _xgboost_task_route(self, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """第二层分类"""
        try:
            if self.xgb_task_model is None:
                print("任务分类器未加载")
                return None
                
            feature_vector = self._prepare_feature_vector(features) # 准备特征向量
            if feature_vector is None:
                return None
            
            # XGBoost任务类型预测
            proba = self.xgb_task_model.predict_proba([feature_vector])[0]
            pred_idx = self.xgb_task_model.predict([feature_vector])[0]
            
            task_type = self.task_label_encoder.inverse_transform([pred_idx])[0] # 预测结果
            confidence = np.max(proba) # 置信度

            task_classes = self.task_label_encoder.classes_
            task_probabilities = {task_classes[i]: proba[i] for i in range(len(task_classes))} # 任务类型概率分布

            return {
                "task_type": task_type,
                "confidence": confidence,
                "probabilities": task_probabilities,
                "reasoning": f"任务类型预测为{task_type}，置信度为{confidence:.3f}"
            }
            
        except Exception as e:
            print(f"任务分类失败: {e}")
            return None


    def _select_specific_model(self, category: str, features: Dict[str, Any]) -> Any:
        """large / medium model的选择机制"""
        if category == "large":
            models = self.model_pool.large_models
        elif category == "medium":
            models = self.model_pool.medium_models
        else:
            models = self.model_pool.small_models
        
        if not models:
            all_models = self.model_pool.get_available_models()
            return all_models[0] if all_models else None
        
        for model_id, model in models.items():
            if model.is_available():
                return model
        
        return None
    
    
    def _select_small_model_by_task(self, task_type: str, features: Dict[str, Any]) -> Any:
        """small model的选择机制"""
        model = self.model_pool.get_model_by_task_type(task_type)
        if model and model.is_available():
            return model
        
        print(f"未找到任务类型 {task_type} 对应的模型，回退到small模型选择")
        models = self.model_pool.small_models
        for model_id, model in models.items():
            if model.is_available():
                return model
        
        return None


    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """准备特征向量"""
        if not self.feature_columns:
            return None
        try:
            feature_values = []
            for col in self.feature_columns:
                if col in features:
                    value = features[col]
                    if isinstance(value, (int, float)):
                        feature_values.append(float(value))
                    elif isinstance(value, str):
                        feature_values.append(self._encode_categorical_feature(col, value)) # 分类特征需要编码
                    elif isinstance(value, bool):
                        feature_values.append(float(value))
                    else:
                        feature_values.append(0.0)
                else:
                    feature_values.append(0.0)
            
            return np.array(feature_values)
            
        except Exception as e:
            print(f"特征向量准备失败: {e}")
            return None
    

    def _encode_categorical_feature(self, feature_name: str, value: str) -> float:
        """编码分类特征"""
        encoding_maps = {
            "primary_domain": {
                "technology": 0.9, "science": 0.8, "business": 0.7,
                "education": 0.6, "health": 0.5, "travel": 0.4,
                "entertainment": 0.3, "lifestyle": 0.2, "general": 0.1
            },
            "primary_task": {
                "analysis": 0.9, "coding": 0.8, "generation": 0.7,
                "qa": 0.6, "summarization": 0.5, "translation": 0.4,
                "chat": 0.3, "general": 0.1
            },
            "overall_complexity": {"high": 0.9, "medium": 0.5, "low": 0.1},
            "overall_urgency": {"high": 0.9, "medium": 0.5, "low": 0.1}
        }
        
        if feature_name in encoding_maps and value in encoding_maps[feature_name]:
            return encoding_maps[feature_name][value]
        else:
            return 0.5 
    
    
    
    def train_models(self, training_data: List[Dict[str, Any]]):
        """训练第一层路由模型"""
        if not training_data:
            print("没有训练数据")
            return
        
        print(f"开始训练第一层路由模型，数据量: {len(training_data)}")

        X, y = self._prepare_training_data("model", training_data) # 准备训练数据
        if X is None or len(X) == 0:
            print("训练数据准备失败")
            return
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # 分割数据
        
        print("训练XGBoost模型...")
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_model.fit(X_train, y_train)
        
        self._evaluate_models(X_test, y_test)# 评估模型
        self._save_models() # 保存模型
        
        print("模型训练完成")

    
    def train_task_classifier(self, task_training_data: List[Dict[str, Any]]):
        """训练第二层路由模型"""
        if not task_training_data:
            print("没有训练数据")
            return
        
        print(f"开始训练第二层任务分类器，数据量: {len(task_training_data)}")

        X, y = self._prepare_training_data("task", task_training_data)
        if X is None or len(X) == 0:
            print("训练数据准备失败")
            return
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print("训练XGBoost任务分类器...")
        self.xgb_task_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.xgb_task_model.fit(X_train, y_train)
        
        self._evaluate_task_models(X_test, y_test)
        self._save_task_models()
        
        print("任务分类器训练完成")


    def _prepare_training_data(self, data_type: str, training_data: List[Dict[str, Any]]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        try:
            features_list = []
            labels = []
            
            for data_point in training_data:
                query = data_point.get("query", "")
                if data_type == "model":
                    label = data_point.get("best_model_category", "")
                else:
                    label = data_point.get("task_type", "")
                context = data_point.get("context")
                
                features = self.feature_extractor.extract_features(query, context) # 提取特征
                features_list.append(features)
                labels.append(label)
            
            if not self.feature_columns:
                self._determine_feature_columns(features_list)
            
            # 转换为数值矩阵
            X = []
            for features in features_list:
                feature_vector = self._prepare_feature_vector(features)
                if feature_vector is not None:
                    X.append(feature_vector)
                else:
                    continue
            
            if not X:
                return None, None
            X = np.array(X)
            
            if data_type == "model":
                y_encoded = self.label_encoder.fit_transform(labels[:len(X)])
            else:
                y_encoded = self.task_label_encoder.fit_transform(labels[:len(X)])
            
            y = y_encoded # 编码标签
            
            return X, y
            
        except Exception as e:
            print(f"训练数据准备失败: {e}")
            return None, None
    
    
    def _determine_feature_columns(self, features_list: List[Dict[str, Any]]):
        """确定特征列"""
        numerical_features = [ # 数值型特征
            "text_length", "word_count", "sentence_count", "avg_word_length",
            "punctuation_count", "digit_count", "uppercase_ratio",
            "lexical_diversity", "sentiment_score", "semantic_complexity",
            "complexity_score", "urgency_score", "domain_confidence", "task_confidence"
        ]
        
        categorical_features = [ # 分类特征
            "primary_domain", "primary_task", "overall_complexity", "overall_urgency"
        ]
        
        boolean_features = [ # 布尔特征
            "has_question_mark", "has_exclamation", "has_context"
        ]
        
        self.feature_columns = numerical_features + categorical_features + boolean_features
        print(f"确定特征列: {len(self.feature_columns)}个特征")
    
     
    def _evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """评估第一层路由模型性能"""
        xgb_pred = self.xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"Router-1 准确率: {xgb_accuracy:.3f}")
        print("\nRoute-1 分类报告:")
        print(classification_report(y_test, xgb_pred, target_names=self.label_encoder.classes_))
    
    
    def _evaluate_task_models(self, X_test: np.ndarray, y_test: np.ndarray):
        """评估第二层路由模型性能"""
        xgb_pred = self.xgb_task_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        print(f"Router-2 准确率: {xgb_accuracy:.3f}")
        print("\nRoute-2 分类报告:")
        print(classification_report(y_test, xgb_pred, target_names=self.task_label_encoder.classes_))
    
    
    def _save_models(self):
        """保存第一层路由模型"""
        try:
            joblib.dump(self.xgb_model, self.xgb_model_path) # 保存模型
            with open(self.feature_columns_path, 'wb') as f: # 保存特征列
                pickle.dump(self.feature_columns, f)
            joblib.dump(self.label_encoder, self.label_encoder_path) # 保存标签编码器
            print("Route-1 保存成功")
            
        except Exception as e:
            print(f"Route-1 保存失败: {e}")

    
    def _save_task_models(self):
        """保存第二层路由模型"""
        try:
            joblib.dump(self.xgb_task_model, self.xgb_task_model_path)
            with open(self.task_feature_columns_path, 'wb') as f:
                pickle.dump(self.task_feature_columns, f)
            joblib.dump(self.task_label_encoder, self.task_label_encoder_path)
            print("Route-2 保存成功")
            
        except Exception as e:
            print(f"Route-2 保存失败: {e}")

    
    def _load_models(self):
        """加载已训练的模型"""
        try:
            if os.path.exists(self.feature_columns_path): # 加载特征列
                with open(self.feature_columns_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                print("Route-1 特征列加载成功")
            
            if os.path.exists(self.xgb_model_path): # 加载模型
                self.xgb_model = joblib.load(self.xgb_model_path)
                print("Route-1 模型加载成功")
            
            if os.path.exists(self.label_encoder_path): # 加载标签编码器
                self.label_encoder = joblib.load(self.label_encoder_path)
                print("Route-1 标签编码器加载成功")
            
            if os.path.exists(self.task_feature_columns_path): 
                with open(self.task_feature_columns_path, 'rb') as f:
                    self.task_feature_columns = pickle.load(f)
                print("Route-2 特征列加载成功")
            
            if os.path.exists(self.xgb_task_model_path): 
                self.xgb_task_model = joblib.load(self.xgb_task_model_path)
                print("Route-2 模型加载成功")
            
            if os.path.exists(self.task_label_encoder_path): 
                self.task_label_encoder = joblib.load(self.task_label_encoder_path)
                print("Route-2 标签编码器加载成功")
                
        except Exception as e:
            print(f"Route-1 加载失败: {e}")
            self.xgb_model = None
            print(f"Route-2 加载失败: {e}")
            self.xgb_task_model = None
    


    def _record_routing(self, features: Dict[str, Any], decision: Dict[str, Any], model: Any):
        """记录路由历史"""
        self.routing_history.append({
            "timestamp": pd.Timestamp.now(),
            "features": features,
            "decision": decision,
            "model": model.model_name if model else None
        })
        
        # 更新统计
        self.performance_stats["total_requests"] += 1
        if model:
            self.performance_stats["successful_routes"] += 1
            category = decision.get("category", "medium")
            self.performance_stats["model_usage"][category] += 1
    


    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        total = self.performance_stats["total_requests"]
        if total == 0:
            return self.performance_stats
        
        stats = self.performance_stats.copy()
        stats["success_rate"] = stats["successful_routes"] / total
        stats["model_distribution"] = {
            k: v / total for k, v in stats["model_usage"].items()
        }
        
        return stats


if __name__ == "__main__":
    from utils.misc import load_data
    

    smart_router = SmartRouter(ModelPool())
    # 训练模型大小分类器
    print("训练模型大小分类器")
    smart_router.train_models(load_data(file_path="data/dataset.json"))
    
    # 训练任务类型分类器
    print("训练任务类型分类器")
    smart_router.train_task_classifier(load_data(file_path="data/task_dataset.json"))