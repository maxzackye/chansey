#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
大模型分析器模块
用于将数据分析结果发送给大模型进行解读
"""

import json
import sys
import requests
from typing import Dict, List, Any, Optional

class LLMAnalyzer:
    """大模型分析器类"""
    
    def __init__(self, gateway: str = "http://ai.caijj.net/chat/completions", 
                 token: str = "sk-6X8QyIvh8MIYHPY3poaa_A",
                 model: str = "pub_qwen3-coder-plus_bailian"):
        """
        初始化大模型分析器
        
        Args:
            gateway: 大模型网关地址
            token: 认证令牌
            model: 使用的模型名称
        """
        self.gateway = gateway
        self.token = token
        self.model = model
        self.headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
    
    def analyze_data(self, analysis_results: List[Dict], analysis_config: Dict) -> Optional[str]:
        """
        将数据分析结果发送给大模型进行解读
        
        Args:
            analysis_results: 分析结果列表
            analysis_config: 分析配置
            
        Returns:
            大模型的解读结果
        """
        # 构造提示词
        prompt = self._construct_prompt(analysis_results, analysis_config)
        
        # 构造请求体
        payload = {
            "stream": False,  # 我们不需要流式输出
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你是一位专业的数据分析师，擅长解读业务指标变化原因。"},
                {"role": "user", "content": prompt}
            ]
        }
        
        try:
            response = requests.post(
                self.gateway, 
                headers=self.headers, 
                json=payload, 
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] 请求大模型失败: {e}")
            return None
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[ERROR] 解析大模型响应失败: {e}")
            return None
    
    def _construct_prompt(self, analysis_results: List[Dict], analysis_config: Dict) -> str:
        """
        构造发送给大模型的提示词
        
        Args:
            analysis_results: 分析结果列表
            analysis_config: 分析配置
            
        Returns:
            构造好的提示词
        """
        prompt = "请帮我分析以下业务指标的变化情况，并给出可能的原因和业务建议:\n\n"
        
        # 添加分析配置信息
        prompt += "分析配置:\n"
        prompt += f"- 观察期: {analysis_config.get('obs_date_range', 'N/A')}\n"
        prompt += f"- 对比期: {analysis_config.get('cmp_date_range', 'N/A')}\n"
        prompt += f"- 分析维度: {analysis_config.get('dimensions', 'N/A')}\n\n"
        
        # 添加每个指标的分析结果
        for i, result in enumerate(analysis_results):
            prompt += f"指标 {i+1}: {result.get('metric_name', 'N/A')}\n"
            prompt += f"- 指标类型: {result.get('metric_type', 'N/A')}\n"
            prompt += f"- 分析维度: {result.get('dimension', 'N/A')}\n"
            prompt += f"- 对比期数值: {result.get('cmp_total', 'N/A')}\n"
            prompt += f"- 观察期数值: {result.get('obs_total', 'N/A')}\n"
            
            # 添加变化情况
            obs_total = result.get('obs_total', 0)
            cmp_total = result.get('cmp_total', 0)
            change = obs_total - cmp_total
            change_pct = (change / cmp_total * 100) if cmp_total != 0 else 0
            
            prompt += f"- 变化量: {change:.4f}\n"
            prompt += f"- 变化率: {change_pct:.2f}%\n"
            
            # 添加主要贡献维度（前5个）
            data = result.get('data', [])
            if data:
                # 按总贡献排序
                sorted_data = sorted(data, key=lambda x: abs(x.get('总贡献', 0)), reverse=True)
                prompt += "- 主要贡献维度:\n"
                for j, item in enumerate(sorted_data[:5]):  # 只取前5个
                    dim_value = list(item.values())[0]  # 第一列是维度值
                    total_contribution = item.get('总贡献', 0)
                    prompt += f"  {j+1}. {dim_value}: {total_contribution:.4f}\n"
            
            prompt += "\n"
        
        prompt += (
            "请根据以上数据分析:\n"
            "1. 整体指标变化的主要原因是什么？\n"
            "2. 哪些维度对指标变化产生了显著影响？\n"
            "3. 根据分析结果，给出具体的业务建议。\n\n"
            "请用中文回答，内容简洁明了，重点突出。"
        )
        
        return prompt

# 兼容旧版本的函数接口
def analyze_with_llm(analysis_results: List[Dict], analysis_config: Dict) -> Optional[str]:
    """
    使用大模型分析数据的便捷函数
    
    Args:
        analysis_results: 分析结果列表
        analysis_config: 分析配置
        
    Returns:
        大模型的解读结果
    """
    analyzer = LLMAnalyzer()
    return analyzer.analyze_data(analysis_results, analysis_config)