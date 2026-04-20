# src/core/__init__.py
"""
核心 RAG 模块，提供默认的 RAGPipeline 单例。
"""
from typing import Optional
from .rag import RAGPipeline

__all__ = ["RAGPipeline", "get_default_pipeline", "reset_default_pipeline"]

# 全局默认管道实例（初始为 None，惰性加载）
_default_pipeline: Optional[RAGPipeline] = None


def get_default_pipeline() -> RAGPipeline:
    """
    获取默认的 RAGPipeline 单例。
    
    首次调用时会初始化管道（加载嵌入模型），后续调用直接返回已创建的实例。
    这样设计可以：
        - 避免在模块导入时立即加载模型
        - 保证整个应用共享同一个向量索引实例
    """
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline


def reset_default_pipeline() -> None:
    """
    重置默认管道实例（主要用于测试环境，或需要重新加载配置的场景）。
    """
    global _default_pipeline
    _default_pipeline = None