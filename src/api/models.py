# src/api/models.py
from typing import Optional, List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """问答请求体"""
    query: str = Field(..., description="用户问题", min_length=1)
    session_id: str = Field("default", description="会话ID，用于区分不同对话")
    index_name: Optional[str] = Field(None, description="指定要使用的索引名称，不指定则使用默认索引")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="生成温度")
    max_history_tokens: Optional[int] = Field(4000, ge=100, description="对话历史最大 token 数")


class ChatResponse(BaseModel):
    """问答响应体"""
    query: str = Field(..., description="原始问题")
    answer: str = Field(..., description="助手回答")
    session_id: str = Field(..., description="会话ID")
    index_used: str = Field(..., description="使用的索引名称")
    error: Optional[str] = Field(None, description="错误信息，仅当出错时存在")


class IndexInfo(BaseModel):
    """索引信息"""
    name: str
    exists: bool
    path: str


class IndexListResponse(BaseModel):
    """索引列表响应"""
    indices: List[IndexInfo]


class PDFImportRequest(BaseModel):
    """PDF 导入请求"""
    pdf_path: str = Field(..., description="PDF 文件的绝对路径")
    index_name: Optional[str] = Field(None, description="索引名称，不指定则使用文件名自动生成")


class PDFImportResponse(BaseModel):
    """PDF 导入响应"""
    success: bool
    message: str
    index_name: Optional[str] = None
    chunks_count: Optional[int] = None


class MemoryResetRequest(BaseModel):
    """重置记忆请求"""
    session_id: str = Field("default", description="要重置的会话ID")


class MemoryResetResponse(BaseModel):
    """重置记忆响应"""
    success: bool
    message: str
    session_id: str


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    deepseek_available: bool
    embedding_model: str
    vector_store_path: str