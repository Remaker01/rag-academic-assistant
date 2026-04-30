# src/api/routes.py
import os
import logging
from pathlib import Path
from typing import Optional, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.agent import AgentManager
from src.core import get_default_pipeline
from src.utils import load_config
from src.api.models import (
    ChatRequest, ChatResponse,
    IndexListResponse, IndexInfo,
    PDFImportRequest, PDFImportResponse,
    MemoryResetRequest, MemoryResetResponse,
    HealthResponse
)
from src import __version__

logger = logging.getLogger("api")


class APIService:
    """封装 Agent 管理与业务逻辑的服务类"""

    def __init__(self):
        self.config = load_config()
        self.pipeline = get_default_pipeline()
        # 存储 AgentManager 实例，key 为 index_name
        self.agent_managers: Dict[str, AgentManager] = {}
        self.default_index_name = "current_paper"

    def get_or_create_agent_manager(self, index_name: str) -> AgentManager:
        """获取或创建指定索引的 AgentManager"""
        if index_name not in self.agent_managers:
            # 确保索引存在
            if not self.pipeline.index_exists(index_name):
                raise HTTPException(
                    status_code=404,
                    detail=f"索引 '{index_name}' 不存在，请先导入 PDF 或指定正确的索引名。"
                )
            self.agent_managers[index_name] = AgentManager(
                index_name=index_name,
                verbose=False,
            )
            logger.info(f"为索引 '{index_name}' 创建 AgentManager 实例")
        return self.agent_managers[index_name]

    def chat(self, request: ChatRequest) -> ChatResponse:
        """处理问答请求"""
        index_name = request.index_name or self.default_index_name
        try:
            agent_mgr = self.get_or_create_agent_manager(index_name)
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"获取 AgentManager 失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Agent 初始化失败: {str(e)}")

        try:
            result = agent_mgr.run(request.query, session_id=request.session_id)
            answer = result.get("output", "")
            error = result.get("error")
            return ChatResponse(
                query=request.query,
                answer=answer,
                session_id=request.session_id,
                index_used=index_name,
                error=error,
            )
        except Exception as e:
            logger.error(f"问答执行失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"问答处理失败: {str(e)}")

    def list_indices(self) -> IndexListResponse:
        """列出所有已存在的向量索引（递归遍历子文件夹）"""
        vector_store_path = str(self.pipeline.vector_store_path)
        indices = []
        for root, dirs, files in os.walk(vector_store_path):
            for file in files:
                if file.endswith('.faiss'):
                    index_name = Path(file).stem
                    pkl_file = os.path.join(root, f"{index_name}.pkl")
                    exists = os.path.isfile(pkl_file)
                    indices.append(IndexInfo(
                        name=index_name,
                        exists=exists,
                        path=os.path.join(root, index_name)
                    ))
        return IndexListResponse(indices=indices)

    def import_pdf(self, request: PDFImportRequest) -> PDFImportResponse:
        """导入 PDF 并构建索引"""
        pdf_path = Path(request.pdf_path)
        if not pdf_path.exists():
            raise HTTPException(status_code=400, detail=f"PDF 文件不存在: {request.pdf_path}")

        # 自动生成索引名
        if request.index_name:
            index_name = request.index_name
        else:
            # 使用文件名（不含扩展名）作为索引名，替换非法字符
            index_name = pdf_path.stem.replace(" ", "_").replace("-", "_")

        try:
            self.pipeline.process_pdf(str(pdf_path), index_name)
            # 清除旧的 AgentManager（如果存在）
            if index_name in self.agent_managers:
                del self.agent_managers[index_name]
            # 获取分块数量（可选）
            if self.pipeline.vector_store:
                chunks_count = self.pipeline.vector_store.index.ntotal
            else:
                chunks_count = None
            return PDFImportResponse(
                success=True,
                message=f"成功导入 {pdf_path.name}，索引名: {index_name}",
                index_name=index_name,
                chunks_count=chunks_count,
            )
        except Exception as e:
            logger.error(f"PDF 导入失败: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"导入失败: {str(e)}")

    def reset_memory(self, request: MemoryResetRequest) -> MemoryResetResponse:
        """重置指定会话的记忆"""
        # 遍历所有 AgentManager 执行重置（因为 session_id 可能跨索引使用）
        success_count = 0
        for mgr in self.agent_managers.values():
            try:
                mgr.reset_memory(request.session_id)
                success_count += 1
            except Exception as e:
                logger.warning(f"重置记忆时出错 (index: {mgr.index_name}): {e}")

        return MemoryResetResponse(
            success=True,
            message=f"已重置 {success_count} 个索引下的会话记忆",
            session_id=request.session_id,
        )

    def health_check(self) -> HealthResponse:
        """健康检查"""
        # 测试 DeepSeek 连通性（轻量检查，不实际调用）
        deepseek_ok = bool(self.config.get("deepseek_api_key"))
        return HealthResponse(
            status="healthy" if deepseek_ok else "degraded",
            deepseek_available=deepseek_ok,
            embedding_model=self.config.get("embedding_model", "unknown"),
            vector_store_path=self.config.get("vector_store_path", "./data/vector_store"),
        )


# 全局服务实例（单例）
_service: Optional[APIService] = None


def get_service() -> APIService:
    global _service
    if _service is None:
        _service = APIService()
    return _service


def create_app() -> FastAPI:
    """创建 FastAPI 应用实例"""
    app = FastAPI(
        title="学术文献智能问答助手 API",
        description="基于 RAG 和 Agent 的学术论文问答服务",
        version=__version__,
    )

    # 配置 CORS（允许跨域请求，便于前端调用）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境建议指定具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    service = get_service()

    @app.get("/health", response_model=HealthResponse, tags=["系统"])
    async def health():
        """健康检查端点"""
        return service.health_check()

    @app.get("/indices", response_model=IndexListResponse, tags=["索引管理"])
    async def list_indices():
        """列出所有已存在的向量索引"""
        return service.list_indices()

    @app.post("/import", response_model=PDFImportResponse, tags=["索引管理"])
    async def import_pdf(request: PDFImportRequest):
        """
        导入 PDF 论文并构建向量索引。
        
        - **pdf_path**: PDF 文件的绝对路径
        - **index_name**: 可选，自定义索引名称；若不指定则使用文件名
        """
        return service.import_pdf(request)

    @app.post("/chat", response_model=ChatResponse, tags=["问答"])
    async def chat(request: ChatRequest):
        """
        向助手提问，获取基于论文内容的回答。
        
        支持多轮对话，通过 session_id 区分不同对话线程。
        """
        return service.chat(request)

    @app.post("/memory/reset", response_model=MemoryResetResponse, tags=["会话管理"])
    async def reset_memory(request: MemoryResetRequest):
        """重置指定会话的对话记忆，开始新的对话"""
        return service.reset_memory(request)

    return app