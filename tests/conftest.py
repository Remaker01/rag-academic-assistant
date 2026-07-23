"""
pytest 共享 fixtures：Mock LLM、temp 向量存储、样本文档。
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_core.documents import Document

# ---------------------------------------------------------------------------
# 环境：确保所有测试使用同样的 HuggingFace 镜像源
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _hf_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", "https://aliendao.cn/models/")

# ---------------------------------------------------------------------------
# Mock LLM —— 所有可能调用 LLM 的测试都通过此 fixture 拦截真实 API 请求
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_llm() -> MagicMock:
    """返回 MagickMock，替换 ChatOpenAI / 任何 LLM 调用。"""
    llm = MagicMock()
    # AI 回复的内容
    ai_content = "这是一个模拟的回答。"

    # 模拟 invoke() 返回 AIMessage
    from langchain_core.messages import AIMessage
    llm.invoke.return_value = AIMessage(content=ai_content)

    # 模拟 stream()
    from langchain_core.messages import AIMessageChunk
    llm.stream.return_value = [AIMessageChunk(content=ai_content)]

    # 模拟 astream()
    async def _astream(*args, **kwargs):
        yield AIMessageChunk(content=ai_content)
    llm.astream = _astream

    return llm

# ---------------------------------------------------------------------------
# Mock Embedding 模型 —— 用固定维度随机向量代替真实句子嵌入
# ---------------------------------------------------------------------------
@pytest.fixture
def mock_embeddings() -> MagicMock:
    """返回 MagickMock 替换 HuggingFaceEmbeddings，动态适配文档数量。

    使用 spec=Embeddings 使 isinstance(emb, Embeddings) == True，
    确保 FAISS / LangChain 内部走正确的 embed_query 代码路径。
    """
    from langchain_core.embeddings import Embeddings

    emb = MagicMock(spec=Embeddings)
    rng = np.random.default_rng(42)  # 固定种子，保证测试可重复
    DIM = 384  # all-MiniLM-L6-v2 的维度

    emb.embed_query.return_value = rng.normal(size=DIM).tolist()

    # embed_documents 接受的文档数量不定，动态生成对应数量的向量
    def _embed_documents(texts, **kwargs):
        return [rng.normal(size=DIM).tolist() for _ in texts]

    emb.embed_documents.side_effect = _embed_documents
    return emb

# ---------------------------------------------------------------------------
# 样本文档（纯文本，无需 PDF 文件）
# ---------------------------------------------------------------------------
SAMPLE_TEXT = """Machine learning is a field of artificial intelligence.
Deep learning is a subset of machine learning using neural networks.
Neural networks consist of layers of interconnected neurons.
Each neuron applies a non-linear activation function to its input.
Training involves backpropagation and gradient descent optimization.
Supervised learning requires labeled training data for model training.
Unsupervised learning finds patterns in unlabeled data automatically.
Reinforcement learning involves agents learning from environment rewards.
Transfer learning leverages pre-trained models for new tasks.
Overfitting occurs when a model memorizes training data too well."""


@pytest.fixture
def sample_documents() -> list[Document]:
    """纯文本 Document 列表，用于索引构建 / 检索测试。"""
    return [
        Document(
            page_content=para,
            metadata={"source": "test.pdf", "page": i // 3, "chunk_id": i},
        )
        for i, para in enumerate(SAMPLE_TEXT.strip().splitlines())
    ]


@pytest.fixture
def sample_chunks() -> list[Document]:
    """较短的样本文本，用于分块测试。"""
    return [
        Document(
            page_content="Introduction to machine learning.",
            metadata={"source": "test.pdf", "page": 1, "chunk_id": 0},
        ),
        Document(
            page_content="Deep learning is a subset of machine learning.",
            metadata={"source": "test.pdf", "page": 1, "chunk_id": 1},
        ),
        Document(
            page_content="Neural networks consist of layers of neurons.",
            metadata={"source": "test.pdf", "page": 2, "chunk_id": 2},
        ),
    ]

# ---------------------------------------------------------------------------
# 临时向量存储目录
# ---------------------------------------------------------------------------
@pytest.fixture
def vector_store_path(tmp_path: Path) -> str:
    """每个测试独享的临时 FAISS 存储目录。"""
    path = tmp_path / "vector_store"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


# ---------------------------------------------------------------------------
# RAGPipeline 实例（Mock Embedding 以跳过模型加载）
# ---------------------------------------------------------------------------
@pytest.fixture
def rag_pipeline(
    mock_embeddings: MagicMock,
    vector_store_path: str,
) -> Generator:
    """返回一个已注入 mock embedding 的 RAGPipeline 实例。"""
    with patch("src.core.rag.HuggingFaceEmbeddings", return_value=mock_embeddings):
        from src.core.rag import RAGPipeline
        pipeline = RAGPipeline(vector_store_path=vector_store_path)
        yield pipeline


# ---------------------------------------------------------------------------
# 预构建好 FAISS 索引的 Pipeline（跳过 save/load，直接使用内存索引）
# ---------------------------------------------------------------------------
@pytest.fixture
def indexed_pipeline(
    rag_pipeline,
    sample_documents: list[Document],
) -> None:
    """返回一个已经 build_index 的 RAGPipeline 实例（索引在内存中）。"""
    rag_pipeline.build_index(sample_documents)
    yield rag_pipeline
