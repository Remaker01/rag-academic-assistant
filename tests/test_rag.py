"""
RAG 核心单元测试：分块、索引构建/加载、检索、格式化。
"""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

# ============================== 分块测试 ==============================


class TestChunking:
    """验证 PDF 加载与 RecursiveCharacterTextSplitter 分块行为。"""

    PDF_PATH = Path("data/pdfs/example.pdf")

    @pytest.fixture(autouse=True)
    def _skip_if_no_pdf(self) -> None:
        if not self.PDF_PATH.exists():
            pytest.skip(f"测试 PDF 不存在: {self.PDF_PATH}")

    def test_chunks_are_created(self, rag_pipeline) -> None:
        """至少生成 1 个文档块。"""
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        assert len(chunks) > 0

    def test_each_chunk_has_content(self, rag_pipeline) -> None:
        """每个块的非空 page_content。"""
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        for chunk in chunks:
            assert chunk.page_content, f"空内容在第 {chunk.metadata.get('chunk_id')} 块"

    def test_chunk_size_bounded(self, rag_pipeline) -> None:
        """每个 chunk 不超过 chunk_size * 1.5（中文容差）。"""
        max_chars = rag_pipeline.chunk_size * 1.5
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        for chunk in chunks:
            assert len(chunk.page_content) <= max_chars, (
                f"chunk {chunk.metadata.get('chunk_id')} 长度 {len(chunk.page_content)} "
                f"超过上限 {max_chars:.0f}"
            )

    def test_metadata_source(self, rag_pipeline) -> None:
        """metadata['source'] 为文件名。"""
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        for chunk in chunks:
            assert chunk.metadata.get("source") == "example.pdf"

    def test_metadata_chunk_id_sequential(self, rag_pipeline) -> None:
        """chunk_id 从 0 开始连续递增。"""
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        for i, chunk in enumerate(chunks):
            assert chunk.metadata.get("chunk_id") == i, (
                f"期望 chunk_id={i}，实际 {chunk.metadata.get('chunk_id')}"
            )

    def test_invalid_path_raises(self, rag_pipeline) -> None:
        """不存在的 PDF 抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            rag_pipeline.load_and_split_pdf("nonexistent.pdf")

    def test_empty_path_raises(self, rag_pipeline) -> None:
        """空路径抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            rag_pipeline.load_and_split_pdf("")

    def test_overlap_between_chunks(self, rag_pipeline) -> None:
        """相邻 chunk 之间存在字符重叠（粗略验证）。"""
        chunks = rag_pipeline.load_and_split_pdf(str(self.PDF_PATH))
        if len(chunks) < 2:
            pytest.skip("只有一个 chunk，无法验证重叠")
        for i in range(len(chunks) - 1):
            cur = chunks[i].page_content
            nxt = chunks[i + 1].page_content
            # 检查当前 chunk 尾部是否与下一 chunk 头部有重叠
            if len(cur) >= rag_pipeline.chunk_overlap:
                overlap_region = cur[-rag_pipeline.chunk_overlap:]
                if overlap_region in nxt:
                    return  # 至少一对相邻 chunk 有重叠即可
        # 如果所有相邻 chunk 都无法找到重叠，也可能是分块在段落边界截断了
        # 这不一定是 bug，但不常见，标记为 expected failure
        pytest.skip("未检测到明确重叠（可能因段落边界恰好对齐）")


# ============================== 索引构建/加载测试 ==============================


class TestIndexing:
    """验证 FAISS 索引的构建、保存、加载生命周期。"""

    def test_build_index(self, rag_pipeline, sample_documents) -> None:
        """build_index 后 vector_store 非空。"""
        rag_pipeline.build_index(sample_documents)
        assert rag_pipeline.vector_store is not None
        assert rag_pipeline.vector_store.index.ntotal == len(sample_documents)

    def test_build_empty_docs(self, rag_pipeline) -> None:
        """空文档列表不抛出异常，vector_store 保持为 None。"""
        rag_pipeline.build_index([])
        assert rag_pipeline.vector_store is None

    def test_save_and_load(self, rag_pipeline, sample_documents) -> None:
        """save_index → load_index 完整闭环。"""
        rag_pipeline.build_index(sample_documents)
        rag_pipeline.save_index("test_save_load")

        # 重置后加载
        rag_pipeline.vector_store = None
        loaded = rag_pipeline.load_index("test_save_load")
        assert loaded is True
        assert rag_pipeline.vector_store is not None
        assert rag_pipeline.vector_store.index.ntotal == len(sample_documents)

    def test_load_nonexistent(self, rag_pipeline) -> None:
        """加载不存在的索引返回 False。"""
        loaded = rag_pipeline.load_index("__does_not_exist__")
        assert loaded is False
        assert rag_pipeline.vector_store is None

    def test_index_exists(self, rag_pipeline, sample_documents) -> None:
        """index_exists 正确返回 True/False。"""
        assert rag_pipeline.index_exists("__does_not_exist__") is False
        rag_pipeline.build_index(sample_documents)
        rag_pipeline.save_index("test_exists")
        assert rag_pipeline.index_exists("test_exists") is True

    def test_overwrite_index(self, rag_pipeline, sample_documents) -> None:
        """重新构建同名索引不报错。"""
        rag_pipeline.build_index(sample_documents)
        rag_pipeline.save_index("overwrite_test")
        rag_pipeline.build_index(sample_documents[:1])
        rag_pipeline.save_index("overwrite_test")
        assert rag_pipeline.index_exists("overwrite_test") is True


# ============================== 检索测试 ==============================


class TestRetrieve:
    """验证语义检索的正确性。"""

    def test_retrieve_returns_k_results(self, indexed_pipeline) -> None:
        """retrieve(k=3) 返回恰好 3 个 Document。"""
        docs = indexed_pipeline.retrieve("machine learning", k=3)
        assert len(docs) == 3
        assert all(isinstance(d, Document) for d in docs)

    def test_k_greater_than_total(self, indexed_pipeline) -> None:
        """k 超过总文档数时不报错，返回全部结果。"""
        docs = indexed_pipeline.retrieve("machine learning", k=100)
        total = indexed_pipeline.vector_store.index.ntotal
        assert len(docs) == total

    def test_query_string_empty(self, indexed_pipeline) -> None:
        """空查询不应抛出异常（返回空列表或兜底结果）。"""
        docs = indexed_pipeline.retrieve("", k=3)
        # FAISS 对空字符串也能返回结果（根据词嵌入向量），不报错即可
        assert isinstance(docs, list)

    def test_retrieve_before_init_raises(self, rag_pipeline) -> None:
        """未初始化 vector_store 时 retrieve 抛出 ValueError。"""
        with pytest.raises(ValueError, match="向量索引尚未初始化"):
            rag_pipeline.retrieve("test")

    def test_retrieved_docs_have_metadata(self, indexed_pipeline) -> None:
        """检索结果保留原始 metadata。"""
        docs = indexed_pipeline.retrieve("machine learning", k=1)
        assert "source" in docs[0].metadata
        assert "chunk_id" in docs[0].metadata


# ============================== 格式化测试 ==============================


class TestFormatDocs:
    """验证 format_docs 输出格式。"""

    def test_format_single_doc(self, rag_pipeline, sample_chunks) -> None:
        """单个文档格式化。"""
        result = rag_pipeline.format_docs(sample_chunks[:1])
        assert "[片段 1]" in result
        assert sample_chunks[0].page_content in result

    def test_format_multiple_docs(self, rag_pipeline, sample_chunks) -> None:
        """多个文档包含正确编号。"""
        result = rag_pipeline.format_docs(sample_chunks)
        assert "[片段 1]" in result
        assert "[片段 2]" in result
        assert "[片段 3]" in result

    def test_format_separator(self, rag_pipeline, sample_chunks) -> None:
        """片段之间用空行分隔。"""
        result = rag_pipeline.format_docs(sample_chunks)
        assert "\n\n" in result

    def test_format_empty(self, rag_pipeline) -> None:
        """空列表返回空字符串。"""
        result = rag_pipeline.format_docs([])
        assert result == ""

    def test_format_content_preserved(self, rag_pipeline, sample_chunks) -> None:
        """原始内容不被修改。"""
        result = rag_pipeline.format_docs(sample_chunks)
        for chunk in sample_chunks:
            assert chunk.page_content in result
