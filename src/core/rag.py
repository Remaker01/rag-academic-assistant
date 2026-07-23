# src/core/rag.py
"""
RAG 核心管道：文档加载、分块、向量嵌入、FAISS 索引、检索与生成。
"""
import os
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import jieba
from sentence_transformers import CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from ..utils import load_config, setup_logger

logger = setup_logger("rag")


def _jieba_tokenize(text: str) -> List[str]:
    """使用 jieba 对中文文本分词，英文保持原样。"""
    return list(jieba.cut(text))


class RAGPipeline:
    """
    检索增强生成管道，负责：
    - PDF 文档加载与智能分块
    - 基于 Sentence-Transformers 的文本嵌入
    - FAISS 向量索引的构建、保存与加载
    - 语义检索与基于 DeepSeek 的答案生成
    """

    _parser = StrOutputParser()
    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        vector_store_path: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        """
        初始化 RAG 管道。

        Args:
            embedding_model_name: Sentence-Transformers 模型名称，默认从配置读取
            vector_store_path: FAISS 索引持久化目录，默认从配置读取
            chunk_size: 文档分块大小（字符数）
            chunk_overlap: 分块重叠字符数
        """
        config = load_config()

        self.embedding_model_name = embedding_model_name or config["embedding_model"]
        self.vector_store_path = Path(vector_store_path or config["vector_store_path"])
        raw_path = vector_store_path or config["vector_store_path"]
        self.vector_store_path = Path(raw_path).resolve()
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 确保向量存储目录存在
        self.vector_store_path.mkdir(parents=True, exist_ok=True)

        # 初始化嵌入模型（使用本地缓存）
        logger.info(f"正在加载嵌入模型: {self.embedding_model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            model_kwargs={"device": "cpu"},  # 强制 CPU 运行
            encode_kwargs={"normalize_embeddings": True},  # 归一化有利于 FAISS 检索
        )

        # FAISS 索引实例（懒加载）
        self.vector_store: Optional[FAISS] = None

        # BM25 稀疏检索器（懒加载）
        self._bm25_retriever: Optional[BM25Retriever] = None
        self._last_documents: List[Document] = []

        # Reranker 模型（懒加载，首次 rerank() 调用时初始化）
        self._reranker: Optional[CrossEncoder] = None
        self._reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        logger.info("RAGPipeline 初始化完成")

    # -------------------- 文档处理 --------------------
    def load_and_split_pdf(self, pdf_path: str) -> List[Document]:
        """
        加载单个 PDF 文件并分割为文档块。

        Args:
            pdf_path: PDF 文件路径

        Returns:
            分割后的 Document 列表
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 文件不存在: {pdf_path}")

        logger.info(f"正在加载 PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        logger.info(f"加载完成，共 {len(documents)} 页。正在进行文本分割...")
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"分割完成，共生成 {len(chunks)} 个文档块。")

        # 为每个块添加来源元数据
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = os.path.basename(pdf_path)
            chunk.metadata["chunk_id"] = i

        return chunks

    # -------------------- 向量索引管理 --------------------
    def build_index(self, documents: List[Document]) -> None:
        """
        根据文档块列表构建 FAISS 向量索引和 BM25 稀疏索引。

        Args:
            documents: Document 列表
        """
        if not documents:
            logger.warning("文档列表为空，跳过索引构建。")
            return

        logger.info(f"正在为 {len(documents)} 个文档块生成嵌入向量...")
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        logger.info("FAISS 索引构建完成。")

        # 同步构建 BM25 索引
        logger.info("正在构建 BM25 稀疏索引...")
        self._bm25_retriever = BM25Retriever.from_documents(
            documents,
            preprocess_func=_jieba_tokenize,
        )
        self._last_documents = documents
        logger.info("BM25 稀疏索引构建完成。")

    def save_index(self, index_name: str = "faiss_index") -> None:
        """
        将当前向量索引持久化到磁盘（包含 BM25 索引）。

        Args:
            index_name: 索引文件名（不含扩展名）
        """
        if self.vector_store is None:
            logger.warning("向量索引为空，无法保存。")
            return

        # 构建保存路径：vector_store_path / index_name （不带扩展名）
        save_path = (self.vector_store_path / index_name).resolve()
        # 确保保存路径的父目录存在（实际上 vector_store_path 已经存在）
        try:
            self.vector_store.save_local(str(save_path))
            logger.info(f"FAISS 索引已保存至: {save_path}")

            # 保存 BM25 索引
            if self._bm25_retriever is not None:
                bm25_path = save_path / "bm25.pkl"
                with open(bm25_path, "wb") as f:
                    pickle.dump(self._bm25_retriever, f)
                logger.info(f"BM25 索引已保存至: {bm25_path}")
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            raise

    def load_index(self, index_name: str = "faiss_index") -> bool:
        """
        从磁盘加载已保存的 FAISS + BM25 索引。

        Args:
            index_name: 索引文件名（不含扩展名）

        Returns:
            是否成功加载
        """
        load_path = (self.vector_store_path / index_name).resolve()
        if not self.index_exists(index_name):
            logger.warning(f"索引目录不存在: {load_path}")
            return False

        try:
            self.vector_store = FAISS.load_local(
                str(load_path),
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            logger.info(f"FAISS 索引已从 {load_path} 加载。")

            # 加载 BM25 索引（可选，不影响主流程）
            bm25_path = load_path / "bm25.pkl"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    self._bm25_retriever = pickle.load(f)
                logger.info(f"BM25 索引已从 {bm25_path} 加载。")
            else:
                logger.info("未找到 BM25 索引文件，仅使用 Dense 检索。")

            return True
        except Exception as e:
            logger.error(f"加载 FAISS 索引失败: {e}")
            return False

    def index_exists(self, index_name: str = "faiss_index") -> bool:
        """检查指定名称的索引文件是否存在。注意索引名称是文件夹，下面还有文件"""
        base_path = (self.vector_store_path / index_name).resolve()
        # print(base_path)
        return (
            (base_path / "index.faiss").exists() and
            (base_path / "index.pkl").exists()
        )

    # -------------------- 检索与生成 --------------------
    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        """
        执行语义检索，返回最相关的文档块。

        Args:
            query: 用户查询
            k: 返回的文档块数量

        Returns:
            相关 Document 列表
        """
        if self.vector_store is None:
            raise ValueError("向量索引尚未初始化，请先构建或加载索引。")

        docs = self.vector_store.similarity_search(query, k=k)
        logger.info(f"检索完成，查询: '{query[:50]}...' 返回 {len(docs)} 个结果。")
        return docs

    def retrieve_hybrid(
        self,
        query: str,
        k: int = 4,
        dense_weight: float = 0.6,
        dense_k: int = 20,
        bm25_k: int = 20,
    ) -> List[Document]:
        """
        混合检索：Dense (FAISS) + Sparse (BM25) 加权融合。

        使用 Reciprocal Rank Fusion (RRF) 合并两路结果，
        再按融合分数取 top-k。

        Args:
            query: 用户查询
            k: 最终返回的文档块数量
            dense_weight: Dense 检索权重（0~1），BM25 权重 = 1 - dense_weight
            dense_k: Dense 检索的宽召回数量
            bm25_k: BM25 检索的宽召回数量

        Returns:
            按相关性排序的 Document 列表
        """
        if self.vector_store is None:
            raise ValueError("向量索引尚未初始化，请先构建或加载索引。")

        # 1. Dense 检索
        dense_docs = self.vector_store.similarity_search(query, k=dense_k)
        # 2. BM25 检索
        bm25_docs: List[Document] = []
        if self._bm25_retriever is not None:
            bm25_docs = self._bm25_retriever.invoke(query)[:bm25_k]

        # 3. RRF 融合
        # 使用 page_content + chunk_id 作为文档唯一标识
        def _doc_key(doc: Document) -> Tuple[str, int]:
            return (doc.metadata.get("source", ""), doc.metadata.get("chunk_id", -1))

        def _rrf_score(rank: int, k: int = 60) -> float:
            return 1.0 / (k + rank + 1)

        # 为 dense 结果打分
        score_map: Dict[Tuple[str, int], float] = {}
        for rank, doc in enumerate(dense_docs):
            key = _doc_key(doc)
            score_map[key] = score_map.get(key, 0.0) + dense_weight * _rrf_score(rank)

        # 为 BM25 结果打分
        bm25_weight = 1.0 - dense_weight
        for rank, doc in enumerate(bm25_docs):
            key = _doc_key(doc)
            score_map[key] = score_map.get(key, 0.0) + bm25_weight * _rrf_score(rank)

        # 4. 按分数降序排序，取 top-k
        ranked = sorted(score_map.items(), key=lambda x: -x[1])

        # 将 key 映射回原始的 Document 对象（优先保留 dense 版本）
        doc_by_key: Dict[Tuple[str, int], Document] = {}
        for doc in dense_docs:
            doc_by_key[_doc_key(doc)] = doc
        for doc in bm25_docs:
            key = _doc_key(doc)
            if key not in doc_by_key:
                doc_by_key[key] = doc

        result = [doc_by_key[key] for key, _ in ranked[:k]]
        logger.info(
            f"混合检索完成，dense={len(dense_docs)}, bm25={len(bm25_docs)}, "
            f"融合后返回 {len(result)} 个结果。"
        )
        return result

    def rerank(
        self,
        query: str,
        docs: List[Document],
        top_k: int = 4,
    ) -> List[Document]:
        """
        使用 Cross-Encoder 对检索结果重排序。

        采用 BAAI/bge-reranker-v2-m3 模型对(query, doc)逐对打分，
        保留 top_k 个最相关的片段。

        Args:
            query: 用户原始查询
            docs: 待重排序的文档列表（通常来自混合检索的宽召回结果）
            top_k: 最终保留的文档数量

        Returns:
            重排序后的 Document 列表（按相关性降序）
        """
        if not docs:
            return []

        # 懒加载 Reranker 模型
        if self._reranker is None:
            logger.info(f"正在加载 Reranker 模型: {self._reranker_model_name}...")
            try:
                self._reranker = CrossEncoder(
                    self._reranker_model_name,
                    device="cpu",
                )
                logger.info("Reranker 模型加载完成。")
            except Exception as e:
                logger.warning(f"Reranker 模型加载失败: {e}，退回原始排序。")
                return docs[:top_k]

        # 构建 (query, doc_text) 对
        pairs = [(query, doc.page_content) for doc in docs]

        # 打分（CrossEncoder 返回每个 pair 的分数）
        try:
            scores = self._reranker.predict(pairs)
        except Exception as e:
            logger.warning(f"Reranker 打分失败: {e}，退回原始排序。")
            return docs[:top_k]

        # 按分数降序重排
        scored = list(zip(docs, scores))
        scored.sort(key=lambda x: -x[1])

        result = [doc for doc, _ in scored[:top_k]]
        logger.info(
            f"Reranker 重排序完成，输入 {len(docs)} 条，返回 {len(result)} 条。"
        )
        return result

    def format_docs(self, docs: List[Document]) -> str:
        """将检索到的文档块拼接为上下文字符串。"""
        return "\n\n".join(f"[片段 {i+1}]\n{doc.page_content}" for i, doc in enumerate(docs))

    def generate(
        self,
        query: str,
        llm_model: str = "deepseek-chat",
        temperature: float = 0.1,
        k: int = 4,
    ) -> Dict[str, Any]:
        """
        执行完整的 RAG 生成流程：检索 + 增强生成。

        Args:
            query: 用户问题
            llm_model: DeepSeek 模型名称
            temperature: 生成温度
            k: 检索文档数量

        Returns:
            包含答案和检索上下文的字典
        """
        # 1. 检索相关文档
        retrieved_docs = self.retrieve(query, k=k)
        context = self.format_docs(retrieved_docs)

        # 2. 构建 LLM 客户端（每次新建以确保线程安全）
        config = load_config()
        llm = ChatOpenAI(
            model=llm_model,
            api_key=config["deepseek_api_key"],
            base_url=config["deepseek_base_url"],
            temperature=temperature,
        )
        system_template = (
            "你是一位严谨的学术文献助手，仅根据提供的论文片段回答问题。\n"
            "规则：\n"
            "1. 回答中引用的任何信息（标题、作者、数值、结论）必须能在片段中找到出处。\n"
            "2. 如果多个片段提供了矛盾信息，请指出冲突并优先采纳最靠近论文开头的片段。\n"
            "3. 若片段未提供足够信息，直接回答“根据现有片段无法确定”，不要猜测。\n"
        )
        # 3. 定义 Prompt 模板
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_template),
            ("human", """论文片段如下：
---
{context}
---

问题：{question}

请回答：""")
        ])

        # 4. 构建链
        chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | self._parser
        )

        # 5. 执行生成
        logger.info(f"正在调用 DeepSeek 生成回答，模型: {llm_model}")
        answer = chain.invoke({"context": context, "question": query})

        return {
            "query": query,
            "answer": answer,
            "retrieved_documents": retrieved_docs,
            "context": context,
        }

    # -------------------- 便捷流程方法 --------------------
    def process_pdf(self, pdf_path: str, index_name: str = "faiss_index") -> None:
        """
        一站式处理：加载 PDF -> 分块 -> 构建索引 -> 保存索引。

        Args:
            pdf_path: PDF 文件路径
            index_name: 保存的索引名称
        """
        chunks = self.load_and_split_pdf(pdf_path)
        self.build_index(chunks)
        self.save_index(index_name)

    def query_with_existing_index(
        self,
        query: str,
        index_name: str = "faiss_index",
        **generate_kwargs
    ) -> Dict[str, Any]:
        """
        在已保存的索引上执行查询（自动加载索引）。

        Args:
            query: 用户问题
            index_name: 索引名称
            **generate_kwargs: 传递给 generate() 的额外参数

        Returns:
            生成结果字典
        """
        if self.vector_store is None:
            if not self.load_index(index_name):
                raise FileNotFoundError(f"索引 {index_name} 不存在，请先导入 PDF。")
        return self.generate(query, **generate_kwargs)