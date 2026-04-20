# main.py
"""
RAGPipeline 测试脚本：处理 data/pdfs/example.pdf 并执行一次问答。
"""
import sys
from pathlib import Path

from src.core import get_default_pipeline
from src.utils import setup_logger

logger = setup_logger("main_test")

def main():
    # PDF 文件路径
    pdf_path = Path("data/pdfs/example.pdf")
    if not pdf_path.exists():
        logger.error(f"测试 PDF 文件不存在: {pdf_path.absolute()}")
        logger.error("请将一篇示例论文放入 data/pdfs/example.pdf 后重试。")
        return False

    # 获取默认管道实例（惰性加载，首次调用时才初始化模型）
    logger.info("正在初始化 RAG 管道...")
    rag = get_default_pipeline()

    # 索引名称（与 PDF 文件对应）
    index_name = "example_pdf"

    # 判断是否需要构建新索引
    if rag.index_exists(index_name):
        logger.info(f"检测到已有索引 '{index_name}'，正在加载...")
        if not rag.load_index(index_name):
            logger.error("索引加载失败，将重新构建。")
            rag.process_pdf(str(pdf_path), index_name)
    else:
        logger.info(f"未找到索引 '{index_name}'，开始处理 PDF 并构建索引...")
        rag.process_pdf(str(pdf_path), index_name)

    # 执行一次问答测试
    test_query = "这篇论文的主要研究内容是什么？"
    logger.info(f"\n{'='*50}")
    logger.info(f"测试查询: {test_query}")
    logger.info(f"{'='*50}")

    try:
        result = rag.generate(query=test_query,llm_model='deepseek-chat', k=3)

        print("\n" + "="*60)
        print("【检索到的文档片段】")
        print("="*60)
        for i, doc in enumerate(result["retrieved_documents"], 1):
            print(f"\n--- 片段 {i} (来源: {doc.metadata.get('source', 'unknown')}) ---")
            print(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)

        print("\n" + "="*60)
        print("【DeepSeek 生成的回答】")
        print("="*60)
        print(result["answer"])
        print("="*60)
    except Exception as e:
        logger.error(f"❌ {e}", exc_info=True)
        return False
    return True


if __name__ == "__main__":
    if not main():
        logger.error("测试出错，请检查日志！")
    else:
        logger.info("测试成功！")