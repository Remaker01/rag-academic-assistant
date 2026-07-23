#!/usr/bin/env python
"""
RAG 检索质量评估脚本。

流程：
1. 从 data/pdfs/example.pdf 构建/加载索引
2. 加载 data/eval_qa.json 中的 QA 对
3. 分别用三种策略检索，统计命中率
4. 输出对比表格

使用方式：
    python scripts/evaluate.py
"""
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core import get_default_pipeline
from src.utils import setup_logger

logger = setup_logger("evaluate")


def load_qa_pairs(path: str = "data/eval_qa.json") -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class EvalResult:
    name: str
    hits: int = 0
    total: int = 0
    details: list[tuple[str, bool]] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total * 100 if self.total > 0 else 0.0


def main():
    qa_pairs = load_qa_pairs()
    # 只取前 2 条 QA 对作为快速验证（避免完整评估耗时过长）
    qa_pairs = qa_pairs[:2]
    logger.info(f"快速模式：取前 {len(qa_pairs)} 条 QA 对进行验证")

    # 初始化管道并加载索引
    pipeline = get_default_pipeline()
    index_name = "example_pdf"

    if pipeline.vector_store is None:
        if pipeline.index_exists(index_name):
            logger.info(f"加载已有索引: {index_name}")
            pipeline.load_index(index_name)
        else:
            pdf_path = Path("data/pdfs/example.pdf")
            if not pdf_path.exists():
                logger.error(f"PDF 文件不存在: {pdf_path}")
                return
            logger.info(f"构建新索引: {index_name}")
            pipeline.process_pdf(str(pdf_path), index_name)

    # 三种检索策略
    strategies = [
        ("dense", "纯 Dense"),
        ("hybrid_rrf", "Hybrid + RRF"),
        ("hybrid_rerank", "Hybrid + RRF + Reranker"),
    ]

    results: dict[str, EvalResult] = {}

    for key, name in strategies:
        r = EvalResult(name=name)
        logger.info(f"\n{'='*50}")
        logger.info(f"策略: {name}")
        logger.info(f"{'='*50}")

        for i, qa in enumerate(qa_pairs):
            question = qa["question"]
            gt = qa["ground_truth"]

            if key == "dense":
                docs = pipeline.retrieve(question, k=4)
            elif key == "hybrid_rrf":
                docs = pipeline.retrieve_hybrid(question, k=4)
            else:
                candidates = pipeline.retrieve_hybrid(question, k=20, dense_k=30, bm25_k=30)
                docs = pipeline.rerank(question, candidates, top_k=4)

            retrieved_texts = [d.page_content for d in docs]
            # 规范化：折叠空白、转小写，忽略换行导致的空格差异
            def _normalize(t: str) -> str:
                import re
                return re.sub(r"\s+", " ", t).lower().strip()
            norm_gt = _normalize(gt)
            # 拆分为多个关键词，命中任意一个即算命中（宽松匹配）
            gt_keywords = [kw.strip() for kw in norm_gt.split() if len(kw.strip()) > 1]
            norm_texts = [_normalize(t) for t in retrieved_texts]
            is_hit = any(
                any(kw in nt for nt in norm_texts)
                for kw in gt_keywords
            )

            if is_hit:
                r.hits += 1
            r.total += 1
            r.details.append((question, is_hit))

            status = "✓" if is_hit else "✗"
            logger.info(f"  [{i+1}/{len(qa_pairs)}] {status} {question[:40]}...")

        results[key] = r
        logger.info(f"\n  >> {name}: {r.hits}/{r.total} = {r.hit_rate:.1f}%")

    # 汇总对比表
    print("\n" + "=" * 55)
    print("📊 检索策略对比汇总")
    print("=" * 55)
    print(f"{'策略':<25} {'命中':<8} {'命中率':<10}")
    print("-" * 55)
    for key, name in strategies:
        r = results[key]
        bar = "█" * int(r.hit_rate / 10) + "░" * (10 - int(r.hit_rate / 10))
        print(f"{name:<25} {r.hits}/{r.total:<4} {r.hit_rate:>5.1f}%  {bar}")
    print("=" * 55)
    print("注：命中率基于 ground_truth 关键词在检索片段中是否出现（忽略空格差异）")


if __name__ == "__main__":
    main()
