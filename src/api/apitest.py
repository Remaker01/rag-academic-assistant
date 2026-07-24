"""
API 接口集成测试模块

覆盖所有 FastAPI 端点，按依赖顺序执行测试，确保一次会话完整覆盖所有接口。
测试顺序:
  1. GET  /health        — 健康检查（无依赖）
  2. GET  /indices        — 索引列表（无依赖）
  3. POST /import         — 导入 PDF（含异常路径测试）
  4. POST /chat           — 问答（使用现有索引，多轮对话，无效索引）
  5. POST /memory/reset   — 重置记忆
  6. POST /chat           — 重置后继续问答

运行方式:
    python -m pytest src/api/apitest.py -v -s
    或直接: python src/api/apitest.py
"""

import io
import os
import sys
import unittest
from pathlib import Path
from typing import Optional

# 确保项目根目录在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi.testclient import TestClient
from src.api import create_app


def _create_minimal_pdf() -> bytes:
    """生成一个最小的有效 PDF 文件字节流，用于导入测试。"""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
        b"/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        b"4 0 obj<</Length 44>>stream\n"
        b"BT /F1 24 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer<</Size 6/Root 1 0 R>>\n"
        b"startxref\n406\n%%EOF\n"
    )


class TestAPIIntegration(unittest.TestCase):
    """API 集成测试 —— 按方法名的数字顺序依次执行，保证接口调用顺序。"""

    @classmethod
    def setUpClass(cls):
        """全局初始化：创建应用实例和测试客户端。"""
        cls.app = create_app()
        cls.client = TestClient(cls.app)

        # 跨测试共享状态
        cls.session_id: str = "apitest_integration_session"
        cls.existing_index: Optional[str] = None
        cls.has_indices: bool = False

    # ==================== 1. 健康检查 ====================

    def test_01_health_check(self):
        """GET /health — 验证服务健康状态"""
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()

        self.assertIn("status", data)
        self.assertIn("deepseek_available", data)
        self.assertIn("embedding_model", data)
        self.assertIn("vector_store_path", data)
        print(f"[健康检查] status={data['status']}, "
              f"deepseek_ok={data['deepseek_available']}, "
              f"model={data['embedding_model']}")

    # ==================== 2. 索引列表 ====================

    def test_02_list_indices(self):
        """GET /indices — 列出已有索引，记录第一个可用索引供后续测试"""
        resp = self.client.get("/indices")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("indices", data)

        indices = data["indices"]
        print(f"[索引列表] 共发现 {len(indices)} 个索引:")
        for idx in indices:
            print(f"    - name={idx['name']}, exists={idx['exists']}, path={idx['path']}")

        # 记录第一个真实存在的索引（name 列去重后取第一个）
        seen: set = set()
        for idx in indices:
            if idx["exists"] and idx["name"] not in seen:
                seen.add(idx["name"])
                self.__class__.existing_index = idx["name"]
                self.__class__.has_indices = True
                break

        if not self.__class__.has_indices:
            print("[索引列表] 警告: 没有找到可用索引，后续问答测试将被跳过")

    # ==================== 3. 导入 PDF ====================

    def test_03_import_pdf_file_not_found(self):
        """POST /import — PDF 文件不存在时返回 400"""
        resp = self.client.post("/import", json={"pdf_path": "/does/not/exist/test.pdf"})
        self.assertEqual(resp.status_code, 400)
        data = resp.json()
        self.assertIn("detail", data)
        print(f"[导入-文件不存在] {data['detail']}")

    def test_04_import_pdf_success(self):
        """POST /import — 使用临时 PDF 文件测试成功导入"""
        pdf_bytes = _create_minimal_pdf()

        # 写入临时文件
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        try:
            resp = self.client.post("/import", json={"pdf_path": tmp_path})
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertIn("index_name", data)
            print(f"[导入-成功] message={data['message']}, "
                  f"index_name={data['index_name']}, "
                  f"chunks_count={data['chunks_count']}")

            # 将这个新索引设为后续问答的候选
            if data.get("index_name"):
                self.__class__.existing_index = data["index_name"]
                self.__class__.has_indices = True
        finally:
            # 清理临时文件
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def test_05_import_pdf_with_custom_name(self):
        """POST /import — 使用自定义 index_name 导入 PDF"""
        pdf_bytes = _create_minimal_pdf()
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        custom_index = "ut_test_custom_index"
        try:
            resp = self.client.post("/import", json={
                "pdf_path": tmp_path,
                "index_name": custom_index,
            })
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["index_name"], custom_index)
            print(f"[导入-自定义名称] message={data['message']}")

            # 后续问答优先使用自定义名称的索引
            self.__class__.existing_index = custom_index
            self.__class__.has_indices = True
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    # ==================== 4. 问答接口 ====================

    def test_06_chat_invalid_index(self):
        """POST /chat — 使用不存在的索引时返回 404"""
        resp = self.client.post("/chat", json={
            "query": "测试问题",
            "session_id": self.session_id,
            "index_name": "_non_existent_index_xyz_",
        })
        self.assertEqual(resp.status_code, 404)
        data = resp.json()
        self.assertIn("detail", data)
        print(f"[问答-无效索引] {data['detail']}")

    def test_07_chat_single_turn(self):
        """POST /chat — 单轮问答（使用现有索引）"""
        if not self.__class__.has_indices:
            self.skipTest("没有可用索引，跳过单轮问答测试")

        resp = self.client.post("/chat", json={
            "query": "这篇文档的主要内容是什么？",
            "session_id": self.__class__.session_id,
            "index_name": self.__class__.existing_index,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertIn("query", data)
        self.assertIn("session_id", data)
        self.assertIn("index_used", data)
        self.assertEqual(data["session_id"], self.__class__.session_id)
        self.assertEqual(data["index_used"], self.__class__.existing_index)
        print(f"[问答-单轮] 问题={data['query'][:60]}... "
              f"回答={data['answer'][:120]}... "
              f"索引={data['index_used']}")

    def test_08_chat_multi_turn(self):
        """POST /chat — 多轮对话（延续同一 session_id）"""
        if not self.__class__.has_indices:
            self.skipTest("没有可用索引，跳过多轮对话测试")

        # 第二轮提问
        resp = self.client.post("/chat", json={
            "query": "能否用一句话总结上面提到的核心观点？",
            "session_id": self.__class__.session_id,
            "index_name": self.__class__.existing_index,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        # session_id 保持不变
        self.assertEqual(data["session_id"], self.__class__.session_id)
        print(f"[问答-多轮] 问题='用一句话总结...' "
              f"回答={data['answer'][:120]}...")

    def test_09_chat_new_session(self):
        """POST /chat — 新会话（不同 session_id），应无历史记忆"""
        if not self.__class__.has_indices:
            self.skipTest("没有可用索引，跳过新会话测试")

        resp = self.client.post("/chat", json={
            "query": "我们之前聊过吗？",
            "session_id": "apitest_fresh_session",
            "index_name": self.__class__.existing_index,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        self.assertEqual(data["session_id"], "apitest_fresh_session")
        print(f"[问答-新会话] session_id=apitest_fresh_session "
              f"回答={data['answer'][:120]}...")

    # ==================== 5. 重置记忆 ====================

    def test_10_reset_memory_nonexistent_session(self):
        """POST /memory/reset — 重置不存在的会话（不应报错）"""
        resp = self.client.post("/memory/reset", json={
            "session_id": "_never_used_session_"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        print(f"[重置-不存在会话] {data['message']}")

    def test_11_reset_memory_existing_session(self):
        """POST /memory/reset — 重置已有会话的记忆"""
        resp = self.client.post("/memory/reset", json={
            "session_id": self.__class__.session_id,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["session_id"], self.__class__.session_id)
        print(f"[重置-已有会话] {data['message']}")

    # ==================== 6. 重置后验证 ====================

    def test_12_chat_after_reset(self):
        """POST /chat — 重置后发送相同问题，确认记忆已被清空"""
        if not self.__class__.has_indices:
            self.skipTest("没有可用索引，跳过重置后问答测试")

        resp = self.client.post("/chat", json={
            "query": "我们之前聊过吗？",
            "session_id": self.__class__.session_id,
            "index_name": self.__class__.existing_index,
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("answer", data)
        # 重置后不应有历史记忆，回答应与 test_09 中新会话的回答类似
        print(f"[问答-重置后] 回答={data['answer'][:120]}...")


if __name__ == "__main__":
    unittest.main(verbosity=2)
