"""
API 集成测试：FastAPI 端点正确性验证。

策略：
- 使用 fastapi.testclient.TestClient 发送 HTTP 请求
- 用 patch 替换 _service 单例，让 create_app() 在闭包中捕获 mock service
- 不启动真实 uvicorn 服务器
"""
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


def _make_mock_service() -> MagicMock:
    """创建一个预配置好的 mock APIService。"""
    svc = MagicMock()
    svc.health_check.return_value = {
        "status": "healthy",
        "deepseek_available": True,
        "embedding_model": "test-model",
        "vector_store_path": "/tmp/test",
    }
    svc.list_indices.return_value = {
        "indices": [{"name": "paper1", "exists": True, "path": "/tmp/paper1"}],
    }
    svc.chat.return_value = {
        "query": "test",
        "answer": "ok",
        "session_id": "default",
        "index_used": "test_index",
        "error": None,
    }
    svc.reset_memory.return_value = {
        "success": True,
        "message": "ok",
        "session_id": "default",
    }
    svc.import_pdf.return_value = {
        "success": True,
        "message": "mock 导入成功",
        "index_name": "mock_paper",
        "chunks_count": 5,
    }
    return svc


@pytest.fixture
def mock_client() -> TestClient:
    """返回一个 route handlers 使用 mock service 的 TestClient。"""
    mock_svc = _make_mock_service()
    # _service 是模块级私有的单例，get_service() 会返回它
    # 替换它使 create_app() 捕获 mock service
    with patch("src.api.routes._service", mock_svc):
        from src.api import create_app

        app = create_app()
        return TestClient(app)


@pytest.fixture
def client() -> TestClient:
    """返回真实 service 的 TestClient（用于不需要 mock 的测试）。"""
    from src.api import create_app

    app = create_app()
    return TestClient(app)


class TestHealth:
    """GET /health 端点测试。"""

    def test_health_returns_200(self, mock_client: TestClient) -> None:
        """健康检查返回 200。"""
        response = mock_client.get("/health")
        assert response.status_code == 200

    def test_health_body_keys(self, mock_client: TestClient) -> None:
        """响应包含预期字段。"""
        response = mock_client.get("/health")
        data = response.json()
        assert "status" in data
        assert "deepseek_available" in data
        assert "embedding_model" in data
        assert "vector_store_path" in data

    def test_health_status_value(self, mock_client: TestClient) -> None:
        """status 值为 healthy。"""
        response = mock_client.get("/health")
        assert response.json()["status"] == "healthy"


class TestIndices:
    """GET /indices 端点测试。"""

    def test_indices_returns_200(self, mock_client: TestClient) -> None:
        """索引列表返回 200。"""
        response = mock_client.get("/indices")
        assert response.status_code == 200

    def test_indices_returns_list(self, mock_client: TestClient) -> None:
        """响应包含 indices 数组。"""
        response = mock_client.get("/indices")
        data = response.json()
        assert "indices" in data

    def test_indices_with_items(self, mock_client: TestClient) -> None:
        """有索引时返回正确的结构。"""
        response = mock_client.get("/indices")
        items = response.json()["indices"]
        assert len(items) == 1
        assert items[0]["name"] == "paper1"


class TestChat:
    """POST /chat 端点测试。"""

    def test_chat_returns_200(self, mock_client: TestClient) -> None:
        """正常问答返回 200。"""
        response = mock_client.post(
            "/chat",
            json={"query": "什么是机器学习", "session_id": "test_session"},
        )
        assert response.status_code == 200

    def test_chat_response_schema(self, mock_client: TestClient) -> None:
        """返回结构包含所有预期字段。"""
        response = mock_client.post("/chat", json={"query": "hello"})
        data = response.json()
        assert "query" in data
        assert "answer" in data
        assert "session_id" in data
        assert "index_used" in data

    def test_chat_with_index_name(self, mock_client: TestClient) -> None:
        """传入 index_name 参数不报错。"""
        response = mock_client.post(
            "/chat",
            json={"query": "test", "index_name": "custom_paper"},
        )
        assert response.status_code == 200

    def test_chat_empty_query_fails(self, client: TestClient) -> None:
        """空 query 返回 422（Pydantic 校验）。"""
        response = client.post("/chat", json={"query": ""})
        assert response.status_code == 422


class TestPDFImport:
    """POST /import 端点测试。"""

    def test_import_nonexistent_pdf(self, client: TestClient) -> None:
        """不存在的 PDF 通过真实 service 校验返回 400。"""
        response = client.post(
            "/import",
            json={"pdf_path": "/nonexistent/test.pdf"},
        )
        assert response.status_code == 400

    def test_import_mocked_service(self, mock_client: TestClient) -> None:
        """mock service 验证返回结构。"""
        response = mock_client.post(
            "/import",
            json={"pdf_path": "C:\\fake\\paper.pdf"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True


class TestMemoryReset:
    """POST /memory/reset 端点测试。"""

    def test_reset_returns_200(self, mock_client: TestClient) -> None:
        """重置记忆返回 200。"""
        response = mock_client.post(
            "/memory/reset",
            json={"session_id": "default"},
        )
        assert response.status_code == 200

    def test_reset_response_schema(self, mock_client: TestClient) -> None:
        """响应包含 success / message / session_id。"""
        response = mock_client.post(
            "/memory/reset",
            json={"session_id": "default"},
        )
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert data["session_id"] == "default"
