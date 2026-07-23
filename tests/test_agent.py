"""
Agent 单元测试：初始化、对话运行、会话隔离、记忆重置。

策略说明：
- create_agent 返回一个简单的 Lambda Runnable，不走真实 LangGraph
- 所有 LLM / 检索调用通过 mock 拦截，不消耗真实 token
- 主要测试 AgentManager 的编排逻辑而非 Agent 内部行为
"""
from typing import Any, Dict, Generator
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableLambda


def _make_fake_agent_output(output_text: str) -> Dict[str, Any]:
    """构造一个模拟 create_agent 返回值的 Runnable。

    模拟 AgentState 结构中包含 messages 列表的最新 AI 回复。
    """
    def _invoke(inputs: dict) -> dict:
        return {"messages": [AIMessage(content=output_text)]}
    return RunnableLambda(_invoke)


@pytest.fixture(autouse=True)
def _mock_agent_and_tool() -> Generator:
    """
    将 create_agent 和 create_retriever_tool 替换为 mock，
    使 AgentManager 初始化不依赖真实 Graph / FAISS / LLM。
    """
    with patch("src.agent.agent.create_agent", return_value=_make_fake_agent_output("模拟回答")), \
         patch("src.agent.agent.create_retriever_tool") as mock_tool:
        from langchain_core.tools import tool

        @tool
        def fake_retriever(query: str) -> str:
            """mock 检索工具"""
            return f"模拟检索结果: {query}"

        mock_tool.return_value = fake_retriever
        yield


@pytest.fixture
def agent_manager() -> Generator:
    """返回一个完全 Mock 底层调用的 AgentManager 实例。"""
    with patch("src.agent.agent.ChatOpenAI"):
        from src.agent.agent import AgentManager

        mgr = AgentManager(verbose=False)
        yield mgr


class TestAgentInit:
    """验证 AgentManager 初始化参数正确传递。"""

    def test_default_parameters(self) -> None:
        """默认参数能创建实例。"""
        with patch("src.agent.agent.create_retriever_tool") as mock_tool:
            with patch("src.agent.agent.ChatOpenAI"):
                from src.agent.agent import AgentManager

                mgr = AgentManager(verbose=False)
                assert mgr.max_history_tokens == 4000
                assert mgr.keep_recent_messages == 10

    def test_custom_parameters(self) -> None:
        """自定义参数被正确设置。"""
        with patch("src.agent.agent.create_retriever_tool") as mock_tool:
            with patch("src.agent.agent.ChatOpenAI"):
                from src.agent.agent import AgentManager

                mgr = AgentManager(
                    index_name="custom_index",
                    llm_model="deepseek-chat",
                    temperature=0.5,
                    max_history_tokens=2000,
                    keep_recent_messages=5,
                    verbose=False,
                )
                assert mgr.index_name == "custom_index"
                assert mgr.max_history_tokens == 2000
                assert mgr.keep_recent_messages == 5

    def test_tools_created(self, agent_manager) -> None:
        """工具列表包含检索工具。"""
        assert len(agent_manager.tools) == 1


class TestAgentRun:
    """验证 AgentManager.run() 的输入输出行为。"""

    def test_run_returns_output_dict(self, agent_manager) -> None:
        """run() 返回包含 'output' 的字典。"""
        result = agent_manager.run("机器学习是什么？")
        assert isinstance(result, dict)
        assert "output" in result
        assert result["output"]  # 非空

    def test_run_no_error_for_valid_input(self, agent_manager) -> None:
        """有效输入下 'error' 键不存在。"""
        result = agent_manager.run("深度学习")
        assert "error" not in result or result["error"] is None

    def test_run_with_custom_session(self, agent_manager) -> None:
        """传入 session_id 正常执行。"""
        result = agent_manager.run("神经网络", session_id="session_xyz")
        assert "output" in result

    def test_multiple_runs_same_session(self, agent_manager) -> None:
        """同一 session 连续两次调用不报错。"""
        agent_manager.run("第一个问题")
        result = agent_manager.run("第二个问题")
        assert "output" in result


class TestAgentMemory:
    """验证会话历史的读写与重置。"""

    def test_memory_stores_messages(self, agent_manager) -> None:
        """run() 后会话历史中新增了 HumanMessage 和 AIMessage。"""
        agent_manager.run("测试问题")
        history = agent_manager._store["default"]
        msg_types = [type(m).__name__ for m in history.messages]
        assert "HumanMessage" in msg_types
        assert "AIMessage" in msg_types

    def test_reset_memory_clears_history(self, agent_manager) -> None:
        """reset_memory 后消息列表为空。"""
        agent_manager.run("问题一")
        agent_manager.run("问题二")
        assert len(agent_manager._store["default"].messages) > 0
        agent_manager.reset_memory("default")
        assert len(agent_manager._store["default"].messages) == 0

    def test_reset_nonexistent_session(self, agent_manager) -> None:
        """重置不存在的 session 不抛出异常。"""
        agent_manager.reset_memory("__never_used__")

    def test_memory_content(self, agent_manager) -> None:
        """存储的消息内容与输入一致。"""
        agent_manager.run("今天天气如何")
        history = agent_manager._store["default"]
        assert any("今天天气如何" in str(m) for m in history.messages)


class TestMultiSession:
    """验证多个 session 之间的隔离性。"""

    def test_sessions_isolated(self, agent_manager) -> None:
        """不同 session 的历史互不影响。"""
        agent_manager.run("会话A的问题", session_id="session_a")
        agent_manager.run("会话B的问题", session_id="session_b")
        # 验证历史只有对应的消息
        hist_a = str(agent_manager._store["session_a"].messages)
        hist_b = str(agent_manager._store["session_b"].messages)
        assert "会话A" in hist_a
        assert "会话B" in hist_b
        assert "会话A" not in hist_b
        assert "会话B" not in hist_a

    def test_default_session_exists(self, agent_manager) -> None:
        """不指定 session_id 时使用 'default' 作为默认值。"""
        agent_manager.run("默认会话")
        assert "default" in agent_manager._store
