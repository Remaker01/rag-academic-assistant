# src/agent/agent.py
"""
Agent 模块 (LangChain 1.0)：基于 create_agent 和中间件构建带记忆的智能体。
"""
from typing import List, Dict, Any

# LangChain 1.0 核心组件
from langchain.agents import create_agent, AgentState
from langchain.agents.middleware import SummarizationMiddleware
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import Runnable

# 社区集成（模型、历史存储）
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_openai import ChatOpenAI

# 本地模块
from ..core import get_default_pipeline
from ..utils import load_config, setup_logger

logger = setup_logger("agent")


def create_retriever_tool(index_name: str = "faiss_index"):
    """
    创建 FAISS 检索工具。
    """
    pipeline = get_default_pipeline()

    if pipeline.vector_store is None:
        if not pipeline.load_index(index_name):
            raise ValueError(f"无法加载索引 '{index_name}'，请先导入 PDF 并构建索引。")

    # 使用 @tool 装饰器定义工具函数 [reference:2]
    @tool
    def document_retriever(query: str) -> str:
        """
        检索已导入的学术论文内容。
        当用户询问论文中的具体信息、概念、方法、数据或结论时使用。
        
        Args:
            query: 要检索的查询问题，例如："论文的创新点是什么"
        
        Returns:
            从论文中检索到的相关文本片段
        """
        docs = pipeline.retrieve(query, k=4)
        if not docs:
            return "未找到相关文档片段。"

        formatted = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            content = doc.page_content.strip()
            formatted.append(f"[片段 {i} - 来源: {source}]\n{content}")

        return "\n\n---\n\n".join(formatted)

    return document_retriever


def create_llm(
    model: str = "deepseek-chat",
    temperature: float = 0.1,
) -> ChatOpenAI:
    """创建 DeepSeek LLM 客户端。"""
    config = load_config()
    return ChatOpenAI(
        model=model,
        api_key=config["deepseek_api_key"],
        base_url=config["deepseek_base_url"],
        temperature=temperature,
    )


class AgentManager:
    """
    LangChain 1.0 Agent 管理器：基于 create_agent + SummarizationMiddleware 实现记忆管理。
    """

    def __init__(
        self,
        index_name: str = "faiss_index",
        llm_model: str = "deepseek-chat",
        temperature: float = 0.1,
        max_history_tokens: int = 4000,      # 对话历史最大 token 数
        keep_recent_messages: int = 10,      # 保留最近 N 条消息（兜底）
        verbose: bool = True,
    ):
        self.index_name = index_name
        self.verbose = verbose
        self.max_history_tokens = max_history_tokens
        self.keep_recent_messages = keep_recent_messages

        # 创建检索工具
        self.tools = [create_retriever_tool(index_name)]

        # 创建 LLM
        self.llm = create_llm(model=llm_model, temperature=temperature)

        # 会话历史存储（内存实现，可按需换为持久化后端）
        self._store: Dict[str, ChatMessageHistory] = {}

        # 系统提示词
        self._system_prompt = (
            "你是一位专业的学术文献助手。请基于提供的对话历史和工具检索结果回答问题。"
            "如果无法从工具结果中找到答案，请如实告知。"
        )

        # 创建基础 Agent（LangChain 1.0 核心 API）
        # create_agent 底层已基于 LangGraph 实现，自带状态管理、错误处理和流式输出[reference:2]
        # self._base_agent = create_agent(
        #     model=self.llm,
        #     tools=self.tools,
        #     system_prompt=self._system_prompt,
        # )

        # 为每个会话单独包装 Agent（注入历史 + 自动摘要中间件）
        self._wrapped_agents: Dict[str, Runnable] = {}

        logger.info("AgentManager 初始化完成")

    def _get_or_create_wrapped_agent(self, session_id: str) -> Runnable:
        """获取或创建带中间件的 Agent 实例。"""
        if session_id not in self._wrapped_agents:
            # 获取会话的历史存储
            history = self._get_session_history(session_id)

            # 使用 SummarizationMiddleware 自动管理对话历史长度
            # 当历史 token 超过阈值时自动压缩，避免超出模型上下文窗口[reference:3]
            middleware = [
                SummarizationMiddleware(
                    model=self.llm,
                    max_input_tokens=self.max_history_tokens,
                    keep_recent_messages=self.keep_recent_messages,
                )
            ]

            # 创建带中间件的 Agent
            wrapped = create_agent(
                model=self.llm,
                tools=self.tools,
                system_prompt=self._system_prompt,  # 复用基础提示词
                middleware=middleware,
            )

            # 注入历史记录（通过预填充 messages 实现）
            # 注意：LangChain 1.0 不再需要 RunnableWithMessageHistory，历史直接通过 messages 参数传入
            self._wrapped_agents[session_id] = wrapped
            self._store[session_id] = history

        return self._wrapped_agents[session_id]

    def _get_session_history(self, session_id: str) -> ChatMessageHistory:
        """获取或创建会话历史存储。"""
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]

    def run(self, user_input: str, session_id: str = "default") -> Dict[str, Any]:
        """
        执行一次用户交互，返回 Agent 的响应。
        """
        logger.info(f"用户输入 (session: {session_id}): {user_input[:100]}...")
        try:
            agent = self._get_or_create_wrapped_agent(session_id)
            history = self._get_session_history(session_id)

            # 构造输入消息：历史消息 + 当前用户输入
            input_messages = history.messages + [HumanMessage(content=user_input)]

            # 调用 Agent（LangChain 1.0 返回 AgentState，包含完整 messages）
            result_state: AgentState = agent.invoke({"messages": input_messages})

            # 提取最新 AI 回复
            ai_messages = [msg for msg in result_state["messages"] if isinstance(msg, AIMessage)]
            if ai_messages:
                latest_ai_message = ai_messages[-1]
                output = latest_ai_message.content
            else:
                output = "（未获取到有效回复）"

            # 更新持久化历史
            history.add_user_message(user_input)
            history.add_ai_message(output)

            logger.info(f"Agent 响应: {output[:100]}...")
            return {"output": output}
        except Exception as e:
            logger.error(f"Agent 执行出错: {e}", exc_info=True)
            return {"output": f"抱歉，处理您的请求时出现错误: {str(e)}", "error": str(e)}

    def reset_memory(self, session_id: str = "default") -> None:
        """清空特定会话的对话记忆。"""
        if session_id in self._store:
            self._store[session_id].clear()
            logger.info(f"会话 '{session_id}' 的对话记忆已清空。")
        else:
            logger.info(f"会话 '{session_id}' 不存在，无需清空。")

    def get_chat_history(self, session_id: str = "default") -> List[Any]:
        """获取特定会话的对话历史。"""
        return self._get_session_history(session_id).messages