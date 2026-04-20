import sys
from pathlib import Path

from src.core import get_default_pipeline
from src.agent import AgentManager
from src.utils import setup_logger

logger = setup_logger("main_agent_test")

def ensure_index_ready():
    """确保 PDF 索引已准备好。"""
    pdf_path = Path("data/pdfs/example.pdf")
    if not pdf_path.exists():
        logger.error(f"测试 PDF 文件不存在: {pdf_path.absolute()}")
        sys.exit(1)
    
    rag = get_default_pipeline()
    index_name = "example_paper"
    
    if not rag.index_exists(index_name):
        logger.info("未找到索引，开始处理 PDF...")
        rag.process_pdf(str(pdf_path), index_name)
    else:
        logger.info("索引已存在，直接加载。")
        rag.load_index(index_name)


def run_interactive_agent():
    """运行交互式 Agent 会话（命令行）。"""
    print("\n" + "="*60)
    print("基于 RAG 的学术文献智能问答助手 - Agent 测试模式")
    print("="*60)
    print("命令说明：")
    print("  - 直接输入问题与助手对话")
    print("  - 输入 /reset 清空对话记忆")
    print("  - 输入 /quit 退出程序")
    print("="*60 + "\n")
    
    # 创建 Agent 管理器（verbose=True 可以看到思考过程）
    agent_manager = AgentManager(
        index_name="example_paper",
        verbose=True,  # 控制台输出详细的 Agent 思考链
    )
    
    while True:
        try:
            user_input = input("\n🧑 您: ").strip()
        except EOFError:
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ("/quit", "/exit"):
            print("再见！")
            break
        
        if user_input.lower() == "/reset":
            agent_manager.reset_memory()
            print("✅ 对话记忆已清空，开始新的对话。")
            continue
        
        # 执行 Agent
        print("\n🤖 助手思考中...\n")
        result = agent_manager.run(user_input)
        
        # 输出最终答案
        print(f"\n🤖 助手: {result['output']}")
if __name__ == "__main__":
    # 确保索引准备就绪
    ensure_index_ready()
    
    # 启动交互式 Agent
    run_interactive_agent()