import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# 加载项目根目录下的 .env 文件
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
def load_config() -> dict:
    """
    加载环境变量配置并返回字典。
    
    必需项:
        DEEPSEEK_API_KEY: DeepSeek API 密钥
    可选项:
        DEEPSEEK_BASE_URL: API 基础地址，默认为 https://api.deepseek.com/v1
        EMBEDDING_MODEL: sentence-transformers 模型名称，默认为 all-MiniLM-L6-v2
        VECTOR_STORE_PATH: FAISS 索引存储路径，默认为 ./data/vector_store
    """
    config = {
        "deepseek_api_key": os.getenv("DEEPSEEK_API_KEY"),
        "deepseek_base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "embedding_model": os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        "vector_store_path": os.getenv("VECTOR_STORE_PATH", "./data/vector_store"),
    }
    
    # 检查必需配置
    if not config["deepseek_api_key"]:
        raise ValueError(
            "未找到 DEEPSEEK_API_KEY。请在项目根目录创建 .env 文件并设置该变量。\n"
            "可参考 .env.example 文件。"
        )
    
    return config

def setup_logger(name: Optional[str] = __name__, level: int = logging.INFO) -> logging.Logger:
    """配置并返回一个 logger 实例，便于统一日志格式。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger