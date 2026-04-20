# main.py
"""
第一阶段测试脚本：验证 DeepSeek API 和本地嵌入模型是否正常工作。
运行命令：python main.py
"""

from src.utils import load_config, setup_logger

logger = setup_logger("phase1_test")

def test_deepseek_api(config: dict):
    """测试 DeepSeek API 连通性（兼容 OpenAI 接口）。"""
    try:
        from openai import OpenAI
    except ImportError:
        logger.error("请先安装 openai 库：pip install openai")
        return False

    client = OpenAI(
        api_key=config["deepseek_api_key"],
        base_url=config["deepseek_base_url"]
    )
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",  # DeepSeek 官方模型名
            messages=[
                {"role": "user", "content": "请回复一个简短的英文单词 以确认连通。"}
            ],
            max_tokens=10,
            temperature=0.0,
        )
        reply = response.choices[0].message.content
        logger.info(f"✅ DeepSeek API 测试成功！回复内容: {reply}")
        return True
    except Exception as e:
        logger.error(f"❌ DeepSeek API 测试失败: {e}")
        logger.error("请检查：1) API Key 是否正确 2) 网络是否可以访问 api.deepseek.com")
        return False

def test_embedding_model(model_name: str):
    """测试 sentence-transformers 嵌入模型加载。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("请先安装 sentence-transformers")
        return False

    try:
        logger.info(f"正在加载嵌入模型: {model_name} ...")
        model = SentenceTransformer(model_name, cache_folder='D:/sentence-transformer')
        # 对测试句子生成嵌入向量
        test_sentence = "This is a test sentence for embedding."
        embedding = model.encode(test_sentence)
        logger.info(f"✅ 嵌入模型加载成功！向量维度: {len(embedding)}")
        return True
    except Exception as e:
        logger.error(f"❌ 嵌入模型加载失败: {e}")
        logger.error("可能原因：1) 网络问题导致模型下载失败 2) 模型名称错误")
        return False

def main():
    logger.info("=" * 50)
    logger.info("第一阶段：环境连通性测试")
    logger.info("=" * 50)
    
    # 1. 加载配置
    try:
        config = load_config()
        logger.info(f"配置加载成功。API Base URL: {config['deepseek_base_url']}")
        logger.info(f"嵌入模型: {config['embedding_model']}")
    except ValueError as e:
        logger.error(str(e))
        return False
    
    # 2. 测试 DeepSeek API
    if not test_deepseek_api(config):
        logger.error("DeepSeek API 测试未通过，请修复后重试。")
        return False
    
    # 3. 测试嵌入模型
    if not test_embedding_model(config["embedding_model"]):
        logger.error("嵌入模型测试未通过，请修复后重试。")
        return False
    
    logger.info("=" * 50)
    logger.info("🎉 第一阶段测试全部通过！可以进入下一阶段开发。")
    logger.info("=" * 50)
    return True

if __name__ == "__main__":
    if not main():
        logger.error("测试出错！请检查日志。")