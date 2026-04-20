## 基于RAG的学术文献智能问答助手

先申请deepseek api key，创建.env文件
```ini
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=./data/vector_store
DEEPSEEK_API_KEY=your_api_key
```

再运行`pip install -r requirements.txt`安装依赖

最后运行main.py