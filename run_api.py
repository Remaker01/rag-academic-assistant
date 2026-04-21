# run_api.py
"""
FastAPI 服务启动入口。
运行方式：python run_api.py
或使用 uvicorn：uvicorn run_api:app --reload --host 0.0.0.0 --port 8000
"""
import uvicorn
from src.api import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "run_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # 开发模式，代码变更自动重启
    )