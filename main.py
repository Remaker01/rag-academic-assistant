"""
应用程序入口：初始化 PySide6 应用并启动主窗口。
"""
import sys
from PySide6.QtWidgets import QApplication
from src.gui.main_window import MainWindow
from src.utils import setup_logger

logger = setup_logger("main")

if __name__ == "__main__":
    # 检查环境配置（可选）
    try:
        from src.utils import load_config
        config = load_config()
        logger.info(f"配置加载成功，API Base URL: {config['deepseek_base_url']}")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        # 不退出，GUI 中可以显示错误
    
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    logger.info("应用程序已启动。")
    sys.exit(app.exec())