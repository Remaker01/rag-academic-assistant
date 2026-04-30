# src/gui/main_window.py
"""
学术文献智能问答助手 - PySide6 前端界面（完整版）
包含 PDF 导入、对话展示（支持 Markdown）、输入区、调用日志面板，并已对接后台业务逻辑。
"""
import sys
from pathlib import Path
from typing import Optional

from src.gui.uiutils import render_markdown
from PySide6.QtCore import Qt, QThread, Signal as QtSignal, QObject
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QLabel, QScrollArea,
    QFrame, QSplitter, QFileDialog, QMessageBox, QSizePolicy
)

from src.agent import AgentManager
from src.core import get_default_pipeline
from src.utils import setup_logger
from src import __version__

logger = setup_logger("gui")

# ==================== 工作线程（保持不变） ====================
class PDFImportWorker(QObject):
    started = QtSignal()
    finished = QtSignal(bool, str)
    progress = QtSignal(str)

    def __init__(self, pdf_path: str, index_name: str = "default"):
        super().__init__()
        self.pdf_path = pdf_path
        self.index_name = index_name

    def run(self):
        self.started.emit()
        self.progress.emit(f"开始处理 PDF: {Path(self.pdf_path).name}")
        try:
            pipeline = get_default_pipeline()
            pipeline.process_pdf(self.pdf_path, self.index_name)
            self.progress.emit("PDF 处理完成，索引已保存。")
            self.finished.emit(True, f"成功导入 {Path(self.pdf_path).name}")
        except Exception as e:
            logger.error(f"PDF 处理失败: {e}", exc_info=True)
            self.progress.emit(f"处理失败: {str(e)}")
            self.finished.emit(False, str(e))


class AgentQueryWorker(QObject):
    started = QtSignal()
    finished = QtSignal(dict)
    log_message = QtSignal(str, str)

    def __init__(self, agent_manager: AgentManager, query: str, session_id: str = "default"):
        super().__init__()
        self.agent_manager = agent_manager
        self.query = query
        self.session_id = session_id

    def run(self):
        self.started.emit()
        self.log_message.emit(f"用户问题: {self.query}", "INFO")
        try:
            result = self.agent_manager.run(self.query, self.session_id)
            self.log_message.emit("Agent 响应完成", "INFO")
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Agent 执行失败: {e}", exc_info=True)
            self.log_message.emit(f"执行出错: {str(e)}", "ERROR")
            self.finished.emit({"output": f"抱歉，处理您的请求时出现错误: {str(e)}", "error": str(e)})


# ==================== UI 组件（部分修改） ====================
class LogPanel(QTextEdit):
    # 保持不变
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setPlaceholderText("调用链路日志将在此显示...")
        self.setMinimumWidth(250)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                padding: 8px;
            }
        """)

    def append_log(self, message: str, level: str = "INFO"):
        color_map = {
            "INFO": "#4ec9b0",
            "WARN": "#dcdcaa",
            "ERROR": "#f14c4c",
            "TOOL": "#ce9178",
        }
        color = color_map.get(level, "#d4d4d4")
        formatted = f'<span style="color:{color};">[{level}] {message}</span>'
        self.append(formatted)
        self.moveCursor(QTextCursor.End)


class ChatBubble(QFrame):
    """支持 Markdown 渲染的气泡控件"""
    def __init__(self, text: str, is_user: bool, use_markdown: bool = False, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setContentsMargins(10, 5, 10, 5)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.label = QLabel()
        self.label.setWordWrap(True)
        self.label.setTextFormat(Qt.TextFormat.RichText)  # 允许富文本

        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.label.setCursor(Qt.CursorShape.IBeamCursor)  # 可选：鼠标悬停显示文本光标
        
        # 根据是否用户消息或是否启用 markdown 决定文本处理
        if is_user:
            # 用户消息不渲染 Markdown，直接显示纯文本（但转义 HTML）
            display_text = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            self.label.setText(f"<b>您：</b><br>{display_text}")
        else:
            if use_markdown:
                # 渲染 Markdown 为 HTML
                html_content = render_markdown(text)
                self.label.setText(html_content)
            else:
                # 普通文本（例如“正在思考...”）
                display_text = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                self.label.setText(f"<b>🤖 助手：</b><br>{display_text}")
        
        self.label.setStyleSheet(f"""
            QLabel {{
                background-color: { "#d0e6ff" if is_user else "#f0f0f0" };
                color: #000000;
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 13px;
            }}
        """)

        if is_user:
            layout.addStretch()
            layout.addWidget(self.label)
        else:
            layout.addWidget(self.label)
            layout.addStretch()

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
    
    def update_text(self, text: str, use_markdown: bool = True):
        """更新气泡内容（用于将“正在思考”替换为真实回答）"""
        if use_markdown:
            html_content = render_markdown(text)
            self.label.setText(html_content)
        else:
            display_text = text.replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            self.label.setText(f"<b>🤖 助手：</b><br>{display_text}")


class ChatArea(QScrollArea):
    """对话展示区域，管理气泡列表，支持临时占位气泡"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        self.setWidget(container)

        self.layout = QVBoxLayout(container)
        self.layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.layout.setSpacing(10)
        self.layout.setContentsMargins(15, 15, 15, 15)

        # 存储临时占位气泡的引用（用于后续更新）
        self.placeholder_bubble: Optional[ChatBubble] = None

        self.add_system_message("欢迎使用学术文献智能问答助手！请先导入一篇 PDF 论文。")

    def add_user_message(self, text: str):
        bubble = ChatBubble(text, is_user=True)
        self.layout.addWidget(bubble)
        self.scroll_to_bottom()

    def add_ai_message(self, text: str, use_markdown: bool = True):
        """添加 AI 回答气泡（支持 Markdown）"""
        bubble = ChatBubble(text, is_user=False, use_markdown=use_markdown)
        self.layout.addWidget(bubble)
        self.scroll_to_bottom()
        return bubble

    def add_placeholder_message(self, text: str = "正在思考，请稍候..."):
        """添加临时占位气泡（不渲染 Markdown），并返回引用"""
        bubble = ChatBubble(text, is_user=False, use_markdown=False)
        self.layout.addWidget(bubble)
        self.scroll_to_bottom()
        self.placeholder_bubble = bubble
        return bubble

    def replace_placeholder(self, final_text: str):
        """用最终回答替换占位气泡的内容"""
        if self.placeholder_bubble:
            self.placeholder_bubble.update_text(final_text, use_markdown=True)
            self.placeholder_bubble = None
        else:
            # 如果没有占位气泡，直接添加新气泡（理论上不会发生）
            self.add_ai_message(final_text)
        self.scroll_to_bottom()

    def add_system_message(self, text: str):
        label = QLabel(f"<i>{text}</i>")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("color: #888888; padding: 8px;")
        self.layout.addWidget(label)
        self.scroll_to_bottom()

    def clear_messages(self):
        while self.layout.count():
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.placeholder_bubble = None
        self.add_system_message("对话已清空。")

    def scroll_to_bottom(self):
        QApplication.processEvents()
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


# PDFImportWidget 和 InputArea 保持不变（略，直接复用前文代码）
class PDFImportWidget(QWidget):
    import_clicked = QtSignal()
    file_selected = QtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.import_btn = QPushButton("📄 导入 PDF")
        self.import_btn.setMinimumHeight(36)
        self.import_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:pressed {
                background-color: #004578;
            }
        """)
        self.import_btn.clicked.connect(self._on_import_clicked)

        self.file_label = QLabel("未导入任何文件")
        self.file_label.setStyleSheet("color: #666666; margin-left: 10px;")

        layout.addWidget(self.import_btn)
        layout.addWidget(self.file_label)
        layout.addStretch()

    def _on_import_clicked(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择学术论文 PDF", "", "PDF 文件 (*.pdf)"
        )
        if file_path:
            self.file_label.setText(Path(file_path).name)
            self.file_label.setToolTip(file_path)
            self.file_selected.emit(file_path)
            self.import_clicked.emit()

    def set_processing(self, is_processing: bool):
        self.import_btn.setEnabled(not is_processing)
        if is_processing:
            self.import_btn.setText("⏳ 处理中...")
        else:
            self.import_btn.setText("📄 导入 PDF")


class InputArea(QWidget):
    send_clicked = QtSignal(str)
    clear_memory_clicked = QtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("输入您的问题...")
        self.input_edit.setMinimumHeight(40)
        self.input_edit.setStyleSheet("""
            QLineEdit {
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #007acc;
            }
        """)
        self.input_edit.returnPressed.connect(self._on_send)

        self.send_btn = QPushButton("发送")
        self.send_btn.setMinimumHeight(40)
        self.send_btn.setMinimumWidth(80)
        self.send_btn.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 6px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
        """)
        self.send_btn.clicked.connect(self._on_send)

        self.clear_btn = QPushButton("🗑 清空记忆")
        self.clear_btn.setMinimumHeight(40)
        self.clear_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #666666;
                border: 1px solid #cccccc;
                border-radius: 6px;
                padding: 0 12px;
            }
            QPushButton:hover {
                background-color: #f0f0f0;
            }
        """)
        self.clear_btn.clicked.connect(self.clear_memory_clicked)

        layout.addWidget(self.input_edit)
        layout.addWidget(self.send_btn)
        layout.addWidget(self.clear_btn)

    def _on_send(self):
        text = self.input_edit.text().strip()
        if text:
            self.send_clicked.emit(text)
            self.input_edit.clear()

    def set_enabled(self, enabled: bool):
        self.input_edit.setEnabled(enabled)
        self.send_btn.setEnabled(enabled)

    def set_placeholder(self, text: str):
        self.input_edit.setPlaceholderText(text)


# ==================== 主窗口（修改了消息发送逻辑） ====================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("学术文献智能问答助手{}".format(f" v{__version__}" if __version__ else ""))
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)

        QApplication.setFont(QFont("Segoe UI", 10))

        self.agent_manager: Optional[AgentManager] = None
        self.current_index_name = "current_paper"
        self.session_id = "default"

        self._import_thread: Optional[QThread] = None
        self._import_worker: Optional[PDFImportWorker] = None
        self._query_thread: Optional[QThread] = None
        self._query_worker: Optional[AgentQueryWorker] = None

        self._setup_ui()
        self._apply_global_style()
        self._connect_signals()

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.pdf_import_widget = PDFImportWidget()
        left_layout.addWidget(self.pdf_import_widget)

        self.chat_area = ChatArea()
        left_layout.addWidget(self.chat_area, stretch=1)

        self.input_area = InputArea()
        left_layout.addWidget(self.input_area)

        self.log_panel = LogPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(self.log_panel)
        splitter.setSizes([800, 300])
        splitter.setHandleWidth(2)

        main_layout.addWidget(splitter)

        self.input_area.set_enabled(False)
        self.input_area.set_placeholder("请先导入一篇 PDF 论文...")

    def _apply_global_style(self):
        style_path = Path(__file__).parent / "styles.qss"
        if style_path.exists():
            with open(style_path, "r", encoding="utf-8") as f:
                self.setStyleSheet(f.read())
        else:
            self.setStyleSheet("""
                QMainWindow { background-color: #ffffff; }
                QSplitter::handle { background-color: #e0e0e0; }
                QSplitter::handle:hover { background-color: #007acc; }
            """)

    def _connect_signals(self):
        self.pdf_import_widget.file_selected.connect(self._on_pdf_selected)
        self.input_area.send_clicked.connect(self._on_send_message)
        self.input_area.clear_memory_clicked.connect(self._on_clear_memory)

    def _on_pdf_selected(self, file_path: str):
        self.pdf_import_widget.set_processing(True)
        self.log_panel.append_log(f"选中文件: {Path(file_path).name}", "INFO")
        self.chat_area.add_system_message(f"正在处理论文：{Path(file_path).name}，请稍候...")

        self._import_thread = QThread()
        self._import_worker = PDFImportWorker(file_path, self.current_index_name)
        self._import_worker.moveToThread(self._import_thread)

        self._import_thread.started.connect(self._import_worker.run)
        self._import_worker.progress.connect(self.log_panel.append_log)
        self._import_worker.finished.connect(self._on_import_finished)
        self._import_worker.finished.connect(self._import_thread.quit)
        self._import_worker.finished.connect(self._import_worker.deleteLater)
        self._import_thread.finished.connect(self._import_thread.deleteLater)

        self._import_thread.start()

    def _on_import_finished(self, success: bool, message: str):
        self.pdf_import_widget.set_processing(False)

        self.chat_area.clear_messages()  # 清空之前的对话
        if success:
            self.log_panel.append_log(f"✅ {message}", "INFO")
            self.chat_area.add_system_message("论文导入成功！您可以开始提问了。")
            self.input_area.set_enabled(True)
            self.input_area.set_placeholder("输入您的问题...")

            if self.agent_manager is None:
                try:
                    self.agent_manager = AgentManager(
                        temperature=0.01,
                        index_name=self.current_index_name,
                        verbose=False,
                    )
                    self.log_panel.append_log("Agent 管理器初始化完成。", "INFO")
                except Exception as e:
                    self.log_panel.append_log(f"Agent 初始化失败: {e}", "ERROR")
                    self.show_error("初始化失败", str(e))
        else:
            self.log_panel.append_log(f"❌ 导入失败: {message}", "ERROR")
            self.chat_area.add_system_message(f"导入失败：{message}")

    def _on_send_message(self, text: str):
        if self.agent_manager is None:
            self.show_error("未就绪", "请先导入一篇 PDF 论文。")
            return

        # 添加用户消息气泡
        self.chat_area.add_user_message(text)
        # 添加临时占位气泡
        self.chat_area.add_placeholder_message("🤔 正在思考，请稍候...")
        
        self.input_area.set_enabled(False)
        self.input_area.set_placeholder("助手思考中...")
        self.log_panel.append_log(f"用户提问: {text[:50]}...", "INFO")

        self._query_thread = QThread()
        self._query_worker = AgentQueryWorker(self.agent_manager, text, self.session_id)
        self._query_worker.moveToThread(self._query_thread)

        self._query_thread.started.connect(self._query_worker.run)
        self._query_worker.log_message.connect(self.log_panel.append_log)
        self._query_worker.finished.connect(self._on_query_finished)
        self._query_worker.finished.connect(self._query_thread.quit)
        self._query_worker.finished.connect(self._query_worker.deleteLater)
        self._query_thread.finished.connect(self._query_thread.deleteLater)

        self._query_thread.start()

    def _on_query_finished(self, result: dict):
        output = result.get("output", "（未收到回复）")
        # 用真实回答替换占位气泡
        self.chat_area.replace_placeholder(output)
        
        self.input_area.set_enabled(True)
        self.input_area.set_placeholder("输入您的问题...")

        if "error" in result:
            self.log_panel.append_log(f"执行过程中出现错误: {result['error']}", "ERROR")

    def _on_clear_memory(self):
        if self.agent_manager:
            self.agent_manager.reset_memory(self.session_id)
            self.chat_area.clear_messages()
            self.log_panel.append_log("对话记忆已清空。", "INFO")
        else:
            self.chat_area.clear_messages()
            self.log_panel.append_log("对话已清空（Agent 未初始化）。", "INFO")

    def show_error(self, title: str, message: str):
        QMessageBox.critical(self, title, message)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())