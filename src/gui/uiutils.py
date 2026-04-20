import markdown

_md = markdown.Markdown(extensions=[
        'fenced_code',      # 支持 ``` 代码块
        'tables',           # 支持表格
        'nl2br',            # 换行转 <br>
        'sane_lists',       # 更好的列表支持
    ])

def render_markdown(text: str) -> str:
    """
    将 Markdown 文本转换为 HTML，用于在 QLabel 中显示。
    支持代码块高亮（需额外安装 Pygments，此处先不做高亮，仅保留样式）。
    """
    # 配置 markdown 扩展

    html = _md.convert(text)
    
    # 添加基础样式（使代码块等有背景色）
    styled_html = f"""
    <html>
    <head>
        <style>
            body {{
                font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif;
                font-size: 13px;
                line-height: 1.5;
                margin: 0;
                padding: 0;
            }}
            pre {{
                background-color: #1e1e1e;
                color: #d4d4d4;
                padding: 12px;
                border-radius: 6px;
                overflow-x: auto;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }}
            code {{
                background-color: #f4f4f4;
                padding: 2px 4px;
                border-radius: 4px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 12px;
            }}
            pre code {{
                background-color: transparent;
                padding: 0;
                color: inherit;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 10px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            blockquote {{
                border-left: 4px solid #007acc;
                margin: 10px 0;
                padding-left: 16px;
                color: #555;
            }}
            a {{
                color: #007acc;
                text-decoration: none;
            }}
            a:hover {{
                text-decoration: underline;
            }}
        </style>
    </head>
    <body>
        {html}
    </body>
    </html>
    """
    return styled_html