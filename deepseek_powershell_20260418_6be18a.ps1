# init_project.ps1
# 在项目根目录下执行此脚本以生成全部文件夹与占位文件

$ErrorActionPreference = "Stop"

# 定义项目根目录（脚本所在位置）
$RootDir = Get-Location

Write-Host "正在项目目录 '$RootDir' 创建文件结构..." -ForegroundColor Cyan

# 1. 创建数据目录
$dataDirs = @(
    "data\pdfs",
    "data\vector_store"
)
foreach ($dir in $dataDirs) {
    New-Item -ItemType Directory -Force -Path (Join-Path $RootDir $dir) | Out-Null
    Write-Host "创建目录: $dir"
}

# 2. 创建源代码目录及占位文件
$srcDirs = @{
    "src\core"     = @("__init__.py", "rag.py")
    "src\agent"    = @("__init__.py", "agent.py")
    "src\gui"      = @("__init__.py", "main_window.py", "styles.qss")
    "src\utils"    = @("__init__.py")
}
foreach ($dir in $srcDirs.Keys) {
    $dirPath = Join-Path $RootDir $dir
    New-Item -ItemType Directory -Force -Path $dirPath | Out-Null
    Write-Host "创建目录: $dir"
    foreach ($file in $srcDirs[$dir]) {
        $filePath = Join-Path $dirPath $file
        New-Item -ItemType File -Force -Path $filePath | Out-Null
        Write-Host "  生成文件: $dir\$file"
    }
}

# 3. 创建 tests 目录及测试占位文件（可选）
$testsDir = "tests"
New-Item -ItemType Directory -Force -Path (Join-Path $RootDir $testsDir) | Out-Null
Write-Host "创建目录: $testsDir"
$testFiles = @("test_rag.py", "test_agent.py")
foreach ($file in $testFiles) {
    $filePath = Join-Path $RootDir "$testsDir\$file"
    New-Item -ItemType File -Force -Path $filePath | Out-Null
    Write-Host "  生成文件: $testsDir\$file"
}


# 4. 创建根目录下的配置文件与入口文件
$rootFiles = @(
    ".env.example",
    ".gitignore",
    "requirements.txt",
    "README.md",
    "main.py"
)
foreach ($file in $rootFiles) {
    $filePath = Join-Path $RootDir $file
    if (-not (Test-Path $filePath)) {
        New-Item -ItemType File -Force -Path $filePath | Out-Null
        Write-Host "生成文件: $file"
    }
}

# 5. 向 .gitignore 写入常见忽略项
$gitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
venv/
.venv/
env.bak/
venv.bak/

# Environment
.env
*.env

# IDE
.vscode/
.idea/
*.swp
*.swo

# Data
data/pdfs/*.pdf
data/vector_store/

# OS
.DS_Store
Thumbs.db
"@
Set-Content -Path (Join-Path $RootDir ".gitignore") -Value $gitignoreContent -Encoding UTF8
Write-Host "已写入 .gitignore 默认内容"

# 6. 向 .env.example 写入示例环境变量
$envExampleContent = @"
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_STORE_PATH=./data/vector_store
"@
Set-Content -Path (Join-Path $RootDir ".env.example") -Value $envExampleContent -Encoding UTF8
Write-Host "已写入 .env.example 模板"

# 7. 向 requirements.txt 写入基础依赖
$requirementsContent = @"
# 核心框架
langchain
langchain-community
langchain-core

# 向量与检索
sentence-transformers
faiss-cpu

# 大模型 API
openai

# 文档解析
pypdf
pdfplumber
unstructured

# GUI
PySide6

# 辅助工具
python-dotenv
tiktoken
rich
"@
Set-Content -Path (Join-Path $RootDir "requirements.txt") -Value $requirementsContent -Encoding UTF8
Write-Host "已写入 requirements.txt"

# 8. 向 README.md 写入基本说明
# $readmeContent = @"
# # 基于RAG的学术文献智能问答助手

# ## 快速开始

# 1. 创建虚拟环境并安装依赖：
#    ```bash
#    conda create -n rag_assistant python=3.10
#    conda activate rag_assistant
#    pip install -r requirements.txt