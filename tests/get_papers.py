#!/usr/bin/env python3
"""
按编号顺序异步下载 arXiv 论文 PDF
用法: python download_arxiv_pdfs.py
"""

import asyncio
import aiohttp
import aiofiles
import random
from pathlib import Path
from typing import List, Tuple
import time


# ============ 配置参数 ============
YEAR = 25
MONTH = 5
START = random.randrange(100, 1000)
N = 30
ARXIV_IDS = []
for i in range(START, START + N):
    ARXIV_IDS.append(f"{YEAR}{MONTH:02d}.{i:05d}")

# 下载并发数（同时下载的论文数量）
MAX_CONCURRENT = 5

# 下载超时时间（秒）
DOWNLOAD_TIMEOUT = 60

# 保存目录
OUTPUT_DIR = Path("./data/pdfs")

# 重试次数
MAX_RETRIES = 3


# ============ 核心函数 ============

def get_pdf_url(arxiv_id: str) -> str:
    """
    根据 arXiv ID 获取 PDF 下载链接
    arXiv PDF 直接下载链接格式: https://arxiv.org/pdf/{id}.pdf
    """
    return f"https://arxiv.org/pdf/{arxiv_id}.pdf"


async def download_paper(
    session: aiohttp.ClientSession,
    arxiv_id: str,
    save_dir: Path,
    semaphore: asyncio.Semaphore,
) -> Tuple[str, bool, str]:
    """
    异步下载单篇论文 PDF
    返回: (arxiv_id, 是否成功, 错误信息)
    """
    async with semaphore:
        url = get_pdf_url(arxiv_id)
        safe_id = arxiv_id.replace("/", "_")
        filepath = save_dir / f"{safe_id}.pdf"

        # 如果文件已存在，跳过下载
        if filepath.exists() and filepath.stat().st_size > 0:
            print(f"⏭️  {arxiv_id}: 已存在，跳过")
            return (arxiv_id, True, "已存在，跳过")

        for attempt in range(MAX_RETRIES):
            try:
                timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
                async with session.get(url, timeout=timeout) as response:
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}"
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"❌ {arxiv_id}: {error_msg} (重试{attempt+1}次)")
                        return (arxiv_id, False, error_msg)

                    total_size = int(response.headers.get("Content-Length", 0))

                    async with aiofiles.open(filepath, "wb") as f:
                        downloaded = 0
                        while True:
                            chunk = await response.content.read(8192)
                            if not chunk:
                                break
                            await f.write(chunk)
                            downloaded += len(chunk)

                    if total_size > 0 and downloaded != total_size:
                        filepath.unlink(missing_ok=True)
                        if attempt < MAX_RETRIES - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        print(f"❌ {arxiv_id}: 大小不匹配 ({downloaded}/{total_size})")
                        return (arxiv_id, False, f"大小不匹配: {downloaded}/{total_size}")

                    print(f"✅ {arxiv_id}: 下载成功")
                    return (arxiv_id, True, "成功")

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                error_msg = str(e)
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                print(f"❌ {arxiv_id}: 网络错误 - {error_msg}")
                return (arxiv_id, False, f"网络错误: {error_msg}")

            except Exception as e:
                print(f"❌ {arxiv_id}: 未知错误 - {e}")
                return (arxiv_id, False, f"未知错误: {e}")

        print(f"❌ {arxiv_id}: 重试次数用尽")
        return (arxiv_id, False, "重试次数用尽")


async def main():
    """主函数"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = min(50, len(ARXIV_IDS))
    print(f"准备下载 {total} 篇论文...")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT * 2)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        for arxiv_id in ARXIV_IDS:
            task = download_paper(session, arxiv_id, OUTPUT_DIR, semaphore)
            tasks.append(task)

        # 并发执行所有任务
        results = await asyncio.gather(*tasks, return_exceptions=True)

    # 统计结果
    success_count = 0
    fail_count = 0
    print("\n" + "=" * 50)
    print("下载完成! 结果统计:")
    for result in results:
        if isinstance(result, Exception):
            fail_count += 1
            print(f"  ❌ 异常: {result}")
        elif isinstance(result, tuple):
            arxiv_id, success, msg = result
            if success:
                success_count += 1
                # 成功信息已在下载时打印，这里不再重复
            else:
                fail_count += 1
                # 失败信息也已在下载时打印
        else:
            fail_count += 1

    print(f"\n总计: 成功 {success_count} 篇, 失败 {fail_count} 篇")
    print(f"文件保存在: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n用户中断下载")