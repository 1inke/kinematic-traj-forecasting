"""
INTERACTION Dataset Resumable Downloader
从 Hugging Face / 镜像源断点续传下载 INTERACTION 完整数据包 TOTAL.zip 并自动解压。
"""
import os
import sys
import time
import zipfile
import requests
from pathlib import Path

HERE = Path(__file__).resolve().parent
URL = "https://hf-mirror.com/datasets/Jeongeun/InteractionDataset/resolve/main/TOTAL.zip"
TARGET_FILE = HERE / "TOTAL.zip"
PART_FILE = HERE / "TOTAL.zip.part"
DEST_DIR = HERE / "interaction-dataset-master" / "recorded_trackfiles"


def download_with_resume(url: str, part_path: Path, target_path: Path):
    if target_path.is_file():
        print(f"目标文件已存在: {target_path} (大小: {target_path.stat().st_size / 1e9:.2f} GB)")
        return

    existing_size = part_path.stat().st_size if part_path.is_file() else 0

    headers = {}
    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(f"检测到未完成下载，正在从 {existing_size / 1e9:.2f} GB 处继续断点续传...")
    else:
        print("开始全新下载 INTERACTION 数据集 (TOTAL.zip, 约 7.87 GB)...")

    # 获取总大小
    r_head = requests.head(url, allow_redirects=True, timeout=10)
    total_size = int(r_head.headers.get("content-length", 0)) + (existing_size if "Range" in headers else 0)

    t0 = time.time()
    last_log = time.time()
    downloaded = existing_size

    with requests.get(url, headers=headers, stream=True, timeout=15) as r:
        r.raise_for_status()
        mode = "ab" if existing_size > 0 else "wb"
        with open(part_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB 缓冲区
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_log >= 5.0:
                        speed = ((downloaded - existing_size) / 1e6) / max(1e-3, now - t0)
                        pct = (downloaded / total_size) * 100.0 if total_size else 0.0
                        print(f"下载进度: {downloaded / 1e9:.2f} GB / {total_size / 1e9:.2f} GB ({pct:.1f}%) | 当前速度: {speed:.2f} MB/s", flush=True)
                        last_log = now

    part_path.rename(target_path)
    print(f"\n✅ 下载完成！文件已保存至: {target_path}")


def extract_zip(zip_path: Path, dest_dir: Path):
    print(f"正在解压 {zip_path.name} 到 {dest_dir} ...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)
    print(f"✅ 解压完成！所有场景轨迹文件已就绪。")


if __name__ == "__main__":
    download_with_resume(URL, PART_FILE, TARGET_FILE)
    if TARGET_FILE.is_file():
        extract_zip(TARGET_FILE, DEST_DIR)
