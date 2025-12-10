#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

# ---------------- 用户设置 ----------------
INPUT_DIR = r"C:\Users\11759\Desktop\test\extract\roi_4"   # <-- 在这里修改为你的输入目录
# -----------------------------------------

# 支持的图片扩展名（小写）
IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.gif', '.webp'}

def gather_image_entries(input_dir: Path):
    folder_name = input_dir.name
    images = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS])
    entries = [f"{folder_name}/{p.name}\t1" for p in images]
    return entries

def read_existing_entries(file_path: Path):
    if not file_path.exists():
        return set()
    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        lines = [line.rstrip('\n').rstrip('\r') for line in f]
    # 只保留非空行
    return set([line for line in lines if line.strip() != ''])

def ensure_file_ends_with_newline(file_path: Path):
    """
    如果文件存在并且最后没有换行符，返回 True（表示需要在写入前先写入一个换行）。
    """
    if not file_path.exists():
        return False
    try:
        with file_path.open('rb') as f:
            if f.seek(0, os.SEEK_END) == 0:
                return False  # 空文件
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            return last != b'\n'
    except OSError:
        return False

def append_entries_to_file(file_path: Path, new_entries):
    if not new_entries:
        print("没有新的图片条目需要追加。")
        return

    need_pre_newline = ensure_file_ends_with_newline(file_path)
    # 以追加模式写入（utf-8）
    with file_path.open('a', encoding='utf-8', newline='') as f:
        if need_pre_newline:
            f.write('\n')  # 确保不会连接到旧内容的最后一行
        for i, entry in enumerate(new_entries):
            # 每条独立一行；最后一行也写入换行，保持文件末尾有换行
            f.write(entry + '\n')
    print(f"已追加 {len(new_entries)} 条到 {file_path}")

def main():
    input_dir = Path(INPUT_DIR)
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"错误：输入目录不存在或不是目录：{input_dir}")
        return

    file_state = input_dir / 'fileState.txt'
    entries = gather_image_entries(input_dir)
    if not entries:
        print("输入目录中没有识别到图片文件。")
        return

    existing = read_existing_entries(file_state)
    # 过滤出还没有存在于 fileState.txt 的条目
    to_add = [e for e in entries if e not in existing]

    append_entries_to_file(file_state, to_add)

if __name__ == '__main__':
    main()
