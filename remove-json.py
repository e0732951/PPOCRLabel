#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
递归删除指定目录下的所有 .json 文件（包含子目录）。
修改 TARGET_ROOT 指向目标路径。设置 DRY_RUN=True 可先预览不实际删除。
运行： python delete_jsons.py
"""

import time
from pathlib import Path
import sys
import traceback

# ---------------- 配置区（在这里修改） ----------------
TARGET_ROOT = r"result"       # 要删除json的根路径（示例为当前目录），改成你的目标路径
DRY_RUN = False          # True -> 仅预览；False -> 实际删除
VERBOSE = True           # 是否打印每个文件的删除信息
# ----------------------------------------------------

def main():
    start = time.time()
    root = Path(TARGET_ROOT)
    if not root.exists():
        print(f"错误：路径不存在 -> {root}")
        sys.exit(1)

    json_files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() == ".json"]
    total = len(json_files)
    if total == 0:
        print("未找到任何 .json 文件。")
        return

    print(f"找到 {total} 个 .json 文件。{'（仅预览）' if DRY_RUN else ''}")
    deleted = 0
    errors = 0

    for p in json_files:
        try:
            if DRY_RUN:
                if VERBOSE:
                    print("[DRY RUN] 将删除：", p)
            else:
                p.unlink()
                deleted += 1
                if VERBOSE:
                    print("已删除：", p)
        except Exception as e:
            errors += 1
            print("删除失败：", p)
            traceback.print_exc()

    elapsed = time.time() - start
    print("\n--- 完成 ---")
    print(f"总文件数: {total}")
    print(f"已删除: {deleted}")
    print(f"出错: {errors}")
    print(f"总用时: {elapsed:.3f} 秒")

if __name__ == "__main__":
    main()
