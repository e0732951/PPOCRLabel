#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理脚本：递归遍历给定根路径下的所有图片，使用 PaddleOCR 对每张图片进行识别，
并将结果（可视化图片与 JSON）按原路径结构保存到输出目录。

说明：
- 不从命令行接收参数，所有配置都在文件开头统一定义。
- 运行结束后打印总用时、处理张数与异常数。

将本文件保存为 batch_paddleocr_recursive.py 后直接运行：
    python batch_paddleocr_recursive.py

依赖：paddleocr（与示例调用方式兼容）
"""

import os
import sys
import time
import traceback
from pathlib import Path

from paddleocr import PaddleOCR

# ------------------ 配置区（全部在这里定义，不要传入参数） ------------------
# 要处理的根路径（可以是图片文件或一个文件夹）
INPUT_ROOT = r"D:\DOWNLOAD\snapshots\2025-12-02"  # <- 在这里填写你要处理的文件/文件夹路径

# 输出根目录（脚本自动在此目录下按原相对路径保存结果）
OUTPUT_ROOT = r"D:\DOWNLOAD\snapshots\2025-12-02"

# OCR 初始化参数（按示例代码）
OCR_CONFIG = {
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
    "ocr_version": "PP-OCRv5",
    "device": "gpu",  # 若不使用 GPU 改为 "cpu"
    "lang": "en",
}

# 支持的图片扩展名（小写）
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 是否在控制台打印每张图片的详细结果（会调用 res.print()）
VERBOSE_PRINT = True

# 是否在每个结果图片/JSON 文件名中保留原图片名
KEEP_ORIGINAL_NAME = True

# 如果你希望将所有结果放到单层目录（不推荐），设置为 True。否则会镜像输入目录结构。
FLATTEN_OUTPUT = False

# -----------------------------------------------------------------------


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def collect_image_paths(root: str):
    """如果 root 是文件则返回该文件（如果是图片）；如果是目录则递归返回所有图片路径"""
    p = Path(root)
    if p.is_file():
        if is_image_file(p):
            return [p.resolve()]
        else:
            return []
    if not p.exists():
        raise FileNotFoundError(f"输入路径不存在: {root}")

    out = []
    for dirpath, dirnames, filenames in os.walk(p):
        for fn in filenames:
            fp = Path(dirpath) / fn
            if is_image_file(fp):
                out.append(fp.resolve())
    return out


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def main():
    start_time = time.time()

    print("初始化 PaddleOCR... (配置：{} )".format(OCR_CONFIG))
    ocr = PaddleOCR(
        use_doc_orientation_classify=OCR_CONFIG["use_doc_orientation_classify"],
        use_doc_unwarping=OCR_CONFIG["use_doc_unwarping"],
        use_textline_orientation=OCR_CONFIG["use_textline_orientation"],
        ocr_version=OCR_CONFIG["ocr_version"],
        device=OCR_CONFIG["device"],
        lang=OCR_CONFIG["lang"],
    )

    try:
        img_paths = collect_image_paths(INPUT_ROOT)
    except Exception as e:
        print("收集图片路径失败：", e)
        sys.exit(1)

    total = len(img_paths)
    if total == 0:
        print("未找到任何图片，退出。")
        return

    print(f"Found {total} images, start processing...")

    processed = 0
    errors = 0

    for img_path in img_paths:
        try:
            rel = os.path.relpath(img_path, start=Path(INPUT_ROOT).resolve())
        except Exception:
            rel = img_path.name

        if FLATTEN_OUTPUT:
            out_dir = Path(OUTPUT_ROOT)
        else:
            out_dir = Path(OUTPUT_ROOT) / Path(rel).parent
        ensure_dir(out_dir)

        base_name = img_path.stem if KEEP_ORIGINAL_NAME else f"img_{processed:06d}"

        # 调用 ocr
        try:
            result = ocr.predict(input=str(img_path))
        except Exception as e:
            print(f"[ERROR] 识别失败: {img_path} -> {e}")
            traceback.print_exc()
            errors += 1
            continue

        # result 可能包含多个页面/段落的结果，逐项保存
        # 我们为每个 res 生成单独的 _page{n} 后缀
        for i, res in enumerate(result):
            try:
                # 控制台打印（可选）
                if VERBOSE_PRINT:
                    try:
                        res.print()
                    except Exception:
                        # 有的 res 对象可能没有 print 方法
                        pass

                # 构造输出文件名
                out_img_path = out_dir / f"{base_name}_res_{i}.png"
                out_json_path = out_dir / f"{base_name}_res_{i}.json"

                # 保存（按示例使用 save_to_img/save_to_json）
                try:
                    res.save_to_img(str(out_img_path))
                except Exception:
                    # 若 save_to_img 失败，则忽略
                    pass
                try:
                    res.save_to_json(str(out_json_path))
                except Exception:
                    pass

            except Exception as e:
                print(f"[WARN] 保存单个结果时出错: {img_path} -> {e}")
                traceback.print_exc()

        processed += 1

    elapsed = time.time() - start_time
    print("\nDone")
    print(f"Total image: {total}")
    print(f"Succeed: {processed}")
    print(f"Fail: {errors}")
    print(f"Total time: {elapsed:.3f} second")


if __name__ == "__main__":
    main()
