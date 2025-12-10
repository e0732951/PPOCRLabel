"""
脚本功能：
- 在文件开头定义多个输入路径（至少两个）和一个输出路径
- 将所有输入路径下的图片文件（包括子目录）复制到输出路径，保持相对目录结构
  - 若多个输入路径中存在同一相对路径/同名的图片文件，脚本会比较文件内容：
    - 若完全相同则跳过重复复制
    - 若不同则在文件名后加上序号（_2, _3...）避免覆盖
- 对于 .txt 文件：按输入路径的顺序合并同相对路径/同名的 txt 文件
  - 即第二个输入路径的同名 txt 内容追加到第一个的后面，以此类推

使用说明：
- 请在脚本开头修改 INPUT_DIRS 列表和 OUTPUT_DIR 字符串为你的路径
- 你也可以在命令行运行时传入路径（可选，命令行优先）

注意：
- 默认以 utf-8 打开 txt 文件，读取失败时会使用 errors='replace'。
- 支持常见图片后缀（不包含非常规 RAW 格式）。

"""

from pathlib import Path
import os
import shutil
import hashlib
import argparse
from collections import defaultdict

# ------------------ 在这里定义输入和输出路径（请修改为你的路径） ------------------
# 至少定义两个输入路径
INPUT_DIRS = [
    r"C:\Users\11759\Desktop\test\extract\roi_0",
    r"C:\Users\11759\Desktop\test\extract\roi_1",
    r"C:\Users\11759\Desktop\test\extract\roi_2",
    r"C:\Users\11759\Desktop\test\extract\roi_3",
    r"C:\Users\11759\Desktop\test\extract\roi_4",
]
# 定义输出路径
OUTPUT_DIR = r"C:\Users\11759\Desktop\test\dataset"
# -------------------------------------------------------------------------------

# 支持的图片扩展名（小写）
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.tif', '.webp'}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def file_hash(path: Path, chunk_size: int = 8192) -> str:
    """计算文件 sha256 哈希"""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def copy_images_preserve_structure(input_dirs, output_dir):
    output_dir = Path(output_dir)
    stats = {
        'copied': 0,
        'skipped_same': 0,
        'renamed_and_copied': 0,
    }

    # 记录已存在目标文件的哈希，避免重复计算目标文件哈希多次
    target_hash_cache = {}

    for base in input_dirs:
        base_path = Path(base)
        if not base_path.exists():
            print(f"警告：输入路径不存在，跳过：{base_path}")
            continue

        for src in base_path.rglob('*'):
            if src.is_file() and is_image_file(src):
                rel = src.relative_to(base_path)
                target = output_dir / rel
                ensure_parent(target)

                if not target.exists():
                    shutil.copy2(src, target)
                    stats['copied'] += 1
                else:
                    # 目标文件已存在，判断是否为相同内容
                    # 计算源哈希
                    src_hash = file_hash(src)
                    # 目标哈希缓存或计算
                    target_key = str(target)
                    if target_key in target_hash_cache:
                        tgt_hash = target_hash_cache[target_key]
                    else:
                        tgt_hash = file_hash(target)
                        target_hash_cache[target_key] = tgt_hash

                    if src_hash == tgt_hash:
                        # 相同文件，跳过
                        stats['skipped_same'] += 1
                    else:
                        # 不同文件：在文件名加后缀直到不冲突
                        stem = target.stem
                        suffix = target.suffix
                        parent = target.parent
                        i = 2
                        while True:
                            new_name = parent / f"{stem}_{i}{suffix}"
                            if not new_name.exists():
                                shutil.copy2(src, new_name)
                                stats['renamed_and_copied'] += 1
                                break
                            else:
                                # 若碰巧新名也存在，比较哈希，若相同则停止并跳过
                                new_key = str(new_name)
                                if new_key in target_hash_cache:
                                    new_hash = target_hash_cache[new_key]
                                else:
                                    new_hash = file_hash(new_name)
                                    target_hash_cache[new_key] = new_hash
                                if new_hash == src_hash:
                                    stats['skipped_same'] += 1
                                    break
                                i += 1
    return stats


def merge_txt_files(input_dirs, output_dir):
    output_dir = Path(output_dir)
    # 收集每个相对路径下的来源文件列表（保持 input_dirs 顺序）
    txt_map = defaultdict(list)  # relpath -> [Path1, Path2, ...]

    for base in input_dirs:
        base_path = Path(base)
        if not base_path.exists():
            print(f"警告：输入路径不存在，跳过：{base_path}")
            continue

        for src in base_path.rglob('*.txt'):
            if src.is_file():
                try:
                    rel = src.relative_to(base_path)
                except Exception:
                    # 如果无法 relativize（理论上不会），使用绝对路径的文件名
                    rel = src.name
                # 统一为 Posix 路径字符串（在不同平台上保持一致）
                rel_str = str(rel.as_posix())
                txt_map[rel_str].append(src)

    stats = {
        'merged_files': 0,
        'single_copied': 0,
    }

    for rel_str, src_list in txt_map.items():
        out_path = output_dir / Path(rel_str)
        ensure_parent(out_path)

        if len(src_list) == 1:
            # 只有一个来源，直接复制
            shutil.copy2(src_list[0], out_path)
            stats['single_copied'] += 1
        else:
            # 多个来源，按顺序合并
            # 以 utf-8 打开，读取失败时使用 errors='replace'
            with out_path.open('w', encoding='utf-8', errors='replace') as outf:
                for idx, s in enumerate(src_list):
                    try:
                        text = s.read_text(encoding='utf-8', errors='replace')
                    except Exception as e:
                        # 作为兜底再用二进制读然后 decode
                        print(f"读取 {s} 时出错：{e}，尝试二进制读取并解码")
                        with s.open('rb') as f:
                            text = f.read().decode('utf-8', errors='replace')

                    # 写入内容
                    outf.write(text)
                    # 如果不是最后一个文件，确保以换行分隔
                    if idx != len(src_list) - 1:
                        if not text.endswith('\n'):
                            outf.write('\n')
            stats['merged_files'] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description='复制图片并合并 txt 文件（保持相对路径）')
    parser.add_argument('--inputs', nargs='+', help='输入文件夹列表，按顺序', default=None)
    parser.add_argument('--output', help='输出文件夹', default=None)
    args = parser.parse_args()

    # 命令行参数优先
    if args.inputs:
        inputs = args.inputs
    else:
        inputs = INPUT_DIRS

    if args.output:
        outdir = args.output
    else:
        outdir = OUTPUT_DIR

    if len(inputs) < 2:
        print("错误：请提供至少两个输入路径（在脚本开头修改 INPUT_DIRS 或通过 --inputs 提供）")
        return

    print("输入路径（按顺序）：")
    for i, p in enumerate(inputs, 1):
        print(f"  {i}. {p}")
    print(f"输出路径：{outdir}\n")

    # 先合并 txt（合并会创建必要的目录）
    print("开始合并 txt 文件...")
    txt_stats = merge_txt_files(inputs, outdir)
    print(f"合并完成：merged_files={txt_stats['merged_files']}, single_copied={txt_stats['single_copied']}")

    # 然后复制图片（避免图片覆盖已存在的 txt 文件）
    print("开始复制图片文件...")
    img_stats = copy_images_preserve_structure(inputs, outdir)
    print(f"图片复制完成：copied={img_stats['copied']}, skipped_same={img_stats['skipped_same']}, renamed_and_copied={img_stats['renamed_and_copied']}")

    print('\n全部完成。')


if __name__ == '__main__':
    main()
