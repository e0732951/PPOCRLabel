#!/usr/bin/env python3
"""
duplicate_rois.py

功能：
- 从 INPUT_DIR/Label.txt 读取模板检测框（使用第一个条目）
- 从 INPUT_DIR/rec_gt.txt 读取模板 crop -> transcription 列表（按 index 排序）
- 对 INPUT_DIR 下的所有图片（作为 extract 目录等同于 INPUT_DIR），截取相同 roi 保存到 INPUT_DIR/crop_img/
- 将对应的 Label 行（修改图片名）追加到 INPUT_DIR/Label.txt（避免重复）
- 将对应的 rec_gt 行追加到 INPUT_DIR/rec_gt.txt（避免重复）
"""
import os
import json
import cv2
import numpy as np
from pathlib import Path

# ====== 在这里设置你的输入路径（目录中包含 Label.txt, rec_gt.txt, extract/ 等） ======
INPUT_DIR = r"C:\Users\11759\Desktop\test\extract\roi_0"  # <-- 修改为你的路径
# =====================================================================================

LABEL_FILE = os.path.join(INPUT_DIR, "Label.txt")
REC_GT_FILE = os.path.join(INPUT_DIR, "rec_gt.txt")
EXTRACT_DIR = INPUT_DIR
CROP_DIR = os.path.join(INPUT_DIR, "crop_img")

os.makedirs(CROP_DIR, exist_ok=True)

def read_label_file(path):
    """读取 Label.txt，返回 dict: { 'extract/img_0000.png': [box_dicts...] , ... }"""
    items = {}
    if not os.path.exists(path):
        return items
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 以第一个 tab 分割
            if '\t' in line:
                key, jsonpart = line.split('\t', 1)
            else:
                # 兜底：按第一个空格
                parts = line.split(None, 1)
                if len(parts) == 2:
                    key, jsonpart = parts
                else:
                    continue
            try:
                boxes = json.loads(jsonpart)
            except Exception as e:
                print(f"[WARN] 无法解析 Label 行 JSON: {line[:100]} ... 跳过 ({e})")
                continue
            items[key] = boxes
    return items

def read_rec_gt(path):
    """读取 rec_gt.txt，返回 list of tuples and set of existing lines
       entries: [(relpath, transcription), ...]"""
    entries = []
    existing_lines = set()
    if not os.path.exists(path):
        return entries, existing_lines
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            raw = line.rstrip('\n')
            if not raw.strip():
                continue
            existing_lines.add(raw)
            if '\t' in raw:
                p, t = raw.split('\t', 1)
            else:
                parts = raw.split(None, 1)
                if len(parts) == 2:
                    p, t = parts
                else:
                    continue
            entries.append((p, t))
    return entries, existing_lines

def _needs_leading_newline(path):
    """
    如果文件存在且最后一个字节不是换行，返回 True（需要在追加前写入一个换行）。
    如果文件不存在或为空，返回 False（不需要）。
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            if size == 0:
                return False
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            return last != b'\n'
    except Exception:
        # 兜底：若检测失败，返回 False（避免误加空行）
        return False

def save_label_append(path, lines_to_append):
    """把 lines_to_append (list of strings) 追加到 Label.txt，确保与原文件换行分隔"""
    if not lines_to_append:
        return
    leading_newline = _needs_leading_newline(path)
    # 使用 'a' 模式以文本方式追加（若不存在会创建），并在必要时先写一个换行
    with open(path, 'a', encoding='utf-8') as f:
        if leading_newline:
            f.write('\n')
        for l in lines_to_append:
            f.write(l.rstrip() + '\n')

def save_rec_gt_append(path, lines_to_append):
    """把 lines_to_append (list of strings) 追加到 rec_gt.txt，确保与原文件换行分隔"""
    if not lines_to_append:
        return
    leading_newline = _needs_leading_newline(path)
    with open(path, 'a', encoding='utf-8') as f:
        if leading_newline:
            f.write('\n')
        for l in lines_to_append:
            f.write(l.rstrip() + '\n')

# -------------------- 截断版的四点透视裁剪实现（truncate） --------------------
def order_points(pts):
    """
    输入 pts: (4,2) array-like
    返回按顺序 [tl, tr, br, bl] 的 np.float32 array
    """
    pts = np.array(pts, dtype="float32")
    if pts.shape[0] != 4:
        raise ValueError("order_points expects 4 points")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype="float32")

def get_rotate_crop_image_truncate(img, points, rotate_if_needed=True, rotate_threshold=1.5):
    """
    采用截断 int(norm(...)) 来计算目标尺寸（与一些官方实现一致）。
    返回 (dst_img or None, (w,h), reason_str)
    """
    try:
        rect = order_points(points)
    except Exception as e:
        return None, (0, 0), f"order_points_error:{e}"

    (tl, tr, br, bl) = rect
    # compute widths (truncate, not round)
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))

    # compute heights (truncate)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))

    if maxWidth <= 0 or maxHeight <= 0:
        return None, (maxWidth, maxHeight), "zero_dim"

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    dst_img = cv2.warpPerspective(img, M, (maxWidth, maxHeight),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # 若需要，按高宽比旋转回正
    if rotate_if_needed and maxHeight / (maxWidth + 1e-9) >= rotate_threshold:
        dst_img = np.rot90(dst_img)
    return dst_img, (maxWidth, maxHeight), "ok"

def clamp_rect(x, y, w, h, W, H):
    """
    将矩形裁剪到图片范围内，返回 (x0,y0,w_clamped,h_clamped)
    """
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    return x0, y0, x1 - x0, y1 - y0

def crop_and_save(image_path, bbox, out_path, rotate_if_needed=True, rotate_threshold=1.5, jpeg_quality=95):
    """
    根据 bbox（polygon points list 或 dict 中 "points"）裁剪并保存为矩形。
    - 当 points 有 4 个点时，优先使用四点透视变换（截断版）
    - 否则回退到 axis-aligned bounding rect（min/max）
    返回 True/False
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] 无法读取图片: {image_path}")
        return False
    pts = None
    if isinstance(bbox, dict):
        pts = bbox.get("points", None)
    elif isinstance(bbox, (list, tuple)):
        pts = bbox
    if not pts:
        print(f"[WARN] bbox points 不存在或格式错误: {bbox}")
        return False

    # 转为 float32 数组
    try:
        pts_arr = np.array(pts, dtype=np.float32)
    except Exception as e:
        print(f"[WARN] points 转换失败: {pts} ({e})")
        return False

    H, W = img.shape[:2]

    # 若恰好有4个点 -> 采用透视变换（truncate 版）
    if pts_arr.shape[0] == 4:
        warped, (w, h), reason = get_rotate_crop_image_truncate(img, pts_arr, rotate_if_needed=rotate_if_needed,
                                                               rotate_threshold=rotate_threshold)
        if warped is None:
            print(f"[WARN] 四点透视裁剪失败({reason})，尝试回退到 bbox 截取")
        else:
            # 显式使用 JPEG quality 保存（当输出为 jpg 时）
            ext = os.path.splitext(out_path)[1].lower()
            if ext in ('.jpg', '.jpeg'):
                ok = cv2.imwrite(out_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            else:
                ok = cv2.imwrite(out_path, warped)
            if not ok:
                print(f"[ERROR] 保存裁剪图失败: {out_path}")
            return ok

    # 回退：axis-aligned bounding rectangle（min/max）
    xs = [int(round(p[0])) for p in pts_arr]
    ys = [int(round(p[1])) for p in pts_arr]
    x1, y1 = max(0, min(xs)), max(0, min(ys))
    x2, y2 = min(W, max(xs)), min(H, max(ys))
    if x2 <= x1 or y2 <= y1:
        print(f"[WARN] 计算出的 bbox 无效: {pts}")
        return False
    crop = img[y1:y2, x1:x2].copy()
    ext = os.path.splitext(out_path)[1].lower()
    if ext in ('.jpg', '.jpeg'):
        ok = cv2.imwrite(out_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
    else:
        ok = cv2.imwrite(out_path, crop)
    if not ok:
        print(f"[ERROR] 保存裁剪图失败: {out_path}")
    return ok

# ------------------------------------------------------------------------------

def main():
    # 1) 读取 Label.txt，选择第一个条目作为模板
    label_items = read_label_file(LABEL_FILE)
    if not label_items:
        print("[ERROR] 找不到有效的 Label.txt 内容，请检查路径和文件格式。")
        return
    # use first key as template
    template_key = next(iter(label_items.keys()))
    template_boxes = label_items[template_key]
    print(f"[INFO] 使用模板 Label 行: {template_key}（包含 {len(template_boxes)} 个检测框）")

    # 2) 读取 rec_gt.txt，找出与模板图片对应的 rec_gt 条目（按 crop index 排序）
    rec_entries, rec_existing_lines = read_rec_gt(REC_GT_FILE)
    # template base name: from template_key like 'extract/img_0000.png' -> basename 'img_0000'
    template_basename = os.path.splitext(os.path.basename(template_key))[0]
    # collect rec_gt entries for template
    template_rec = []
    for p, t in rec_entries:
        # 兼容 path 中是否有前缀 crop_img/
        if p.startswith("crop_img/"):
            fname = os.path.basename(p)
            if fname.startswith(template_basename + "_crop_"):
                # extract index if needed
                template_rec.append((p, t))
    # sort by crop index
    def crop_index_from_name(p):
        fn = os.path.basename(p)
        # pattern img_0000_crop_0.jpg
        try:
            part = fn.split('_crop_')[-1]
            idx = int(os.path.splitext(part)[0])
            return idx
        except:
            return 0
    template_rec.sort(key=lambda x: crop_index_from_name(x[0]))
    if not template_rec:
        # 如果没有在 rec_gt 中找到模板对应项，则尝试用 template_boxes 的 transcription 字段提取顺序
        print("[WARN] 在 rec_gt.txt 中未找到模板图片的 crop 项。将尝试从 Label 模板中的 transcription 构建 rec 列表。")
        template_rec = []
        for i, b in enumerate(template_boxes):
            trans = b.get("transcription", "")
            fname = f"crop_img/{template_basename}_crop_{i}.jpg"
            template_rec.append((fname, trans))
    print(f"[INFO] 模板 rec_gt 项数: {len(template_rec)}")

    # 3) 列出 extract 目录下所有图片
    if not os.path.isdir(EXTRACT_DIR):
        print(f"[ERROR] 找不到 extract 目录: {EXTRACT_DIR}")
        return
    img_files = sorted([p for p in os.listdir(EXTRACT_DIR) if os.path.isfile(os.path.join(EXTRACT_DIR, p))])
    # only include common image extensions
    img_files = [f for f in img_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    if not img_files:
        print(f"[ERROR] extract 目录下没有图片: {EXTRACT_DIR}")
        return
    print(f"[INFO] 在 extract 目录中发现 {len(img_files)} 张图片。")

    # Build set of existing Label keys to avoid duplicate addition
    existing_label_keys = set(label_items.keys())

    # We'll collect label lines and rec_gt lines to append
    label_lines_to_append = []
    rec_gt_lines_to_append = []

    # compute last folder name of INPUT_DIR (e.g., 'roi_0')
    last_folder = os.path.basename(os.path.normpath(INPUT_DIR))

    for img_name in img_files:
        full_img_path = os.path.join(EXTRACT_DIR, img_name)
        base = os.path.splitext(img_name)[0]  # e.g. img_0001
        # label_key now follows INPUT_DIR 最后一项文件夹名，例如 "roi_0/img_0001.jpg"
        label_key = f"{last_folder}/{img_name}"
        # skip if label already exists
        if label_key in existing_label_keys:
            print(f"[SKIP] Label 已存在: {label_key}")
        else:
            # write label line: use same JSON (but make no change to points/transcription)
            json_text = json.dumps(template_boxes, ensure_ascii=False)
            label_lines_to_append.append(f"{label_key}\t{json_text}")
            existing_label_keys.add(label_key)
            print(f"[APPEND] Label 行 添加: {label_key}")

        # create crops for this image according to template_boxes
        for i, bbox in enumerate(template_boxes):
            crop_fname = f"{base}_crop_{i}.jpg"
            crop_relpath = f"crop_img/{crop_fname}"
            crop_fullpath = os.path.join(CROP_DIR, crop_fname)
            # if crop not present, do it
            if not os.path.exists(crop_fullpath):
                ok = crop_and_save(full_img_path, bbox, crop_fullpath, rotate_if_needed=True, rotate_threshold=1.5, jpeg_quality=95)
                if ok:
                    print(f"[SAVE] 裁剪并保存: {crop_fullpath}")
                else:
                    print(f"[WARN] 裁剪失败: {crop_fullpath}")
            else:
                print(f"[SKIP] 裁剪文件已存在: {crop_fullpath}")

            # find transcription for this crop from template_rec (by index)
            if i < len(template_rec):
                _, transcription = template_rec[i]
            else:
                transcription = template_boxes[i].get("transcription", "")
            rec_line = f"{crop_relpath}\t{transcription}"
            # avoid duplicate rec_gt lines
            if rec_line in rec_existing_lines:
                print(f"[SKIP] rec_gt 已存在: {rec_line}")
            else:
                rec_gt_lines_to_append.append(rec_line)
                rec_existing_lines.add(rec_line)
                print(f"[APPEND] rec_gt 添加: {rec_line}")

    # 4) 将收集的行追加到 Label.txt 和 rec_gt.txt
    if label_lines_to_append:
        print(f"[INFO] 追加 {len(label_lines_to_append)} 行到 {LABEL_FILE}")
        save_label_append(LABEL_FILE, label_lines_to_append)
    else:
        print("[INFO] 无需向 Label.txt 追加新行。")

    if rec_gt_lines_to_append:
        print(f"[INFO] 追加 {len(rec_gt_lines_to_append)} 行到 {REC_GT_FILE}")
        save_rec_gt_append(REC_GT_FILE, rec_gt_lines_to_append)
    else:
        print("[INFO] 无需向 rec_gt.txt 追加新行。")

    print("[DONE] 所有操作完成。")

if __name__ == "__main__":
    main()
