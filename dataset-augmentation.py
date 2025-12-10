#!/usr/bin/env python3
# coding: utf-8
"""
增强 + 基于4点透视拉伸裁剪并保存为矩形
保持旋转不扩展（output size == input size），支持 scale, contrast, brightness
裁剪策略：
 - 若 annotation.points 有 4 个点 -> 做四点透视变换（拉伸为矩形，采用截断 int() 计算尺寸）
 - 否则 -> 使用 boundingRect 裁剪
"""
import os
import cv2
import json
import glob
import random
import hashlib
import numpy as np

# --------------- 配置 ---------------
dataset_root = r"C:\Users\11759\Desktop\test\extract\roi_0"
n = 2
contrast_range = (0.9, 1.15)
rotation_range = (-10, 10)
brightness_range = (-30, 30)
scale_range = (0.95, 1.05)
crop_dir = os.path.join(dataset_root, "crop_img")
label_txt_path = os.path.join(dataset_root, "Label.txt")
rec_gt_path = os.path.join(dataset_root, "rec_gt.txt")
os.makedirs(crop_dir, exist_ok=True)

# 透视裁剪相关配置
rotate_if_tall = True        # 裁剪后高度/宽度 >= rotate_threshold 则 np.rot90 旋转回正
rotate_threshold = 1.5
jpeg_quality = 95            # 保存 jpg 时的 JPEG 质量
# -------------------------------------

# 读取 Label.txt -> map basename -> list of (left_path, anns)
label_map = {}
if os.path.exists(label_txt_path):
    with open(label_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                left, json_str = s.split("\t", 1)
                anns = json.loads(json_str)
            except Exception:
                continue
            base = os.path.basename(left)
            label_map.setdefault(base, []).append((left, anns))

def get_images(root):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    res = []
    for r, d, f in os.walk(root):
        if os.path.abspath(r) == os.path.abspath(crop_dir):
            continue
        for fn in f:
            if os.path.splitext(fn)[1].lower() in exts:
                res.append(os.path.join(r, fn))
    return res

images = get_images(dataset_root)
print(f"[INFO] Found {len(images)} images")

def unique_crop_name(orig_rel_path, suffix, counter, ext):
    """
    生成裁剪文件名：基于原始相对路径（但去掉扩展名）+ 后缀 + crop_counter + hash + ext
    例子：orig_rel_path="img_0399.png", suffix="_C..."; 返回 "img_0399_C..._crop_123_abcd1234.png"
    """
    # 保留用来做 hash 的完整原始路径（包含扩展名），以保证唯一性
    h = hashlib.sha1(orig_rel_path.encode("utf-8")).hexdigest()[:8]
    # 但 safe 名称不包含原始扩展名（避免重复后缀）
    noext = os.path.splitext(orig_rel_path)[0]
    safe = noext.replace(os.sep, "__").replace("/", "__")
    return f"{safe}{suffix}_crop_{counter}_{h}{ext}"

def apply_affine_noexpand(img, angle_deg, scale):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    out = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, M

def transform_points_with_M(pts, M):
    if not pts:
        return []
    pts_arr = np.array(pts, dtype=np.float32)
    ones = np.ones((pts_arr.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([pts_arr, ones], axis=1)  # (N,3)
    M_full = np.vstack([M, [0, 0, 1]])  # 3x3
    res = hom @ M_full.T
    return [[float(r[0]), float(r[1])] for r in res]

# ---------- 四点顺序化与截断版透视变换 ----------
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

def get_rotate_crop_image_truncate(image, pts, rotate_if_needed=True, rotate_threshold=1.5):
    """
    采用截断 int(norm(...)) 来计算目标尺寸（尽量与某些官方实现对齐）。
    返回：warped_image (rectangle) 或 None, (w,h), reason
    """
    try:
        rect = order_points(pts)  # tl,tr,br,bl
    except Exception as e:
        return None, (0, 0), f"order_points_error:{e}"

    (tl, tr, br, bl) = rect
    # compute widths (truncate)
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
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if rotate_if_needed and maxHeight / (maxWidth + 1e-9) >= rotate_threshold:
        warped = np.rot90(warped)
    return warped, (maxWidth, maxHeight), "ok"

# ---------- clamp rect ----------
def clamp_rect(x, y, w, h, W, H):
    x0 = max(0, int(round(x)))
    y0 = max(0, int(round(y)))
    x1 = min(W, int(round(x + w)))
    y1 = min(H, int(round(y + h)))
    return x0, y0, x1 - x0, y1 - y0

# ---------- 进行 crop（优先透视拉伸，降级为 bbox） ----------
def crop_and_save_by_polygon(img, poly_pts, save_path, rotate_if_needed=True, rotate_threshold=1.5, jpeg_quality=95):
    """
    poly_pts: list of [x,y] (int or float). If len==4 -> perspective warp.
    Returns (ok:bool, reason:str)
    """
    H, W = img.shape[:2]
    if not poly_pts or len(poly_pts) == 0:
        return False, "empty_points"

    if len(poly_pts) == 4:
        # try perspective transform (truncate version)
        pts_f = np.array(poly_pts, dtype=np.float32)
        warped, (w, h), reason = get_rotate_crop_image_truncate(img, pts_f, rotate_if_needed=rotate_if_needed, rotate_threshold=rotate_threshold)
        if warped is None:
            return False, f"four_point_failed:{reason}"
        # save warped result with jpeg quality if jpg
        ext = os.path.splitext(save_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            ok = cv2.imwrite(save_path, warped, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            ok = cv2.imwrite(save_path, warped)
        return ok, "saved_perspective" if ok else "write_failed"
    else:
        # fallback: bounding rect
        pts = np.array(poly_pts, dtype=np.int32)
        x, y, w_box, h_box = cv2.boundingRect(pts)
        x0, y0, wc, hc = clamp_rect(x, y, w_box, h_box, W, H)
        if wc <= 0 or hc <= 0:
            return False, "empty_bbox_after_clamp"
        crop = img[y0:y0+hc, x0:x0+wc].copy()
        ext = os.path.splitext(save_path)[1].lower()
        if ext in ('.jpg', '.jpeg'):
            ok = cv2.imwrite(save_path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
        else:
            ok = cv2.imwrite(save_path, crop)
        return ok, "saved_bbox" if ok else "write_failed"

# ---------- 主流程 ----------
label_lines_to_append = []
rec_gt_lines_to_append = []
existing = glob.glob(os.path.join(crop_dir, "*"))
crop_counter = len(existing)

for img_path in images:
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] cannot read {img_path}")
        continue
    base = os.path.basename(img_path)
    name_noext, ext = os.path.splitext(base)
    rel_dir = os.path.relpath(os.path.dirname(img_path), dataset_root)
    if rel_dir == ".":
        rel_dir = ""
    label_entries = label_map.get(base, None)

    for i_aug in range(n):
        contrast = random.uniform(*contrast_range)
        angle = random.uniform(*rotation_range)
        brightness = random.uniform(*brightness_range)
        scale = random.uniform(*scale_range)
        c_val = int(round((contrast - 1.0) * 100))
        r_val = int(round(angle))
        b_val = int(round(brightness))
        s_val = round(scale, 3)
        suffix = f"_C{('+' if c_val>=0 else '')}{c_val}_R{('+' if r_val>=0 else '')}{r_val}_B{('+' if b_val>=0 else '')}{b_val}_S{s_val}"
        aug_basename = f"{name_noext}{suffix}{ext}"
        aug_rel = os.path.join(rel_dir, aug_basename).replace("\\", "/") if rel_dir else aug_basename
        try:
            aug_img, M = apply_affine_noexpand(img, angle, scale)
            aug_img = cv2.convertScaleAbs(aug_img, alpha=contrast, beta=brightness)
            save_dir = os.path.dirname(img_path) or dataset_root
            save_path = os.path.join(save_dir, aug_basename)
            cv2.imwrite(save_path, aug_img)
        except Exception as e:
            print(f"[ERROR] augment & save failed for {img_path}: {e}")
            continue

        # handle labels (if any)
        if label_entries:
            for left_path, anns in label_entries:
                new_anns = []
                for ann in anns:
                    pts = ann.get("points", [])
                    new_pts_f = transform_points_with_M(pts, M)
                    new_pts_i = [[int(round(x)), int(round(y))] for x, y in new_pts_f]
                    ann_copy = ann.copy()
                    ann_copy["points"] = new_pts_i
                    new_anns.append(ann_copy)
                # construct new left path with same prefix as original left_path
                left_dirname = os.path.dirname(left_path)
                if left_dirname == "":
                    new_left = aug_basename
                else:
                    new_left = os.path.join(left_dirname, aug_basename).replace("\\", "/")
                label_lines_to_append.append(f"{new_left}\t{json.dumps(new_anns, ensure_ascii=False)}")

                # crop each annotation using perspective warp (if 4 points) or bbox fallback
                for ann in new_anns:
                    pts = ann.get("points", [])
                    if not pts:
                        print(f"[DEBUG] skip ann with empty pts for {aug_basename}")
                        continue
                    crop_fn = unique_crop_name(os.path.join(rel_dir, base).replace("\\", "/"), suffix, crop_counter, ext)
                    crop_path = os.path.join(crop_dir, crop_fn)
                    crop_counter += 1

                    ok, reason = crop_and_save_by_polygon(aug_img, pts, crop_path,
                                                         rotate_if_needed=rotate_if_tall,
                                                         rotate_threshold=rotate_threshold,
                                                         jpeg_quality=jpeg_quality)
                    if ok:
                        rel_crop = os.path.join("crop_img", crop_fn).replace("\\", "/")
                        transcription = ann.get("transcription", "")
                        rec_gt_lines_to_append.append(f"{rel_crop}\t{transcription}")
                        print(f"[SAVED CROP] {crop_path} (src={base}, suffix={suffix}) reason={reason}")
                    else:
                        print(f"[SKIP CROP] {crop_path} skipped: {reason} (src={base}, suffix={suffix})")
        else:
            # no label entry: append empty anns line
            if rel_dir:
                new_left = aug_rel
            else:
                new_left = aug_basename
            label_lines_to_append.append(f"{new_left}\t{json.dumps([], ensure_ascii=False)}")
            print(f"[INFO] no label for {base}, appended empty anns for {aug_basename}")

# 写文件
if label_lines_to_append:
    with open(label_txt_path, "a", encoding="utf-8") as f:
        for l in label_lines_to_append:
            f.write(l + "\n")
    print(f"[INFO] appended {len(label_lines_to_append)} label lines to {label_txt_path}")

if rec_gt_lines_to_append:
    with open(rec_gt_path, "a", encoding="utf-8") as f:
        for l in rec_gt_lines_to_append:
            f.write(l + "\n")
    print(f"[INFO] appended {len(rec_gt_lines_to_append)} rec_gt lines to {rec_gt_path}")

print("[DONE]")
