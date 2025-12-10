import os

# === 修改这里：你的 train_data 根目录 ===
BASE_DIR = r"C:\GITHUB\PaddleOCR\train_data"

# 支持的图像后缀
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def find_image_by_name(root, filename):
    """在指定目录下递归寻找匹配文件名的图片"""
    for dirpath, _, filenames in os.walk(root):
        if filename in filenames:
            return os.path.join(dirpath, filename)
    return None


def fix_txt_image_paths(root):
    """遍历 root 下所有 txt，替换每行中图片路径为真实存在的路径"""
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if not file.lower().endswith(".txt"):
                continue

            txt_path = os.path.join(dirpath, file)
            print(f"处理 TXT: {txt_path}")

            new_lines = []
            with open(txt_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split("\t", 1)
                if len(parts) != 2:
                    new_lines.append(line)
                    continue

                old_img_path, json_part = parts
                img_name = os.path.basename(old_img_path)

                # 找图片真实路径
                real_img_path = find_image_by_name(root, img_name)
                if real_img_path is None:
                    print(f"  ⚠ 找不到图片: {img_name}")
                    new_lines.append(line)
                else:
                    # 替换成绝对路径
                    new_line = real_img_path.replace("/", "\\") + "\t" + json_part
                    new_lines.append(new_line + "\n")

            # 写回文件
            with open(txt_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            print(f"✔ 完成修复: {txt_path}\n")


if __name__ == "__main__":
    fix_txt_image_paths(BASE_DIR)
    print("🎉 全部处理完成！")
