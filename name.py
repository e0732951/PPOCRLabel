import os
import re

def rename_trim_to_png(folder_path):
    for filename in os.listdir(folder_path):
        old_path = os.path.join(folder_path, filename)

        # 跳过目录
        if os.path.isdir(old_path):
            continue

        # 查找以 .png 结束的部分
        match = re.search(r'.+?\.png', filename, re.IGNORECASE)
        if not match:
            # 文件名里没有 .png 的跳过
            continue

        new_filename = match.group(0)
        new_path = os.path.join(folder_path, new_filename)

        # 避免覆盖已存在文件
        if os.path.exists(new_path):
            print(f"跳过（目标已存在）：{new_path}")
            continue

        os.rename(old_path, new_path)
        print(f"已重命名: {filename} -> {new_filename}")


# 示例路径
folder = r"processed"   # 修改为你的路径
rename_trim_to_png(folder)
