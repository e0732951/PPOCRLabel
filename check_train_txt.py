# check_train_txt.py
import sys, json

def try_parse(line):
    try:
        # 取第一个空格后作为 label（这是常见格式）
        parts = line.rstrip("\n\r").split(maxsplit=1)
        if len(parts) != 2:
            return False, "no two parts (path + label)"
        label = parts[1].strip()
        json.loads(label)
        return True, ""
    except Exception as e:
        return False, str(e)

def main(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, 1):
            ok, msg = try_parse(line)
            if not ok:
                print(f"Line {i} parse fail: {msg}")
                snippet = line[:400] if len(line)>400 else line
                print(f"  snippet: {snippet!r}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_train_txt.py <train.txt>")
    else:
        main(sys.argv[1])
