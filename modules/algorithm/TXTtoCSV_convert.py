import csv
import os

# =============================
# Streamlit でもズレない相対パス
# =============================
TXT_REL_PATH = "assets/csv/G15.txt"   # app.py から見た相対パスで書く


def convert_txt_to_csv(relative_path: str):
    """相対パスで txt を読み込み、同じフォルダに csv を作る"""

    # Streamlit では cwd が app.py の階層になる
    base_dir = os.getcwd()
    txt_path = os.path.join(base_dir, relative_path)

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"相対パスが見つかりません → {txt_path}")

    # 出力パス（拡張子だけ変える）
    csv_path = os.path.splitext(txt_path)[0] + ".csv"

    # txt 読み込み
    with open(txt_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    edge_lines = lines[1:]  # 1行目はヘッダ

    # CSV 出力
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "weight"])

        for line in edge_lines:
            parts = line.split()
            if len(parts) < 2:
                continue
            source = parts[0]
            target = parts[1]
            weight = 1
            writer.writerow([source, target, weight])

    print(f"✓ CSV 出力完了：{csv_path}")


if __name__ == "__main__":
    convert_txt_to_csv(TXT_REL_PATH)
