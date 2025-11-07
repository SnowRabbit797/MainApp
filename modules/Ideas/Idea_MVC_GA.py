import pandas as pd

# --- ファイル読み込み ---
file_path = "assets/csv/G_set1.csv"  # あなたのCSVファイル名に変更
df = pd.read_csv(file_path)

# --- ノード数の計算 ---
# source列とtarget列の両方に含まれるノードをまとめて集合にする
nodes = set(df["source"]).union(set(df["target"]))
num_nodes = len(nodes)

# --- エッジ数の計算 ---
num_edges = len(df)

# --- 結果を出力 ---
print(f"ノード数: {num_nodes}")
print(f"エッジ数: {num_edges}")
