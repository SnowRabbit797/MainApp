# # demo_compare_anyxy.py
# # streamlit run demo_compare_anyxy.py

# import streamlit as st
# import plotly.graph_objects as go

# st.set_page_config(page_title="Comparison Plot (Any X-Y)", layout="wide")
# st.title("比較グラフ（任意X-Y点列・系列可変）")

# # --------------------------
# # Helpers
# # --------------------------

# def parse_points(s: str):
#     """
#     "0:227, 1:226, 3:225" -> [(0.0,227.0),(1.0,226.0),(3.0,225.0)]
#     形式は "x:y" をカンマ区切り
#     - パース不能は無視
#     - x昇順にソート
#     - x重複は後勝ち
#     """
#     pts = []
#     for chunk in s.split(","):
#         chunk = chunk.strip()
#         if not chunk or ":" not in chunk:
#             continue
#         xs, ys = chunk.split(":", 1)
#         try:
#             x = float(xs.strip())
#             y = float(ys.strip())
#         except ValueError:
#             continue
#         pts.append((x, y))

#     if not pts:
#         return []

#     pts.sort(key=lambda t: t[0])

#     merged = []
#     for x, y in pts:
#         if merged and merged[-1][0] == x:
#             merged[-1] = (x, y)
#         else:
#             merged.append((x, y))
#     return merged


# def step_curve_from_points(pts):
#     """
#     段差表示（piecewise-constant）にするための点列を作る。
#     pts = [(x1,y1),(x2,y2),...] のとき、
#     x1..x2間はy1を保持、x2以降はy2…みたいに見せる。
#     Plotlyの線で段差っぽくするために、
#     (x2, y1) を挟む形の点列にする。
#     """
#     if len(pts) <= 1:
#         return [p[0] for p in pts], [p[1] for p in pts]

#     xs, ys = [pts[0][0]], [pts[0][1]]
#     for i in range(1, len(pts)):
#         x_prev, y_prev = pts[i-1]
#         x_now,  y_now  = pts[i]
#         # 段差の横線終端
#         xs.append(x_now)
#         ys.append(y_prev)
#         # 段差の縦落ち（同じxで次のy）
#         xs.append(x_now)
#         ys.append(y_now)
#     return xs, ys


# # --------------------------
# # session_state defaults
# # --------------------------

# def ensure_defaults():
#     if "axis" not in st.session_state:
#         st.session_state["axis"] = {
#             "x_title": "計算ステップ（例：世代 / 時間 / 実験回数 / データサイズ）",
#             "y_title": "評価値（例：頂点被覆サイズ / 目的関数値）",
#         }

#     if "series" not in st.session_state:
#         st.session_state["series"] = [
#             {
#                 "name": "Normal_GA",
#                 "desc": "通常GAの推移（例：ステップ:評価値）",
#                 "points": "1:227, 2:226, 3:225, 5:224, 7:223, 10:222, 13:221, 19:220, 20:219, 141:218",
#                 "style": "line_step",  # line_linear / line_step / markers
#             },
#             {
#                 "name": "Strong_Perturbation_GA",
#                 "desc": "強い摂動付きGAの推移（例：摂動で段階的に落ちる）",
#                 "points": "1:227, 3:226, 4:225, 6:224, 7:223, 11:222, 12:221, 17:220, 33:219, 44:218, 66:217, 106:215, 146:214, 150:213, 170:212",
#                 "style": "line_step",
#             },
#             {
#                 "name": "ILP (PuLP/CBC) Optimal",
#                 "desc": "線形/整数計画法の解（単一点 or 水平線で表示）",
#                 "points": "0:179",
#                 "style": "hline",  # hline / line_linear / line_step / markers
#             },
#         ]

# ensure_defaults()

# # --------------------------
# # Axis UI
# # --------------------------

# with st.container(border=True):
#     st.subheader("軸ラベル（自由に指定）")
#     c1, c2 = st.columns(2)
#     with c1:
#         st.session_state["axis"]["x_title"] = st.text_input("x軸は何を指す？", value=st.session_state["axis"]["x_title"])
#     with c2:
#         st.session_state["axis"]["y_title"] = st.text_input("y軸は何を指す？", value=st.session_state["axis"]["y_title"])

# # --------------------------
# # Series controls
# # --------------------------

# with st.container(border=True):
#     st.subheader("系列（グラフ）の追加・削除")

#     col_add, col_del, col_reset = st.columns([1, 1, 1])
#     with col_add:
#         if st.button("＋ 系列を追加"):
#             st.session_state["series"].append(
#                 {
#                     "name": f"Series_{len(st.session_state['series'])+1}",
#                     "desc": "",
#                     "points": "0:0, 1:0",
#                     "style": "line_linear",
#                 }
#             )
#             st.rerun()

#     with col_del:
#         if st.session_state["series"]:
#             del_idx = st.number_input(
#                 "削除する系列番号（1〜）",
#                 min_value=1,
#                 max_value=len(st.session_state["series"]),
#                 value=len(st.session_state["series"]),
#                 step=1,
#             )
#             if st.button("－ 指定番号の系列を削除"):
#                 st.session_state["series"].pop(int(del_idx) - 1)
#                 st.rerun()
#         else:
#             st.info("系列がありません。追加してください。")

#     with col_reset:
#         if st.button("初期状態に戻す"):
#             for k in ["series", "axis"]:
#                 if k in st.session_state:
#                     del st.session_state[k]
#             ensure_defaults()
#             st.rerun()

# # --------------------------
# # Series editors
# # --------------------------

# st.markdown("### 各系列の情報（名前・説明・点列・表示方法）")

# style_labels = {
#     "line_linear": "折れ線（線形）",
#     "line_step": "段差（キー点を保持する感じ）",
#     "markers": "点のみ（散布図）",
#     "hline": "水平線（ILP/LPの最良値を基準線表示）",
# }

# for i, ser in enumerate(st.session_state["series"]):
#     with st.container(border=True):
#         st.markdown(f"#### 系列 {i+1}")

#         c1, c2 = st.columns([1, 2])
#         with c1:
#             ser["name"] = st.text_input("表示名", value=ser["name"], key=f"name_{i}")
#             ser["style"] = st.selectbox(
#                 "表示方法",
#                 options=list(style_labels.keys()),
#                 format_func=lambda k: style_labels[k],
#                 index=list(style_labels.keys()).index(ser.get("style", "line_linear")),
#                 key=f"style_{i}",
#             )
#         with c2:
#             ser["desc"] = st.text_area("この系列が指すもの（説明）", value=ser["desc"], height=70, key=f"desc_{i}")
#             ser["points"] = st.text_area(
#                 "点列（形式: x:y をカンマ区切り）",
#                 value=ser["points"],
#                 height=90,
#                 help="例: 0:227, 1:226, 3:225",
#                 key=f"pts_{i}",
#             )

# run_btn = st.button("グラフ生成", type="primary")

# # --------------------------
# # Plot
# # --------------------------

# if run_btn:
#     if not st.session_state["series"]:
#         st.error("系列が0本です。1本以上追加してください。")
#         st.stop()

#     fig = go.Figure()
#     used_points = []

#     for ser in st.session_state["series"]:
#         pts = parse_points(ser["points"])
#         used_points.append((ser["name"], pts))

#         if not pts:
#             # 点が無い系列はスキップ（入力ミス対策）
#             continue

#         style = ser.get("style", "line_linear")

#         # 水平線（ILP/LPの最終値比較向け）
#         if style == "hline":
#             # 最初の点のyを使う（例: 0:179）
#             y0 = pts[0][1]
#             fig.add_hline(y=y0, line_dash="dot", annotation_text=ser["name"], annotation_position="top left")
#             continue

#         if style == "line_step":
#             xs, ys = step_curve_from_points(pts)
#             fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=ser["name"]))
#         elif style == "markers":
#             fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=ser["name"]))
#         else:  # line_linear
#             fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="lines", name=ser["name"]))

#     fig.update_layout(
#         xaxis_title=st.session_state["axis"]["x_title"],
#         yaxis_title=st.session_state["axis"]["y_title"],
#         template="plotly_white",
#         legend_title_text="Series",
#     )

#     st.subheader("比較グラフ")
#     st.plotly_chart(fig, use_container_width=True, key="cmp_anyxy")

#     st.markdown("### 各系列の説明（何を指すか）")
#     for ser in st.session_state["series"]:
#         with st.container(border=True):
#             st.markdown(f"**{ser['name']}**")
#             st.write(ser["desc"] if ser["desc"].strip() else "（説明なし）")

#     st.markdown("### 実際に使われた点列（整形後）")
#     for name, pts in used_points:
#         st.write(f"{name}:", pts)

# else:
#     st.info("軸ラベルと各系列の点列を入力して「グラフ生成」を押してください。")
