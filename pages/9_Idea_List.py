import streamlit as st

st.set_page_config(page_title="アイディアリスト", layout="wide")

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds_dict = st.secrets["gcp_service_account"]
credentials = Credentials.from_service_account_info(creds_dict, scopes=scopes)
client = gspread.authorize(credentials)

SPREADSHEET_ID = "1dNzJx9smj9YiXool8n3UEIB_B_V4xVI2Z0ON8XUjkwc"
sheet = client.open_by_key(SPREADSHEET_ID).sheet1


def displayIL(category: str):
    if len(mat) <= 1:
        st.info("まだアイディアはありません。")
    else:
        count = 0
        for i, row in enumerate(mat[1:], start=1): 
            if row[1] == category:
                count += 1
                st.markdown(f"・{row[2]} （{row[0]}）")
        if count == 0:
            st.info("このカテゴリにはまだアイディアがありません。")
            
            
with st.container(border=True):
  st.subheader("アイディアの追加", divider="gray")

  with st.form(key="idea_form", clear_on_submit=True, border=True):
      idea = st.text_area("①新しいアイディアを入力してください")

      selected = st.multiselect(
          "②カテゴリを選択してください。",
          ["IL:初期個体群", "IL:交叉と突然変異", "IL:貪欲補正", "IL:その他"],
      )
      submitted = st.form_submit_button("＋ 追加")

      if submitted:
          if not idea.strip():
              st.warning("⚠ 空のアイディアは追加できません")
          elif not selected:
              st.warning("⚠ カテゴリを選択してください")
          else:
              now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
              for cat in selected:
                  sheet.append_row([now, cat, idea.strip()])
              st.success("アイディアリストに追加しました。")


st.markdown("---")
# --- UI：一覧表示 ---
st.subheader("登録済みアイディア一覧")

mat = sheet.get_all_values()


tab1, tab2, tab3, tab4 = st.tabs(["IL:初期個体群", "IL:交叉と突然変異", "IL:貪欲補正", "IL:その他"])

with tab1:
  displayIL("IL:初期個体群")
  
with tab2:
  displayIL("IL:交叉と突然変異")
  
with tab3:
  displayIL("IL:貪欲補正")

with tab4:
  displayIL("IL:その他")




# # グラフ作成
# G = nx.random_geometric_graph(200, 0.125)

# # ノードの座標を取得（random_geometric_graph は pos を自動で持ってる）
# pos = nx.get_node_attributes(G, 'pos')

# # ノードの座標を x, y に分ける
# x_nodes = [pos[i][0] for i in G.nodes()]
# y_nodes = [pos[i][1] for i in G.nodes()]

# # エッジの座標を作成
# edge_x = []
# edge_y = []
# for edge in G.edges():
#     x0, y0 = pos[edge[0]]
#     x1, y1 = pos[edge[1]]
#     edge_x.extend([x0, x1, None])
#     edge_y.extend([y0, y1, None])

# # Plotly グラフを作成
# fig = go.Figure()

# # エッジ描画
# fig.add_trace(go.Scatter(
#     x=edge_x, y=edge_y,
#     line=dict(width=0.5, color='#888'),
#     hoverinfo='none',
#     mode='lines'))

# # ノード描画
# fig.add_trace(go.Scatter(
#     x=x_nodes, y=y_nodes,
#     mode='markers',
#     marker=dict(size=5, color='blue'),
#     hoverinfo='text',
#     text=[str(i) for i in G.nodes()]))

# fig.update_layout(
#     showlegend=False,
#     margin=dict(l=0, r=0, b=0, t=0),
#     xaxis=dict(showgrid=False, zeroline=False),
#     yaxis=dict(showgrid=False, zeroline=False)
# )

# # Streamlit で表示
# st.plotly_chart(fig, use_container_width=True)
