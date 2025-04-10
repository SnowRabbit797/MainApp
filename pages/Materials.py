import streamlit as st
from modules.zemi import z20250411

st.set_page_config(page_title="ゼミ発表", layout="wide")

st.title("🧪 ゼミ発表ページ")

page = st.sidebar.selectbox(
    "ページを選択してください",
    ("4/11(第1回)", "4/25(第2回)"),
)

if page == "4/11(第1回)":
    z20250411.main()
elif page == "4/25(第2回)":
    st.write("ページが見つかりません。")
else:
    st.write("ページが見つかりません。")
