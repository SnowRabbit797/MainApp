import streamlit as st
from modules.Ideas import Idea_MVC_GA

page = st.sidebar.selectbox(
    "ページを選択してください",
    ("GAによるMVC問題のアイディア(作成中)"),
)

if page == "GAによるMVC問題のアイディア":
    Idea_MVC_GA.main()
else:
    st.write("ページが見つかりません。")
