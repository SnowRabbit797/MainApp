import streamlit as st

st.set_page_config(page_title="ゼミ発表", layout="wide")

from modules.zemi import z20250411
from modules.zemi import z20250425
from modules.zemi import z20250523
from modules.zemi import z20250613
from modules.zemi import z20250711
from modules.zemi import z20251017
from modules.zemi import z20251107

page = st.sidebar.selectbox(
    "ページを選択してください",
    ("4/11(第1回)", "4/25(第2回)", "5/23(第3回)", "6/13(第4回)", "7/11(第5回)", "10/17(後期第1回)", "11/07(後期第2回)"),
)

if page == "4/11(第1回)":
    z20250411.main()
elif page == "4/25(第2回)":
    z20250425.main()
elif page == "5/23(第3回)":
    z20250523.main()
elif page == "6/13(第4回)":
    z20250613.main()
elif page == "7/11(第5回)":
    z20250711.main()
elif page == "10/17(後期第1回)":
    z20251017.main()
elif page == "11/07(後期第2回)":
    z20251107.main()
else:
    st.write("ページが見つかりません。")
