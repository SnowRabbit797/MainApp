import streamlit as st
st.set_page_config(page_title="ゼミの資料アプリ", layout="wide")

from modules.zemi import z20250411
from modules.zemi import z20250425
from modules.zemi import z20250523
from modules.zemi import z20250613
from modules.zemi import z20250711

z20250711.main()



