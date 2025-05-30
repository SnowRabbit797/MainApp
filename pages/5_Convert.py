import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from pulp import LpProblem, LpVariable, LpMinimize, lpSum, LpBinary
import time
import io


section = st.sidebar.radio("目次", ["隣接行列とエッジリスト", "jpgTogif"])

if section == "隣接行列とエッジリスト":
    st.subheader("隣接行列とエッジリストの変換ページです。")
    
    df = st.file_uploader("CSVファイルをアップロードしてください", type=["csv"])
    if df is not None:
        df = pd.read_csv(df)
        G = nx.Graph()
        G.add_edges_from(df.values)
        array = nx.to_numpy_array(G)
        array_int = array.astype(int)
        csv_data = pd.DataFrame(array_int).to_csv(index=False).encode('utf-8')
        st.dataframe(pd.read_csv(io.BytesIO(csv_data)))
        
        st.download_button(
            "ダウンロード",
            csv_data,
            "st_download.csv",
            "text/csv",
            key="download-csv"
        )
        
        
elif section == "jpgTogif":
    st.write("イメージ")
