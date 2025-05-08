import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.sidebar.title("25/05/23 ゼミ発表資料")
    section = st.sidebar.radio("目次", ["section1", "section2", "section3", "section4", "section5"])

    if section == "section1":
        st.title("2025年5月23日 M2ゼミ発表(3回目)")
