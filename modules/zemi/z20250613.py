import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from modules.algorithm import kenchoXY
from pulp import LpMaximize, LpProblem, LpVariable, value, LpMinimize, LpBinary, lpSum, LpStatus


def main():
    st.title("6月13日(第4回)の発表")
    st.markdown("""<br>""", unsafe_allow_html=True)
    with st.container(border = True):
        st.subheader("本日の発表内容", divider="red")
        st.markdown("""
            <ol>
                <li>前回の復習</li>
                <li>整数計画法と線形計画法の導入</li>
                <li>最小頂点被覆問題を整数計画法で解く</li>
                <li>最小頂点被覆問題を整数計画法で解く②</li>
            </ol>
        """, unsafe_allow_html=True)
