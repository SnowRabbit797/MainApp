import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import time
import random
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

def main():
    st.title("7月11日(第5回)の発表")
    st.markdown("""<br>""", unsafe_allow_html=True)

    with st.container(border=False):
        st.subheader("1. 今回のコードについて", divider="orange")
        st.markdown("""
        今回の実験では、従来のコードを再利用せず、**完全に一から新しい実装**を行った。<br><br>
        この方針にした理由は以下のとおりである：
        
        - 処理フローの本質的理解を深めるため
        - 古いコードに不要な依存やバグがあったため
        - よりシンプルで拡張性の高い設計を目指した
        """, unsafe_allow_html=True)
    
    st.markdown("""ということで、まずはコードがうまく動いているか調べるため、以下のような問題を作成した。<br><br><br>""", unsafe_allow_html=True)
        
    with st.container(border=True):
        st.subheader("1. 簡素なビット列最適化問題", divider="orange")

        st.markdown("""
        最小頂点被覆問題（MVC）に進む前の準備段階として、**極めてシンプルなGAモデル**を用いて挙動の確認を行った。<br>
        これはMVC問題とは全く関係ないが、基本的なGAの流れを確認し、GAの動作が正しく行われているか、個人的に確認するために作成した。<br>
        せっかく作成したので、ここで一応紹介する。摂動等を用いず、単純なことしかしていないので、多少は容赦いただきたい...<br><br>

        ---
        #### 問題設定
        - 各個体は800ビット（0 or 1）の配列
        - **適応度関数：ビット列中の「1の数」**
        - 最終的に **「すべて0」に近づける**ことが目標
        - 評価関数の形はMVCの「カバーサイズ最小化」に相当する簡易版

        ---
        #### 実験の目的
        - GAの基本的な流れ（選択・交叉・突然変異・エリート保存）の動作確認
        - 適応度の減少傾向を可視化し、アルゴリズムの収束特性を理解
        - 評価関数の単純化により、**GA全体の構造のデバッグが容易**

        ---
        #### 実行内容
        - 初期個体群をランダムに生成（個体数 = 1000）
        - 選択はルーレット方式（ただし逆数適応度で高スコアほど低確率）
        - 交叉は一点交叉、突然変異は無効化（mutationRate = 0）
        - エリート戦略を適用（上位10%を次世代へ無条件継承）
        - 世代数 = 1000、**最良適応度の推移を記録しグラフ化**

        ---
        #### 結果と考察
        - 初期世代では「1」の数はランダムに偏在（例：平均400前後）
        - 世代を経るごとに、ビット列の「1の数」が減少
        - 1000個体, 3000世代で回した結果、3000世代目には50以下に収束した。
        - 摂動を用いず、単純な選択と交叉のみである程度の値まで収束することが確認できた。<br>
        下の画像は1000個体、1000世代で回した結果のグラフである。<br>
        とりあえず、GAの基本的な流れは確認できたので、次のステップに進める。
        
        """, unsafe_allow_html=True)
        st.image("data/image/image0709/070901.jpeg")
