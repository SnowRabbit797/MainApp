import streamlit as st

def main():
    st.sidebar.title("25/04/25 ゼミ発表資料(テスト)")
    section = st.sidebar.radio("目次", ["はじめに", "2枚目", "結論"])

    st.title("テスト")

    if section == "はじめに":
        st.markdown("コードテスト")
        
        code = """
        import streamlit as st
        import pandas as pd
        
        print("Hello World")
        
        """
        st.latex(r'''
            a + ar + a r^2 + a r^3 + \cdots + a r^{n-1} =
            \sum_{k=0}^{n-1} ar^k =
            a \left(\frac{1-r^{n}}{1-r}\right)
            ''')
        
        st.code(code, language="python")

    elif section == "2枚目":
        st.write("テスト")

    elif section == "結論":
        st.write("テスト")
