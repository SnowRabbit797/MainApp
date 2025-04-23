import streamlit as st

def main():
    st.sidebar.title("25/04/11 ゼミ発表資料(テスト)")
    section = st.sidebar.radio("目次", ["はじめに", "2枚目", "結論"])

    st.title("テスト")

    if section == "はじめに":
        st.markdown("### テスト")

    elif section == "2枚目":
        st.write("テスト")

    elif section == "結論":
        st.write("テスト")

