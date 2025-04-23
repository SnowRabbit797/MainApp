import streamlit as st
from pdf2image import convert_from_path

def main():
    images = convert_from_path("data/pdfData/250411mate.pdf")

    st.sidebar.title("25/04/11 ゼミ発表資料(テスト)")
    section = st.sidebar.radio("目次", ["はじめに", "2枚目", "結論"])

    st.title("テスト")

    if section == "はじめに":
        st.image(images[0], caption="1ページ目", use_container_width=True)
        st.markdown("### テスト")

    elif section == "2枚目":
        st.image(images[3], caption="2ページ目", use_container_width=True)
        st.write("テスト")

    elif section == "結論":
        st.image(images[5], caption="3ページ目", use_container_width=True)
