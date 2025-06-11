import streamlit as st
import time

progress_text = "Operation in progress. Please wait."
my_bar = st.progress(0, text=progress_text)

i = 100

for j in range(i):
    time.sleep(0.01)
    percent = int((j + 1) / i * 100)
    st.write(percent)
    my_bar.progress(percent, text=f"{progress_text} ({percent}%)")

my_bar.empty()  # プログレスバーをクリア
