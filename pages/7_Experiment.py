import streamlit as st
from filelock import FileLock, Timeout
import time

lock_path = "my_task.lock"

st.title("排他制御デモ")

if st.button("処理を開始"):
    st.write("[処理を開始]を押すと、10秒間擬似的な処理をしているようなプログラムが実行されます。")
    try:
        with FileLock(lock_path, timeout=1):
            st.success("ロックを取得しました。処理中です...")
            time.sleep(10)
            st.success("処理が完了しました")
    except Timeout:
        st.warning("他のユーザーが現在処理中です。")
