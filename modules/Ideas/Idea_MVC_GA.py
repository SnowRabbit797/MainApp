import streamlit as st

def main():
    st.title("一人ブレスト：アイディアメモ")

    # セッション状態の初期化
    if 'ideas' not in st.session_state:
        st.session_state.ideas = []

    # アイディア入力欄
    new_idea = st.text_area("新しいアイディアを入力してください")

    # 追加ボタン
    if st.button("アイディアを追加"):
        if new_idea.strip():
            st.session_state.ideas.append(new_idea.strip())
            st.success("アイディアを追加しました！")

    # アイディア一覧表示
    st.subheader("これまでのアイディア一覧")
    for idx, idea in enumerate(st.session_state.ideas, 1):
        st.markdown(f"**{idx}.** {idea}")

  
  


# めっちゃいい！
# 今回はMVCをGAで解く時のアイディアリストを作ってる。(他にも追加予定)
# それでその中でも、「交叉のアイディア」「初期個体群生成のアイディア」とか、色々おいておきたい。
# ユーザーは、どのアイディアに入れるか選択した上で、入力して追加するようにしたい。
