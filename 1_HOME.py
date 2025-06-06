import streamlit as st

st.set_page_config(page_title="佐藤葵のStreamlit", layout="wide")

st.markdown("""
    <h1>2026年度ゼミ資料アプリ(ver1.0)</h1>
    <p>2026年度のゼミ資料をまとめたページです。このページは、Streamlitを使用して作成しています。</p>
    <p>Streamlitは、PythonでWebアプリケーションを簡単に作成できるライブラリです。</p>
    <p>以下にStreamlitの基本的な使い方を紹介します。</p>
    <br>
    <h3>ゼミの資料は画面左側のサイドバーから飛べます。</h3>
    <ul>
        <li>HOME：ホーム画面(streamlitの使い方等の紹介ページ)</li>
        <li>Today's Materials：今日のゼミ資料</li>
        <li>Materials Collection：ゼミの資料置き場</li>
        <li>Dataset：過去に使用したデータの一覧</li>
        <li>others：その他(作成中、データなどを入れる予定)</li>
    </ul>
    """, unsafe_allow_html=True)
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")


#streamlitのインストール方法-----------------------------
st.markdown("""
    <h2> streamlitのインストール方法</h2>
""", unsafe_allow_html=True)

st.markdown("""
    以下のコマンドでStreamlitをインストールできます。：
""")

st.code("pip install streamlit", language="bash")

st.markdown("""
    streamlitの起動方法(stappの部分は任意のアプリ名に変更する。)
""")

st.code("streamlit run stapp.py", language="bash")
st.markdown("詳細は [公式ドキュメント](https://docs.streamlit.io/) から。")

st.markdown("---")


#文字列の表示方法-----------------------------------------
st.markdown("""
    <h2>文字列の表示方法(st形式)</h2>
""", unsafe_allow_html=True)

st.markdown("""
    st形式はstremlitにおいて基本となる書き方です。
""", unsafe_allow_html=True)

st.markdown("""<h3>コードの例</h3>""", unsafe_allow_html=True)

st.code("""
    import streamlit as st

    st.write("Hello, World!")
""")
st.write("Hello, World!")

st.markdown("---")


#文字列の表示方法-----------------------------------------
st.markdown("""
    <h2>文字列の表示方法(Markdown形式)</h2>
""", unsafe_allow_html=True)

st.markdown("""
    Markdown形式は、Markdown記法を使用して文字列を表示する方法です。
    Markdownは、テキストを装飾するための軽量マークアップ言語です。
""", unsafe_allow_html=True)

st.markdown("""<h3>コードの例</h3>""", unsafe_allow_html=True)

st.code('''
    import streamlit as st

    st.markdown("""
        # 見出し大
        ## 見出し中
        ### 見出し小
        **太字**
        *斜体*
        Hello World
    """)
''')

st.markdown("""
    # 見出し大
    ## 見出し中
    ### 見出し小
    **太字**<br>
    *斜体*<br>
    Hello World
""", unsafe_allow_html=True)
st.markdown("---")

st.latex(r'a + ar + ar^2 + \cdots + ar^{n-1}')
st.latex(r'= \sum_{k=0}^{n-1} ar^k')
st.latex(r'= a \left( \frac{1 - r^n}{1 - r} \right)')


