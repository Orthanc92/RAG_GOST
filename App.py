import streamlit as st
from qa_chain import answer_with_rag

st.set_page_config(page_title="ГОСТ RAG", layout="wide")

st.title("📘 Поиск по ГОСТам")

query = st.text_input("Введите вопрос:")

if query:
    with st.spinner("Ищем ответ..."):
        answer, sources = answer_with_rag(query)

    st.markdown("### 🧠 Ответ:")
    st.success(answer)

    st.markdown("### 📄 Использованные фрагменты:")
    for i, doc in enumerate(sources, 1):
        st.markdown(f"**{i}. Источник:** `{doc.metadata.get('source', 'неизвестно')}`")
        st.markdown(f"> {doc.page_content.strip()}")
