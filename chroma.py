import os
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings
from transformers import AutoTokenizer, AutoModel
from langchain.docstore.document import Document
from typing import List


def build_chroma_index(docs: List[Document], index_path="chroma_index") -> Chroma:
    """
    Строит Chroma векторную базу и сохраняет её на диск
    """
    if not docs:
        raise ValueError("Нет документов для индексации.")

    # 💡 Добавим метаданные для отслеживания чанков
    for i, doc in enumerate(docs):
        doc.metadata["chunk_id"] = f"CHUNK-{i+1}"

    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        persist_directory=index_path,
    )

    print("Chroma индекс создан и сохранён.")
    return vectorstore




def load_chroma_index(index_path: str = "chroma_index") -> Chroma:
    """
    Загружает Chroma индекс с диска
    """
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")
    return Chroma(persist_directory=index_path, embedding_function=embeddings)

def get_retriever(k: int = 10, fetch_k: int = 128):
    chroma = load_chroma_index()
    return chroma.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, 'fetch_k': fetch_k}
    )
# Пример использования
if __name__ == "__main__":
    from load_docs import load_docx_documents
    from chunking import split_documents

    docs = load_docx_documents(r"D:\R-STYLE WORK\study_rag")
    print(f"Загружено документов: {len(docs)}")  # Проверка загрузки
    chunks = split_documents(docs)
    print(f"Получено чанков: {len(chunks)}")  # Проверка чанков
    build_chroma_index(chunks)
    retriever = get_retriever()
    results = retriever.invoke("Дай точное определение скальный грунт")
    print(f"🔍 Найдено: {len(results)}")
    for i, doc in enumerate(results, 1):
        print(f"\n=== Фрагмент {i} ===")
        print(f"Источник: {doc.metadata.get('source', 'неизвестно')}")
        print(doc.page_content.strip())