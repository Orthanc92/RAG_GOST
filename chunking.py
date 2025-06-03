from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import re

def trim_gost_text(full_text: str) -> str:
    start_index = full_text.find("1 Область применения")
    return full_text[start_index:] if start_index != -1 else full_text


def clean_documents(docs: List[Document]) -> List[Document]:
    cleaned = []
    for doc in docs:
        trimmed_text = trim_gost_text(doc.page_content)
        cleaned.append(Document(page_content=trimmed_text, metadata=doc.metadata))
    return cleaned

def clean_intro(text: str) -> str:
    # Удалим всё до раздела "3 Термины и определения" или первого заголовка вида "1."
    match = re.search(r"(3\s*Термины и определения|1\.)", text, re.IGNORECASE)
    return text[match.start():] if match else text

def preprocess_documents(docs: List[Document]) -> List[Document]:
    """
    Применяет очистку к каждому документу.
    """
    cleaned_docs = []
    for doc in docs:
        cleaned_text = clean_intro(doc.page_content)
        cleaned_docs.append(Document(page_content=cleaned_text, metadata=doc.metadata))
    return cleaned_docs


def split_documents(documents, chunk_size=3000, chunk_overlap=300):
    """
    Разделяет документы на чанки с увеличенным размером и метит фрагменты с определениями.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    all_chunks = []
    term_pattern = re.compile(r"\b\d+\.\d+\s+[а-яА-ЯёЁa-zA-Z0-9\s\-]+?:\s")
    for doc in documents:
        chunks = splitter.create_documents([doc.page_content], metadatas=[doc.metadata])

        for chunk in chunks:
            if term_pattern.search(chunk.page_content):
                chunk.metadata["term_chunk"] = True
        all_chunks.extend(chunks)

    return all_chunks

# Пример использования
if __name__ == "__main__":
    from load_docs import load_docx_documents

    docs = load_docx_documents(r"D:\R-STYLE WORK\study_rag")
    chunks = split_documents(clean_documents(docs))
    print(f"Чанков получено: {len(chunks)}")
    print(chunks[0].page_content[:1000])
