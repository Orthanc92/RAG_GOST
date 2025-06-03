import re
from typing import List
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

# ⚙️ Настройка LLM
llm = OllamaLLM(
    model="infidelis/GigaChat-20B-A3B-instruct-v1.5:bf16",
    temperature=0
)

# 🧠 Промпт для выбора лучших чанков
rerank_prompt = PromptTemplate.from_template(
    """
Вот пользовательский вопрос:
"{question}"

Ниже даны фрагменты текста (с указанием номера CHUNK-N):

{context}
Выбери и верни только те фрагменты, которые содержат точные формулировки определений из ГОСТа.
Если среди фрагментов есть формулировка "термин: определение" — обязательно выбери её. Это приоритетный фрагмент.

"""
)

rerank_chain = rerank_prompt | llm | StrOutputParser()


def rerank_documents(question: str, docs: List[Document], fallback_k: int = 5) -> List[Document]:
    if not docs:
        return []

    combined_text = "\n\n".join([f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs)])
    selected_text = rerank_chain.invoke({"question": question, "context": combined_text})

    selected_docs = []
    found_chunks = re.findall(r"\[(\d+)\]", selected_text)

    for num in found_chunks:
        idx = int(num) - 1
        if 0 <= idx < len(docs):
            selected_docs.append(docs[idx])

    # Приоритизируем фрагменты с терминами
    selected_docs.sort(key=lambda d: not d.metadata.get("term_chunk", False))

    return selected_docs or docs[:fallback_k]