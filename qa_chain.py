from typing import List
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from chroma import load_chroma_index
from reranker import rerank_documents
from langchain_ollama import OllamaLLM
from chroma import get_retriever
# Настройка модели (Mistral через API- интерфейс)
llm = OllamaLLM(
    model="infidelis/GigaChat-20B-A3B-instruct-v1.5:bf16",
    temperature=0
)
# Промпт для ответа
qa_prompt = PromptTemplate.from_template(
    """
    Вопрос: {question}

    Ниже приведены фрагменты из ГОСТов. Если среди них есть определение в формате
    "термин: определение", и оно напрямую относится к вопросу — верни **именно его**, без изменений.
    Если определения нет — напиши "нет данных".

    Контекст:
    {context}

    Ответ:
    """
)
qa_chain = qa_prompt | llm | StrOutputParser()

def filter_unique_docs(docs: List[Document]) -> List[Document]:
    """
    Удаляет дубликаты документов по содержимому.
    """
    seen = set()
    unique_docs = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            unique_docs.append(doc)
            seen.add(content)
    return unique_docs
def answer_with_rag(question: str, rerank: bool = True, k: int = 10) -> tuple[str, List[Document]]:
    """
    Возвращает финальный ответ на основе запроса пользователя
    """
    retriever = get_retriever(k=k, fetch_k=128)
    retrieved_docs = retriever.invoke(question)
    if rerank:
        top_docs = rerank_documents(question, retrieved_docs)
    else:
        top_docs = retrieved_docs

    context = "\n\n".join([doc.page_content for doc in top_docs])
    answer = qa_chain.invoke({"question": question, "context": context})
    return answer, top_docs


# Пример
if __name__ == "__main__":
    question = input("Введите ваш вопрос по ГОСТам: ")
    answer, used_chunks = answer_with_rag(question)
    print("\nОтвет:\n", answer)

    print("\nИспользованные документы:")
    for i, doc in enumerate(used_chunks, 1):

        print(f"{i}. {doc.metadata['source']}")
        print(doc.page_content.strip())
