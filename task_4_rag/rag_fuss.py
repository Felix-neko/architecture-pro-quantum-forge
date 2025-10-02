#!/usr/bin/env python3
"""RAG чат-бот с собственной функцией поиска"""

from typing import List
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


def your_search_function(query: str) -> List[str]:
    """
    Ваша функция поиска по внутренней базе знаний.
    Заменить на реальную реализацию.

    Args:
        query: текст вопроса

    Returns:
        список из 4 самых релевантных документов
    """
    # Пример заглушки - заменить на вашу функцию
    return [
        "SQLite - это встроенная SQL база данных, не требующая отдельного сервера.",
        "SQLite широко используется в мобильных приложениях на iOS и Android.",
        "База данных SQLite хранится в одном файле на диске.",
        "SQLite поддерживает транзакции ACID и большинство стандартных SQL операций."
    ]


def create_rag_chain():
    """Создаёт RAG chain с Qwen 3"""

    # Инициализация Qwen 3 8B
    llm = OllamaLLM(
        model="qwen3:8b",
        temperature=0.7
    )

    # Промпт для RAG на русском
    template = """Используй следующий контекст для ответа на вопрос. Отвечай на русском языке.
Если не знаешь ответа на основе контекста, так и скажи - не придумывай.

Контекст:
{context}

Вопрос: {question}

Ответ:"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    # Современный LCEL синтаксис вместо LLMChain
    chain = prompt | llm

    return chain


def answer_question(chain, question: str, search_func) -> tuple[str, List[str]]:
    """
    Отвечает на вопрос используя RAG

    Args:
        chain: LangChain цепочка
        question: вопрос пользователя
        search_func: функция поиска документов

    Returns:
        (ответ, список найденных документов)
    """
    # Поиск релевантных документов через вашу функцию
    documents = search_func(question)

    # Объединение документов в контекст
    context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(documents)])

    # Генерация ответа
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response, documents


def main():
    """Основная функция"""

    print("Инициализация RAG чат-бота...")
    chain = create_rag_chain()
    print("✓ Готово!\n")

    print("RAG чат-бот готов! (введите 'exit' или 'выход' для завершения)")
    print("Задавайте вопросы на русском языке\n")

    while True:
        question = input("Вопрос: ").strip()

        if question.lower() in ('exit', 'quit', 'выход'):
            print("До свидания!")
            break

        if not question:
            continue

        # Получение ответа с использованием вашей функции поиска
        answer, sources = answer_question(chain, question, your_search_function)

        print(f"\nОтвет: {answer}\n")

        # Показать найденные документы
        print("Найденные документы:")
        for i, doc in enumerate(sources, 1):
            snippet = doc[:150].replace('\n', ' ')
            print(f"  [{i}] {snippet}...")
        print()


if __name__ == "__main__":
    main()