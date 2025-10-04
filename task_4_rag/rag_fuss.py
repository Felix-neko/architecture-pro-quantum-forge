#!/usr/bin/env python3
"""RAG чат-бот с собственной функцией поиска"""

import logging
from typing import List, Tuple, Callable
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

# Настройка логирования
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class ChatHistory:
    """Простое управление историей диалога с окном последних N обменов"""

    def __init__(self, window_size: int = 3):
        self.window_size: int = window_size
        self.messages: List[HumanMessage | AIMessage] = []

    def add_exchange(self, question: str, answer: str) -> None:
        """Добавить обмен вопрос-ответ в историю"""
        self.messages.append(HumanMessage(content=question))
        self.messages.append(AIMessage(content=answer))

        # Оставить только последние N обменов (N * 2 сообщения)
        max_messages = self.window_size * 2
        if len(self.messages) > max_messages:
            self.messages = self.messages[-max_messages:]

    def get_messages(self) -> List[HumanMessage | AIMessage]:
        """Получить текущую историю"""
        return self.messages

    def clear(self) -> None:
        """Очистить историю"""
        self.messages = []


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
        "SQLite поддерживает транзакции ACID и большинство стандартных SQL операций.",
    ]


def create_rag_chain() -> Tuple[ChatPromptTemplate, OllamaLLM, ChatHistory]:
    """Создаёт RAG chain с Qwen 3 и памятью"""

    # Инициализация Qwen 3 8B
    llm: OllamaLLM = OllamaLLM(model="qwen3:8b", temperature=0.7)

    # История диалога (последние 3 обмена)
    history: ChatHistory = ChatHistory(window_size=3)

    # ChatPromptTemplate: Few-Shot примеры в system message, история отдельно
    prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Ты полезный ассистент. Отвечай на вопросы на основе предоставленного контекста.
Если в контексте нет информации для ответа, честно скажи об этом - не придумывай.
Учитывай историю диалога для лучшего понимания контекста.

Примеры правильных ответов:

Пример 1:
Контекст:
[1] PostgreSQL - это объектно-реляционная СУБД с открытым исходным кодом.
[2] PostgreSQL поддерживает JSON, полнотекстовый поиск и расширения.
Вопрос: Что такое PostgreSQL?
Ответ: PostgreSQL - это объектно-реляционная СУБД с открытым исходным кодом, которая поддерживает JSON, полнотекстовый поиск и расширения.

Пример 2:
Контекст:
[1] Redis - это in-memory хранилище данных типа ключ-значение.
[2] Redis часто используется для кэширования и очередей сообщений.
Вопрос: Какие языки программирования поддерживает Redis?
Ответ: В предоставленном контексте нет информации о языках программирования, которые поддерживает Redis. Могу только сказать, что Redis - это in-memory хранилище данных.""",
            ),
            # История диалога (если есть)
            ("placeholder", "{chat_history}"),
            # Актуальный запрос пользователя
            (
                "human",
                """Контекст:
{context}

Вопрос: {question}""",
            ),
        ]
    )

    return prompt, llm, history


def answer_question(
    prompt_template: ChatPromptTemplate,
    llm: OllamaLLM,
    history: ChatHistory,
    question: str,
    search_func: Callable[[str], List[str]],
) -> Tuple[str, List[str]]:
    """
    Отвечает на вопрос используя RAG с памятью диалога

    Args:
        prompt_template: шаблон промпта
        llm: языковая модель
        history: история диалога
        question: вопрос пользователя
        search_func: функция поиска документов

    Returns:
        (ответ, список найденных документов)
    """
    # Поиск релевантных документов через вашу функцию
    documents: List[str] = search_func(question)

    # Объединение документов в контекст
    context: str = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(documents)])

    # Получить историю диалога
    chat_history = history.get_messages()

    # Форматирование промпта с историей
    messages = prompt_template.format_messages(chat_history=chat_history, context=context, question=question)

    # Логирование промпта
    logger.info("=" * 80)
    logger.info("ПРОМПТ, ОТПРАВЛЯЕМЫЙ В LLM:")
    logger.info("-" * 80)
    for msg in messages:
        logger.info(f"[{msg.type.upper()}]: {msg.content}")
        logger.info("-" * 80)

    # Генерация ответа
    response: str = llm.invoke(messages)

    # Логирование ответа
    logger.info("ОТВЕТ ОТ LLM:")
    logger.info("-" * 80)
    logger.info(response)
    logger.info("=" * 80)

    # Сохранить обмен в историю
    history.add_exchange(question, response)

    return response, documents


def main() -> None:
    """Основная функция"""

    print("Инициализация RAG чат-бота с памятью диалога...")
    prompt_template, llm, history = create_rag_chain()
    print("✓ Готово!\n")

    print("RAG чат-бот готов! (введите 'exit' или 'выход' для завершения)")
    print("Задавайте вопросы на русском языке")
    print("Бот помнит последние 3 обмена для понимания контекста\n")

    while True:
        question: str = input("Вопрос: ").strip()

        if question.lower() in ("exit", "quit", "выход"):
            print("До свидания!")
            break

        if not question:
            continue

        # Получение ответа с использованием вашей функции поиска и истории
        answer, sources = answer_question(prompt_template, llm, history, question, your_search_function)

        print(f"\nОтвет: {answer}\n")

        # Показать найденные документы
        print("Найденные документы:")
        for i, doc in enumerate(sources, 1):
            snippet: str = doc[:150].replace("\n", " ")
            print(f"  [{i}] {snippet}...")
        print()


if __name__ == "__main__":
    main()
