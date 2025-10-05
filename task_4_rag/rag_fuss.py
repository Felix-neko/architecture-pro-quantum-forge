#!/usr/bin/env python3
"""RAG чат-бот с собственной функцией поиска и RetrievalQAWithSourcesChain"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

# Настройка логирования
logging.basicConfig(level=logging.WARNING, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class CustomRetriever(BaseRetriever):
    """Кастомный ретривер с встроенной функцией поиска"""

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Получить релевантные документы по запросу.
        Заменить на реальную реализацию поиска.

        Args:
            query: текст вопроса
            run_manager: callback manager (опционально)

        Returns:
            список Document с метаданными (source, token_range, char_range)
        """
        # Пример заглушки - заменить на вашу реальную функцию поиска
        # Здесь должна быть логика поиска по вашей базе знаний
        
        documents = [
            Document(
                page_content="SQLite - это встроенная SQL база данных, не требующая отдельного сервера.",
                metadata={
                    "source": "https://example.com/docs/sqlite_intro.html",
                    "token_range": (0, 15),
                    "char_range": (0, 85),
                },
            ),
            Document(
                page_content="SQLite широко используется в мобильных приложениях на iOS и Android.",
                metadata={
                    "source": "https://example.com/docs/sqlite_mobile.html",
                    "token_range": (120, 135),
                    "char_range": (580, 650),
                },
            ),
            Document(
                page_content="База данных SQLite хранится в одном файле на диске.",
                metadata={
                    "source": "/path/to/local/sqlite_storage.txt",
                    "token_range": (45, 58),
                    "char_range": (220, 273),
                },
            ),
            Document(
                page_content="SQLite поддерживает транзакции ACID и большинство стандартных SQL операций.",
                metadata={
                    "source": "https://example.com/docs/sqlite_features.html",
                    "token_range": (200, 218),
                    "char_range": (1050, 1128),
                },
            ),
        ]
        
        return documents


def load_prompt_template(template_path: Path = Path(__file__).parent / "prompt_template.txt") -> str:
    """Загрузить шаблон промпта из файла"""
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def create_rag_chain() -> RetrievalQAWithSourcesChain:
    """Создаёт RAG chain с Qwen 3 и кастомным ретривером"""

    # Инициализация Qwen 3 8B
    llm: OllamaLLM = OllamaLLM(model="qwen3:8b", temperature=0.7)

    # Загрузить шаблон промпта из файла
    system_template = load_prompt_template()

    # Создать промпт для QA chain
    qa_prompt = PromptTemplate(
        template=system_template
            + "\n\nКонтекст:\n{summaries}\n\nВопрос: {question}\n\nОтвет:",
        input_variables=["summaries", "question"],
    )

    # Создать кастомный ретривер
    retriever = CustomRetriever()

    # Создать цепочку RetrievalQAWithSourcesChain
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": qa_prompt},
    )

    return chain


def answer_question(chain: RetrievalQAWithSourcesChain, question: str) -> Dict[str, Any]:
    """
    Отвечает на вопрос используя RAG chain

    Args:
        chain: RAG chain с кастомным ретривером
        question: вопрос пользователя

    Returns:
        словарь с ответом и найденными документами
    """
    logger.info("=" * 80)
    logger.info(f"Вопрос: {question}")
    logger.info("-" * 80)

    # Получить ответ от chain
    result = chain.invoke({"question": question})

    logger.info("ОТВЕТ ОТ LLM:")
    logger.info("-" * 80)
    logger.info(result.get("answer", ""))
    logger.info("=" * 80)

    return result


def main() -> None:
    """Основная функция"""

    print("Инициализация RAG чат-бота с кастомным ретривером...")
    chain = create_rag_chain()
    print("✓ Готово!\n")

    print("RAG чат-бот готов! (введите 'exit' или 'выход' для завершения)")
    print("Задавайте вопросы на русском языке\n")

    while True:
        question: str = input("Вопрос: ").strip()

        if question.lower() in ("exit", "quit", "выход"):
            print("До свидания!")
            break

        if not question:
            continue

        # Получение ответа через RAG chain
        result = answer_question(chain, question)

        print(f"\nОтвет: {result.get('answer', 'Нет ответа')}\n")

        # Показать источники
        if result.get("sources"):
            print(f"Источники: {result['sources']}\n")

        # Показать найденные чанки с метаданными
        if result.get("source_documents"):
            print("Найденные чанки:")
            for i, doc in enumerate(result["source_documents"], 1):
                source = doc.metadata.get("source", "Неизвестно")
                token_range = doc.metadata.get("token_range", "N/A")
                char_range = doc.metadata.get("char_range", "N/A")
                snippet: str = doc.page_content[:150].replace("\n", " ")
                print(f"  [{i}] Источник: {source}")
                print(f"      Токены: {token_range}, Символы: {char_range}")
                print(f"      {snippet}...")
            print()


if __name__ == "__main__":
    main()
