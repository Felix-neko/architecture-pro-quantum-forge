#!/usr/bin/env python3
"""RAG чат-бот с собственной функцией поиска и RetrievalQAWithSourcesChain"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_ollama import OllamaLLM, ChatOllama
from langchain.chains import RetrievalQAWithSourcesChain, LLMChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, BaseCallbackHandler
from langchain_core.outputs import LLMResult

# Настройка логирования
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)


class PromptLoggingCallback(BaseCallbackHandler):
    """Callback для логирования промптов, отправляемых в LLM"""

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Вызывается при начале работы LLM"""
        logger.info("=" * 80)
        logger.info("ПРОМПТ, ОТПРАВЛЯЕМЫЙ В LLM:")
        logger.info("=" * 80)
        for i, prompt in enumerate(prompts, 1):
            logger.info(f"\n--- Промпт #{i} ---")
            logger.info(prompt)
        logger.info("=" * 80)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Вызывается при завершении работы LLM"""
        logger.info("-" * 80)
        logger.info("ОТВЕТ ОТ LLM:")
        logger.info("-" * 80)
        for generation in response.generations:
            for gen in generation:
                logger.info(gen.text)
        logger.info("=" * 80)


class CustomChromaRetriever(BaseRetriever):
    """Кастомный ретривер для работы с ChromaDB коллекцией"""

    suffix: str = "4B"  # Модельный суффикс: "0.6B" или "4B"
    collection_name: str = "kb_embeddings"
    k: int = 4  # количество результатов
    use_cpu: bool = False  # Использовать CPU для embeddings

    def __init__(self, suffix: str = "4B", use_cpu: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.suffix = suffix
        self.use_cpu = use_cpu

        # Путь к ChromaDB с учётом суффикса
        chroma_db_path = Path(__file__).parent.parent / "task_3_vector_index" / "chroma" / f"chroma-{suffix}"

        # Инициализация ChromaDB клиента
        self._chroma_client = chromadb.PersistentClient(
            path=str(chroma_db_path), settings=Settings(anonymized_telemetry=False)
        )
        # Получить коллекцию
        self._collection = self._chroma_client.get_collection(name=self.collection_name)

        # Инициализация embeddings модели (Qwen3-Embedding)
        self._model = SentenceTransformer(
            f"Qwen/Qwen3-Embedding-{suffix}",
            device="cpu" if use_cpu else None,
            tokenizer_kwargs={"padding_side": "left"},
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        """
        Получить релевантные документы из ChromaDB по запросу.

        Args:
            query: текст вопроса
            run_manager: callback manager (опционально)

        Returns:
            список Document с метаданными (source, token_range, char_range)
        """
        # Получить эмбеддинг для запроса (так же, как в extract_embeddings_for_kb.py)
        query_embedding = self._model.encode(query).tolist()

        # Поиск в ChromaDB
        results = self._collection.query(query_embeddings=[query_embedding], n_results=self.k)

        # Преобразовать результаты в Document объекты с нумерацией
        documents = []
        if results["metadatas"] and len(results["metadatas"]) > 0:
            for i, metadata in enumerate(results["metadatas"][0], start=1):
                # Добавляем нумерацию прямо в page_content
                doc = Document(
                    page_content=f"[{i}] {metadata['text']}",
                    metadata={
                        "source": metadata["source_path"],
                        "token_range": (metadata["token_range_start"], metadata["token_range_end"]),
                        "char_range": (metadata["char_range_start"], metadata["char_range_end"]),
                    },
                )
                documents.append(doc)

        return documents


class StubRetriever(BaseRetriever):
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
                page_content="[1] SQLite - это встроенная SQL база данных, не требующая отдельного сервера.",
                metadata={
                    "source": "https://example.com/docs/sqlite_intro.html",
                    "token_range": (0, 15),
                    "char_range": (0, 85),
                },
            ),
            Document(
                page_content="[2] SQLite широко используется в мобильных приложениях на iOS и Android.",
                metadata={
                    "source": "https://example.com/docs/sqlite_mobile.html",
                    "token_range": (120, 135),
                    "char_range": (580, 650),
                },
            ),
            Document(
                page_content="[3] База данных SQLite хранится в одном файле на диске.",
                metadata={
                    "source": "/path/to/local/sqlite_storage.txt",
                    "token_range": (45, 58),
                    "char_range": (220, 273),
                },
            ),
            Document(
                page_content="[4] SQLite поддерживает транзакции ACID и большинство стандартных SQL операций.",
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


def load_ginecarum_prompts() -> Dict[str, str]:
    """Загрузить все секции промпта Ginecarum из папки"""
    prompt_dir = Path(__file__).parent / "ginecarum_prompt"
    
    sections = {
        "system": "system.txt",
        "example1_user": "example1_user.txt",
        "example1_assistant": "example1_assistant.txt",
        "example2_user": "example2_user.txt",
        "example2_assistant": "example2_assistant.txt",
        "query_user": "query_user.txt",
    }
    
    prompts = {}
    for key, filename in sections.items():
        file_path = prompt_dir / filename
        with open(file_path, "r", encoding="utf-8") as f:
            prompts[key] = f.read()
    
    return prompts


def create_rag_chain() -> RetrievalQAWithSourcesChain:
    """Создаёт RAG chain с Qwen 3 и кастомным ретривером"""

    # Создать callback для логирования промптов
    prompt_callback = PromptLoggingCallback()

    # Инициализация ChatOllama (вместо OllamaLLM) с callback
    llm = ChatOllama(model="qwen3:8b", temperature=0.5, callbacks=[prompt_callback])

    # Создать кастомный ретривер
    # retriever = StubRetriever()  # Для тестирования без ChromaDB
    retriever = CustomChromaRetriever(suffix="4B", use_cpu=True)  # Реальный поиск через ChromaDB

    # Загрузить промпт-темплейты из файлов
    prompts = load_ginecarum_prompts()

    # Создать ChatPromptTemplate с разными ролями для разных секций
    combine_prompt = ChatPromptTemplate.from_messages([
        # System message: основные инструкции
        SystemMessagePromptTemplate.from_template(prompts["system"]),
        # Few-shot Example 1 - User message
        HumanMessagePromptTemplate.from_template(prompts["example1_user"]),
        # Few-shot Example 1 - Assistant response
        AIMessagePromptTemplate.from_template(prompts["example1_assistant"]),
        # Few-shot Example 2 - User message
        HumanMessagePromptTemplate.from_template(prompts["example2_user"]),
        # Few-shot Example 2 - Assistant response
        AIMessagePromptTemplate.from_template(prompts["example2_assistant"]),
        # Actual query - User message with context and question
        HumanMessagePromptTemplate.from_template(prompts["query_user"]),
    ])

    # Промпт для форматирования отдельных документов (без index, т.к. он не поддерживается)
    document_prompt = PromptTemplate(template="{page_content}", input_variables=["page_content"])

    # Создаём кастомный document separator с нумерацией
    document_separator = "\n\n"

    # Создать цепочку RetrievalQAWithSourcesChain с правильными параметрами
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": combine_prompt,
            "document_prompt": document_prompt,
            "document_variable_name": "summaries",
            "document_separator": document_separator,
        },
        verbose=False,
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
    logger.info("\n" + "=" * 80)
    logger.info(f"ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}")
    logger.info("=" * 80)

    # Получить ответ от chain (промпт будет залогирован через callback)
    result = chain.invoke({"question": question})

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
