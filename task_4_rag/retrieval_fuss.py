"""
RAG чат-бот с указанием источников на базе Ollama + LangChain
Требует: pip install langchain langchain-ollama chromadb
"""

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.schema import Document


def create_sample_documents():
    """Создаём примеры документов для демонстрации"""
    docs = [
        Document(
            page_content="Питон был создан Гвидо ван Россумом в 1991 году. "
            "Язык назван в честь комедийной группы Monty Python.",
            metadata={"source": "python_history.txt"},
        ),
        Document(
            page_content="Julia — высокопроизводительный язык для научных вычислений. "
            "Разработан в MIT, впервые выпущен в 2012 году.",
            metadata={"source": "julia_overview.txt"},
        ),
        Document(
            page_content="Zig — современный системный язык программирования. "
            "Фокусируется на простоте, производительности и безопасности. "
            "Создан Эндрю Келли.",
            metadata={"source": "zig_intro.txt"},
        ),
        Document(
            page_content="SQLite — встраиваемая SQL база данных. "
            "Это самая используемая СУБД в мире. "
            "Код отличается высоким качеством и надёжностью.",
            metadata={"source": "databases.txt"},
        ),
    ]
    return docs


def setup_rag_system():
    """Настраивает RAG-систему с векторным хранилищем"""

    # LLM модель из Ollama для генерации ответов
    llm = OllamaLLM(model="qwen3:8b", temperature=0)

    # Embeddings модель — ВАЖНО: используем специализированную модель!
    # Альтернативы: "mxbai-embed-large", "all-minilm"
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    # Загружаем документы
    documents = create_sample_documents()

    # Разбиваем на чанки (здесь документы уже маленькие)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splits = text_splitter.split_documents(documents)

    # Создаём векторное хранилище
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, collection_name="knowledge_base")

    # Настраиваем цепочку QA с источниками
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 'stuff' помещает все документы в один промпт
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
    )

    return chain


def ask_question(chain, question: str):
    """Задаёт вопрос и выводит ответ с источниками"""
    print(f"\n{'='*70}")
    print(f"Вопрос: {question}")
    print("=" * 70)

    result = chain.invoke({"question": question})

    print(f"\nОтвет:\n{result['answer']}")

    if result.get("sources"):
        print(f"\nИсточники: {result['sources']}")

    if result.get("source_documents"):
        print(f"\nНайденные документы ({len(result['source_documents'])}):")
        for i, doc in enumerate(result["source_documents"], 1):
            print(f"\n  {i}. [{doc.metadata['source']}]")
            print(f"     {doc.page_content[:150]}...")


def main():
    print("🔧 Инициализация RAG-системы...")
    chain = setup_rag_system()

    # Примеры вопросов
    questions = [
        "Кто создал язык Python и когда?",
        "Расскажи о языке Julia",
        "Какие языки программирования упоминаются в базе знаний?",
        "Что известно про SQLite?",
    ]

    for q in questions:
        ask_question(chain, q)

    # Интерактивный режим
    print(f"\n\n{'='*70}")
    print("💬 Интерактивный режим (введите 'exit' для выхода)")
    print("=" * 70)

    while True:
        question = input("\nВаш вопрос: ").strip()
        if question.lower() in ("exit", "quit", "выход"):
            break
        if question:
            ask_question(chain, question)


if __name__ == "__main__":
    main()
