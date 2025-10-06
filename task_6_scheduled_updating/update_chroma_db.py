import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict

import chromadb
from chromadb.config import Settings
from more_itertools import chunked
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

# Импорт функций из task_3_vector_index
sys.path.insert(0, str(Path(__file__).parent.parent))
from task_3_vector_index.extract_embeddings_for_kb import encode_documents
from task_6_scheduled_updating.file_hashing import calculate_hashes, compare_hashes
from task_6_scheduled_updating.sqla_models import Base, VectorIndexVersion, DocHash

N_DOCS_PER_BATCH = 3


def update_embeddings(
    old_chroma_dir_path: Optional[Path],
    new_chroma_dir_path: Path,
    new_files: List[Path],
    modified_files: List[Path],
    deleted_files: List[Path],
    qwen3_suffix="4B",
) -> Path:
    """
    Копирует старую директорию ChromaDB в new_chroma_dir (если old_chroma_dir_path == None, то создаёт там новую базу).
    Затем удаляет оттуда modified_files и deleted_files.
    Затем считает эмбеддинги по new_files и modified_files и добавляет их в базу.

    Args:
        old_chroma_dir_path: путь к старой ChromaDB (или None для создания новой)
        new_chroma_dir_path: путь для новой ChromaDB
        new_files: список новых файлов для добавления
        modified_files: список измененных файлов для обновления
        deleted_files: список удаленных файлов
        qwen3_suffix: суффикс модели Qwen ("0.6B" или "4B")

    Returns:
        путь к новой ChromaDB
    """
    # 1. Копировать или создать базу
    if old_chroma_dir_path is not None and old_chroma_dir_path.exists() and old_chroma_dir_path.is_dir():
        # Копируем старую базу
        if new_chroma_dir_path.exists() and new_chroma_dir_path.is_dir():
            shutil.rmtree(new_chroma_dir_path, ignore_errors=True)
        shutil.copytree(old_chroma_dir_path, new_chroma_dir_path)
        logging.info(f"✅ Скопирована база из {old_chroma_dir_path} в {new_chroma_dir_path}")
    else:
        # Создаем новую базу (директория создастся автоматически при создании клиента)
        logging.info(f"🆕 Создается новая база в {new_chroma_dir_path}")

    # 2. Открыть ChromaDB
    chroma_client = chromadb.PersistentClient(
        path=str(new_chroma_dir_path), settings=Settings(anonymized_telemetry=False)
    )

    collection_name = "kb_embeddings"

    # Получить или создать коллекцию
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except:
        collection = chroma_client.create_collection(name=collection_name, metadata={"hnsw:space": "l2"})

    # 3. Удалить записи для modified_files и deleted_files
    files_to_delete = deleted_files + modified_files
    if files_to_delete:
        logging.info(f"🗑️  Удаление {len(files_to_delete)} файлов из базы...")
        for file_path in files_to_delete:
            # Удаляем все чанки, связанные с этим файлом
            try:
                collection.delete(where={"source_path": str(file_path)})
            except Exception as e:
                logging.warning(f"⚠️  Ошибка при удалении {file_path}: {e}")

    # 4. Вычислить эмбеддинги для new_files и modified_files
    files_to_add = new_files + modified_files
    if files_to_add:
        logging.info(f"➕ Добавление {len(files_to_add)} файлов в базу (батчами по {N_DOCS_PER_BATCH})...")

        # Загрузить модель эмбеддингов
        use_cpu = True  # Можно сделать параметром
        model = SentenceTransformer(
            f"Qwen/Qwen3-Embedding-{qwen3_suffix}",
            device="cpu" if use_cpu else None,
            tokenizer_kwargs={"padding_side": "left"},
        )

        # Обработка файлов батчами
        chunk_idx = 0
        total_chunks_added = 0
        total_files = len(files_to_add)

        for batch_num, file_batch in enumerate(chunked(files_to_add, N_DOCS_PER_BATCH), start=1):
            start_doc = (batch_num - 1) * N_DOCS_PER_BATCH + 1
            end_doc = min(batch_num * N_DOCS_PER_BATCH, total_files)
            logging.info(f"  📦 Батч {batch_num}: обработка документов с {start_doc} по {end_doc}...")

            # Загрузить документы из текущего батча
            documents = {}
            for file_path in file_batch:
                if file_path.exists() and file_path.is_file():
                    with open(file_path, "r", encoding="utf-8") as f:
                        documents[file_path] = f.read()

            if not documents:
                logging.warning(f"  ⚠️  Батч {batch_num}: нет валидных файлов для обработки")
                continue

            # Вычислить эмбеддинги для батча
            embeddings_list = encode_documents(model, documents)

            # Подготовить данные для добавления в ChromaDB
            batch_embeddings = []
            batch_metadatas = []
            batch_ids = []

            for doc_info in embeddings_list:
                for i, chunk_info in enumerate(doc_info.chunks):
                    # Используем .dict() для получения метаданных
                    metadata = chunk_info.dict()
                    metadata["source_path"] = str(doc_info.original_text_path)

                    batch_embeddings.append(doc_info.embeddings[i].tolist())
                    batch_metadatas.append(metadata)
                    batch_ids.append(f"chunk_{chunk_idx}")
                    chunk_idx += 1

            # Добавить батч в базу
            if batch_embeddings:
                collection.add(embeddings=batch_embeddings, metadatas=batch_metadatas, ids=batch_ids)
                total_chunks_added += len(batch_embeddings)
                logging.info(f"  ✅ Батч {batch_num}: добавлено {len(batch_embeddings)} чанков")

        if total_chunks_added > 0:
            logging.info(f"✅ Всего добавлено {total_chunks_added} чанков из {len(files_to_add)} файлов")

    logging.info(f"✅ База обновлена: {new_chroma_dir_path}")
    return new_chroma_dir_path


def update_kb_index(
    doc_dir_path: Path,
    new_chroma_dir_path: Path,
    metadata_db_path: Path = Path(__file__).parent / "metadata.db",
    qwen3_suffix: str = "4B",
) -> Path:
    """
    Обновляем векторный индекс базы знаний:
    - открываем SQLite-БД метаданных;
    - ищем там последнюю по времени версию векторного индекса (если есть) -- т.е. предыдущую версию хешей;
    - считаем хеши всех документов из doc_dir_path (с помощью calculate_hashes) -- т.е. актуальную версию хешей;
    - Сравниваем актуальную версию хешей с предыдущей (с помощью compare_hashes)
    - обновляем ChromaDB с помощью update_embeddings (новую базу размещаем в new_chroma_dir_path)
    - в случае успеха записываем в БД метаданных информацию о новом векторном индексе (путь к Chroma-базе и набор хешей)

    Returns:
        путь к новой ChromaDB
    """
    logging.info("=== Начало обновления векторного индекса ===")

    # 1. Открыть или создать БД метаданных
    engine = create_engine(f"sqlite:///{metadata_db_path}", echo=False)
    Base.metadata.create_all(engine)
    logging.info(f"✅ БД метаданных: {metadata_db_path}")

    # 2. Найти последнюю версию векторного индекса
    old_chroma_path: Optional[Path] = None
    old_hashes: Dict[Path, str] = {}

    with Session(engine) as session:
        # Запрос последней версии по дате создания
        stmt = select(VectorIndexVersion).order_by(VectorIndexVersion.created_at.desc()).limit(1)
        last_version = session.scalars(stmt).first()

        if last_version:
            old_chroma_path = Path(last_version.path)
            logging.info(f"📂 Найдена предыдущая версия индекса: {old_chroma_path}")
            logging.info(f"   Создана: {last_version.created_at}")
            logging.info(f"   Документов: {len(last_version.doc_hashes)}")

            # Загрузить старые хеши
            for doc_hash in last_version.doc_hashes:
                old_hashes[Path(doc_hash.path)] = doc_hash.hash
        else:
            logging.info("🆕 Предыдущих версий не найдено, создаём индекс с нуля")

    # 3. Вычислить актуальные хеши документов
    logging.info(f"🔍 Вычисление хешей документов в {doc_dir_path}...")
    new_hashes = calculate_hashes(doc_dir_path)
    logging.info(f"✅ Найдено документов: {len(new_hashes)}")

    # 4. Сравнить хеши
    new_files, modified_files, deleted_files = compare_hashes(old_hashes, new_hashes)
    logging.info(f"📊 Изменения:")
    logging.info(f"   Новые файлы: {len(new_files)}")
    logging.info(f"   Изменённые файлы: {len(modified_files)}")
    logging.info(f"   Удалённые файлы: {len(deleted_files)}")

    # 5. Обновить ChromaDB
    logging.info(f"🔄 Обновление ChromaDB...")
    new_chroma_path = update_embeddings(
        old_chroma_dir_path=old_chroma_path,
        new_chroma_dir_path=new_chroma_dir_path,
        new_files=new_files,
        modified_files=modified_files,
        deleted_files=deleted_files,
        qwen3_suffix=qwen3_suffix,
    )

    # 6. Записать информацию о новой версии в БД
    logging.info(f"💾 Сохранение метаданных новой версии...")
    with Session(engine) as session:
        # Создать новую версию индекса
        new_version = VectorIndexVersion(path=str(new_chroma_dir_path.resolve()), created_at=datetime.utcnow())

        # Добавить хеши документов
        for file_path, file_hash in new_hashes.items():
            doc_hash = DocHash(path=str(file_path.resolve()), hash=file_hash)
            new_version.doc_hashes.append(doc_hash)

        session.add(new_version)
        session.commit()

        logging.info(f"✅ Версия индекса сохранена (ID: {new_version.id})")
        logging.info(f"   Путь: {new_version.path}")
        logging.info(f"   Документов: {len(new_version.doc_hashes)}")

    logging.info("=== Обновление векторного индекса завершено ===")
    return new_chroma_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    old_doc_folder_path = Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced"

    # Создаём путь с временной меткой: chroma_versions/4B/YYYY-MM-DD_hh-mm-ss
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_chroma_path = Path(__file__).parent / "chroma_versions" / "4B" / timestamp

    update_kb_index(old_doc_folder_path, new_chroma_path)

    # old_hashes = calculate_hashes(old_folder_path)
    # update_embeddings(
    #     Path(__file__).parent.parent / "task_3_vector_index/chroma/chroma-4B",
    #     Path("new_chroma-4B"),
    #     new_files=list(old_hashes.keys()),
    #     modified_files=[],
    #     deleted_files=[],
    # )
