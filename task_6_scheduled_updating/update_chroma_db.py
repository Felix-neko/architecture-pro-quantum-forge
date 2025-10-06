import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from more_itertools import chunked
from sentence_transformers import SentenceTransformer

# Импорт функций из task_3_vector_index
sys.path.insert(0, str(Path(__file__).parent.parent))
from task_3_vector_index.extract_embeddings_for_kb import encode_documents
from task_6_scheduled_updating.file_hashing import calculate_hashes, compare_hashes

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

        for batch_num, file_batch in enumerate(chunked(files_to_add, N_DOCS_PER_BATCH), start=1):
            logging.info(f"  📦 Батч {batch_num}: обработка {len(file_batch)} файлов...")

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
):
    """
    Обновляем векторный индекс базы знаний:
    - j
    -  Ищем последний по порядку .pck-файл в hashes_dir_path (считает его хешами предыдущей версии документов)
    - Считаем хеши всех документов doc_dir_path (с помощью calculate_hashes)
    - Актуальные хеши сохраняем в `hashes_dir_path` / f"{yyyy}-{mm}-{dd}_{hh}-{mm}-{ss}.pck"
    - Делаем сравнение хешей с помощью compare_hashes и обновление ChromaDB с помощью update_embeddings

    """


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    old_folder_path = Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced"

    old_hashes = calculate_hashes(old_folder_path)
    update_embeddings(
        Path(__file__).parent.parent / "task_3_vector_index/chroma/chroma-4B",
        Path("new_chroma-4B"),
        new_files=list(old_hashes.keys()),
        modified_files=[],
        deleted_files=[],
    )
