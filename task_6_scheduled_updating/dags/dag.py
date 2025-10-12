from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator, PythonOperator

import sys

sys.path.append(str(Path(__file__).parent.parent.parent.resolve()))

# Параметры по умолчанию для DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


def update_vector_db_task(project_root_path: str, doc_dir_path: str, metadata_db_path: str, chroma_versions_dir: str):
    """
    Функция для выполнения в виртуальном окружении.
    Запускает обновление векторного индекса.

    Args:
        project_root_path: абсолютный путь к корню проекта
        doc_dir_path: путь к директории с документами базы знаний
        metadata_db_path: путь к БД метаданных
        chroma_versions_dir: путь к папке для хранения версий Chroma-базы
    """
    import logging
    import sys
    from datetime import datetime
    from pathlib import Path

    # Преобразуем строку в Path
    project_root = Path(project_root_path)

    # Добавляем путь к проекту в sys.path
    sys.path.insert(0, str(project_root))

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from task_6_scheduled_updating.update_chroma_db import update_kb_index

    # Преобразуем строки в Path
    doc_dir_path = Path(doc_dir_path)
    metadata_db_path = Path(metadata_db_path)
    chroma_versions_dir = Path(chroma_versions_dir)

    # Создаём путь с временной меткой: chroma_versions/4B/YYYY-MM-DD_hh-mm-ss
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_chroma_path = chroma_versions_dir / "4B" / timestamp

    logging.info(f"Запуск обновления векторного индекса...")
    logging.info(f"Директория документов: {doc_dir_path}")
    logging.info(f"Путь к новой ChromaDB: {new_chroma_path}")
    logging.info(f"БД метаданных: {metadata_db_path}")

    # Запуск обновления
    result_path = update_kb_index(
        doc_dir_path=doc_dir_path, new_chroma_dir_path=new_chroma_path, metadata_db_path=metadata_db_path
    )

    logging.info(f"✅ Обновление завершено. Результат сохранён в: {result_path}")
    return str(result_path)


def cleanup_old_indexes_task(project_root_path: str, metadata_db_path: str):
    """
    Функция для выполнения в виртуальном окружении.
    Удаляет старые векторные индексы (старше max_age).

    Args:
        project_root_path: абсолютный путь к корню проекта
        metadata_db_path: путь к БД метаданных
    """
    import logging
    import sys
    from datetime import timedelta
    from pathlib import Path

    # Преобразуем строку в Path
    project_root = Path(project_root_path)

    # Добавляем путь к проекту в sys.path
    sys.path.insert(0, str(project_root))

    # Настройка логирования
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    from task_6_scheduled_updating.cleanup_indexes import drop_old_indexes

    # Преобразуем строку в Path
    metadata_db_path = Path(metadata_db_path)

    max_age = timedelta(minutes=30)

    logging.info(f"🧹 Запуск очистки старых индексов (возраст > {max_age})...")
    logging.info(f"БД метаданных: {metadata_db_path}")

    # Запуск очистки
    result = drop_old_indexes(metadata_db_path=metadata_db_path, max_age=max_age)

    if result:
        logging.info(f"✅ Очистка завершена. Удалено индексов: {result.get('deleted_indexes', 0)}")
        return result
    else:
        logging.info("✅ Очистка завершена. Ничего не удалено.")
        return {"deleted_indexes": 0, "deleted_dirs": []}


# Вычисляем пути один раз
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())
DOC_DIR_PATH = str(Path(PROJECT_ROOT) / "task_2_sample_dataset/arcanum_articles/text_output_replaced")
METADATA_DB_PATH = str(Path(PROJECT_ROOT) / "task_6_scheduled_updating" / "metadata.db")
CHROMA_VERSIONS_DIR = str(Path(PROJECT_ROOT) / "task_6_scheduled_updating" / "chroma_versions")

# Определение DAG
with DAG(
    dag_id="update_kb_vector_index",
    default_args=default_args,
    description="Обновление векторного индекса базы знаний (ChromaDB)",
    schedule="*/5 * * * *",  # Запуск каждые 5 минут
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["knowledge-base", "vector-index", "chromadb"],
) as dag:

    # Оператор для обновления векторной БД в виртуальном окружении
    update_vector_index = PythonVirtualenvOperator(
        task_id="update_vector_db",
        python_callable=update_vector_db_task,
        op_kwargs={
            "project_root_path": PROJECT_ROOT,
            "doc_dir_path": DOC_DIR_PATH,
            "metadata_db_path": METADATA_DB_PATH,
            "chroma_versions_dir": CHROMA_VERSIONS_DIR,
        },
        requirements=[
            "chromadb>=0.4.0",
            "sentence-transformers>=2.2.0",
            "sqlalchemy>=2.0.0",
            "more-itertools>=10.0.0",
            "contexttimer",
            # Torch ставим не слишком новый, чтобы работал с моей старушкой GTX 1080Ti
            "torch<=2.7.1",
            # Для 8-битного квантования:
            "bitsandbytes",
            "accelerate>=0.26.0",
        ],
        system_site_packages=True,  # Использовать системные пакеты
        python_version="3.11",  # Укажите вашу версию Python
    )

    # Оператор для очистки старых индексов (запускается только после успешного обновления)
    # Используем обычный PythonOperator, так как зависимости уже установлены в основном окружении
    cleanup_old_indexes = PythonOperator(
        task_id="cleanup_old_indexes",
        python_callable=cleanup_old_indexes_task,
        op_kwargs={"project_root_path": PROJECT_ROOT, "metadata_db_path": METADATA_DB_PATH},
    )

    # Устанавливаем зависимость: сначала обновление, потом очистка
    update_vector_index >> cleanup_old_indexes


if __name__ == "__main__":
    """
    Тестовый запуск DAG в debug-режиме.
    Напрямую вызывает функцию обновления без Airflow инфраструктуры.
    """
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("🧪 ТЕСТОВЫЙ ЗАПУСК DAG В DEBUG-РЕЖИМЕ")
    logging.info("=" * 60)

    try:
        # Передаём путь к проекту в функцию
        project_root = str(Path(__file__).parent.parent.parent.resolve())

        # Вычисляем пути
        doc_dir = str(Path(project_root) / "task_2_sample_dataset/arcanum_articles/text_output_replaced")
        metadata_db = str(Path(project_root) / "task_6_scheduled_updating" / "metadata.db")
        chroma_versions = str(Path(project_root) / "task_6_scheduled_updating" / "chroma_versions")

        # 1. Обновление индекса
        result = update_vector_db_task(project_root, doc_dir, metadata_db, chroma_versions)
        logging.info("=" * 60)
        logging.info(f"✅ Обновление завершено!")
        logging.info(f"📁 Результат: {result}")

        # 2. Очистка старых индексов
        logging.info("\n" + "=" * 60)
        cleanup_result = cleanup_old_indexes_task(project_root, metadata_db)
        logging.info("=" * 60)
        logging.info(f"✅ Очистка завершена!")
        logging.info(f"🗑️  Результат: {cleanup_result}")

    except Exception as e:
        logging.error("=" * 60)
        logging.error(f"❌ Ошибка при выполнении теста: {e}")
        import traceback

        traceback.print_exc()
        raise
