from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator

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


def update_vector_db_task(project_root_path: str):
    """
    Функция для выполнения в виртуальном окружении.
    Запускает обновление векторного индекса.
    
    Args:
        project_root_path: абсолютный путь к корню проекта
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

    # Пути к данным
    doc_dir_path = project_root / "task_2_sample_dataset/arcanum_articles/text_output_replaced"

    # Создаём путь с временной меткой: chroma_versions/4B/YYYY-MM-DD_hh-mm-ss
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_chroma_path = project_root / "task_6_scheduled_updating" / "chroma_versions" / "4B" / timestamp

    # Путь к БД метаданных
    metadata_db_path = project_root / "task_6_scheduled_updating" / "metadata.db"

    logging.info(f"Запуск обновления векторного индекса...")
    logging.info(f"Директория документов: {doc_dir_path}")
    logging.info(f"Путь к новой ChromaDB: {new_chroma_path}")
    logging.info(f"БД метаданных: {metadata_db_path}")

    # Запуск обновления
    result_path = update_kb_index(
        doc_dir_path=doc_dir_path,
        new_chroma_dir_path=new_chroma_path,
        metadata_db_path=metadata_db_path,
        qwen3_suffix="4B",
    )

    logging.info(f"✅ Обновление завершено. Результат сохранён в: {result_path}")
    return str(result_path)


# Вычисляем путь к проекту один раз
PROJECT_ROOT = str(Path(__file__).parent.parent.parent.resolve())

# Определение DAG
with DAG(
    dag_id="update_kb_vector_index",
    default_args=default_args,
    description="Обновление векторного индекса базы знаний (ChromaDB) каждый час",
    schedule="@hourly",  # Запуск раз в час
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["knowledge-base", "vector-index", "chromadb"],
) as dag:

    # Оператор для обновления векторной БД в виртуальном окружении
    update_vector_index = PythonVirtualenvOperator(
        task_id="update_vector_db",
        python_callable=update_vector_db_task,
        op_kwargs={"project_root_path": PROJECT_ROOT},
        requirements=["chromadb>=0.4.0", "sentence-transformers>=2.2.0", "sqlalchemy>=2.0.0", "more-itertools>=10.0.0"],
        system_site_packages=True,  # Использовать системные пакеты
        python_version="3.11",  # Укажите вашу версию Python
    )


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
        result = update_vector_db_task(project_root)
        logging.info("=" * 60)
        logging.info(f"✅ Тест успешно завершён!")
        logging.info(f"📁 Результат: {result}")
    except Exception as e:
        logging.error("=" * 60)
        logging.error(f"❌ Ошибка при выполнении теста: {e}")
        import traceback

        traceback.print_exc()
        raise
