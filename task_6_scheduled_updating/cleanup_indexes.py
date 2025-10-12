import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from task_6_scheduled_updating.sqla_models import Base, VectorIndexVersion


def drop_old_indexes(metadata_db_path: Path, max_age=timedelta(days=3)):
    """
    Открывает БД метаданных, ищет VectorIndex старше max_age от datetime.now(), удаляет их папки c ChromaDB,
    потом удаляет из базы метаданных и информацию об этих VectorIndex

    Args:
        metadata_db_path: путь к БД метаданных
        max_age: максимальный возраст индекса (по умолчанию 3 дня)
    """
    logging.info("=== Начало удаления старых индексов ===")

    # 1. Открыть БД метаданных
    engine = create_engine(f"sqlite:///{metadata_db_path}", echo=False)
    Base.metadata.create_all(engine)
    logging.info(f"✅ БД метаданных: {metadata_db_path}")

    # 2. Найти старые версии индекса
    cutoff_time = datetime.utcnow() - max_age
    logging.info(f"🔍 Поиск индексов старше {max_age} (до {cutoff_time})")

    deleted_count = 0
    deleted_dirs = []

    with Session(engine) as session:
        # Запрос всех версий старше cutoff_time
        stmt = select(VectorIndexVersion).where(VectorIndexVersion.created_at < cutoff_time)
        old_versions = session.scalars(stmt).all()

        if not old_versions:
            logging.info("✅ Старых индексов не найдено")
            return

        logging.info(f"📊 Найдено старых индексов для удаления: {len(old_versions)}")

        for version in old_versions:
            chroma_path = Path(version.path)

            # Удалить папку ChromaDB
            if chroma_path.exists() and chroma_path.is_dir():
                try:
                    shutil.rmtree(chroma_path, ignore_errors=False)
                    logging.info(f"🗑️  Удалена папка: {chroma_path}")
                    deleted_dirs.append(str(chroma_path))
                except Exception as e:
                    logging.warning(f"⚠️  Ошибка при удалении папки {chroma_path}: {e}")
            else:
                logging.warning(f"⚠️  Папка не существует: {chroma_path}")

            # Удалить запись из БД (каскадно удалятся и DocHash)
            logging.info(
                f"🗑️  Удаление VectorIndexVersion (ID: {version.id}, "
                f"created: {version.created_at}, docs: {len(version.doc_hashes)})"
            )
            session.delete(version)
            deleted_count += 1

        # Сохранить изменения
        session.commit()

    logging.info(f"✅ Удалено индексов: {deleted_count}")
    logging.info(f"✅ Удалено папок: {len(deleted_dirs)}")
    logging.info("=== Удаление старых индексов завершено ===")

    return {"deleted_indexes": deleted_count, "deleted_dirs": deleted_dirs}
