from datetime import datetime
from typing import List

from sqlalchemy import String, DateTime, ForeignKey, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Базовый класс для всех моделей"""

    pass


class VectorIndexVersion(Base):
    """Версия векторного индекса с информацией о ChromaDB"""

    __tablename__ = "vector_index_versions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    path: Mapped[str] = mapped_column(String, nullable=False, comment="Путь к директории с ChromaDB")
    created_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=datetime.utcnow, comment="Дата создания"
    )

    # Связь один-ко-многим с DocHash
    doc_hashes: Mapped[List["DocHash"]] = relationship(
        "DocHash", back_populates="vector_index", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<VectorIndexVersion(id={self.id}, path={self.path}, created_at={self.created_at}, hashes={len(self.doc_hashes)})>"


class DocHash(Base):
    """Хеш документа в конкретной версии индекса"""

    __tablename__ = "doc_hashes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    vector_index_id: Mapped[int] = mapped_column(
        ForeignKey("vector_index_versions.id"), nullable=False, comment="ID версии индекса"
    )
    path: Mapped[str] = mapped_column(String, nullable=False, comment="Путь к документу")
    hash: Mapped[str] = mapped_column(String, nullable=False, comment="Blake2b хеш документа")

    # Обратная связь к VectorIndexVersion
    vector_index: Mapped["VectorIndexVersion"] = relationship("VectorIndexVersion", back_populates="doc_hashes")

    def __repr__(self) -> str:
        return f"<DocHash(id={self.id}, path={self.path}, hash={self.hash[:16]}...)>"


def create_db(db_path: str = "metadata.db") -> None:
    """Создать базу данных и все таблицы"""
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    print(f"✅ База данных создана: {db_path}")


if __name__ == "__main__":
    # Пример создания базы данных
    create_db("metadata.db")
