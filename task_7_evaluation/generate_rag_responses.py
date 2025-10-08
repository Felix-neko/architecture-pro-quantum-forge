#!/usr/bin/env python3
"""
Скрипт для генерации ответов RAG-системы на золотые вопросы из gold_questions.yaml
Сохраняет расширенные данные включая context (отфильтрованный контекст) и answer
"""

import logging
from pathlib import Path
from typing import List, Dict, Any
import yaml

from task_4_rag.rag_example import create_rag_chain, answer_question


logger = logging.getLogger(__name__)


def load_gold_questions(yaml_path: Path) -> List[Dict[str, str]]:
    """
    Загружает золотые вопросы из YAML файла

    Args:
        yaml_path: путь к gold_questions.yaml

    Returns:
        список словарей с вопросами и ожидаемыми ответами
    """
    with open(yaml_path, "r", encoding="utf-8") as f:
        questions = yaml.safe_load(f)
    return questions


def generate_enriched_questions(gold_questions: List[Dict[str, str]], chain) -> List[Dict[str, Any]]:
    """
    Генерирует обогащенные вопросы с ответами RAG и контекстом

    Args:
        gold_questions: список золотых вопросов
        chain: RAG цепочка

    Returns:
        список обогащенных словарей с дополнительными полями
    """
    enriched = []

    for i, q_item in enumerate(gold_questions, 1):
        question = q_item["question"]
        expected_answer = q_item["expected_answer"]

        logger.info(f"\n{'='*80}")
        logger.info(f"Обработка вопроса {i}/{len(gold_questions)}: {question}")
        logger.info(f"{'='*80}")

        # Получить ответ от RAG
        result = answer_question(chain, question)

        # Извлечь контекст из source_documents
        source_documents = result.get("source_documents", [])
        context = [doc.page_content for doc in source_documents]

        # Извлечь финальный ответ
        answer = result.get("final_answer", result.get("answer", ""))

        # Создать обогащенный словарь
        enriched_item = {"question": question, "expected_answer": expected_answer, "context": context, "answer": answer}

        enriched.append(enriched_item)

        logger.info(f"✓ Получен ответ ({len(answer)} символов)")
        logger.info(f"  Context chunks: {len(context)}")

    return enriched


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Пути к файлам (относительно этого скрипта)
    script_dir = Path(__file__).parent
    gold_questions_path = script_dir / "gold_questions.yaml"
    output_path = script_dir / "enriched_gold_questions.yaml"

    # Путь к ChromaDB (относительно корня проекта)
    chroma_db_path = script_dir.parent / "task_3_vector_index" / "chroma" / "chroma-4B"

    logger.info("=" * 80)
    logger.info("Генерация RAG ответов на золотые вопросы")
    logger.info("=" * 80)
    logger.info(f"Gold questions: {gold_questions_path}")
    logger.info(f"ChromaDB path: {chroma_db_path}")
    logger.info(f"Output: {output_path}")

    # Проверить существование файлов
    if not gold_questions_path.exists():
        logger.error(f"Файл не найден: {gold_questions_path}")
        return

    if not chroma_db_path.exists():
        logger.error(f"ChromaDB не найдена: {chroma_db_path}")
        return

    # Загрузить золотые вопросы
    logger.info("\nЗагрузка золотых вопросов...")
    gold_questions = load_gold_questions(gold_questions_path)
    logger.info(f"✓ Загружено {len(gold_questions)} вопросов")

    # Создать RAG chain с фильтрацией
    logger.info("\nИнициализация RAG системы...")
    chain = create_rag_chain(use_chunk_filtering=True)
    logger.info("✓ RAG система инициализирована")

    # Генерировать обогащенные вопросы
    logger.info("\nОбработка вопросов...")
    enriched_questions = generate_enriched_questions(gold_questions, chain)

    # Сохранить результаты
    logger.info(f"\nСохранение результатов в {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(enriched_questions, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    logger.info("=" * 80)
    logger.info("✓ Готово!")
    logger.info(f"Результаты сохранены в: {output_path}")
    logger.info(f"Обработано вопросов: {len(enriched_questions)}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
