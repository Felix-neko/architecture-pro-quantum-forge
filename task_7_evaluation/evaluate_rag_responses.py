#!/usr/bin/env python3
"""
Скрипт для запуска Ragas evaluation с локальной Ollama-моделью qwen3:8b.
Использует РУССКИЕ ИНСТРУКЦИИ из russian_prompts.py для оценки RAG-системы.

Требования:
  - ollama установлен и запущен (https://ollama.com)
  - модель qwen3:8b загружена: `ollama pull qwen3:8b`
  - Python-пакеты: ragas, langchain-ollama, pyyaml, pydantic
  - sentence-transformers (автоматически установится с ragas)
  - Файл russian_prompts.py в том же каталоге (содержит русские промпты)

Метрики оценки:
  1. answer_relevancy - релевантность ответа вопросу
  2. faithfulness - достоверность (отсутствие галлюцинаций)
  3. context_precision - точность ранжирования контекста
  4. context_recall - полнота извлечения контекста

Пример запуска:
  ollama pull qwen3:8b
  python evaluate_rag_responses.py --input enriched_gold_questions_with_unfiltered_context.yaml --out results.json
  
  # Для использования русских промптов:
  python evaluate_rag_responses.py --input data.yaml --out results.json --use-russian-templates

Примечание: Скрипт автоматически применяет русские промпты к метрикам Ragas.
Если russian_prompts.py не найден, будут использованы стандартные англоязычные промпты.
"""
import argparse
import json
import logging
import sys
import yaml
from pathlib import Path
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, faithfulness, context_precision, context_recall

# Импортируем русские промпты и шаблоны
try:
    from russian_prompts import (
        PROMPTS,
        PROMPT_EXAMPLES,
        ANSWER_RELEVANCY_TEMPLATE,
        FAITHFULNESS_TEMPLATE,
        CONTEXT_PRECISION_TEMPLATE,
        CONTEXT_RECALL_TEMPLATE,
    )

    RUSSIAN_PROMPTS_AVAILABLE = True
except ImportError:
    RUSSIAN_PROMPTS_AVAILABLE = False
    logging.warning("Не удалось импортировать russian_prompts.py - будут использованы стандартные промпты")

# Описание метрик:
# - answer_relevancy (релевантность ответа): оценивает, насколько ответ релевантен вопросу
# - faithfulness (достоверность): проверяет, насколько ответ основан на предоставленном контексте
# - context_precision (точность контекста): оценивает, насколько релевантные контексты находятся выше в списке
# - context_recall (полнота контекста): проверяет, весь ли необходимый контекст был извлечен

# ВАЖНО: Как проверить, что русские промпты применились:
# 1. При запуске скрипта проверьте вывод "🌍 ПРИМЕНЕНИЕ РУССКИХ ПРОМПТОВ К МЕТРИКАМ RAGAS"
# 2. Убедитесь, что видите "✅ Русские промпты с few-shot примерами успешно применены"
# 3. В логах запросов к LLM (🔵 LLM REQUEST) проверьте язык инструкций - они должны быть на русском
# 4. Если видите английские промпты, проверьте:
#    - Наличие файла russian_prompts.py в каталоге task_7_evaluation/
#    - Правильность импорта (нет ошибок при импорте)
#    - Структуру метрик Ragas (может отличаться в разных версиях)

# LangChain imports
try:
    from langchain_ollama import ChatOllama
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
    from sentence_transformers import SentenceTransformer
except Exception as e:
    print(f"Ошибка импорта: {e}")
    print("Требуются пакеты: pip install langchain-ollama langchain-huggingface")
    sys.exit(1)


class LLMLoggingCallback(BaseCallbackHandler):
    """Callback для логирования всех запросов к LLM и ответов"""

    def __init__(self):
        self.request_count = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        """Вызывается при начале запроса к LLM"""
        self.request_count += 1
        print(f"\n{'='*80}")
        print(f"🔵 LLM REQUEST #{self.request_count}")
        print(f"{'='*80}")
        for i, prompt in enumerate(prompts, 1):
            print(f"\n--- Промпт {i}/{len(prompts)} ---")
            print(prompt)
            print(f"--- Конец промпта {i} ---")

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Вызывается при получении ответа от LLM"""
        print(f"\n{'='*80}")
        print(f"🟢 LLM RESPONSE #{self.request_count}")
        print(f"{'='*80}")
        for i, generation in enumerate(response.generations, 1):
            for j, gen in enumerate(generation, 1):
                print(f"\n--- Ответ {i}.{j} ---")
                print(gen.text)
                print(f"--- Конец ответа {i}.{j} ---")
        print(f"{'='*80}\n")


def apply_russian_prompts_to_metrics():
    """
    Применяет русские промпты к метрикам Ragas через метод set_prompts().

    В Ragas каждая метрика имеет внутренние промпты, которые можно заменить.
    Мы заменяем их на подготовленные русские промпты с few-shot примерами.
    """
    if not RUSSIAN_PROMPTS_AVAILABLE:
        logging.warning("Русские промпты недоступны, используются стандартные")
        return False

    try:
        # Проверяем доступность Prompt объектов
        if PROMPTS is None:
            logging.warning("Prompt объекты не созданы в russian_prompts.py")
            return False

        logging.info("Применение русских Prompt объектов к метрикам...")

        # Сопоставление метрик с их промптами и ключами
        # Каждая метрика имеет свои специфичные ключи промптов
        prompt_mappings = {
            "answer_relevancy": {
                "metric": answer_relevancy,
                "prompt_key": "response_relevance_prompt",  # Ключ в set_prompts()
                "russian_prompt": PROMPTS["answer_relevancy"],
            },
            "faithfulness": {
                "metric": faithfulness,
                # У faithfulness 2 промпта, заменяем statement_generator
                "prompt_key": "statement_generator_prompt",
                "russian_prompt": PROMPTS["faithfulness"],
            },
            "context_precision": {
                "metric": context_precision,
                "prompt_key": "context_precision_prompt",
                "russian_prompt": PROMPTS["context_precision"],
            },
            "context_recall": {
                "metric": context_recall,
                "prompt_key": "context_recall_classification_prompt",
                "russian_prompt": PROMPTS["context_recall"],
            },
        }

        applied_count = 0
        for metric_name, config in prompt_mappings.items():
            try:
                metric_obj = config["metric"]
                prompt_key = config["prompt_key"]
                russian_prompt = config["russian_prompt"]

                # Используем set_prompts(**{key: prompt})
                metric_obj.set_prompts(**{prompt_key: russian_prompt})

                logging.info(f"  ✓ {metric_name}: промпт '{prompt_key}' заменён на русский")
                applied_count += 1

            except Exception as e:
                logging.error(f"  ✗ {metric_name}: ошибка при замене промпта: {e}")
                continue

        if applied_count > 0:
            logging.info(f"✅ Применено {applied_count} русских промптов к метрикам")
            return True
        else:
            logging.warning("⚠️  Не удалось применить ни одного промпта")
            return False

    except Exception as e:
        logging.error(f"Ошибка при применении русских промптов: {e}", exc_info=True)
        return False


def load_dataset_yaml(path: str):
    """Загружает датасет из YAML.

    Ожидаемая структура (список записей):
    - question: "вопрос1"
      expected_answer: "эталонный ответ"
      context: ["контекст1", "контекст2", ...]
      answer: "ответ RAG-системы"
    - question: "вопрос2"
      ...

    Преобразует в формат для Ragas EvaluationDataset:
    - user_input: вопрос
    - response: ответ RAG-системы
    - reference: эталонный ответ
    - retrieved_contexts: список контекстов
    """
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # data должен быть списком записей
    if not isinstance(data, list):
        raise ValueError(f"Ожидается список записей в YAML, получен {type(data)}")

    examples = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            logging.warning(f"Запись {i} не является словарем, пропускаем")
            continue

        question = item.get("question", "")
        expected_answer = item.get("expected_answer", "")
        context = item.get("context", [])
        answer = item.get("answer", "")

        if not question:
            logging.warning(f"Запись {i} не содержит вопроса, пропускаем")
            continue

        example = {
            "user_input": question,
            "response": answer,
            "reference": expected_answer,
            "retrieved_contexts": context,
        }

        examples.append(example)

    logging.info(f"Загружено {len(examples)} записей из {path}")
    return examples


def main():
    # Настраиваем логирование в самом начале
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, help="Path to dataset YAML file (e.g., enriched_gold_questions.yaml)"
    )
    parser.add_argument("--out", "-o", default="ragas_results_ollama.json", help="Output path for results")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name (default qwen3:8b)")
    parser.add_argument(
        "--embedding_model",
        default="Qwen/Qwen3-Embedding-4B",
        help="HuggingFace embedding model (default: Qwen3-Embedding-4B)",
    )
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument(
        "--use-russian-templates",
        action="store_true",
        help="Use Russian templates for evaluation (default: False)"
    )
    args = parser.parse_args()

    samples = load_dataset_yaml(args.input)
    samples = [s for s in samples if s["user_input"] and s["response"]]
    if not samples:
        print("Нет валидных примеров в входном файле.")
        return

    eval_dataset = EvaluationDataset.from_list(samples)

    # Создаём локальные embeddings (HuggingFace) на CPU
    print(f"Загрузка embedding модели на CPU: {args.embedding_model}...")

    # Для Qwen3-Embedding используем те же параметры, что и в extract_embeddings_for_kb.py
    # HuggingFaceEmbeddings принимает model_name (строку) и model_kwargs
    if "Qwen3-Embedding" in args.embedding_model:
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={"device": "cpu", "tokenizer_kwargs": {"padding_side": "left"}},
            encode_kwargs={"normalize_embeddings": True},
        )
    else:
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    # Оборачиваем для ragas (с подавлением deprecation warning)
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    # Создаём ChatOllama из langchain-ollama (новая версия)
    print(f"Подключение к Ollama-модели: {args.model}...")

    # Создаём callback для логирования всех запросов и ответов
    llm_callback = LLMLoggingCallback()

    # Включаем verbose и добавляем callback для детального логирования,
    # reasoning убираем, чтобы не было <think-секции> в ответе
    chat_ollama = ChatOllama(model=args.model, temperature=0.0, verbose=True, reasoning=False, callbacks=[llm_callback])
    llm_wrapper = LangchainLLMWrapper(chat_ollama)

    print("Запускаю evaluate() с локальными моделями — это может занять немного времени...")
    print(f"Параметры: batch_size={args.batch_size} (количество задач оценки, обрабатываемых параллельно)")

    # Применяем русские промпты к метрикам, если запрошено
    if args.use_russian_templates:
        print("\n" + "=" * 80)
        print("🌍 ПРИМЕНЕНИЕ РУССКИХ ПРОМПТОВ К МЕТРИКАМ RAGAS")
        print("=" * 80)

        if RUSSIAN_PROMPTS_AVAILABLE:
            print(f"📄 Файл russian_prompts.py найден и загружен")
            print(f"📦 PydanticPrompt объекты: {'✓ Доступны' if PROMPTS else '✗ Недоступны'}")
            
            russian_prompts_applied = apply_russian_prompts_to_metrics()
            
            if russian_prompts_applied:
                print("\n✅ Русские промпты с few-shot примерами успешно применены ко всем метрикам!")
                print("   Модель-критик будет использовать русскоязычные инструкции и примеры.")
            else:
                print("\n⚠️  Не удалось применить русские промпты. Используются стандартные промпты Ragas.")
        else:
            print("⚠️  Файл russian_prompts.py не найден. Используются стандартные промпты Ragas.")
            print("   Для использования русских промптов убедитесь, что файл russian_prompts.py доступен.")
            
        print("=" * 80 + "\n")
    else:
        print("\nℹ️  Использование русских промптов отключено. Применяются стандартные промпты Ragas.")
        print("   Для включения используйте флаг --use-russian-templates\n")

    # Определяем метрики для оценки
    # Так как в YAML есть все необходимые поля (retrieved_contexts, reference), используем все метрики
    metrics = [
        answer_relevancy,  # Релевантность ответа: насколько ответ отвечает на вопрос
        faithfulness,  # Достоверность: ответ основан только на фактах из контекста (без галлюцинаций)
        context_precision,  # Точность контекста: насколько точно RAG ранжирует релевантные фрагменты
        context_recall,  # Полнота контекста: все ли необходимые фрагменты извлечены из базы знаний
    ]

    # Требования к данным для каждой метрики:
    # - answer_relevancy: user_input, response
    # - faithfulness: response, retrieved_contexts
    # - context_precision: user_input, retrieved_contexts, reference
    # - context_recall: retrieved_contexts, reference

    print(f"\n📊 Метрики для оценки ({len(metrics)} метрик):")
    for i, metric in enumerate(metrics, 1):
        print(f"  {i}. {metric.name}: {metric.__doc__.split('.')[0] if metric.__doc__ else 'N/A'}")

    total_tasks = len(samples) * len(metrics)
    num_batches = (total_tasks + args.batch_size - 1) // args.batch_size
    print(f"\n🔢 Общее количество задач: {len(samples)} примеров × {len(metrics)} метрик = {total_tasks} задач")
    print(f"   Будет обработано в {num_batches} батчах по {args.batch_size} задач\n")

    result = evaluate(
        eval_dataset,
        llm=llm_wrapper,
        embeddings=ragas_embeddings,
        metrics=metrics,  # Явно передаём метрики
        show_progress=True,
        batch_size=args.batch_size,
    )

    # Конвертируем результат в словарь для JSON-сериализации
    print(f"\n📈 Результаты оценки:")
    result_dict = result.to_pandas().to_dict("records")

    # Также выводим средние значения метрик
    print(f"  Средние значения метрик:")
    for metric_name in result.to_pandas().columns:
        if metric_name not in ["user_input", "response", "retrieved_contexts", "reference"]:
            mean_value = result.to_pandas()[metric_name].mean()
            print(f"    {metric_name}: {mean_value:.4f}")

    # Сохраняем результаты в JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
