#!/usr/bin/env python3
"""
Скрипт для запуска Ragas evaluation с локальной Ollama-моделью qwen3:8b.
Требования:
  - ollama установлен и запущен (https://ollama.com)
  - модель qwen3:8b загружена: `ollama pull qwen3:8b`
  - Python-пакеты: ragas, langchain-ollama
  - sentence-transformers (автоматически установится с ragas)

Пример запуска:
  ollama pull qwen3:8b
  python evaluate_rag_responses.py --input dataset.json --out results.json

"""
import argparse
import json
import logging
import sys
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

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


def load_dataset_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Конвертируем в список примеров для EvaluationDataset
    n = len(data.get("questions", []))
    examples = []
    for i in range(n):
        q = data["questions"][i]
        resp = data["answers"][i] if i < len(data.get("answers", [])) else ""
        ref = data["ground_truths"][i] if i < len(data.get("ground_truths", [])) else None
        contexts = data["contexts"][i] if i < len(data.get("contexts", [])) else None
        examples.append({"user_input": q, "response": resp, "reference": ref, "retrieved_contexts": contexts})
    return examples


def main():
    # Настраиваем логирование в самом начале
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to dataset.json")
    parser.add_argument("--out", "-o", default="ragas_results_ollama.json", help="Output path")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model name (default qwen3:8b)")
    parser.add_argument("--embedding_model", default="Qwen/Qwen3-Embedding-4B", 
                        help="HuggingFace embedding model (default: Qwen3-Embedding-4B)")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    samples = load_dataset_json(args.input)
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
            encode_kwargs={"normalize_embeddings": True}
        )
    else:
        langchain_embeddings = HuggingFaceEmbeddings(
            model_name=args.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
    
    # Оборачиваем для ragas (с подавлением deprecation warning)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)

    # Создаём ChatOllama из langchain-ollama (новая версия)
    print(f"Подключение к Ollama модели: {args.model}...")
    
    # Создаём callback для логирования всех запросов и ответов
    llm_callback = LLMLoggingCallback()
    
    # Включаем verbose и добавляем callback для детального логирования
    chat_ollama = ChatOllama(
        model=args.model, 
        temperature=0.0, 
        verbose=True,
        callbacks=[llm_callback]
    )
    llm_wrapper = LangchainLLMWrapper(chat_ollama)

    print("Запускаю evaluate() с локальными моделями — это может занять немного времени...")
    print(f"Параметры: batch_size={args.batch_size} (количество задач оценки, обрабатываемых параллельно)")
    
    # Определяем метрики для оценки
    metrics = [
        faithfulness,         # Верность ответа контексту (нет галлюцинаций)
        answer_relevancy,     # Релевантность ответа вопросу
        context_precision,    # Точность извлечённого контекста
        context_recall        # Полнота контекста (покрывает ли ground truth)
    ]
    
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
        batch_size=args.batch_size
    )

    # Конвертируем результат в словарь для JSON-сериализации
    print(f"\n📈 Результаты оценки:")
    result_dict = result.to_pandas().to_dict('records')
    
    # Также выводим средние значения метрик
    print(f"  Средние значения метрик:")
    for metric_name in result.to_pandas().columns:
        if metric_name not in ['user_input', 'response', 'retrieved_contexts', 'reference']:
            mean_value = result.to_pandas()[metric_name].mean()
            print(f"    {metric_name}: {mean_value:.4f}")
    
    # Сохраняем результаты в JSON
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Результаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
