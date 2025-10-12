# Быстрая проверка русских промптов

## Проверка создания промптов

```bash
cd task_7_evaluation
uv run python -c "from russian_prompts import PROMPTS; print('✓ Создано' if PROMPTS else '✗ Ошибка'); [print(f'  {k}: {type(v).__name__}') for k,v in PROMPTS.items()] if PROMPTS else None"
```

**Ожидаемый вывод:**
```
✓ Успешно создано 4 русских PydanticPrompt объектов с few-shot примерами
✓ Создано
  answer_relevancy: ResponseRelevancePrompt
  faithfulness: StatementGeneratorPrompt
  context_precision: ContextPrecisionPrompt
  context_recall: ContextRecallClassificationPrompt
```

## Проверка применения к метрикам

```bash
uv run python -c "
import sys; sys.path.insert(0, '.')
from evaluate_rag_responses import apply_russian_prompts_to_metrics
from ragas.metrics import answer_relevancy

print('До:', answer_relevancy.get_prompts()['response_relevance_prompt'].language)
apply_russian_prompts_to_metrics()
print('После:', answer_relevancy.get_prompts()['response_relevance_prompt'].language)
"
```

**Ожидаемый вывод:**
```
До: english
После: russian
```

## Полный запуск оценки

```bash
uv run python evaluate_rag_responses.py \
  --input enriched_gold_questions_with_unfiltered_context.yaml \
  --out results_russian.json \
  --batch_size 8
```

**Ожидаемый вывод в начале:**
```
================================================================================
🌍 ПРИМЕНЕНИЕ РУССКИХ ПРОМПТОВ К МЕТРИКАМ RAGAS
================================================================================
📄 Файл russian_prompts.py найден и загружен
📦 PydanticPrompt объекты: ✓ Доступны

✅ Русские промпты с few-shot примерами успешно применены к метрикам!
   Модель-критик будет использовать русскоязычные инструкции и примеры.
================================================================================
```

## Проверка промптов в логах LLM

В выводе `🔵 LLM REQUEST` вы должны увидеть **русские инструкции**:

```
Инструкция: Оцени, насколько ответ модели отвечает на заданный вопрос...
```

Вместо английских:
```
Generate a question for the given answer and Identify if answer is noncommittal...
```

## Устранение проблем

### ✗ PydanticPrompt объекты: Недоступны

**Причина:** Ошибка при импорте классов промптов

**Решение:**
1. Проверьте версию Ragas: `pip show ragas`
2. Убедитесь, что версия >= 0.1.0
3. Проверьте, есть ли модули:
   ```bash
   uv run python -c "from ragas.metrics._answer_relevance import ResponseRelevancePrompt; print('OK')"
   ```

### Промпты остаются на английском

**Причина:** Метод `set_prompts()` не сработал

**Решение:**
1. Проверьте логи на уровне INFO
2. Убедитесь, что видите "✓ answer_relevancy: промпт ... заменён на русский"
3. Если нет, проверьте, что ключи промптов совпадают (можно посмотреть через `metric.get_prompts().keys()`)

## Структура русских промптов

Каждый промпт содержит:
- **instruction**: Русская инструкция для модели
- **examples**: 2 few-shot примера (положительный + проблемный случай)
- **response_model**: Pydantic модель для валидации ответа
- **language**: "russian"

## Метрики

1. **answer_relevancy** - Релевантность ответа вопросу
2. **faithfulness** - Достоверность (отсутствие галлюцинаций)
3. **context_precision** - Точность ранжирования контекста
4. **context_recall** - Полнота извлечения контекста

Все метрики используют русские промпты с примерами!
