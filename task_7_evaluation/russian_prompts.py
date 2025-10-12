# pydantic_prompts_with_examples.py
# Pydantic модели + примеры few-shot для PydanticPrompt (RAGAS)
from typing import List, Optional
from pydantic import BaseModel, Field


# --- 1) answer_relevancy ---------------------------------------------------
class AnswerRelevancyInput(BaseModel):
    question: str
    context: Optional[List[str]] = None
    model_output: str


class AnswerRelevancyOutput(BaseModel):
    score: float = Field(..., ge=0.0, le=5.0)
    rationale: str
    relevant_context_chunks: List[int] = []
    improvement: Optional[str] = None


# --- 2) faithfulness -------------------------------------------------------
class UnsupportedClaim(BaseModel):
    text: str
    why: str


class FaithfulnessInput(BaseModel):
    context: Optional[List[str]] = None
    model_output: str


class FaithfulnessOutput(BaseModel):
    hallucination: bool
    unsupported_claims: List[UnsupportedClaim] = []
    score: float = Field(..., ge=0.0, le=1.0)
    recommendation: Optional[str] = None


# --- 3) context_precision --------------------------------------------------
class ContextPrecisionInput(BaseModel):
    question: str
    retrieved_chunks: List[str]


class ContextPrecisionOutput(BaseModel):
    precision: float = Field(..., ge=0.0, le=1.0)
    relevant_chunks: List[int] = []
    total_returned: int
    rationale: str


# --- 4) context_recall -----------------------------------------------------
class ContextRecallInput(BaseModel):
    question: str
    retrieved_chunks: List[str]
    ground_truth: List[str]


class ContextRecallOutput(BaseModel):
    recall: float = Field(..., ge=0.0, le=1.0)
    covered_facts: List[str] = []
    missing_facts: List[str] = []
    rationale: str


# --------------------------------------------------------------------------
# ТЕКСТОВЫЕ ШАБЛОНЫ (инструкции) — оставлены в удобочитаемом виде
ANSWER_RELEVANCY_TEMPLATE = """
Инструкция: Оцени, насколько ответ модели отвечает на заданный вопрос с учётом предоставленного контекста.
Требуется вернуть объект, соответствующий схеме AnswerRelevancyOutput.
Вход: question, context (список строк или null), model_output.
Выход: JSON с полями score (0..5), rationale, relevant_context_chunks (индексы), improvement (опционально).
"""

FAITHFULNESS_TEMPLATE = """
Инструкция: Проверь, содержит ли ответ модели утверждения, не подтверждённые контекстом (галлюцинации).
Требуется вернуть объект, соответствующий схеме FaithfulnessOutput.
Вход: context (список строк или null), model_output.
Выход: JSON с полями hallucination (bool), unsupported_claims (list of {text, why}), score (0..1), recommendation (опционально).
"""

CONTEXT_PRECISION_TEMPLATE = """
Инструкция: Оцени precision среди возвращённых контекстных чанков: какая доля релевантна вопросу.
Требуется вернуть объект, соответствующий схеме ContextPrecisionOutput.
Вход: question, retrieved_chunks (список строк).
Выход: JSON с полями precision (0..1), relevant_chunks (индексы), total_returned, rationale.
"""

CONTEXT_RECALL_TEMPLATE = """
Инструкция: Оцени recall: из фактов, необходимых для корректного эталонного ответа (ground_truth), какая доля покрыта возвращённым контекстом.
Требуется вернуть объект, соответствующий схеме ContextRecallOutput.
Вход: question, retrieved_chunks, ground_truth (список фактов).
Выход: JSON с полями recall (0..1), covered_facts, missing_facts, rationale.
"""

# --------------------------------------------------------------------------
# Few-shot примеры (2 примера для каждой метрики)
ANSWER_RELEVANCY_EXAMPLES = [
    # Пример 1 — полностью релевантный ответ
    {
        "input": {
            "question": "Какой срок гарантии на ноутбук X?",
            "context": ["Ноутбук X продаётся с гарантией 2 года.", "Доставка бесплатна по России."],
            "model_output": "Гарантия на ноутбук X — 2 года.",
        },
        "output": {
            "score": 5.0,
            "rationale": "Ответ точно совпадает с информацией в первом фрагменте контекста.",
            "relevant_context_chunks": [0],
            "improvement": "",
        },
    },
    # Пример 2 — частично нерелевантный ответ
    {
        "input": {
            "question": "Сколько часов работает аккумулятор у телефона Y?",
            "context": ["Телефон Y доступен в трёх цветах.", "Процессор: Z-Gen, есть поддержка быстрой зарядки."],
            "model_output": "Аккумулятор телефона Y держит 18 часов активного использования.",
        },
        "output": {
            "score": 1.5,
            "rationale": "Контекст не содержит информации о времени работы батареи — ответ не подтверждён и слабо релевантен.",
            "relevant_context_chunks": [],
            "improvement": "Указать источник для параметра батареи или отметить, что данных нет в контексте.",
        },
    },
]

FAITHFULNESS_EXAMPLES = [
    # Пример 1 — есть галлюцинация
    {
        "input": {
            "context": ["Компания Y основана в 2010 году.", "Штаб-квартира в Берлине."],
            "model_output": "Компания Y была основана в 2008 году и базируется в Берлине.",
        },
        "output": {
            "hallucination": True,
            "unsupported_claims": [
                {
                    "text": "основана в 2008 году",
                    "why": "В контексте указан 2010 год, утверждение конфликтует с источником.",
                }
            ],
            "score": 0.2,
            "recommendation": "Проверить год основания по официальному сайту компании или другим первоисточникам.",
        },
    },
    # Пример 2 — без галлюцинаций
    {
        "input": {
            "context": ["API v1.2 поддерживает авторизацию через OAuth2."],
            "model_output": "API v1.2 поддерживает авторизацию через OAuth2.",
        },
        "output": {"hallucination": False, "unsupported_claims": [], "score": 1.0, "recommendation": ""},
    },
]

CONTEXT_PRECISION_EXAMPLES = [
    # Пример 1 — частично релевантные чанки
    {
        "input": {
            "question": "Какой срок гарантии на продукт?",
            "retrieved_chunks": [
                "Характеристики модели и размеры.",
                "Условия гарантии: 1 год с момента покупки.",
                "Инструкции по эксплуатации и уходу.",
            ],
        },
        "output": {
            "precision": 1.0 / 3.0,
            "relevant_chunks": [1],
            "total_returned": 3,
            "rationale": "Только второй чанк прямо содержит информацию о гарантии.",
        },
    },
    # Пример 2 — все чанки релевантны
    {
        "input": {
            "question": "Как отключить звук уведомлений в приложении Z?",
            "retrieved_chunks": [
                "Настройки -> Звук: отключить уведомления.",
                "Как управлять звуком уведомлений в приложении Z: шаги 1..3.",
                "Если звук не выключается — перезапустите приложение и проверьте настройки системы.",
            ],
        },
        "output": {
            "precision": 1.0,
            "relevant_chunks": [0, 1, 2],
            "total_returned": 3,
            "rationale": "Все возвращённые фрагменты напрямую помогают ответить на вопрос о звуках уведомлений.",
        },
    },
]

CONTEXT_RECALL_EXAMPLES = [
    # Пример 1 — частичное покрытие фактов
    {
        "input": {
            "question": "Кто и когда основал компанию Z?",
            "retrieved_chunks": ["Информация: компания Z основана в 1999 году"],
            "ground_truth": ["основана в 1999 году", "основатель — Иван Иванов"],
        },
        "output": {
            "recall": 0.5,
            "covered_facts": ["основана в 1999 году"],
            "missing_facts": ["основатель — Иван Иванов"],
            "rationale": "Из двух ключевых фактов найден только один — год основания.",
        },
    },
    # Пример 2 — полное покрытие
    {
        "input": {
            "question": "Когда и кем основана организация A?",
            "retrieved_chunks": ["Организация A основана в 2005 году.", "Основатель — Мария Петрова."],
            "ground_truth": ["основана в 2005 году", "основатель — Мария Петрова"],
        },
        "output": {
            "recall": 1.0,
            "covered_facts": ["основана в 2005 году", "основатель — Мария Петрова"],
            "missing_facts": [],
            "rationale": "Все факты из ground_truth присутствуют в retrieved_chunks.",
        },
    },
]

# --------------------------------------------------------------------------
# Конвертация примеров в формат для Prompt (List[Tuple[Pydantic, Pydantic]])
def convert_examples_to_prompt_format(examples_list, input_model, output_model):
    """
    Конвертирует примеры из формата {input: {...}, output: {...}} 
    в формат (input_pydantic, output_pydantic)
    
    Ragas требует Pydantic модели, а не обычные словари!
    """
    converted = []
    for ex in examples_list:
        # Создаем Pydantic объекты из словарей
        input_obj = input_model(**ex["input"])
        output_obj = output_model(**ex["output"])
        converted.append((input_obj, output_obj))
    return converted


PROMPT_EXAMPLES = {
    "answer_relevancy": ANSWER_RELEVANCY_EXAMPLES,
    "faithfulness": FAITHFULNESS_EXAMPLES,
    "context_precision": CONTEXT_PRECISION_EXAMPLES,
    "context_recall": CONTEXT_RECALL_EXAMPLES,
}

PROMPTS = None

try:
    # Импортируем конкретные классы промптов и их Input/Output модели для каждой метрики
    from ragas.metrics._answer_relevance import (
        ResponseRelevancePrompt, ResponseRelevanceInput, ResponseRelevanceOutput
    )
    from ragas.metrics._faithfulness import (
        StatementGeneratorPrompt, StatementGeneratorInput, StatementGeneratorOutput
    )
    from ragas.metrics._context_precision import (
        ContextPrecisionPrompt, QAC, Verification
    )
    from ragas.metrics._context_recall import (
        ContextRecallClassificationPrompt, QCA, ContextRecallClassifications, ContextRecallClassification
    )

    # Создаем русские промпты для каждой метрики
    # Используем правильные Pydantic модели для примеров
    
    # 1. Answer Relevancy - требует: Input(response), Output(question, noncommittal)
    p_answer_relevancy = ResponseRelevancePrompt(language="russian")
    p_answer_relevancy.instruction = ANSWER_RELEVANCY_TEMPLATE
    p_answer_relevancy.examples = [
        (
            ResponseRelevanceInput(response="Альберт Эйнштейн родился в Германии."),
            ResponseRelevanceOutput(question="Где родился Альберт Эйнштейн?", noncommittal=0)
        ),
        (
            ResponseRelevanceInput(response="Я не знаю о революционных функциях смартфона 2023 года, так как моя информация ограничена 2022 годом."),
            ResponseRelevanceOutput(question="Какова была революционная функция смартфона 2023 года?", noncommittal=1)
        ),
    ]

    # 2. Faithfulness - требует: Input(question, answer), Output(statements)
    p_faithfulness = StatementGeneratorPrompt(language="russian")
    p_faithfulness.instruction = FAITHFULNESS_TEMPLATE
    p_faithfulness.examples = [
        (
            StatementGeneratorInput(
                question="Кто такой Альберт Эйнштейн и чем он известен?",
                answer="Он был немецким физиком-теоретиком, признанным одним из величайших физиков всех времён. Он наиболее известен разработкой теории относительности, а также внёс важный вклад в квантовую механику."
            ),
            StatementGeneratorOutput(statements=[
                "Альберт Эйнштейн был немецким физиком-теоретиком.",
                "Альберт Эйнштейн признан одним из величайших физиков всех времён.",
                "Альберт Эйнштейн наиболее известен разработкой теории относительности.",
                "Альберт Эйнштейн внёс важный вклад в квантовую механику."
            ])
        ),
    ]

    # 3. Context Precision - требует: Input(question, context, answer), Output(reason, verdict)
    p_context_precision = ContextPrecisionPrompt(language="russian")
    p_context_precision.instruction = CONTEXT_PRECISION_TEMPLATE
    p_context_precision.examples = [
        (
            QAC(
                question="Что можно рассказать об Альберте Эйнштейне?",
                context="Альберт Эйнштейн (14 марта 1879 – 18 апреля 1955) был немецким физиком-теоретиком, признанным одним из величайших учёных всех времён.",
                answer="Альберт Эйнштейн, родившийся 14 марта 1879 года, был немецким физиком-теоретиком."
            ),
            Verification(
                reason="Предоставленный контекст действительно был полезен для получения ответа. Контекст включает ключевую информацию об Эйнштейне.",
                verdict=1
            )
        ),
    ]

    # 4. Context Recall - требует: Input(question, context, answer), Output(classifications)
    p_context_recall = ContextRecallClassificationPrompt(language="russian")
    p_context_recall.instruction = CONTEXT_RECALL_TEMPLATE
    p_context_recall.examples = [
        (
            QCA(
                question="Что можно рассказать об Альберте Эйнштейне?",
                context="Альберт Эйнштейн (14 марта 1879 – 18 апреля 1955) был немецким физиком-теоретиком.",
                answer="Альберт Эйнштейн, родившийся 14 марта 1879 года, был немецким физиком-теоретиком."
            ),
            ContextRecallClassifications(classifications=[
                ContextRecallClassification(
                    statement="Альберт Эйнштейн, родившийся 14 марта 1879 года, был немецким физиком-теоретиком.",
                    reason="Дата рождения Эйнштейна чётко упомянута в контексте.",
                    attributed=1
                )
            ])
        ),
    ]

    PROMPTS = {
        "answer_relevancy": p_answer_relevancy,
        "faithfulness": p_faithfulness,
        "context_precision": p_context_precision,
        "context_recall": p_context_recall,
    }
    
    print(f"✓ Успешно создано {len(PROMPTS)} русских PydanticPrompt объектов с инструкциями и few-shot примерами")

except Exception as e:
    # Если не удалось создать промпты, сохраняем PROMPTS = None
    PROMPTS = None
    import traceback
    print(f"✗ Ошибка при создании PydanticPrompt объектов: {e}")
    print("  Traceback:")
    traceback.print_exc()
    print("  Будут использованы стандартные промпты Ragas")


# --------------------------------------------------------------------------
# Usage notes:
# - Если PydanticPrompt примет examples, они будут встроены в prompt и использованы как few-shot.
# - В противном случае используйте PROMPT_EXAMPLES при создании/регистрации метрик вручную.
# --------------------------------------------------------------------------

if __name__ == "__main__":
    # Быстрая проверка: вывести количество примеров для каждой метрики
    for k, v in PROMPT_EXAMPLES.items():
        print(f"{k}: {len(v)} примера")
