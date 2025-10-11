# Задание 1. Исследование моделей и инфраструктуры

## open-weights vs проприетарные облачные LLM

Учитывая, что в базе знаний ожидается много конфиденциальных данных (в т.ч. персональных данных от заказчиков), а компании нужно сертифицироваться по SOC 2 – я рассматривал **только те модели, которые можно развернуть локально** (т.е. open-weights).

### Если оставить за скобками конфиденциальность?

Не будь у нас конфиденциальной информации и персональных данных – мы действовали бы в 3 этапа.

1\) Делаем всю систему на проприетарных облачных LLM (эмбеддинги и реранкинг тоже у себя не держим, все эти модели есть и облачные), пробуя разные модели и двигаясь от дорогих к дешёвым (в идеале – дорогие модели используем только в качестве модели-критика для оценки качества в RAGAS).

2\) Кроме облачных проприетарных LLM пробуем облачные open-weights LLM.

3\) Когда система заработает, мы быстро поймём, какая у нас предельная длина промптов по токенам – и сколько у нас запросов и токенов в месяц. Считать стоимость владения можно тут.

## Многоязычность

Учитывая, что документация будет на разных языках – смотрим **только многоязычные модели** (причём действительно многоязычные, даже Llama 3.1 не подходит ибо [языков она знает мало](https://huggingface.co/meta-llama/Llama-3.1-70B#:~:text=Supported%20languages%3A%20English%2C%20German%2C,refer%20to%20pretraining%20data%20only.)).

## Размер модели против качества

- модель эмбеддингов: стараемся не экономить, берём что получше (от этого зависит качество поиска чанков);  
- модель реранкинга: тоже стараемся не экономить, берём что получше (от этого тоже зависит качество поиска чанков);  
- собственно LLM: это самая затратная часть, но тут можно попробовать сэкономить:  
  - если поиск и реранкинг чанков работает хорошо, то вытащить из отфильтрованных чанков информацию для ответа на вопрос может даже не самая мощная LLM (простецкий Qwen3-8B, квантованный в 4 бита, уже мог);  
  - если поиск чанков работает плохо, то даже лучшая LLM нам уже не поможет.

## Сравнение моделей эмбеддингов

Я базировался на результатах MTEB Leaderboard: [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard)

Тамошний топ:

- gemini-embedding-001 (внешний сервис, нам не подходит);  
- Qwen3-Embedding-8B;  
- Qwen3-Embedding-4B;  
- Qwen3-Embedding-0.6B

Семейство Qwen3-Embedding нам отлично подходит: и локальный запуск, и многоязычность, и качество.

Согласно той таблице, Qwen3-Embedding-4B по качеству очень близко приближается к Qwen3-Embedding-8B, так что берём **Qwen3-Embedding-4B**.

## Сравнение моделей реранкинга

Я пользовался вот этими данными: [https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3\_602308](https://www.alibabacloud.com/blog/mastering-text-embedding-and-reranker-with-qwen3_602308)

| Модель | MTEB-R | CMTEB-R | MMTEB-R | MLDR | MTEB-Code |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Jina-multilingual-reranker-v2-base | 58.22 | 63.37 | 63.73 | 39.66 | 58.98 |
| gte-multilingual-reranker-base | 59.51 | 74.08 | 59.44 | 66.33 | 54.18 |
| BGE-reranker-v2-m3 | 57.03 | 72.16 | 58.36 | 59.51 | 41.38 |
| Qwen3-Reranker-0.6B | 65.80 | 71.31 | 66.36 | 67.28 | 73.42 |
| Qwen3-Reranker-4B | 69.76 | 75.94 | 72.74 | 69.97 | 81.20 |
| Qwen3-Reranker-8B | 69.02 | 77.45 | 72.94 | 70.19 | 81.22 |

Там свеженькая специализированная модель Qwen3-Reranker сравнивалась с другими специализированными моделями для реранкинга. Есть ли что-нибудь от DeepSeek – не нашёл, так что будем закладываться на **Qwen3-Reranker-4B** (судя по таблице, от 8B он отстаёт лишь немного).

## Сравнение моделей LLM

Я смотрел вот эту таблицу (\<40B, open-weights)  
[https://artificialanalysis.ai/leaderboards/models?open\_weights=open\_source\&size\_class=small](https://artificialanalysis.ai/leaderboards/models?open_weights=open_source&size_class=small)

И вот это сравнение конкретно по семейству Qwen3:  
[https://dev.to/best\_codes/qwen-3-benchmarks-comparisons-model-specifications-and-more-4hoa](https://dev.to/best_codes/qwen-3-benchmarks-comparisons-model-specifications-and-more-4hoa)

[https://medium.com/@marketing\_novita.ai/which-qwen3-model-is-right-for-you-a-practical-guide-e576569e3c78](https://medium.com/@marketing_novita.ai/which-qwen3-model-is-right-for-you-a-practical-guide-e576569e3c78)

Судя по этим данным, нам **идеально подойдёт Qwen3-30B, но может сгодиться и Qwen3-14B, и даже Qwen3-8B.**

Спойлер: Qwen3-8B и правда сгодился.

## ChromaDB vs FAISS

Будем считать, что у нас примерно 500К векторов (по максимуму) размерности 2560 (как для модели Qwen3-Embedding-4B).

| Параметр | ChromaDB | FAISS |
| :-- | :-- | :-- |
| Скорость поиска | ~100-150 мс на поиск 1000 ближайших из 500k при размерности 2560, зависит от конфигурации и индекса (HNSW, IVF) [^1][^2] | ~10-50 мс на поиск 1000 ближайших из 500k, при использовании CPU; с GPU — до 5-20 мс [^3][^4] |
| Скорость индексации | >5 часов на 1M векторов размерности ~512, пропорционально больше при 2560; ожидать часы для 500k, зависит от апстрим-конфигурации [^1][^2] | Несколько минут на 500k, зависит от индекса (IVF, PQ) и железа; GPU сильно ускоряет [^3][^4] |
| Сложность внедрения | Низкая — простой Python API, быстрый старт, встроенное управление метаданными [^5][^6][^2] | Средняя-Высокая — требует понимания индексов, настройка C++ и Python биндингов [^3][^7] |
| Сложность поддержки | Низкая — встроенные механизмы, проще масштабировать как БД [^5][^2] | Средняя — требует опытной команды для оптимизации, особенно при кластеризации [^3][^4] |
| Удобство работы | Высокое — богатый API, поддержка фильтров, поиск + метаданные в одном [^5][^8] | Среднее — мощный, но более "низкоуровневый", требует дополнительной разработки [^3][^7] |
| Стоимость владения | Средняя — требует выделенного сервера/К8s, дисковое хранение, меньше GPU [^2][^9] | От средней до высокой — если GPU используется, стоимость оборудования растёт [^3][^9] |


Источники:

- Сравнение ChromaDB и FAISS по скорости и функциям от Capella Solutions (2024)[^5]
- Официальная документация FAISS (Meta) с информацией о поиске и индексации[^3][^10]
- Отчёты пользователей о времени индексации и поиске в ChromaDB (GitHub, 2023-2024)[^1][^2]
- Руководство по выбору индексов в FAISS (2024)[^7]
- Статьи и обзоры по TCO и инфраструктуре в vector DB (2024-2025)[^9][^2][^4]

Таким образом, FAISS обеспечивает намного более высокую скорость поиска и индексации при использовании специализированного оборудования, однако требует больше усилий при внедрении и поддержке, а также может быть дороже в эксплуатации. ChromaDB более удобна в работе и проще в управлении, но скорость несколько ниже, особенно при больших объёмах и высоких размерностях. Для 500000 векторов размерности 2560 FAISS предпочтительнее, если критична производительность, ChromaDB — если важна простота и быстрое развертывание.

Все цифры являются порядковыми, зависят от конкретной конфигурации и оборудования.[^2][^4][^5][^3][^1]
<span style="display:none">[^11][^12][^13][^14][^15][^16][^17][^18][^19][^20]</span>

<div align="center">⁂</div>

[^1]: https://github.com/chroma-core/chroma/issues/335

[^2]: https://metadesignsolutions.com/chroma-db-the-ultimate-vector-database-for-ai-and-machine-learning-revolution/

[^3]: https://faiss.ai/index.html

[^4]: https://milvus.io/ai-quick-reference/how-do-you-utilize-faiss-or-a-similar-vector-database-with-sentence-transformer-embeddings-for-efficient-similarity-search

[^5]: https://www.capellasolutions.com/blog/faiss-vs-chroma-lets-settle-the-vector-database-debate

[^6]: https://docs.trychroma.com/docs/collections/configure?lang=typescript

[^7]: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

[^8]: https://www.trychroma.com

[^9]: https://community.hpe.com/t5/insight-remote-support/comparing-pinecone-chroma-db-and-faiss-exploring-vector/td-p/7210879

[^10]: https://faiss.ai

[^11]: https://github.com/chroma-core/chroma/issues/3011

[^12]: https://github.com/facebookresearch/faiss/issues/1444

[^13]: https://stackoverflow.com/questions/77093395/size-of-indexed-vectors-in-chroma-db

[^14]: https://www.reddit.com/r/LangChain/comments/15a447w/chroma_or_faiss/

[^15]: https://apxml.com/posts/best-databases-for-rag

[^16]: https://zilliz.com/comparison/chroma-vs-faiss

[^17]: https://stackoverflow.com/questions/77694864/invaliddimensionexception-embedding-dimension-384-does-not-match-collection-dim

[^18]: https://opensearch.org/blog/lucene-on-faiss-powering-opensearchs-high-performance-memory-efficient-vector-search/

[^19]: https://myscale.com/blog/faiss-vs-chroma-efficiency-vector-storage-battle/

[^20]: https://github.com/facebookresearch/faiss


**Решение: начинаем с ChromaDB, нам её вполне хватит. В крайнем случае – перейти на FAISS мы всегда успеем \= )**

## Рекомендуемая конфигурация сервера

## Для быстрого старта (рекомендую начать с него)

Для быстрого старта можно развернуть ollama на недорогой конфигурации на игровых комплектующих (должны уложиться в 500К).

* **Материнка: MSI X870 GAMING PLUS**  
* **Процессор: AMD Ryzen 9 9950X**  
* **Видео: 2x RTX 3090 (24 Gb каждая)**  
* **Память: 128 Gb DDR5 (4x 32 Gb)**  
* **Диски: 2x 2Tb SSD;**  
* **Блок питания: Corsair AX1600i**

Я закладывался на то, что здесь мы уже сможем запустить:

- Qwen3-30B с 4-битным квантованием или Qwen3-14B с 8-битным квантованием или Qwen3-8B без квантования;  
- Qwen3-Embedding-4B без квантования или с 8-битным квантованием;  
- Qwen3-Reranking-4B без квантования или с 8-битным квантованием.

## Чтобы иметь запас

Для более тяжёлых LLM нужна машина с 4 видеокартами, а значит, нам нужно что-то с большим количеством линий PCIе, это либо Threadripper’ы, либо Intel Core X.

Вот один из вариантов (в районе 1.2M):

| Компонент | Модель / примечание | Цена за единицу (₽) | Кол-во |
| ----- | ----- | ----- | ----- |
| CPU | AMD Threadripper PRO 7965WX (24 ядер, sTR5) | **273 507** | 1 |
| Материнская плата | Gigabyte TRX50 AI TOP (E-ATX, 4 × PCIe 5.0 x16) | **157 200** | 1 |
| Видеокарты | NVIDIA GeForce RTX 3090 (новая) | **135 001** | 4 |
| ОЗУ | DDR5 128 GB (4 × 32 GB, Kingston или Corsair kit) | **47 450** | 1 |
| SSD | Samsung 990 PRO 2 TB (M.2 NVMe) | **16 299** | 2 |
| Блок питания | Corsair AX1600i (1600 W, Titan Efficiency) | **48 460** | 2 |
| Корпус | Lian Li O11 Dynamic EVO XL (E-ATX, Big-Tower) | **24 381** | 1 |
| Охлаждение CPU | Enermax LiqTech TR4 II 360 (совместимо с TR5) | **16 481** | 1 |
| Кабели и аксессуары | Адаптеры, доп. вентиляторы, крепления и т.д. | **8 000** | 1 |

В перспективе – RTX 3090 можно заменить на RTX 4090 48Gb (\~350K за штуку).

Хинт: для быстрого тестирования разных LLM лучше не возиться с локальным запуском тяжёлых поискать облачных провайдеров open-weights-моделей (заливать туда всю базу не требуется, достаточно скармливать проблемные запросы).

Хинт: более серьёзные LLM понадобятся для более серьёзных задач (например, Coding Assistance), но конфигу под них надо считать отдельно.