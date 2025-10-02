from pathlib import Path
from typing import List, Tuple
import pickle

import faiss
import numpy as np

# Число outgoing edges (связей) для каждой вершины в графе.
# Обычно 16–64.
# Чем больше M → лучше recall, но выше память и время построения.
HNSW_M = 32

# Количество соседей, рассматриваемых при построении графа.
# Обычно 100–500.
# Чем больше → качественнее граф, но дольше построение.
HNSW_EF_CONSTRUCTION = 200

# Количество соседей, рассматриваемых при поиске.
# Обычно 50–500.
# Чем больше → выше точность, но медленнее поиск.
# Этот параметр можно менять динамически перед поиском:
HNSW_EF_SEARCH = 128

# Сколько ближайших векторов выдавать при поиске
K_SEARCH = 4


def load_and_index_kb_embeddings(kb_embeddings_path: Path) -> Tuple[faiss.IndexHNSWFlat, List[Path]]:
    kb_embeddings_dict = pickle.load(open(kb_embeddings_path, "rb"))
    assert isinstance(kb_embeddings_dict, dict)
    kb_embeddings = np.vstack(list(kb_embeddings_dict.values()))
    assert len(kb_embeddings.shape) == 2
    n_docs, d = kb_embeddings.shape

    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_L2)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    index.add(kb_embeddings)
    return index, list(kb_embeddings_dict.keys())


# def load_questions_embeddings(questions_embeddings_path: Path) -> np.ndarray


if __name__ == "__main__":
    qwen_suffixes = ["4B", "0.6B"]
    for suffix in qwen_suffixes:
        print("=========")
        print(f"Using Qwen3-{suffix}")

        index, paths = load_and_index_kb_embeddings(
            kb_embeddings_path=Path(__file__).parent / f"embeddings-{suffix}.pck"
        )

        questions_embeddings_path = Path(__file__).parent / f"questions_embeddings-{suffix}.pck"
        q_embeddings = pickle.load(open(questions_embeddings_path, "rb"))
        n_questions = q_embeddings.shape[0]

        d_vals, i_vals = index.search(q_embeddings, K_SEARCH)
        print(d_vals)
        print(i_vals)

        for i in range(n_questions):
            print("====")
            print(f"Question {i}")
            for j in range(i_vals.shape[1]):
                print(paths[i_vals[i, j]].name)
