from pathlib import Path
from typing import List, Tuple
import logging
import pickle

import faiss
import numpy as np
import yaml

from task_3_vector_index.pydantic_dto import TextEmbeddingsInfo, TextChunkInfo

# Число outgoing edges (связей) для каждой вершины в графе.
# Обычно 16–64.
# Чем больше M → лучше recall, но выше память и время построения.
HNSW_M = 64

# Количество соседей, рассматриваемых при построении графа.
# Обычно 100–500.
# Чем больше → качественнее граф, но дольше построение.
HNSW_EF_CONSTRUCTION = 200

# Количество соседей, рассматриваемых при поиске.
# Обычно 50–500.
# Чем больше → выше точность, но медленнее поиск.
# Этот параметр можно менять динамически перед поиском:
HNSW_EF_SEARCH = 32

# Сколько ближайших векторов выдавать при поиске
K_SEARCH = 5


def load_and_index_kb_embeddings(kb_embeddings_path: Path) -> Tuple[faiss.IndexHNSWFlat, List[Path]]:
    """
    Load chunked document embeddings and create a FAISS index.

    Args:
        kb_embeddings_path: Path to pickle file containing List[TextEmbeddingsInfo]

    Returns:
        Tuple of (FAISS index, list of source paths for each embedding vector)
    """
    kb_embeddings_list = pickle.load(open(kb_embeddings_path, "rb"))
    assert isinstance(kb_embeddings_list, list), f"Expected list, got {type(kb_embeddings_list)}"

    # Collect all embeddings and their corresponding source paths
    all_embeddings = []
    source_paths = []

    for doc_info in kb_embeddings_list:
        assert isinstance(doc_info, TextEmbeddingsInfo), f"Expected TextEmbeddingsInfo, got {type(doc_info)}"

        # doc_info.embeddings has shape (n_chunks, embedding_dim)
        n_chunks = doc_info.embeddings.shape[0]
        all_embeddings.append(doc_info.embeddings)

        # Each chunk gets associated with the original document path
        source_paths.extend([doc_info.original_text_path] * n_chunks)

    # Stack all embeddings into a single matrix
    kb_embeddings = np.vstack(all_embeddings)
    assert len(kb_embeddings.shape) == 2
    n_vectors, d = kb_embeddings.shape
    logging.info(f"Total vectors (chunks): {n_vectors},\tembedding dimension: {d}")
    logging.info(f"From {len(kb_embeddings_list)} documents")

    # Create and populate FAISS index
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_L2)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    index.add(kb_embeddings)
    return index, source_paths


# def load_questions_embeddings(questions_embeddings_path: Path) -> np.ndarray


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    qwen_suffixes = ["4B", "0.6B"]

    questions = yaml.safe_load(open(Path(__file__).parent / f"questions.yaml", "r"))

    for suffix in qwen_suffixes:
        print("=========")
        print(f"Using Qwen3-{suffix}")

        index, paths = load_and_index_kb_embeddings(
            kb_embeddings_path=Path(__file__).parent / f"doc_embeddings_chunked-{suffix}.pck"
        )

        questions_embeddings_path = Path(__file__).parent / f"questions_embeddings-{suffix}.pck"
        q_embeddings = pickle.load(open(questions_embeddings_path, "rb"))
        n_questions = q_embeddings.shape[0]

        d_vals, i_vals = index.search(q_embeddings, K_SEARCH)
        print(d_vals)
        print(i_vals)

        for i in range(n_questions):
            print("====")
            print(f"Question {i}: {questions[i]}")
            print("Docs found:")
            for j in range(i_vals.shape[1]):
                print(f"{j}: {paths[i_vals[i, j]].name}")
