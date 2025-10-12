from pathlib import Path
from typing import List, Tuple
import logging
import pickle

import faiss
import numpy as np
import yaml
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection

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


def load_and_index_kb_embeddings(
    kb_embeddings_path: Path, suffix: str
) -> Tuple[faiss.IndexHNSWFlat, Collection, List[Path]]:
    """
    Load chunked document embeddings and create FAISS and Chroma DB indexes.

    Args:
        kb_embeddings_path: Path to pickle file containing List[TextEmbeddingsInfo]
        suffix: Model suffix (e.g., "0.6B" or "4B") for separate ChromaDB storage

    Returns:
        Tuple of (FAISS index, Chroma collection, list of source paths)
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

    # Ensure embeddings are in float32 format and C-contiguous for FAISS
    kb_embeddings = np.ascontiguousarray(kb_embeddings, dtype=np.float32)

    # Create and populate FAISS index
    index = faiss.IndexHNSWFlat(d, HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.hnsw.efSearch = HNSW_EF_SEARCH

    index.add(kb_embeddings)

    # Create Chroma DB index with metadata
    chroma_db_path = Path(__file__).parent / "chroma" / f"chroma-{suffix}"
    chroma_client = chromadb.PersistentClient(path=str(chroma_db_path), settings=Settings(anonymized_telemetry=False))

    # Create or get collection with cosine similarity (equivalent to L2 normalized)
    collection_name = "kb_embeddings"
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    chroma_collection = chroma_client.create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}  # L2 distance like FAISS
    )

    # Prepare data for Chroma DB
    chunk_metadata_list = []
    chunk_idx = 0

    for doc_info in kb_embeddings_list:
        n_chunks = doc_info.embeddings.shape[0]

        for i, chunk_info in enumerate(doc_info.chunks):
            # Используем .model_dump() для получения метаданных (Pydantic v2) и добавляем source_path
            metadata = chunk_info.model_dump() if hasattr(chunk_info, "model_dump") else chunk_info.dict()
            metadata["source_path"] = str(doc_info.original_text_path)
            chunk_metadata_list.append(metadata)
            chunk_idx += 1

    # Add embeddings with metadata to Chroma
    chroma_collection.add(
        embeddings=kb_embeddings.tolist(),
        metadatas=chunk_metadata_list,
        ids=[f"chunk_{i}" for i in range(len(chunk_metadata_list))],
    )

    logging.info(f"Created Chroma collection '{collection_name}' with {len(chunk_metadata_list)} chunks")

    return index, chroma_collection, source_paths


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    qwen3_emb_params = [
        ("0.6B", False, False),
        ("4B", False, True),
        ("4B", True, False),
    ]  # (model_suffix, use_cpu, use_8_bit)

    questions = yaml.safe_load(open(Path(__file__).parent / f"questions.yaml", "r"))

    for sfx, use_cpu, use_8bit in qwen3_emb_params:
        # Формируем полный суффикс с учётом квантизации
        full_suffix = f"{sfx}{'-8bit' if use_8bit else ''}"
        
        print("")
        print("=========")
        print(f"Using Qwen3-{sfx}" + (" (8-bit quantized)" if use_8bit else "") + (" (CPU)" if use_cpu else " (GPU)"))

        index, chroma_collection, paths = load_and_index_kb_embeddings(
            kb_embeddings_path=Path(__file__).parent / f"doc_embeddings_chunked-{full_suffix}.pck", 
            suffix=full_suffix
        )

        questions_embeddings_path = Path(__file__).parent / f"questions_embeddings-{full_suffix}.pck"
        q_embeddings = pickle.load(open(questions_embeddings_path, "rb"))
        n_questions = q_embeddings.shape[0]

        # Ensure questions embeddings are in float32 format for FAISS
        q_embeddings = np.ascontiguousarray(q_embeddings, dtype=np.float32)

        d_vals, i_vals = index.search(q_embeddings, K_SEARCH)
        print(d_vals)
        print(i_vals)

        for i in range(n_questions):
            print("")
            print("====")
            print(f"Question {i}: {questions[i]}")
            print("")
            print("FAISS Docs found:")
            for j in range(i_vals.shape[1]):
                print(f"{j}: {paths[i_vals[i, j]].name}")

            # Query Chroma DB
            print("\nChroma DB results:")
            chroma_results = chroma_collection.query(query_embeddings=[q_embeddings[i].tolist()], n_results=K_SEARCH)
            print(chroma_results["distances"])
            for j, metadata in enumerate(chroma_results["metadatas"][0]):
                # Debug: print metadata keys to see what's actually returned
                if j == 0 and i == 0:
                    print(f"DEBUG - Metadata keys: {metadata.keys()}")

                print(f"{j}: {Path(metadata['source_path']).name}")
                print(
                    f"   Token range: ({metadata.get('token_range_start', 'N/A')}, {metadata.get('token_range_end', 'N/A')})"
                )
                print(
                    f"   Char range: ({metadata.get('char_range_start', 'N/A')}, {metadata.get('char_range_end', 'N/A')})"
                )
                print(f"   Text: {metadata.get('text', '')[:100]}...")
