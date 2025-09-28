import logging

logging.getLogger().setLevel(logging.INFO)

import pickle
import yaml

from pathlib import Path
from typing import Dict, Union

import numpy as np
from more_itertools import chunked, ichunked
from sentence_transformers import SentenceTransformer

N_DOCS_PER_BATCH = 4


def load_documents_in_folder(folder_path: Union[str, Path]) -> Dict[Path, str]:

    logging.info(f"Loading documents from {folder_path}")
    result = {}
    folder = Path(folder_path)

    # Iterate through all files in the directory (not recursively)
    for doc_path in folder.iterdir():
        if doc_path.is_file():
            result[doc_path.absolute()] = open(doc_path, "r", encoding="utf-8").read()
    logging.info(f"Loaded {len(result)} documents from {folder_path}")
    return result


# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", tokenizer_kwargs={"padding_side": "left"})
# model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", device="cpu", tokenizer_kwargs={"padding_side": "left"})

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-4B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
# queries = [
#     "What is the capital of China?",
#     "Explain gravity",
# ]

documents = load_documents_in_folder(
    Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced"
)


def encode_documents(
    mdl: SentenceTransformer, documents: Dict[Path, str], n_docs_per_batch: int = N_DOCS_PER_BATCH
) -> Dict[Path, np.ndarray]:

    result_embeddings: Dict[Path, np.ndarray] = {}

    for chunk_idx, items in enumerate(chunked(documents.items(), n_docs_per_batch)):
        logging.info(
            f"Processing chunk {chunk_idx} "
            f"(docs #{n_docs_per_batch * chunk_idx} -- #{n_docs_per_batch * chunk_idx + len(items) - 1})"
        )
        paths = []
        texts = []
        for path, text in items:
            paths.append(path)
            texts.append(text)

        embeddings = mdl.encode(texts)
        assert embeddings.shape[0] == len(paths)
        for i, path in enumerate(paths):
            result_embeddings[path] = embeddings[i][:]

    return result_embeddings


# docs_embeddings = encode_documents(model, documents)

# pickle.dump(result_embeddings, open("embeddings-0.6B.pck", "wb"))
# pickle.dump(docs_embeddings, open(Path(__file__).parent / "embeddings-4B.pck", "wb"))


questions = yaml.safe_load(open("questions.yaml", "r"))
questions_embeddings = model.encode(questions)
# pickle.dump(questions_embeddings, open(Path(__file__).parent / "questions_embeddings-4B.pck", "wb"))
pickle.dump(questions_embeddings, open(Path(__file__).parent / "questions_embeddings-0.6B.pck", "wb"))


# # Compute the (cosine) similarity between the query and document embeddings
# similarity = model.similarity(query_embeddings, document_embeddings)
# print(similarity)
# # tensor([[0.7534, 0.1147],
# #         [0.0320, 0.6258]])
