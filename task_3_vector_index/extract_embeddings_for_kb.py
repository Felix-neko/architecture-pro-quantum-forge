from pathlib import Path
from typing import Dict, List, Tuple, Union

import logging
import pickle

import numpy as np
from sentence_transformers import SentenceTransformer
import yaml
from contexttimer import Timer

from task_3_vector_index.pydantic_dto import TextChunkInfo, TextEmbeddingsInfo


DOC_CHUNK_SIZE = 500
DOC_OVERLAPPING_SIZE = 50


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


def get_model_context_size(model: SentenceTransformer, tokenizer) -> int:
    # Сначала пробуем config.max_position_embeddings
    try:
        cfg = getattr(model, "auto_model", None)
        if cfg is not None and hasattr(cfg, "config"):
            val = getattr(cfg.config, "max_position_embeddings", None)
            if val is not None and val > 0:
                return int(val)
    except Exception:
        pass
    # fallback на tokenizer
    return int(getattr(tokenizer, "model_max_length", 512))


def chunk_text_tokenwise(text: str, tokenizer, chunk_size: int, stride: int = 5) -> List[TextChunkInfo]:
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    # маска спецтокенов (1 — спецтокен)
    special_mask = tokenizer.get_special_tokens_mask(input_ids, already_has_special_tokens=True)
    token_indices = [i for i, m in enumerate(special_mask) if m == 0]

    chunks = []
    i = 0
    step = max(1, chunk_size - stride)
    while i < len(token_indices):
        window = token_indices[i : i + chunk_size]
        if not window:
            break
        tok_start, tok_end = window[0], window[-1]
        char_start = offsets[tok_start][0]
        char_end = offsets[tok_end][1]
        chunk_text = text[char_start:char_end]
        chunks.append(
            TextChunkInfo(
                token_range_start=tok_start,
                token_range_end=tok_end,
                char_range_start=char_start,
                char_range_end=char_end,
                text=chunk_text,
            )
        )
        i += step
    return chunks


def encode_documents(
    mdl: SentenceTransformer,
    documents: Dict[Path, str],
    chunk_size: int = DOC_CHUNK_SIZE,
    overlap_size: int = DOC_OVERLAPPING_SIZE,
) -> List[TextEmbeddingsInfo]:
    """
    Encode documents by splitting them into overlapping chunks.

    Args:
        mdl: SentenceTransformer model for encoding
        documents: Dictionary mapping file paths to document text
        chunk_size: Size of each chunk in tokens
        overlap_size: Number of overlapping tokens between chunks

    Returns:
        List of TextEmbeddingsInfo objects, one per document
    """
    tokenizer = mdl.tokenizer
    result: List[TextEmbeddingsInfo] = []

    with Timer() as timer:
        for doc_idx, (path, text) in enumerate(documents.items()):
            logging.info(f"Processing document {doc_idx + 1}/{len(documents)}: {path.name}")

            # Split document into chunks
            chunks = chunk_text_tokenwise(text, tokenizer, chunk_size=chunk_size, stride=overlap_size)

            if not chunks:
                logging.warning(f"Document {path.name} produced no chunks, skipping")
                continue

            logging.info(f"  Split into {len(chunks)} chunks")

            # Extract chunk texts for encoding
            chunk_texts = [chunk.text for chunk in chunks]

            # Encode all chunks at once

            embeddings = mdl.encode(chunk_texts, show_progress_bar=False)

            embeddings = np.astype(embeddings, np.float32)

            # Create TextEmbeddingsInfo object
            result.append(TextEmbeddingsInfo(original_text_path=path, embeddings=embeddings, chunks=chunks))
    logging.info(f"Encoding time: {timer.elapsed:.2f} seconds")
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    qwen3_emb_params = [
        ("0.6B", False, False),
        ("4B", False, True),
        ("4B", True, False),
    ]  # (model_suffix, use_cpu, use_8_bit)
    documents = load_documents_in_folder(
        Path(__file__).parent.parent / "task_2_sample_dataset/arcanum_articles/text_output_replaced"
    )
    questions = yaml.safe_load(open(Path(__file__).parent / "questions.yaml", "r"))

    for sfx, use_cpu, use_8bit in qwen3_emb_params:
        logging.info("=========")
        logging.info(f"Qwen3-{sfx}, cpu: {use_cpu}, 8bit: {use_8bit}")
        logging.info(f"encoding documents...")

        # Load model with 8-bit quantization using bitsandbytes
        # Requires: pip install bitsandbytes (or: uv add bitsandbytes)
        model = SentenceTransformer(
            f"Qwen/Qwen3-Embedding-{sfx}",
            device="cpu" if use_cpu else "cuda",
            model_kwargs={"load_in_8bit": use_8bit},  # Enable 8-bit quantization (~4GB RAM вместо ~8GB)
            tokenizer_kwargs={"padding_side": "left"},
        )

        docs_embeddings = encode_documents(model, documents)
        questions_embeddings = model.encode(questions)  # Считаем вопросы короткими, по каждому из них просто 1 вектор
        del model

        logging.info("saving embeddings...")
        with open(Path(__file__).parent / f"doc_embeddings_chunked-{sfx}{'-8bit' if use_8bit else ''}.pck", "wb") as f:
            pickle.dump(docs_embeddings, f)
        with open(Path(__file__).parent / f"questions_embeddings-{sfx}{'-8bit' if use_8bit else ''}.pck", "wb") as f:
            pickle.dump(questions_embeddings, f)
        logging.info(f"Saved {len(docs_embeddings)} documents with chunked embeddings")
        logging.info("embeddings saved...")
