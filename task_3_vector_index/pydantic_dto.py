from pathlib import Path
from typing import List, Tuple

import numpy as np

from pydantic import BaseModel


class TextChunkInfo(BaseModel):
    """Represents a chunk of text with token and character ranges."""

    token_range_start: int
    token_range_end: int
    char_range_start: int
    char_range_end: int
    text: str


class TextEmbeddingsInfo(BaseModel):
    """Contains embeddings for all chunks of a document."""

    original_text_path: Path
    embeddings: np.ndarray  # Shape: (n_chunks, embedding_dim)
    chunks: List[TextChunkInfo]

    class Config:
        arbitrary_types_allowed = True
