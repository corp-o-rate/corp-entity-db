"""
Embedding service for company name matching.

Uses sentence-transformers with Gemma3 embedding model for high-quality
semantic similarity matching of company names.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class CompanyEmbedder:
    """
    Embedding service for company names.

    Uses Google's embedding models for high-quality semantic embeddings
    suitable for company name matching.
    """

    # Default model - good balance of quality and speed
    DEFAULT_MODEL = "google/embeddinggemma-300m"
    # Alternative: smaller but faster
    # DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
    ):
        """
        Initialize the embedder.

        Args:
            model_name: HuggingFace model ID for embeddings
            device: Device to use (cuda, mps, cpu, or None for auto)
        """
        self._model_name = model_name
        self._device = device
        self._model = None
        self._embedding_dim: Optional[int] = None

    @property
    def embedding_dim(self) -> int:
        """Get the embedding dimension (loads model if needed)."""
        if self._embedding_dim is None:
            self._load_model()
        return self._embedding_dim

    def _load_model(self) -> None:
        """Load the embedding model (lazy loading)."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            device = self._device
            if device is None:
                if torch.cuda.is_available():
                    device = "cuda"
                elif torch.backends.mps.is_available():
                    device = "mps"
                else:
                    device = "cpu"

            logger.info(f"Loading embedding model '{self._model_name}' on {device}...")
            self._model = SentenceTransformer(self._model_name, device=device)
            self._embedding_dim = self._model.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded (dim={self._embedding_dim})")

        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for embeddings. "
                "Install with: pip install sentence-transformers"
            ) from e

    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text string.

        Args:
            text: Text to embed

        Returns:
            Normalized embedding vector as numpy array
        """
        self._load_model()

        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embedding.astype(np.float32)

    def embed_batch(self, texts: list[str], batch_size: int = 192) -> np.ndarray:
        """
        Embed multiple texts in batches.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Array of normalized embeddings (N x dim)
        """
        self._load_model()

        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 100,
            batch_size=batch_size,
            normalize_embeddings=True,
        )
        return embeddings.astype(np.float32)

    def quantize_to_int8(self, embedding: np.ndarray) -> np.ndarray:
        """
        Quantize L2-normalized float32 embedding to int8.

        For normalized embeddings (values in [-1, 1]), this provides
        75% storage reduction with ~92% recall at top-100.

        Args:
            embedding: L2-normalized float32 embedding vector

        Returns:
            int8 embedding vector
        """
        return np.clip(np.round(embedding * 127), -127, 127).astype(np.int8)

    def embed_and_quantize(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """
        Embed text and return both float32 and int8 embeddings.

        Args:
            text: Text to embed

        Returns:
            Tuple of (float32_embedding, int8_embedding)
        """
        fp32 = self.embed(text)
        return fp32, self.quantize_to_int8(fp32)

    def embed_batch_and_quantize(
        self, texts: list[str], batch_size: int = 192
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Embed multiple texts and return both float32 and int8 embeddings.

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing

        Returns:
            Tuple of (float32_embeddings, int8_embeddings) matrices
        """
        fp32 = self.embed_batch(texts, batch_size=batch_size)
        int8 = np.array([self.quantize_to_int8(e) for e in fp32])
        return fp32, int8

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First embedding (normalized)
            embedding2: Second embedding (normalized)

        Returns:
            Cosine similarity score (0-1 for normalized vectors)
        """
        return float(np.dot(embedding1, embedding2))

    def search_similar(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray,
        top_k: int = 20,
    ) -> list[tuple[int, float]]:
        """
        Find most similar embeddings to query.

        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: Matrix of candidate embeddings (N x dim)
            top_k: Number of results to return

        Returns:
            List of (index, similarity) tuples, sorted by similarity descending
        """
        # Compute similarities (dot product for normalized vectors)
        similarities = np.dot(candidate_embeddings, query_embedding)

        # Get top-k indices
        if len(similarities) <= top_k:
            indices = np.argsort(similarities)[::-1]
        else:
            indices = np.argpartition(similarities, -top_k)[-top_k:]
            indices = indices[np.argsort(similarities[indices])[::-1]]

        return [(int(idx), float(similarities[idx])) for idx in indices]


# Singleton instance for shared use
_default_embedder: Optional[CompanyEmbedder] = None


def get_embedder(model_name: str = CompanyEmbedder.DEFAULT_MODEL) -> CompanyEmbedder:
    """
    Get or create a shared embedder instance.

    Args:
        model_name: HuggingFace model ID

    Returns:
        CompanyEmbedder instance
    """
    global _default_embedder

    if _default_embedder is None or _default_embedder._model_name != model_name:
        _default_embedder = CompanyEmbedder(model_name=model_name)

    return _default_embedder
