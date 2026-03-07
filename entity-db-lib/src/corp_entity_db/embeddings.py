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
        self._segment_cache: dict[str, np.ndarray] = {}

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

    def embed_composite_person(
        self,
        name: str,
        role: str | None,
        org: str | None,
        weights: tuple[float, float, float] = (8.0, 1.0, 4.0),
        segment_dim: int = 256,
    ) -> np.ndarray:
        """
        Embed a person as a composite 768-dim vector from three 256-dim segments.

        Each segment is the first `segment_dim` dims of the full embedding,
        scaled by the corresponding weight, then the full vector is L2-normalized.

        Args:
            name: Person name
            role: Role/job title (None → zero vector segment)
            org: Organization name (None → zero vector segment)
            weights: (name_weight, role_weight, org_weight)
            segment_dim: Dimensions per segment (default 256, 3×256=768)

        Returns:
            L2-normalized float32 vector of shape (segment_dim * 3,)
        """
        self._load_model()

        # Collect texts to embed (deduplicate)
        texts_to_embed: list[str] = [name]
        if role:
            texts_to_embed.append(role)
        if org:
            texts_to_embed.append(org)

        raw = self._model.encode(
            texts_to_embed,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        ).astype(np.float32)

        # Extract + L2-normalize each segment independently (Matryoshka requirement),
        # then scale by weight so weights control relative importance in cosine sim.
        def _normed_segment(vec: np.ndarray) -> np.ndarray:
            seg = vec[:segment_dim].copy()
            n = np.linalg.norm(seg)
            if n > 0:
                seg /= n
            return seg

        name_seg = _normed_segment(raw[0]) * weights[0]
        idx = 1
        if role:
            role_seg = _normed_segment(raw[idx]) * weights[1]
            idx += 1
        else:
            role_seg = np.zeros(segment_dim, dtype=np.float32)
        if org:
            org_seg = _normed_segment(raw[idx]) * weights[2]
        else:
            org_seg = np.zeros(segment_dim, dtype=np.float32)

        composite = np.concatenate([name_seg, role_seg, org_seg])
        norm = np.linalg.norm(composite)
        if norm > 0:
            composite /= norm
        return composite.astype(np.float32)

    def embed_composite_person_batch(
        self,
        names: list[str],
        roles: list[str | None],
        orgs: list[str | None],
        weights: tuple[float, float, float] = (8.0, 1.0, 4.0),
        segment_dim: int = 256,
        batch_size: int = 192,
    ) -> np.ndarray:
        """
        Batch-embed people as composite 768-dim vectors.

        Deduplicates all unique texts across names/roles/orgs, embeds once,
        then assembles composite vectors.

        Args:
            names: List of person names
            roles: List of role strings (None for missing)
            orgs: List of org strings (None for missing)
            weights: (name_weight, role_weight, org_weight)
            segment_dim: Dimensions per segment
            batch_size: Embedding batch size

        Returns:
            (N, segment_dim*3) float32 array, L2-normalized
        """
        self._load_model()
        n = len(names)

        # Collect all unique texts
        unique_texts: dict[str, int] = {}
        text_list: list[str] = []
        for text in names:
            if text not in unique_texts:
                unique_texts[text] = len(text_list)
                text_list.append(text)
        for text in roles:
            if text is not None and text not in unique_texts:
                unique_texts[text] = len(text_list)
                text_list.append(text)
        for text in orgs:
            if text is not None and text not in unique_texts:
                unique_texts[text] = len(text_list)
                text_list.append(text)

        # Filter to uncached texts only
        uncached = [t for t in text_list if t not in self._segment_cache]
        logger.debug(f"Composite person batch: {n} people, {len(text_list)} unique texts, {len(uncached)} uncached")

        if uncached:
            raw = self._model.encode(
                uncached,
                convert_to_numpy=True,
                show_progress_bar=len(uncached) > 100,
                batch_size=batch_size,
                normalize_embeddings=False,
            ).astype(np.float32)

            # Truncate to segment_dim and L2-normalize (Matryoshka requirement)
            raw_truncated = raw[:, :segment_dim].copy()
            norms = np.linalg.norm(raw_truncated, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            raw_truncated /= norms

            for j, text in enumerate(uncached):
                self._segment_cache[text] = raw_truncated[j]

        # Build segment lookup array from cache in text_list order
        cached_segments = np.array([self._segment_cache[t] for t in text_list])

        # Vectorized assembly: fancy-index into cached_segments
        name_indices = np.array([unique_texts[nm] for nm in names])
        name_segs = cached_segments[name_indices] * weights[0]

        role_indices = np.array([unique_texts[r] if r is not None else 0 for r in roles])
        role_segs = cached_segments[role_indices] * weights[1]
        role_mask = np.array([r is not None for r in roles], dtype=np.float32)[:, np.newaxis]
        role_segs *= role_mask

        org_indices = np.array([unique_texts[o] if o is not None else 0 for o in orgs])
        org_segs = cached_segments[org_indices] * weights[2]
        org_mask = np.array([o is not None for o in orgs], dtype=np.float32)[:, np.newaxis]
        org_segs *= org_mask

        out = np.concatenate([name_segs, role_segs, org_segs], axis=1)

        # L2-normalize each row
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        out /= norms
        return out

    def clear_segment_cache(self) -> None:
        """Clear the cross-batch segment embedding cache to free memory."""
        count = len(self._segment_cache)
        self._segment_cache.clear()
        logger.info(f"Cleared segment cache ({count:,} entries)")

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
