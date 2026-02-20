"""Tests for corp_entity_db.embeddings — CompanyEmbedder and get_embedder."""

from unittest.mock import MagicMock

import numpy as np
import pytest

from corp_entity_db.embeddings import CompanyEmbedder, get_embedder


class TestQuantizeToInt8:
    def test_range(self):
        """Quantized values should be int8 within [-127, 127]."""
        embedder = CompanyEmbedder()
        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)

        result = embedder.quantize_to_int8(vec)

        assert result.dtype == np.int8
        assert result.min() >= -127
        assert result.max() <= 127

    def test_boundary_values(self):
        """Boundary inputs [1.0, -1.0, 0.0] map to [127, -127, 0]."""
        embedder = CompanyEmbedder()
        vec = np.array([1.0, -1.0, 0.0], dtype=np.float32)

        result = embedder.quantize_to_int8(vec)

        np.testing.assert_array_equal(result, np.array([127, -127, 0], dtype=np.int8))


class TestSimilarity:
    def test_identical_vectors(self):
        """Cosine similarity of a normalized vector with itself should be ~1.0."""
        embedder = CompanyEmbedder()
        vec = np.random.randn(768).astype(np.float32)
        vec /= np.linalg.norm(vec)

        score = embedder.similarity(vec, vec)

        assert score == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors(self):
        """Cosine similarity of orthogonal unit vectors should be ~0.0."""
        embedder = CompanyEmbedder()
        v1 = np.zeros(768, dtype=np.float32)
        v1[0] = 1.0
        v2 = np.zeros(768, dtype=np.float32)
        v2[1] = 1.0

        score = embedder.similarity(v1, v2)

        assert score == pytest.approx(0.0, abs=1e-5)


class TestSearchSimilar:
    def test_finds_nearest(self):
        """The query vector itself should appear first among candidates."""
        embedder = CompanyEmbedder()
        rng = np.random.default_rng(42)
        candidates = rng.standard_normal((10, 768)).astype(np.float32)
        # normalise each row
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

        query = candidates[3].copy()
        results = embedder.search_similar(query, candidates, top_k=5)

        assert results[0][0] == 3
        assert results[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_respects_top_k(self):
        """search_similar with top_k=3 should return exactly 3 results."""
        embedder = CompanyEmbedder()
        rng = np.random.default_rng(99)
        candidates = rng.standard_normal((10, 768)).astype(np.float32)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)

        query = candidates[0].copy()
        results = embedder.search_similar(query, candidates, top_k=3)

        assert len(results) == 3


class TestEmbed:
    def test_embed_calls_model(self):
        """embed() should call model.encode with the given text."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 768
        fake_out = np.random.randn(768).astype(np.float32)
        mock_model.encode.return_value = fake_out

        embedder = CompanyEmbedder(device="cpu")
        # Bypass _load_model by injecting the mock directly
        embedder._model = mock_model
        embedder._embedding_dim = 768

        embedder.embed("Acme Corp")

        mock_model.encode.assert_called_once()
        call_args = mock_model.encode.call_args
        assert call_args[0][0] == "Acme Corp"


class TestLazyLoading:
    def test_model_not_loaded_at_init(self):
        """CompanyEmbedder should not load the model on construction."""
        embedder = CompanyEmbedder()
        assert embedder._model is None


class TestGetEmbedderSingleton:
    def test_returns_same_instance(self):
        """get_embedder() should return the same object on repeated calls."""
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2
