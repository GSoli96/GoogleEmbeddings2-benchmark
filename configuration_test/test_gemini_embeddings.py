"""
Standalone test for Gemini 2 text embeddings via Vertex AI.

Requirements:
    pip install google-genai

Auth:
    gcloud auth application-default login
"""

import unittest
from google import genai
from google.genai import types

PROJECT_ID = "provatest-495410"
LOCATION   = "us-central1"
MODEL      = "text-embedding-004"   # Gemini 2 embedding model on Vertex AI


def get_client() -> genai.Client:
    return genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


class TestGeminiEmbeddings(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = get_client()

    # ── basic ──────────────────────────────────────────────────────────────────

    def test_single_text_returns_embedding(self):
        """A single text should return a non-empty float vector."""
        response = self.client.models.embed_content(
            model=MODEL,
            contents="Hello, Gemini embeddings!",
        )
        embedding = response.embeddings[0].values
        self.assertIsInstance(embedding, list)
        self.assertGreater(len(embedding), 0)
        self.assertIsInstance(embedding[0], float)

    def test_embedding_dimension_is_consistent(self):
        """Two different texts should produce vectors of the same length."""
        texts = ["first sentence", "second sentence"]
        response = self.client.models.embed_content(
            model=MODEL,
            contents=texts,
        )
        dims = [len(e.values) for e in response.embeddings]
        self.assertEqual(dims[0], dims[1])

    def test_batch_embed_returns_one_per_input(self):
        """Batch of N texts should return exactly N embeddings."""
        texts = ["cat", "dog", "fish", "bird"]
        response = self.client.models.embed_content(
            model=MODEL,
            contents=texts,
        )
        self.assertEqual(len(response.embeddings), len(texts))

    # ── semantic ───────────────────────────────────────────────────────────────

    def test_similar_texts_are_closer_than_dissimilar(self):
        """Cosine similarity: 'king' vs 'queen' > 'king' vs 'banana'."""
        import math

        def cosine(a, b):
            dot  = sum(x * y for x, y in zip(a, b))
            norm = lambda v: math.sqrt(sum(x * x for x in v))
            return dot / (norm(a) * norm(b))

        response = self.client.models.embed_content(
            model=MODEL,
            contents=["king", "queen", "banana"],
        )
        vecs = [e.values for e in response.embeddings]
        sim_related    = cosine(vecs[0], vecs[1])   # king ↔ queen
        sim_unrelated  = cosine(vecs[0], vecs[2])   # king ↔ banana
        self.assertGreater(sim_related, sim_unrelated)

    def test_identical_texts_have_similarity_near_one(self):
        """Embedding of identical texts should yield cosine similarity ≈ 1."""
        import math

        text = "The quick brown fox jumps over the lazy dog"
        response = self.client.models.embed_content(
            model=MODEL,
            contents=[text, text],
        )
        a, b = [e.values for e in response.embeddings]
        dot  = sum(x * y for x, y in zip(a, b))
        norm = lambda v: math.sqrt(sum(x * x for x in v))
        sim  = dot / (norm(a) * norm(b))
        self.assertAlmostEqual(sim, 1.0, delta=1e-4)

    # ── task type ──────────────────────────────────────────────────────────────

    def test_task_type_retrieval_query(self):
        """task_type=RETRIEVAL_QUERY should work without errors."""
        response = self.client.models.embed_content(
            model=MODEL,
            contents="What is the capital of Italy?",
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
        )
        self.assertGreater(len(response.embeddings[0].values), 0)

    def test_task_type_retrieval_document(self):
        """task_type=RETRIEVAL_DOCUMENT should work without errors."""
        response = self.client.models.embed_content(
            model=MODEL,
            contents="Rome is the capital of Italy.",
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        self.assertGreater(len(response.embeddings[0].values), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
