"""Dobles de prueba compartidos por los tests (sin red ni API)."""

import math


class FakeEmbedder:
    """Bolsa de palabras sobre un vocabulario fijo: determinista y sin red."""

    VOCAB = ["playa", "teide", "comer", "fiesta", "barranco", "norte", "sur"]

    def _vector(self, text):
        words = text.lower()
        values = [float(words.count(term)) + 0.01 for term in self.VOCAB]
        norm = math.sqrt(sum(v * v for v in values))
        return [v / norm for v in values]

    def embed_documents(self, texts, titles):
        return [self._vector(text) for text in texts]

    def embed_query(self, query):
        return self._vector(query)
