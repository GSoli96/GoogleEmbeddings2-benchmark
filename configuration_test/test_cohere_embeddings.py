#!/usr/bin/env python3
"""
Test connessione e embedding per Cohere Embed v3
Modello: embed-multilingual-v3.0

Requisiti:
    pip install cohere python-dotenv

Uso:
    export COHERE_API_KEY="la_tua_api_key"
    python test_cohere_embed.py
"""

import os
import sys
from typing import List

import cohere
from dotenv import load_dotenv


# =========================
# CONFIG MODELLO
# =========================
MODEL_CONFIG = {
    "name": "Cohere Embed v3",
    "short_name": "Cohere-v3",
    "provider": "cohere",
    "model_id": "embed-multilingual-v3.0",
    "dimension": 1024,
    "max_tokens": 512,
    "multilingual": True,
    "cost_per_million_tokens": 0.100,
    "api_env_var": "COHERE_API_KEY",
    "batch_size": 96,
}


def load_api_key() -> str:
    """
    Carica API key da environment variable o file .env
    """
    load_dotenv()

    api_key = os.getenv(MODEL_CONFIG["api_env_var"])

    if not api_key:
        raise ValueError(
            f"Variabile ambiente {MODEL_CONFIG['api_env_var']} non trovata.\n"
            "Esempio:\n"
            'export COHERE_API_KEY="your_api_key"'
        )

    return api_key


def create_client(api_key: str) -> cohere.Client:
    """
    Crea client Cohere
    """
    return cohere.Client(api_key)


def test_connection(client: cohere.Client) -> bool:
    """
    Test semplice di connessione
    """
    print("🔌 Test connessione Cohere...")

    try:
        response = client.embed(
            model=MODEL_CONFIG["model_id"],
            texts=["hello world"],
            input_type="search_document",
            embedding_types=["float"],
        )

        embedding = response.embeddings.float[0]

        print("✅ Connessione OK")
        print(f"📏 Dimension embedding: {len(embedding)}")

        expected_dim = MODEL_CONFIG["dimension"]

        if len(embedding) != expected_dim:
            print(
                f"⚠️  Dimensione inattesa: {len(embedding)} "
                f"(atteso: {expected_dim})"
            )
        else:
            print(f"✅ Dimensione corretta: {expected_dim}")

        return True

    except Exception as e:
        print(f"❌ Errore connessione: {e}")
        return False


def generate_embeddings(
    client: cohere.Client,
    texts: List[str],
    input_type: str = "search_document",
):
    """
    Genera embeddings per una lista di testi

    input_type:
        - search_document
        - search_query
        - classification
        - clustering
    """

    print(f"\n🧠 Generazione embeddings ({len(texts)} testi)...")

    response = client.embed(
        model=MODEL_CONFIG["model_id"],
        texts=texts,
        input_type=input_type,
        embedding_types=["float"],
    )

    embeddings = response.embeddings.float

    print(f"✅ Embeddings generati: {len(embeddings)}")

    for i, emb in enumerate(embeddings):
        print(f"   Testo {i+1}: dimension={len(emb)}")

    return embeddings


def main():
    print("=" * 60)
    print(f"MODELLO: {MODEL_CONFIG['name']}")
    print("=" * 60)

    try:
        # Load API key
        api_key = load_api_key()

        # Create client
        client = create_client(api_key)

        # Test connessione
        if not test_connection(client):
            sys.exit(1)

        # Test embedding multilingua
        sample_texts = [
            "Ciao mondo",
            "Hello world",
            "Bonjour le monde",
            "Hola mundo",
            "Questo è un test di embedding multilingua",
        ]

        embeddings = generate_embeddings(
            client=client,
            texts=sample_texts,
            input_type="search_document",
        )

        # Mostra anteprima embedding
        first_embedding = embeddings[0]

        print("\n📌 Preview primo embedding:")
        print(first_embedding[:10])  # primi 10 valori

        print("\n🎉 Test completato con successo")

    except Exception as e:
        print(f"\n❌ Errore generale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()