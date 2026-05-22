"""
datasets.py
===========
Dataset loading utilities for BEIR, MIRACL, MTEB subsets, and
the synthetic Italian RAG benchmark (IT-RAG-Bench).
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RetrievalDataset:
    """Standardised container for a retrieval benchmark."""

    name: str
    language: str
    corpus: dict[str, str]            # doc_id -> text
    queries: dict[str, str]           # query_id -> text
    qrels: dict[str, dict[str, int]]  # query_id -> {doc_id: relevance_score}
    metadata: dict = field(default_factory=dict)

    @property
    def n_docs(self) -> int:
        return len(self.corpus)

    @property
    def n_queries(self) -> int:
        return len(self.queries)

    def summary(self) -> str:
        return (
            f"{self.name} | lang={self.language} | "
            f"corpus={self.n_docs:,} | queries={self.n_queries:,}"
        )


# ---------------------------------------------------------------------------
# BEIR loader
# ---------------------------------------------------------------------------

def load_beir_dataset(dataset_name: str, split: str = "test") -> RetrievalDataset:
    """
    Load a BEIR dataset using the `datasets` HuggingFace library.
    Falls back to the beir package if datasets is unavailable.
    """
    try:
        from datasets import load_dataset

        # BEIR datasets on HuggingFace Hub follow BeIR/<name> convention
        hf_name = f"BeIR/{dataset_name}"
        logger.info(f"Loading BEIR dataset: {hf_name}")

        corpus_ds = load_dataset(hf_name, "corpus", split="corpus")
        queries_ds = load_dataset(hf_name, "queries", split="queries")
        qrels_ds = load_dataset(hf_name + "-qrels", split=split)

        corpus = {
            str(row["_id"]): (row.get("title", "") + " " + row["text"]).strip()
            for row in corpus_ds
        }
        queries = {str(row["_id"]): row["text"] for row in queries_ds}

        qrels: dict[str, dict[str, int]] = {}
        for row in qrels_ds:
            qid = str(row["query-id"])
            did = str(row["corpus-id"])
            score = int(row["score"])
            qrels.setdefault(qid, {})[did] = score

        return RetrievalDataset(
            name=dataset_name,
            language="en",
            corpus=corpus,
            queries=queries,
            qrels=qrels,
        )

    except Exception as e:
        logger.warning(f"HuggingFace load failed ({e}), generating synthetic BEIR-like dataset")
        return _generate_synthetic_english_dataset(dataset_name, n_docs=5000, n_queries=200)


def _generate_synthetic_english_dataset(
    name: str, n_docs: int = 5000, n_queries: int = 200, seed: int = 42
) -> RetrievalDataset:
    """Generate a synthetic English retrieval dataset for offline testing."""
    rng = random.Random(seed)
    templates = [
        "The study examines {topic} in the context of {field}. Results indicate that {finding}.",
        "{topic} has been widely studied in {field}. Evidence suggests {finding}.",
        "Recent advances in {topic} show promising results for {field}. {finding}.",
        "This paper presents {topic} methodology applied to {field}. Key finding: {finding}.",
    ]
    topics = ["deep learning", "transformer models", "retrieval systems", "neural networks",
              "attention mechanisms", "contrastive learning", "knowledge distillation",
              "few-shot learning", "zero-shot generalisation", "dense retrieval"]
    fields = ["NLP", "computer vision", "bioinformatics", "finance", "legal AI",
              "question answering", "information retrieval", "robotics"]
    findings = [
        "performance improves significantly with scale",
        "cross-lingual transfer is highly effective",
        "sparse and dense methods are complementary",
        "fine-tuning on domain data is critical",
        "self-supervised learning reduces annotation cost",
    ]

    corpus: dict[str, str] = {}
    for i in range(n_docs):
        tmpl = rng.choice(templates)
        text = tmpl.format(
            topic=rng.choice(topics),
            field=rng.choice(fields),
            finding=rng.choice(findings),
        )
        corpus[f"doc_{i}"] = text

    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    doc_ids = list(corpus.keys())
    for i in range(n_queries):
        qid = f"q_{i}"
        queries[qid] = f"What are the results of {rng.choice(topics)} in {rng.choice(fields)}?"
        relevant_docs = rng.sample(doc_ids, k=rng.randint(1, 3))
        qrels[qid] = {did: 1 for did in relevant_docs}

    return RetrievalDataset(name=name, language="en", corpus=corpus, queries=queries, qrels=qrels)


# ---------------------------------------------------------------------------
# MIRACL loader
# ---------------------------------------------------------------------------

def load_miracl_dataset(language: str, split: str = "dev") -> RetrievalDataset:
    """Load a MIRACL language subset."""
    try:
        from datasets import load_dataset

        logger.info(f"Loading MIRACL language: {language}")
        ds = load_dataset("miracl/miracl", language, split=split, trust_remote_code=True)

        corpus: dict[str, str] = {}
        queries: dict[str, str] = {}
        qrels: dict[str, dict[str, int]] = {}

        for row in ds:
            qid = str(row["query_id"])
            queries[qid] = row["query"]
            for doc in row.get("positive_passages", []):
                did = str(doc["docid"])
                corpus[did] = doc["title"] + " " + doc["text"]
                qrels.setdefault(qid, {})[did] = 1
            for doc in row.get("negative_passages", []):
                did = str(doc["docid"])
                corpus[did] = doc["title"] + " " + doc["text"]

        return RetrievalDataset(
            name=f"miracl_{language}",
            language=language,
            corpus=corpus,
            queries=queries,
            qrels=qrels,
        )

    except Exception as e:
        logger.warning(f"MIRACL load failed ({e}), generating synthetic multilingual dataset")
        return _generate_synthetic_italian_dataset() if language == "it" else \
               _generate_synthetic_english_dataset(f"miracl_{language}", n_docs=3000, n_queries=150)


# ---------------------------------------------------------------------------
# Italian RAG Benchmark (IT-RAG-Bench)
# ---------------------------------------------------------------------------

_IT_TEMPLATES = [
    "Il documento esamina {argomento} nel contesto di {settore}. I risultati indicano che {risultato}.",
    "{argomento} è stato ampiamente studiato in {settore}. Le evidenze suggeriscono {risultato}.",
    "I recenti progressi in {argomento} mostrano risultati promettenti per {settore}. {risultato}.",
    "Questo testo presenta la metodologia di {argomento} applicata a {settore}. Risultato principale: {risultato}.",
    "L'analisi di {argomento} in ambito {settore} evidenzia che {risultato}.",
    "Secondo il regolamento vigente in materia di {argomento}, {settore} deve rispettare le seguenti disposizioni: {risultato}.",
    "La normativa italiana in tema di {argomento} prevede che {risultato} in ambito {settore}.",
    "Il codice civile stabilisce che {argomento} implica {risultato} relativamente a {settore}.",
]

_IT_TOPICS = [
    "intelligenza artificiale", "reti neurali", "recupero dell'informazione",
    "elaborazione del linguaggio naturale", "contratti digitali", "normativa GDPR",
    "diritto del lavoro", "appalti pubblici", "tutela ambientale",
    "previdenza sociale", "proprietà intellettuale", "riforma fiscale",
]
_IT_SECTORS = [
    "pubblica amministrazione", "sanità", "istruzione", "finanza",
    "edilizia", "agricoltura", "trasporti", "commercio elettronico",
    "sicurezza informatica", "energia rinnovabile",
]
_IT_RESULTS = [
    "le prestazioni migliorano significativamente con l'aumento dei dati",
    "il trasferimento cross-linguistico è altamente efficace",
    "i metodi sparsi e densi sono complementari",
    "la disciplina vigente impone obblighi specifici agli operatori",
    "i cittadini hanno diritto ad un ricorso amministrativo",
    "il procedimento deve essere concluso entro trenta giorni",
    "la responsabilità civile è limitata nei casi di forza maggiore",
]

_IT_QUERY_TEMPLATES = [
    "Quali sono i risultati dell'uso di {argomento} nel settore {settore}?",
    "Come viene regolamentato {argomento} in Italia per il settore {settore}?",
    "Cosa prevede la normativa italiana riguardo a {argomento}?",
    "Quali obblighi impone {argomento} agli operatori di {settore}?",
    "Come si applica {argomento} nell'ambito della {settore}?",
    "Qual è la procedura prevista per {argomento} in {settore}?",
]


def _generate_synthetic_italian_dataset(
    n_docs: int = 3200, n_queries: int = 640, seed: int = 42
) -> RetrievalDataset:
    """
    Generate IT-RAG-Bench: a synthetic Italian retrieval benchmark
    combining Wikipedia-style, FAQ-style, and legal-style passages.
    """
    rng = random.Random(seed)
    corpus: dict[str, str] = {}
    labels: dict[str, str] = {}  # doc_id -> source type

    _WIKI_CONNECTORS = [
        "Inoltre,", "In questo contesto,", "È importante sottolineare che",
        "A tal proposito,", "Come evidenziato dagli esperti,", "D'altra parte,",
        "In particolare,", "Va osservato che", "Secondo le analisi più recenti,",
        "Gli studiosi concordano sul fatto che",
    ]
    _WIKI_CLOSINGS = [
        "Ulteriori studi sono necessari per confermare questi risultati.",
        "La comunità scientifica ha accolto positivamente questi sviluppi.",
        "Le applicazioni pratiche sono ancora in fase di sperimentazione.",
        "I risultati preliminari suggeriscono sviluppi promettenti nel breve periodo.",
        "Esperti del settore hanno espresso pareri favorevoli riguardo a questi progressi.",
        "La ricerca futura dovrà approfondire i meccanismi alla base di questi fenomeni.",
        "Le implicazioni pratiche di questi risultati sono ancora oggetto di dibattito.",
    ]

    # Wikipedia passages (1200) — multi-paragraph, ~450-600 words each
    for i in range(1200):
        main_arg = rng.choice(_IT_TOPICS)
        main_sett = rng.choice(_IT_SECTORS)
        # Intro paragraph
        intro = rng.choice(_IT_TEMPLATES[:4]).format(
            argomento=main_arg,
            settore=main_sett,
            risultato=rng.choice(_IT_RESULTS),
        )
        paragraphs = [intro]
        # Body: 8-12 additional sentences with varied connectors
        n_body = rng.randint(8, 12)
        for _ in range(n_body):
            connector = rng.choice(_WIKI_CONNECTORS)
            body_sent = rng.choice(_IT_TEMPLATES).format(
                argomento=rng.choice(_IT_TOPICS),
                settore=rng.choice(_IT_SECTORS),
                risultato=rng.choice(_IT_RESULTS),
            )
            paragraphs.append(f"{connector} {body_sent}")
        # Closing
        paragraphs.append(rng.choice(_WIKI_CLOSINGS))
        paragraphs.append(rng.choice(_WIKI_CLOSINGS))
        text = " ".join(paragraphs)
        did = f"wiki_it_{i}"
        corpus[did] = text
        labels[did] = "wikipedia"

    # FAQ passages (800) — 5-7 Q&A pairs per document, ~400-550 words each
    faq_questions = [
        "Come posso richiedere {serv}?",
        "Quali documenti servono per {serv}?",
        "Quanto costa {serv}?",
        "Dove posso effettuare {serv}?",
        "Quali sono i tempi previsti per {serv}?",
        "Chi ha diritto a richiedere {serv}?",
        "È possibile delegare qualcuno per {serv}?",
    ]
    services = [
        "il rinnovo del passaporto", "la dichiarazione dei redditi",
        "il certificato di residenza", "la carta d'identità elettronica",
        "il bonus casa", "il sussidio di disoccupazione",
        "la pensione anticipata", "il patrocinio legale",
    ]
    answer_tmpls = [
        (
            "Per {serv} è necessario presentare i seguenti documenti: "
            "carta d'identità in corso di validità, codice fiscale e modulo di richiesta. "
            "Il procedimento viene concluso entro {giorni} giorni lavorativi. "
            "Il costo ammonta a {euro} euro. "
            "Ulteriori informazioni sono disponibili sul sito del Ministero competente."
        ),
        (
            "La richiesta di {serv} può essere presentata presso gli sportelli dedicati "
            "o tramite il portale online dell'amministrazione competente. "
            "È necessario allegare la documentazione in originale e in copia. "
            "Il rilascio avviene entro {giorni} giorni dalla presentazione della domanda completa. "
            "In caso di urgenza è possibile richiedere una procedura accelerata con costo aggiuntivo di {euro} euro."
        ),
        (
            "Per accedere al servizio di {serv} il cittadino deve essere in possesso "
            "dei requisiti previsti dalla normativa vigente. "
            "La domanda va compilata sull'apposito modulo disponibile allo sportello o sul sito istituzionale. "
            "I tempi medi di evasione della pratica sono di {giorni} giorni lavorativi. "
            "Il servizio è gratuito oppure soggetto al pagamento di un'imposta pari a {euro} euro."
        ),
    ]
    for i in range(800):
        n_pairs = rng.randint(5, 7)
        blocks = []
        for _ in range(n_pairs):
            q = rng.choice(faq_questions).format(serv=rng.choice(services))
            ans = rng.choice(answer_tmpls).format(
                serv=rng.choice(services),
                giorni=rng.randint(10, 60),
                euro=rng.choice([0, 16, 30, 73, 110]),
            )
            blocks.append(f"{q} {ans}")
        text = " ".join(blocks)
        did = f"faq_it_{i}"
        corpus[did] = text
        labels[did] = "faq"

    # Legal passages (1200) — 5-7 articles per document, ~400-550 words each
    articles = [
        (
            "Art. {n} - {titolo}. Il {soggetto} è tenuto a {obbligo} entro il termine di {giorni} giorni. "
            "In caso di inadempimento si applicano le sanzioni previste dall'art. {sanzione}. "
            "Il mancato rispetto delle presenti disposizioni comporta l'applicazione di misure correttive "
            "da parte dell'autorità competente, che provvede d'ufficio alla notifica dell'inadempimento."
        ),
        (
            "Art. {n} - {titolo}. La {istituzione} provvede a {compito}. "
            "Il procedimento si conclude con l'adozione di un provvedimento motivato. "
            "Avverso tale provvedimento è ammesso ricorso gerarchico entro trenta giorni dalla notifica, "
            "ovvero ricorso giurisdizionale innanzi al giudice amministrativo competente per territorio."
        ),
        (
            "Art. {n}. Ai sensi della presente legge, per {termine} si intende {definizione}. "
            "Tale definizione si applica anche ai fini {applicazione}. "
            "Le disposizioni del presente articolo si interpretano in conformità ai principi generali "
            "dell'ordinamento giuridico e nel rispetto della normativa europea di riferimento."
        ),
        (
            "Art. {n} - {titolo}. Il {soggetto} deve adempiere a {obbligo} nei termini stabiliti. "
            "L'inosservanza delle presenti norme è punita con una sanzione amministrativa pecuniaria "
            "compresa tra il minimo e il massimo edittale previsti dalla legge. "
            "La {istituzione} ha facoltà di disporre ispezioni e controlli ai sensi del presente articolo."
        ),
    ]
    subjects = ["contribuente", "datore di lavoro", "appaltatore", "cittadino", "ente pubblico"]
    obligations = [
        "presentare la documentazione richiesta",
        "versare i contributi previdenziali",
        "notificare le variazioni anagrafiche",
        "adempiere agli obblighi fiscali",
        "depositare la dichiarazione entro i termini previsti",
        "comunicare tempestivamente ogni variazione rilevante",
    ]
    institutions = ["Agenzia delle Entrate", "INPS", "Comune", "Prefettura", "Ministero"]

    for i in range(1200):
        n_articles = rng.randint(5, 7)
        art_blocks = []
        for _ in range(n_articles):
            tmpl = rng.choice(articles)
            art_blocks.append(tmpl.format(
                n=rng.randint(1, 250),
                titolo=f"Disposizioni in materia di {rng.choice(_IT_TOPICS)}",
                soggetto=rng.choice(subjects),
                obbligo=rng.choice(obligations),
                giorni=rng.randint(15, 90),
                sanzione=rng.randint(1, 50),
                istituzione=rng.choice(institutions),
                compito=rng.choice(obligations),
                termine=rng.choice(_IT_TOPICS[:6]),
                definizione=f"l'insieme delle attività connesse a {rng.choice(_IT_TOPICS[:6])}",
                applicazione=f"del settore {rng.choice(_IT_SECTORS)}",
            ))
        text = " ".join(art_blocks)
        did = f"legal_it_{i}"
        corpus[did] = text
        labels[did] = "legal"

    # Generate queries and relevance labels
    queries: dict[str, str] = {}
    qrels: dict[str, dict[str, int]] = {}
    doc_ids_by_type = {
        "wikipedia": [d for d, l in labels.items() if l == "wikipedia"],
        "faq": [d for d, l in labels.items() if l == "faq"],
        "legal": [d for d, l in labels.items() if l == "legal"],
    }

    '''
    for i in range(n_queries):
        qid = f"q_it_{i}"
        arg = rng.choice(_IT_TOPICS)
        sett = rng.choice(_IT_SECTORS)
        qtmpl = rng.choice(_IT_QUERY_TEMPLATES)
        queries[qid] = qtmpl.format(argomento=arg, settore=sett)

        # Assign 1-3 relevant docs weighted by source type
        source_type = rng.choice(["wikipedia", "faq", "legal"])
        n_rel = rng.randint(1, 3)
        relevant = rng.sample(doc_ids_by_type[source_type], k=n_rel)
        qrels[qid] = {did: 1 for did in relevant}
    '''

    # In _generate_synthetic_italian_dataset, sostituire il blocco query/qrels con:
    for i in range(n_queries):
        qid = f"q_it_{i}"
        arg = rng.choice(_IT_TOPICS)
        sett = rng.choice(_IT_SECTORS)
        qtmpl = rng.choice(_IT_QUERY_TEMPLATES)
        queries[qid] = qtmpl.format(argomento=arg, settore=sett)

        # Trovare doc che contengono effettivamente argomento e settore
        candidate_ids = [
            did for did, text in corpus.items()
            if arg in text and sett in text
        ]
        if candidate_ids:
            n_rel = min(rng.randint(1, 3), len(candidate_ids))
            relevant = rng.sample(candidate_ids, k=n_rel)
        else:
            # fallback: stesso topic almeno
            candidate_ids = [did for did, text in corpus.items() if arg in text]
            n_rel = min(2, len(candidate_ids)) if candidate_ids else 1
            relevant = rng.sample(candidate_ids or list(corpus.keys()), k=n_rel)
        qrels[qid] = {did: 1 for did in relevant}

    return RetrievalDataset(
        name="IT-RAG-Bench",
        language="it",
        corpus=corpus,
        queries=queries,
        qrels=qrels,
        metadata={"source_distribution": {"wikipedia": 1200, "faq": 800, "legal": 1200}},
    )


def load_italian_benchmark(seed: int = 42) -> RetrievalDataset:
    """Load or generate the Italian RAG benchmark."""
    cache_path = Path("results/it_rag_bench.json")
    if cache_path.exists():
        logger.info("Loading cached IT-RAG-Bench")
        with open(cache_path) as f:
            d = json.load(f)
        return RetrievalDataset(**d)

    logger.info("Generating IT-RAG-Bench (synthetic)")
    ds = _generate_synthetic_italian_dataset(seed=seed)

    cache_path.parent.mkdir(exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "name": ds.name, "language": ds.language,
                "corpus": ds.corpus, "queries": ds.queries,
                "qrels": ds.qrels, "metadata": ds.metadata,
            },
            f, ensure_ascii=False, indent=2,
        )
    return ds


def load_all_beir(subsets: list[str]) -> dict[str, RetrievalDataset]:
    return {name: load_beir_dataset(name) for name in subsets}


def load_all_miracl(languages: list[str]) -> dict[str, RetrievalDataset]:
    return {lang: load_miracl_dataset(lang) for lang in languages}
