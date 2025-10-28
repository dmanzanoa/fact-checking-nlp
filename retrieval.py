"""
retrieval.py
============

This module implements simple information–retrieval baselines for matching
claims to supporting evidence passages.  Two retrieval strategies are
provided:

* A TF‑IDF‑based cosine similarity model that ranks evidence passages based on
  the cosine similarity between the TF‑IDF vectors of the claim and each
  evidence.

* A BM25 model (Okapi BM25) that ranks evidence passages using a classic
  probabilistic retrieval framework.  This implementation relies on the
  ``rank_bm25`` library.

The retrieval functions return, for each claim, the identifiers of the top
``k`` evidence passages and the concatenated evidence texts.  These outputs
can be passed directly to downstream classifiers, for example by concatenating
the claim text with its top retrieved evidence.

Note that these retrieval methods are unsupervised and serve as a baseline.
More sophisticated approaches (e.g. neural retrieval) can be plugged in
following the same interface.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import List, Tuple, Sequence, Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    # ``rank_bm25`` is an optional dependency.  Import inside a try/except so
    # that the module can still be imported if the library is unavailable.
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

from nltk.tokenize import word_tokenize

__all__ = [
    "retrieve_top_evidence_tfidf",
    "retrieve_top_evidence_bm25",
]


def _build_tfidf_vectors(claim_texts: Sequence[str], evidence_texts: Sequence[str]) -> Tuple:
    """Fit a TF‑IDF vectoriser on the union of claim and evidence texts.

    To minimise vocabulary mismatch between claim and evidence documents, the
    vectoriser is fitted on the combined corpus.  It returns the matrix
    representations for the claim texts and evidence texts respectively.

    Parameters
    ----------
    claim_texts : sequence of str
        The cleaned claim texts.

    evidence_texts : sequence of str
        The cleaned evidence texts.

    Returns
    -------
    tuple
        A tuple ``(claims_vec, evidence_vec, vectoriser)`` where ``claims_vec``
        and ``evidence_vec`` are TF‑IDF matrices and ``vectoriser`` is the
        fitted ``TfidfVectorizer`` instance.
    """
    vectoriser = TfidfVectorizer(tokenizer=word_tokenize, ngram_range=(1, 3))
    # Fit on combined corpus.
    combined = list(claim_texts) + list(evidence_texts)
    vectoriser.fit(combined)
    claims_vec = vectoriser.transform(claim_texts)
    evidence_vec = vectoriser.transform(evidence_texts)
    return claims_vec, evidence_vec, vectoriser


def retrieve_top_evidence_tfidf(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    k: int = 5,
    output_file: str | None = None,
) -> Tuple[List[List[str]], List[str]]:
    """Retrieve the top ``k`` evidence passages per claim using TF‑IDF.

    This function computes TF‑IDF vectors for both claim and evidence texts
    and ranks the evidence for each claim by cosine similarity.  The IDs of
    the top evidence passages are stored in a dictionary keyed by claim ID.
    Optionally, the result can be serialised to disk as JSON.

    Parameters
    ----------
    claims_df : pandas.DataFrame
        DataFrame containing claim information with columns ``claim_id`` and
        ``cleaned_text``.

    evidence_df : pandas.DataFrame
        DataFrame containing evidence passages with columns ``evidence_id`` and
        ``cleaned_evidence``.

    k : int, default ``5``
        Number of top evidence passages to retrieve per claim.

    output_file : str or None, default ``None``
        If specified, a JSON file will be written mapping claim IDs to lists
        of evidence IDs.

    Returns
    -------
    tuple
        A pair ``(concatenated_texts, top_ids)`` where ``concatenated_texts``
        is a list of strings containing each claim concatenated with its top
        evidence texts, and ``top_ids`` is a list of lists containing the
        corresponding evidence IDs.
    """

    claims_vec, evidence_vec, _ = _build_tfidf_vectors(
        claims_df["cleaned_text"], evidence_df["cleaned_evidence"]
    )

    top_evidence_per_claim: Dict[str, List[str]] = defaultdict(list)
    concatenated_texts: List[str] = []
    top_ids_list: List[List[str]] = []

    for i, claim_row in claims_df.iterrows():
        claim_id = claim_row["claim_id"]
        # Compute cosine similarities between this claim and all evidence.
        cos_sim = cosine_similarity(claims_vec[i], evidence_vec).flatten()
        top_indices = np.argsort(cos_sim)[-k:][::-1]
        ids = evidence_df.iloc[top_indices]["evidence_id"].tolist()
        top_ids_list.append(ids)
        top_evidence_per_claim[claim_id] = ids
        # Concatenate the claim text with the retrieved evidence texts.
        evidence_texts = evidence_df.iloc[top_indices]["cleaned_evidence"]
        concatenated_text = f"{claim_row['cleaned_text']} " + " ".join(evidence_texts)
        concatenated_texts.append(concatenated_text)

    # Persist the dictionary if an output path is provided.
    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(top_evidence_per_claim, f, indent=4)

    return concatenated_texts, top_ids_list


def retrieve_top_evidence_bm25(
    claims_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    k: int = 5,
    output_file: str | None = None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Retrieve top ``k`` evidence passages per claim using BM25.

    Parameters
    ----------
    claims_df : pandas.DataFrame
        DataFrame containing claim information with columns ``claim_id`` and
        ``cleaned_text``.

    evidence_df : pandas.DataFrame
        DataFrame containing evidence passages with columns ``evidence_id`` and
        ``cleaned_evidence``.

    k : int, default ``5``
        Number of top evidence passages to retrieve per claim.

    output_file : str or None, default ``None``
        If specified, a JSON file will be written mapping claim IDs to lists
        of evidence IDs.

    Returns
    -------
    tuple
        A pair ``(concatenated_texts, top_evidence_per_claim)`` where
        ``concatenated_texts`` is a list of strings containing each claim
        concatenated with its top evidence texts, and ``top_evidence_per_claim``
        is a dictionary mapping claim IDs to lists of evidence IDs.
    """
    if BM25Okapi is None:
        raise ImportError(
            "rank_bm25 package is required for BM25 retrieval. Install it via pip."
        )

    tokenised_corpus = [doc.split() for doc in evidence_df["cleaned_evidence"]]
    bm25 = BM25Okapi(tokenised_corpus)

    top_evidence_per_claim: Dict[str, List[str]] = defaultdict(list)
    concatenated_texts: List[str] = []

    for _, claim_row in claims_df.iterrows():
        claim_id = claim_row["claim_id"]
        tokens = claim_row["cleaned_text"].split()
        scores = bm25.get_scores(tokens)
        top_indices = np.argsort(scores)[-k:][::-1]
        ids = evidence_df.iloc[top_indices]["evidence_id"].tolist()
        top_evidence_per_claim[claim_id] = ids
        evidence_texts = evidence_df.iloc[top_indices]["cleaned_evidence"]
        concatenated_texts.append(
            f"{claim_row['cleaned_text']} " + " ".join(evidence_texts)
        )

    if output_file is not None:
        with open(output_file, "w") as f:
            json.dump(top_evidence_per_claim, f, indent=4)

    return concatenated_texts, top_evidence_per_claim
