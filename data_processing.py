"""
data_processing.py
====================

This module provides functions for loading and preprocessing the fact–checking
datasets used in this project.  The data consists of JSON files containing
claims, their labels (for the training and development sets), and supporting
evidence passages.  The functions here read these files into pandas
dataframes, perform minimal text cleaning (lower‑casing, removing
non‑alphanumeric characters and stop words, and optional stemming/lemmatizing),
and return objects that are ready to be passed into the retrieval and
classification pipelines.

The default directory structure expects the following files under a ``data``
directory:

* ``train-claims.json`` – a dictionary keyed by claim identifiers with
  ``claim_text``, ``claim_label`` and a list of ``evidences`` identifiers.
* ``dev-claims.json`` – the same format as the training claims but used for
  validation.
* ``test-claims-unlabelled.json`` – contains claim identifiers and text for
  the held‑out test set.
* ``evidence.json`` – a dictionary mapping evidence identifiers to evidence
  passages.

If you wish to use your own datasets, adjust the filenames in
``load_datasets`` accordingly.  See the README for more details on the
expected format.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

# Ensure required NLTK corpora are downloaded at import time.  If these
# downloads hang behind a proxy you can call ``nltk.download`` manually in
# your environment.
for resource in ["stopwords", "wordnet", "punkt"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

__all__ = [
    "DataSets",
    "load_datasets",
    "clean_text",
]


@dataclass
class DataSets:
    """Container for the various datasets used in this project.

    Attributes
    ----------
    claims_train : pandas.DataFrame
        Training claims.  Contains columns ``claim_id``, ``claim_text``,
        ``claim_label``, ``evidences`` (list of evidence IDs) and
        ``cleaned_text`` (preprocessed claim text).

    claims_dev : pandas.DataFrame
        Development claims.  Same columns as ``claims_train`` but used for
        validation.

    claims_test : pandas.DataFrame
        Test claims.  Contains ``claim_id``, ``claim_text`` and
        ``cleaned_text``.  There is no ``claim_label`` column for the test
        claims.

    evidences : pandas.DataFrame
        Evidence passages with columns ``evidence_id``, ``evidence_text`` and
        ``cleaned_evidence``.
    """

    claims_train: pd.DataFrame
    claims_dev: pd.DataFrame
    claims_test: pd.DataFrame
    evidences: pd.DataFrame


def clean_text(text: str) -> str:
    """Clean a piece of text by normalising and removing noise.

    The cleaning pipeline performs the following operations:

    * Lower‑case the text.
    * Unicode normalisation (NFKD).
    * Remove all characters other than lowercase letters, digits and spaces.
    * Tokenise on whitespace.
    * Remove English stop words.
    * Apply stemming and lemmatisation.

    Parameters
    ----------
    text : str
        The raw text to clean.

    Returns
    -------
    str
        A cleaned string ready for vectorisation.
    """
    # Lowercase and Unicode normalisation.
    text = text.lower()
    text = unicodedata.normalize("NFKD", text)

    # Remove punctuation and non‑alphanumeric characters.
  

def _load_claims(file_path: Path, labelled: bool = True) -> pd.DataFrame:
    """Load claim data from a JSON file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file containing the claim data.  The JSON is expected
        to be a dictionary keyed by claim identifiers, where each value is a
        dict with a ``claim_text`` and, if ``labelled`` is True, a
        ``claim_label`` and list of ``evidences`` IDs.

    labelled : bool, default ``True``
        Whether the dataset includes claim labels and evidence identifiers.

    Returns
    -------
    pandas.DataFrame
        The claim data in tabular form with a ``cleaned_text`` column.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    records: List[Dict] = []
    for claim_id, info in data.items():
        record = {
            "claim_id": claim_id,
            "claim_text": info["claim_text"] if labelled else info["claim_text"],
        }
        if labelled:
            record["claim_label"] = info["claim_label"]
            record["evidences"] = info["evidences"]
        records.append(record)

    df = pd.DataFrame(records)
    df["cleaned_text"] = df["claim_text"].apply(clean_text)
    return df


def _load_evidence(file_path: Path) -> pd.DataFrame:
    """Load evidence passages from a JSON file into a DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the JSON file mapping evidence identifiers to strings.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ``evidence_id``, ``evidence_text`` and
        ``cleaned_evidence``.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    records = [
        {"evidence_id": eid, "evidence_text": text} for eid, text in data.items()
    ]
    df = pd.DataFrame(records)
    df["cleaned_evidence"] = df["evidence_text"].apply(clean_text)
    return df


def load_datasets(data_dir: str | Path = "data") -> DataSets:
    """Load the training, development, test and evidence sets from a directory.

    Parameters
    ----------
    data_dir : str or Path, default ``"data"``
        The directory containing the JSON files.  See module docstring for the
        expected file names.

    Returns
    -------
    DataSets
        A dataclass containing the loaded and cleaned datasets.
    """
    data_dir = Path(data_dir)
    claims_train = _load_claims(data_dir / "train-claims.json", labelled=True)
    claims_dev = _load_claims(data_dir / "dev-claims.json", labelled=True)
    claims_test = _load_claims(data_dir / "test-claims-unlabelled.json", labelled=False)
    evidences = _load_evidence(data_dir / "evidence.json")
    return DataSets(claims_train, claims_dev, claims_test, evidences)
  text = re.sub(r"[^a-z0-9\s]", "", text)

    tokens = text.split()

    # Remove stop words.
    stop_words = set(nltk_stopwords.words("english"))
    tokens = [t for t in tokens if t not in stop_words]

    # Apply stemming and lemmatisation.  In practice, lemmatisation alone
    # suffices, but we demonstrate both here as in the original notebook.
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(stemmer.stem(t)) for t in tokens]

    return " ".join(tokens)


