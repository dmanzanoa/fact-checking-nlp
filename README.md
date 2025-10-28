# Fact Checking Pipeline

This repository implements a simple fact‑checking pipeline for natural language
processing research.  The goal of the project is to take a set of textual
claims and automatically determine their veracity by (1) retrieving
supporting evidence passages from a large corpus and (2) classifying the
claim–evidence pairs using a lightweight neural network.


## Project structure

| File | Purpose |
|-----|---------|
| `data_processing.py` | Functions for loading the claim/evidence datasets and cleaning text.  The default dataset format expects JSON files with claim identifiers, claim text, (optionally) labels and lists of evidence identifiers, and a separate JSON mapping evidence identifiers to passages. |
| `retrieval.py` | Baseline retrieval methods using TF‑IDF cosine similarity or BM25.  These functions return the top‑`k` evidence passages for each claim and prepare concatenated texts for downstream classification. |
| `model.py` | Defines a simple recurrent neural network (RNN) classifier and helper routines for batching, training and evaluating.  The model consumes the concatenated claim–evidence texts and predicts one of several claim labels. |
| `main.py` | Example script that ties the modules together: it loads the data, performs retrieval, builds a vocabulary, trains the RNN classifier and prints basic training diagnostics. |
| `requirements.txt` | List of Python dependencies. |

## Dataset

The pipeline expects the following JSON files under a `data` directory:

* `train-claims.json` – a dictionary mapping claim IDs to a dictionary with keys `claim_text`, `claim_label` and a list of evidence IDs.
* `dev-claims.json` – the same structure as the training data but used for validation.
* `test-claims-unlabelled.json` – contains claim IDs and texts only.  This set is used for final evaluation.
* `evidence.json` – a dictionary mapping evidence IDs to their textual content.

These files are not included in the repository.  You should prepare your own
data following the same structure.  The `data_processing.load_datasets`
function will load these files and perform minimal cleaning (lower‑casing,
unicode normalisation, removal of stop words, stemming/lemmatisation).

## Installation

Install the project dependencies with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The script relies on `nltk` for tokenisation and lemmatisation.  When you run
the code for the first time, it will attempt to download the required NLTK
resources automatically.  If your environment restricts internet access, you
should download these resources manually using `nltk.download()`.

## Usage

1. Place your JSON datasets in a `data` folder at the root of the repository.
2. Run the example pipeline using the default BM25 retrieval and an RNN
   classifier:

   ```bash
   python main.py --bm25 --epochs 5 --batch-size 8
   ```

   The script will load the data, retrieve the top evidence passages for each
   claim, build a vocabulary over the concatenated texts, train the RNN and
   print training losses.  You can switch to TF‑IDF retrieval by omitting
   the `--bm25` flag.

3. Modify or extend the modules to experiment with other retrieval methods
   (e.g. neural search) or classification models (e.g. Transformers).

## Background

The workflow implemented here reflects common practice in fact‑checking
research: a claim is paired with a small set of candidate evidence passages
retrieved from a large corpus, and a classifier predicts the label based on
the combined text.  While this repository provides a minimalist baseline,
successful systems often incorporate advanced techniques such as pre‑trained
language models and sophisticated retrieval mechanisms.  Good research
projects balance clear presentation with substantive experimentation and
sound methodology.  Typical evaluation criteria for projects like this
include clarity of writing, effectiveness of tables/figures, methodological
soundness and convincing results【533169095099516†L640-L673】.

## Contributing

Contributions are welcome.  If you build additional retrieval or
classification components, please submit a pull request or open an issue to
discuss your ideas.
