# HR — FAQ RAG Chatbot

## Project Overview

This project is a **Retrieval-Augmented Generation (RAG)** chatbot for HR SaaS customer support. It answers employee and support questions using an internal FAQ knowledge base instead of relying on the language model’s general knowledge alone. When a user asks a question, the system first **retrieves the most relevant text chunks** from a plain-text FAQ document, then **generates an answer** grounded in that retrieved context. Responses are returned as structured JSON (`user_question`, `system_answer`, `chunks_related`) so answers are transparent and auditable. A rule-based evaluator scores each response, and successful queries are saved automatically for later review.

---

## Features

- **Document chunking** — Sliding-window segmentation of the FAQ source document
- **OpenAI embeddings** — Vector representations for chunks and queries
- **FAISS vector search** — Fast similarity search over stored chunk embeddings
- **Retrieval-Augmented Generation** — Retrieve context first, then generate with an LLM
- **Evaluator agent** — Rule-based quality score (0–10) with justification
- **Historical query persistence** — Timestamped JSON files per successful query
- **Prompt governance** — System prompt restricts answers to provided context
- **Input validation** — Length limits and basic prompt-injection pattern blocking
- **Environment-based configuration** — API keys and models via `.env`

---

## Project Structure

```text
data/
outputs/
src/
  build_index.py
  config.py
  indexing/
  querying/
  utils/
storage/
```

| Path | Purpose |
|------|---------|
| `data/` | Source FAQ plain-text document (`faq_document.txt`) |
| `storage/` | Persisted FAISS index and chunk metadata (`faiss.index`, `chunks.json`) |
| `outputs/` | Sample query outputs and historical run logs |
| `src/indexing/` | Document loading, chunking, embeddings, and vector-store persistence |
| `src/querying/` | Query pipeline, evaluator, and input validation |
| `src/utils/` | Shared utilities (e.g. historical result persistence) |
| `src/build_index.py` | Entry point for the indexing pipeline |
| `src/config.py` | Central constants (chunk size, overlap, `TOP_K`, paths) |

---

## Installation

Use **Python 3.11+** and run all commands from the **repository root**.

### 1. Clone the repository

```bash
git clone https://github.com/pablo-salituri/M2-hr-saas-rag-chatbot.git
cd M2-hr-saas-rag-chatbot
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

**Linux / macOS:**

```bash
source .venv/bin/activate
```

**Windows (PowerShell):**

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

Dependencies are listed in `requirements.txt`. Refer to that file for the exact package versions.

---

## Environment Variables

Copy the template and set your values:

```bash
cp .env.example .env
```

On Windows (PowerShell):

```powershell
Copy-Item .env.example .env
```

Required variables (see `.env.example`):

```env
OPENAI_API_KEY=
EMBEDDING_MODEL=
CHAT_MODEL=
```

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | Authenticates requests to the OpenAI API for embeddings and chat completions |
| `EMBEDDING_MODEL` | Model used to embed FAQ chunks and user questions (default in template: `text-embedding-3-small`) |
| `CHAT_MODEL` | Model used to generate answers from retrieved context (default in template: `gpt-5-mini`) |


---

## Index Creation

Build the vector database from the FAQ document:

```bash
python src/build_index.py
```

This script:

1. **Loads** `data/faq_document.txt` (UTF-8)
2. **Creates chunks** with the configured sliding-window strategy
3. **Generates embeddings** for every chunk via OpenAI
4. **Builds** a FAISS index for similarity search
5. **Persists** artifacts under `storage/` (`faiss.index`, `chunks.json`)


---

## Query Pipeline

Start the interactive FAQ chatbot:

```bash
python src/querying/query.py
```

Type a question and press Enter. Type `exit` or `quit` to leave.

### Retrieval and generation flow

1. **User question** — Input is validated (non-empty, length limit, blocked injection patterns)
2. **Query embedding** — The question is embedded with the same model as the chunks
3. **Vector search** — FAISS returns the top similar chunks (`TOP_K`)
4. **Context construction** — Retrieved chunk texts are formatted into the LLM prompt
5. **Answer generation** — OpenAI chat completions produce a grounded answer
6. **Evaluation** — A rule-based evaluator returns `score` (0–10) and `reason`

### Example question

```text
How many failed login attempts are allowed?
```

---

## JSON Output Format

Each successful query prints JSON like the following (structure matches the live pipeline; chunk text shortened for readability):

```json
{
  "user_question": "How many failed login attempts are allowed?",
  "system_answer": "Five. Reaching five unsuccessful attempts triggers a 30-minute cryptographic lock on the account identifier.",
  "chunks_related": [
    {
      "chunk_id": 6,
      "text": "...",
      "score": 0.505
    }
  ],
  "evaluation": {
    "score": 8,
    "reason": "The answer is supported by multiple relevant chunks and appears to address the user's question."
  }
}
```

- `chunks_related` — Retrieved chunks with `chunk_id`, `text`, and FAISS similarity `score`
- `evaluation` — Present on interactive runs; sample file `outputs/sample_queries.json` shows the three core keys without evaluation for static examples

---

## Chunking Strategy

Chunking uses a **sliding window** over the FAQ text:

- Text is split on **words**
- Each chunk contains up to `CHUNK_SIZE` words
- Consecutive chunks share `CHUNK_OVERLAP` words

**Why overlap?** Shared words between adjacent chunks reduce the risk of cutting an important sentence or fact across a boundary, so related context stays intact for retrieval.

Configuration lives in `src/config.py`:

```python
CHUNK_SIZE = 75
CHUNK_OVERLAP = 15
```

Adjust these values and re-run `python src/build_index.py` to rebuild the index.

---

## Search Strategy

Vector search uses **FAISS** as the vector store:

1. Chunk and query embeddings are converted to NumPy arrays
2. Vectors are **L2-normalized** (`faiss.normalize_L2`)
3. **Cosine similarity** is computed via **inner product** on normalized vectors (`faiss.IndexFlatIP`) — equivalent to cosine similarity for unit vectors
4. **k-NN search** returns the nearest neighbors to the query embedding

The number of results is controlled in `src/config.py`:

```python
TOP_K = 3
```

**Why limit to top results?** Sending only the most similar chunks keeps the LLM context focused, reduces noise from weakly related passages, and stays within practical context limits.

---

## RAG Architecture

**Retrieval-Augmented Generation (RAG)** combines search over a private knowledge base with text generation:

1. **Retrieve** — Find FAQ chunks semantically similar to the user question
2. **Generate** — Pass those chunks as context to the LLM so the answer cites internal documentation

This project uses RAG because:

- **Fewer hallucinations** — The model is instructed to answer only from supplied context; if the FAQ does not contain the answer, it should say so
- **Updatable knowledge** — Replace or extend `data/faq_document.txt`, rebuild the index, and the chatbot reflects new policies without retraining the model
- **Transparency** — `chunks_related` shows which passages supported the answer

The two-step flow is implemented in `run_query_pipeline()` in `src/querying/query.py`: embedding and search happen before `generate_answer()` is called.

---

## Evaluator Agent

After each answer, a **rule-based** evaluator (not an LLM judge) scores quality:

- **Score** — Integer from **0 to 10**
- **Reason** — Textual justification (length varies by outcome)

It considers at least two dimensions:

1. **Relevance** — Whether retrieved chunks align with the question (lexical overlap and retrieval scores)
2. **Completeness** — Whether the answer appears sufficient and grounded in chunk text

Implementation: `src/querying/evaluator.py` (`evaluate_answer()`).

---

## Historical Persistence

Successful queries are automatically saved under:

```text
outputs/historical/
```

Each file is named with a timestamp (e.g. `20260531_132833.json`) and includes `timestamp` plus the full response payload.

**Not persisted:** Questions rejected by input validation (empty input, over length limit, or blocked patterns) — the pipeline returns an error message and does not call `save_query_result()`.

---

## Dependencies

Install from:

```text
requirements.txt
```

Current packages (see file for pinned versions):

- `openai` — Embeddings and chat completions
- `faiss-cpu` — Vector index and similarity search
- `numpy` — Embedding arrays for FAISS
- `python-dotenv` — Load `.env` configuration

---

## Additional Outputs

| File | Description |
|------|-------------|
| `outputs/sample_queries.json` | Three example query–response pairs demonstrating the JSON format |
| `storage/faiss.index` | Serialized FAISS index (generated by `build_index.py`) |
| `storage/chunks.json` | Chunk text and ids aligned with index positions |

---

## Quick Start Summary

```bash
# 1. Setup
python -m venv .venv
source .venv/bin/activate          # or .venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
cp .env.example .env               # then edit OPENAI_API_KEY

# 2. Build index
python src/build_index.py

# 3. Run chatbot
python src/querying/query.py
```
