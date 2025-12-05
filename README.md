# PaperTrail

***Your offline margin notes, supercharged.***

PaperTrail is an **on-device, offline** research paper reading companion built with **llama.cpp**, **GPT-OSS**, and Python. It helps you **summarize sections, navigate long papers, and map claims to evidence** with traceable context snippets — so you can form your *own* understanding faster.

This project is designed to support **personal study and literature review workflows** while keeping your data local.

## Why this exists

Reading papers efficiently is hard:

* Dense methods
* Long experimental sections
* Buried assumptions and limitations
* Time-consuming claim verification

PaperTrail aims to reduce the *search and organization overhead* of reading without replacing human judgment.

## What it does

* **Ingests Markdown** versions of papers (PDF ingestion can be added later; Markdown-first keeps things robust.)
* **Chunks + indexes** the paper locally.
* **Retrieves relevant passages** for a question.
* Uses a **local LLM** to generate answers **grounded in retrieved context**.
* Supports quick workflows:
  * high-level summary
  * section-wise outline
  * novelty + limitations scan
  * claim-to-evidence lookup
  * “where do they justify X?”

## What it does *not* do

* ❌ Automatically write peer reviews to submit as your own.
* ❌ Replace your independent assessment.

## Responsible Use

PaperTrail is a **reading and note-building aid**.

If you’re a reviewer:

* Treat outputs as **reading scaffolding**.
* Verify key points in the paper.
* Use the tool to locate evidence and summarize structure - not to outsource judgment.

**Suggested practice:**
Copy any output you want to keep into your own notes and add your personal evaluation and reasoning.

## Architecture (simple)

1. **Local LLM** via `llama.cpp` server (OpenAI-compatible endpoint)
2. **Ingestion**: Markdown → chunking
3. **Embeddings**: local sentence-transformer
4. **Index**: FAISS
5. **RAG prompts** that cite internal “Context 1/2/3…” blocks

## Getting started

### 1) Run a local model server

Install `llama.cpp` and start the server:

```bash
# macOS
brew install llama.cpp

# Example server invocation (adjust model/quant/ctx as needed)
llama-server \
  -hf ggml-org/gpt-oss-20b-GGUF \
  --ctx-size 32768 \
  --jinja
```

This should expose an endpoint like:

* `http://localhost:8080/v1/chat/completions`

### 2) Install Python dependencies

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```txt
requests
sentence-transformers
faiss-cpu
numpy
```

### 3) Add a paper

Place a Markdown file in:

```bash
papers/
  my_paper.md
```

### 4) Run the CLI

```bash
python papertrail.py --paper papers/my_paper.md --mode summary
python papertrail.py --paper papers/my_paper.md --mode outline
python papertrail.py --paper papers/my_paper.md --ask "What is the main contribution?"
python papertrail.py --paper papers/my_paper.md --ask "Where do they justify the dataset choice?"
```

## Example usage

**High-level summary**

```bash
python papertrail.py --paper papers/my_paper.md --mode summary
```

**Critique prompts**

```bash
python papertrail.py --paper papers/my_paper.md --mode critique
```

**Claim-to-evidence**

```bash
python papertrail.py --paper papers/my_paper.md \
  --claim "The proposed method outperforms prior SOTA on X under Y constraints."
```

## Roadmap

* [ ] PDF → Markdown helper script
* [ ] Table extraction to Markdown
* [ ] Multi-paper library + comparison mode
* [ ] Minimal Streamlit UI
* [ ] Citation-aware navigation (jump to references)

## FAQ

**Is this “cheating”?**
PaperTrail is meant for **comprehension and navigation**. It doesn’t replace your responsibility to evaluate novelty, correctness, or impact. The tool emphasizes **traceable, context-grounded answers** to support human reading.

**Can I use it for confidential manuscripts?**
Yes — the primary goal is to keep everything **offline and local**.

## License

**Apache-2.0**

## Acknowledgments

* Built on **llama.cpp** for local inference.
* Uses **GPT-OSS** models and local embedding tools.
