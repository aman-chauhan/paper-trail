#!/usr/bin/env python3
"""
PaperTrail - Offline research paper reading companion.

A RAG-based tool for navigating and understanding research papers using
a local LLM (via llama.cpp) and local embeddings (sentence-transformers).
"""

import argparse
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# =============================================================================
# Configuration
# =============================================================================

# LLM server settings (llama-server with OpenAI-compatible API)
LLAMA_API_URL = "http://localhost:8080/v1/chat/completions"
LLAMA_MODEL = "gpt-oss"  # Model name for logging; llama-server typically ignores this

# Embedding model (runs locally via sentence-transformers)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Chunking parameters
DEFAULT_CHUNK_SIZE = 1200  # characters
DEFAULT_CHUNK_OVERLAP = 200  # characters

# RAG parameters
DEFAULT_TOP_K = 6  # number of chunks to retrieve


# =============================================================================
# LLM Client
# =============================================================================

def llm_chat(
    messages: list[dict],
    temperature: float = 0.2,
    max_tokens: int = 4096,
    stream: bool = True
) -> str:
    """
    Send a chat completion request to the local llama-server.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        temperature: Sampling temperature (lower = more deterministic).
        max_tokens: Maximum tokens to generate.
        stream: If True, stream tokens and print them as they are generated.

    Returns:
        The assistant's response content as a string.

    Raises:
        requests.RequestException: If the server request fails.
        KeyError: If the response format is unexpected.
    """
    import json as json_module

    payload = {
        "model": LLAMA_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
    }

    if stream:
        # Streaming request - print tokens as they arrive
        response = requests.post(
            LLAMA_API_URL,
            json=payload,
            timeout=300,
            stream=True
        )
        response.raise_for_status()

        full_content = []
        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue

            data = line[6:]  # Remove "data: " prefix
            if data == "[DONE]":
                break

            try:
                chunk = json_module.loads(data)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    full_content.append(content)
            except json_module.JSONDecodeError:
                continue

        print()  # Final newline after streaming completes
        return "".join(full_content)
    else:
        # Non-streaming request
        response = requests.post(LLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()

        try:
            return response.json()["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise KeyError(
                f"Unexpected response format from LLM server: {response.json()}"
            ) from e


# =============================================================================
# Document Processing
# =============================================================================

def load_markdown(path: str | Path) -> str:
    """Load a Markdown file and return its contents."""
    return Path(path).read_text(encoding="utf-8")


def split_into_chunks(
    text: str,
    max_chars: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into overlapping chunks for embedding.

    Strategy: Split by double newlines (paragraph boundaries), then pack
    paragraphs into chunks up to max_chars. Long paragraphs are hard-split.

    Args:
        text: The input text to chunk.
        max_chars: Maximum characters per chunk.
        overlap: Character overlap between chunks (for long paragraph splits).

    Returns:
        List of text chunks.
    """
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        candidate = (current + "\n\n" + paragraph).strip() if current else paragraph

        if len(candidate) <= max_chars:
            current = candidate
        else:
            # Save current chunk if non-empty
            if current:
                chunks.append(current)
                # Apply overlap for next chunk
                current = current[-overlap:] if overlap > 0 and len(current) > overlap else ""

            # Handle paragraph that exceeds max_chars
            while len(paragraph) > max_chars:
                chunks.append(paragraph[:max_chars])
                # Apply overlap for next chunk
                paragraph = paragraph[max_chars - overlap:] if overlap > 0 and len(paragraph) > overlap else paragraph[max_chars:]

            current = paragraph

    # Don't forget the last chunk
    if current:
        chunks.append(current)

    return chunks


# =============================================================================
# Paper Index (Embeddings + Vector Search)
# =============================================================================

class PaperIndex:
    """
    Vector index for a paper's text chunks.

    Provides semantic search over paper content using sentence embeddings
    and FAISS for efficient similarity search.
    """

    # Lazy-loaded embedding model (shared across instances)
    _embedding_model: SentenceTransformer | None = None

    @classmethod
    def _get_embedding_model(cls) -> SentenceTransformer:
        """Get or initialize the shared embedding model."""
        if cls._embedding_model is None:
            print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}", file=sys.stderr)
            cls._embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._embedding_model

    def __init__(self, chunks: list[str], index: faiss.Index, embeddings: np.ndarray):
        """
        Initialize a PaperIndex.

        Args:
            chunks: List of text chunks from the paper.
            index: FAISS index containing chunk embeddings.
            embeddings: Numpy array of chunk embeddings.
        """
        self.chunks = chunks
        self.index = index
        self.embeddings = embeddings

    @classmethod
    def from_markdown(cls, path: str | Path) -> "PaperIndex":
        """
        Build a PaperIndex from a Markdown file.

        Args:
            path: Path to the Markdown file.

        Returns:
            A PaperIndex instance ready for semantic search.
        """
        print(f"Loading paper from: {path}", file=sys.stderr)
        text = load_markdown(path)

        print("Chunking document...", file=sys.stderr)
        chunks = split_into_chunks(text)
        print(f"Created {len(chunks)} chunks", file=sys.stderr)

        print("Generating embeddings...", file=sys.stderr)
        model = cls._get_embedding_model()
        embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

        # Normalize for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity after normalization
        index.add(embeddings)

        return cls(chunks, index, embeddings)

    def search(self, query: str, k: int = DEFAULT_TOP_K) -> list[tuple[str, float]]:
        """
        Search for chunks most relevant to the query.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            List of (chunk_text, similarity_score) tuples, sorted by relevance.
        """
        model = self._get_embedding_model()
        query_embedding = model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for j, idx in enumerate(indices[0]):
            if idx < len(self.chunks):  # Safety check
                results.append((self.chunks[idx], float(scores[0][j])))

        return results


# =============================================================================
# RAG Query Functions
# =============================================================================

SYSTEM_PROMPT = """You are an expert research assistant helping me critically read one specific scientific paper.
You ONLY know the content of the provided context, which consists of snippets from the same paper.

Guidelines:
- If a question cannot be answered from the context, say so clearly.
- When possible, cite which context snippet (Context 1, 2, etc.) supports your answer.
- Prefer precise, concise explanations suitable for a graduate-level reader.
- Use bullet points and structured formatting where helpful."""


def ask_paper(paper: PaperIndex, query: str, k: int = DEFAULT_TOP_K) -> str:
    """
    Ask a question about the paper using RAG.

    Retrieves relevant chunks and uses them as context for the LLM.

    Args:
        paper: The PaperIndex to query.
        query: The question to answer.
        k: Number of context chunks to retrieve.

    Returns:
        The LLM's response as a string.
    """
    # Retrieve relevant chunks
    results = paper.search(query, k=k)

    # Format context blocks
    context_blocks = []
    for i, (chunk, score) in enumerate(results, start=1):
        context_blocks.append(f"### Context {i} (relevance: {score:.3f})\n{chunk}")
    context_text = "\n\n".join(context_blocks)

    # Build the user prompt
    user_prompt = f"""I have a question about this paper.

**Question:**
{query}

**Paper context:**

{context_text}

Please answer based on the provided context. If relevant, reference specific contexts like (see Context 2)."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    return llm_chat(messages)


# =============================================================================
# Predefined Review Workflows
# =============================================================================

def high_level_summary(paper: PaperIndex) -> str:
    """Generate a high-level summary of the paper."""
    query = (
        "Give a concise high-level summary of this paper. Cover: "
        "1) The problem being addressed, "
        "2) The proposed method or approach, "
        "3) Key results and findings, "
        "4) Significance or implications."
    )
    return ask_paper(paper, query)


def section_outline(paper: PaperIndex) -> str:
    """Generate a section-by-section outline of the paper."""
    query = (
        "Outline this paper section-by-section (e.g., Introduction, Related Work, "
        "Method, Experiments, Discussion, Conclusion). For each section you can identify, "
        "summarize the key points in 2-3 bullet points."
    )
    return ask_paper(paper, query)


def critique_novelty(paper: PaperIndex) -> str:
    """Identify claimed contributions and potential weaknesses."""
    query = (
        "Based on this paper, identify: "
        "1) The claimed contributions and novelty, "
        "2) Potential weaknesses or limitations mentioned, "
        "3) Any threats to validity you notice in their claims or methodology."
    )
    return ask_paper(paper, query)


def verify_claim(paper: PaperIndex, claim: str) -> str:
    """Search for evidence in the paper that may support, refute, or describe a specific claim."""
    query = (
        f"Search the paper for any evidence, discussion, or context that may support, refute, or describe the following claim. "
        f"Summarize what the paper says about it, and note if the claim is not addressed:\n\n\"{claim}\""
    )
    return ask_paper(paper, query)


# =============================================================================
# CLI Interface
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="papertrail",
        description="PaperTrail - Offline research paper reading companion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --paper papers/my_paper.md --mode summary
  %(prog)s --paper papers/my_paper.md --mode outline
  %(prog)s --paper papers/my_paper.md --mode critique
  %(prog)s --paper papers/my_paper.md --ask "What is the main contribution?"
  %(prog)s --paper papers/my_paper.md --claim "Method X outperforms baseline Y"
        """
    )

    parser.add_argument(
        "--paper", "-p",
        required=True,
        help="Path to the paper Markdown file"
    )

    parser.add_argument(
        "--mode", "-m",
        choices=["summary", "outline", "critique"],
        help="Run a predefined review workflow"
    )

    parser.add_argument(
        "--ask", "-a",
        metavar="QUESTION",
        help="Ask a custom question about the paper"
    )

    parser.add_argument(
        "--claim", "-c",
        metavar="CLAIM",
        help="Find evidence for a specific claim"
    )

    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of context chunks to retrieve (default: {DEFAULT_TOP_K})"
    )

    parser.add_argument(
        "--server-url",
        default=LLAMA_API_URL,
        help=f"LLM server URL (default: {LLAMA_API_URL})"
    )

    return parser


def main():
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Update server URL if specified
    global LLAMA_API_URL
    LLAMA_API_URL = args.server_url

    # Validate that at least one action is specified
    if not any([args.mode, args.ask, args.claim]):
        parser.error("Please specify --mode, --ask, or --claim")

    # Check that the paper file exists
    paper_path = Path(args.paper)
    if not paper_path.exists():
        print(f"Error: Paper file not found: {paper_path}", file=sys.stderr)
        sys.exit(1)

    # Build the paper index
    try:
        paper = PaperIndex.from_markdown(paper_path)
    except UnicodeDecodeError as e:
        print(f"Error: Could not decode paper file (encoding issue): {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error: Problem accessing paper file: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error loading paper: {e}", file=sys.stderr)
        sys.exit(1)
    print("", file=sys.stderr)  # Blank line before output

    # Execute the requested action(s)
    # Note: Functions use streaming by default, so output is printed inline
    try:
        if args.mode == "summary":
            print("=== High-Level Summary ===\n")
            high_level_summary(paper)

        elif args.mode == "outline":
            print("=== Section Outline ===\n")
            section_outline(paper)

        elif args.mode == "critique":
            print("=== Novelty & Critique ===\n")
            critique_novelty(paper)

        if args.ask:
            if args.mode:
                print("\n" + "=" * 40 + "\n")
            print(f"=== Question: {args.ask} ===\n")
            ask_paper(paper, args.ask, k=args.top_k)

        if args.claim:
            if args.mode or args.ask:
                print("\n" + "=" * 40 + "\n")
            print(f"=== Claim Evidence: {args.claim} ===\n")
            verify_claim(paper, args.claim)

    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to LLM server.\n"
            f"Make sure llama-server is running at: {LLAMA_API_URL}\n\n"
            "To start the server:\n"
            "  llama-server -hf ggml-org/gpt-oss-20b-GGUF --ctx-size 32768 --jinja",
            file=sys.stderr
        )
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with LLM server: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
