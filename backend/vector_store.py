
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent

MODEL = SentenceTransformer("all-MiniLM-L6-v2")

INDEX_PATH = BASE_DIR / "vector_index.faiss"
DOCUMENTS_PATH = BASE_DIR / "vector_documents.json"

index = faiss.IndexFlatL2(384)
documents = []


def load_store():
    global index, documents

    if not INDEX_PATH.exists() or not DOCUMENTS_PATH.exists():
        print("No existing vector store found. Starting with an empty index.")
        index = faiss.IndexFlatL2(384)
        documents = []
        return

    try:
        loaded_index = faiss.read_index(str(INDEX_PATH))

        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as file:
            loaded_documents = json.load(file)

        if loaded_index.ntotal != len(loaded_documents):
            raise ValueError(
                "FAISS index and document metadata have different sizes."
            )

        if loaded_index.d != 384:
            raise ValueError(
                f"Unexpected embedding dimension: {loaded_index.d}"
            )

        index = loaded_index
        documents = loaded_documents

        print(f"Loaded {len(documents)} indexed documents.")

    except Exception as error:
        print(f"Could not load the existing vector store: {error}")
        print("Starting with an empty index.")
        index = faiss.IndexFlatL2(384)
        documents = []


def save_store():
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_PATH))

    with open(DOCUMENTS_PATH, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=4,
        )


def add_document(
    text: str,
    metadata: dict,
    embedding_text: str | None = None,
    save: bool = True,
):
    """
    Add a document to the vector store.

    text:
        Actual text returned as retrieval context.

    embedding_text:
        Optional richer text used only for embedding generation.
        For example, it can combine a video title with a transcript chunk.

    save:
        Whether to immediately persist the index and documents.
    """
    text = (text or "").strip()

    if not text:
        return

    text_for_embedding = (embedding_text or text).strip()

    if not text_for_embedding:
        text_for_embedding = text

    embedding = MODEL.encode(
        [text_for_embedding],
        convert_to_numpy=True,
    ).astype("float32")

    if embedding.shape != (1, 384):
        raise ValueError(
            f"Unexpected embedding shape: {embedding.shape}"
        )

    index.add(embedding)

    documents.append({
        "text": text,
        "metadata": metadata,
    })

    if save:
        save_store()


def search(
    query: str,
    top_k: int = 8,
    max_distance: float = 1.4,
):
    """
    Return the nearest semantic matches.

    A larger max_distance permits weaker matches. The caller should
    still assess relevance and filter by collection before answering.
    """
    query = (query or "").strip()

    if not query or index.ntotal == 0 or top_k <= 0:
        return []

    query_embedding = MODEL.encode(
        [query],
        convert_to_numpy=True,
    ).astype("float32")

    top_k = min(top_k, index.ntotal)

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, index_id in zip(distances[0], indices[0]):
        if index_id < 0 or index_id >= len(documents):
            continue

        distance = float(distance)

        if distance > max_distance:
            continue

        document = documents[index_id]

        results.append({
            "text": document["text"],
            "metadata": document["metadata"],
            "distance": distance,
        })

    return results


def get_documents_by_id(reel_id: str):
    """
    Return all indexed chunks belonging to one video.

    The Reel ID is the identity key. Filename is not used because
    different videos may share the same filename.
    """
    results = []

    for document in documents:
        metadata = document.get("metadata", {})

        if metadata.get("id") == reel_id:
            results.append(document)

    results.sort(
        key=lambda document: (
            document.get("metadata", {}).get("start", float("inf")),
            document.get("metadata", {}).get("chunk_number", float("inf")),
        )
    )

    return results


def expand_results(results, window: int = 2):
    """
    Expand each matched chunk with nearby chunks from the same video.

    Neighbors are determined by the chunk's position in the sorted list,
    not by assuming chunk_number is a list index.
    """
    if not results:
        return []

    expanded = {}

    # Cache each video's ordered chunks during expansion.
    reel_cache = {}

    for result in results:
        metadata = result.get("metadata", {})

        reel_id = metadata.get("id")
        target_chunk = metadata.get("chunk_number")

        if reel_id is None:
            continue

        if reel_id not in reel_cache:
            reel_cache[reel_id] = get_documents_by_id(reel_id)

        reel_documents = reel_cache[reel_id]

        if not reel_documents:
            continue

        target_position = None

        # Find the actual matching document, not a guessed list offset.
        for position, document in enumerate(reel_documents):
            document_metadata = document.get("metadata", {})

            if (
                document_metadata.get("chunk_number") == target_chunk
                and document.get("text") == result.get("text")
            ):
                target_position = position
                break

        # Fall back to chunk number only if it identifies a real document.
        if target_position is None:
            for position, document in enumerate(reel_documents):
                if (
                    document.get("metadata", {}).get("chunk_number")
                    == target_chunk
                ):
                    target_position = position
                    break

        if target_position is None:
            continue

        start_position = max(0, target_position - window)
        end_position = min(
            len(reel_documents),
            target_position + window + 1,
        )

        for document in reel_documents[start_position:end_position]:
            document_metadata = document.get("metadata", {})

            key = (
                reel_id,
                document_metadata.get("chunk_number"),
                document.get("text", ""),
            )

            expanded[key] = {
                "text": document["text"],
                "metadata": document_metadata,
            }

    expanded_results = list(expanded.values())

    expanded_results.sort(
        key=lambda result: (
            result["metadata"].get("id", ""),
            result["metadata"].get("start", float("inf")),
            result["metadata"].get("chunk_number", float("inf")),
        )
    )

    return expanded_results


def get_index_size():
    """Return the number of indexed chunks."""
    return index.ntotal


load_store()
