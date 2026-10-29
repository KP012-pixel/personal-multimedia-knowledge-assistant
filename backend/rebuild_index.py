import hashlib
import json
from pathlib import Path

import faiss

import vector_store


DATA_DIR = Path("data")


def create_content_fingerprint(reel):
    """
    Creates a fingerprint for detecting the same Reel stored
    under multiple JSON files / IDs.

    This is only for deduplication during indexing.
    The Reel's actual ID remains its identity.
    """

    content = {
        "caption": reel.get("caption"),
        "transcript": reel.get("transcript"),
        "transcript_segments": reel.get("transcript_segments"),
        "ocr_text": reel.get("ocr_text"),
        "visual_description": reel.get("visual_description"),
        "comments": reel.get("comments"),
    }

    raw = json.dumps(
        content,
        ensure_ascii=False,
        sort_keys=True
    )

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()


def build_contextual_chunks(reel, window_size=3):
    """
    Creates overlapping retrieval chunks.

    Each chunk contains neighboring transcript segments so that
    short Whisper segments do not lose their surrounding context.
    """

    segments = reel.get("transcript_segments") or []

    if not segments:
        return []

    chunks = []

    half_window = window_size // 2

    for i in range(len(segments)):
        start_index = max(0, i - half_window)
        end_index = min(
            len(segments),
            i + half_window + 1
        )

        window = segments[start_index:end_index]

        chunk_text = " ".join(
            segment.get("text", "").strip()
            for segment in window
            if segment.get("text")
        ).strip()

        if not chunk_text:
            continue

        chunks.append({
            "text": chunk_text,
            "start": window[0].get("start", 0),
            "end": window[-1].get("end", 0),
            "chunk_number": i
        })

    return chunks


def main():
    print("Loading weights...")

    # Completely rebuild the FAISS store.
    vector_store.index = faiss.IndexFlatL2(384)
    vector_store.documents = []

    seen_fingerprints = set()
    reel_count = 0

    json_files = sorted(DATA_DIR.glob("*.json"))

    for json_file in json_files:

        with open(json_file, "r", encoding="utf-8") as f:
            reel = json.load(f)

        fingerprint = create_content_fingerprint(reel)

        if fingerprint in seen_fingerprints:
            print(
                f"Skipped duplicate: "
                f"{reel.get('filename', json_file.name)}"
            )
            continue

        seen_fingerprints.add(fingerprint)

        reel_id = reel["id"]
        filename = reel.get("filename", json_file.name)

        chunks = build_contextual_chunks(
            reel,
            window_size=3
        )

        if not chunks:
            print(f"Skipped: {filename} (no transcript segments)")
            continue

        # Use the beginning of the Reel as lightweight document context.
        first_segments = reel.get("transcript_segments", [])[:2]

        video_context = " ".join(
            segment.get("text", "").strip()
            for segment in first_segments
            if segment.get("text")
        ).strip()

        for chunk in chunks:

            metadata = {
                "id": reel_id,
                "filename": filename,
                "source": reel.get("source", "local"),
                "source_url": reel.get("source_url"),
                "start": chunk["start"],
                "end": chunk["end"],
                "chunk_number": chunk["chunk_number"],
            }

            # The actual chunk remains clean.
            #
            # The richer embedding text gives the vector model
            # information about which Reel the chunk belongs to.
            embedding_text = (
                f"Reel context: {video_context}\n"
                f"Content: {chunk['text']}"
            )

            vector_store.add_document(
                text=chunk["text"],
                metadata=metadata,
                embedding_text=embedding_text,
                save=False
            )

        reel_count += 1

        print(
            f"Indexed: {filename} "
            f"({len(chunks)} contextual chunks)"
        )

    vector_store.save_store()

    print()
    print(f"Reels indexed: {reel_count}")
    print(f"Vector count: {vector_store.index.ntotal}")


if __name__ == "__main__":
    main()