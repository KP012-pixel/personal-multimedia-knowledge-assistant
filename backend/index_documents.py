import json
from pathlib import Path

from vector_store import add_document


DATA_DIR = Path("data")


def build_reel_text(reel):
    parts = []

    if reel.get("transcript"):
        parts.append(
            "AUDIO TRANSCRIPT:\n" + reel["transcript"]
        )

    if reel.get("caption"):
        parts.append(
            "CAPTION:\n" + reel["caption"]
        )

    if reel.get("ocr_text"):
        ocr_text = "\n".join(
            item["text"] for item in reel["ocr_text"]
        )

        parts.append(
            "ON-SCREEN TEXT:\n" + ocr_text
        )

    return "\n\n".join(parts)


for file_path in DATA_DIR.glob("*.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        reel = json.load(f)

    text = build_reel_text(reel)

    if not text.strip():
        continue

    add_document(
        text,
        {
            "id": reel["id"],
            "filename": reel["filename"],
            "source": reel["source"],
            "source_url": reel.get("source_url")
        }
    )

    print(f"Indexed: {reel['filename']}")

print(f"\nIndexed {len(list(DATA_DIR.glob('*.json')))} JSON document(s).")