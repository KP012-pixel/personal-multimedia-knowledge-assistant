
import json
import re
from pathlib import Path
import vector_store

import ollama

from vector_store import search, index, expand_results


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

MODEL_NAME = "llama3:8b"

NO_ANSWER = (
    "The saved content does not provide enough evidence "
    "to answer this confidently."
)

MAX_CONTEXT_CHUNKS = 14
MAX_CONTEXT_CHARS = 16000

# Keep semantic retrieval conservative. This uses the existing
# FAISS index and does not require rebuilding it.
MAX_SEARCH_DISTANCE = 1.4

STOP_WORDS = {
    "what", "did", "does", "do", "she", "he", "they", "i",
    "we", "you", "it", "the", "a", "an", "on", "in", "at",
    "to", "of", "for", "and", "or", "was", "were", "is",
    "are", "be", "been", "being", "her", "his", "their",
    "this", "that", "there", "here", "with", "from", "about",
    "tell", "me", "please", "which", "who", "how", "many",
    "much", "name", "called", "visited", "video", "reel",
    "mention", "mentioned", "talk", "talked", "say", "said",
    "explain", "explained", "show", "shown", "give", "gave",
    "find", "specific", "detail", "details", "content",
}

# These are only lightweight transcript-quality indicators.
# They flag suspicious text; they cannot reliably repair ASR output.
GARBLED_PHRASES = {
    "inaudible",
    "unintelligible",
    "unknown speaker",
    "[unintelligible]",
    "[inaudible]",
}


def load_reel(reel_id: str):
    """Load saved video metadata and timestamped transcript."""
    path = DATA_DIR / f"{reel_id}.json"

    if not path.exists():
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            reel = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        print(f"Could not load saved video {reel_id}: {error}")
        return None

    if reel.get("id") != reel_id:
        return None

    segments = []

    for number, segment in enumerate(
        reel.get("transcript_segments") or []
    ):
        text = str(segment.get("text", "")).strip()

        if not text:
            continue

        try:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
        except (TypeError, ValueError):
            continue

        if start < 0 or end < start:
            continue

        segments.append({
            "segment_id": number,
            "start": start,
            "end": end,
            "text": text,
        })

    transcript = "\n".join(
        f"[{segment['start']:.1f}s-{segment['end']:.1f}s] "
        f"{segment['text']}"
        for segment in segments
    )

    if not transcript:
        transcript = reel.get("transcript") or ""

    return {
        "id": reel_id,
        "filename": reel.get("filename", "Unknown"),
        "source": reel.get("source"),
        "source_url": reel.get("source_url"),
        "caption": reel.get("caption", ""),
        "transcript": transcript,
        "segments": segments,
        "document": reel,
    }


def format_timestamp(seconds):
    try:
        seconds = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        seconds = 0

    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def get_chunk_key(result):
    metadata = result.get("metadata", {})

    return (
        metadata.get("id"),
        metadata.get("chunk_number"),
        result.get("text", ""),
    )


def get_query_terms(text):
    return {
        word
        for word in re.findall(r"\w+", str(text).lower())
        if len(word) > 2 and word not in STOP_WORDS
    }


def get_video_title(metadata):
    return str(
        metadata.get("filename")
        or metadata.get("title")
        or ""
    )


def looks_like_garbled_transcript(text):
    """
    Identify obvious transcript-quality problems.

    This deliberately uses conservative checks. A grammatical-looking
    but incorrect transcription cannot be reliably detected by rules
    alone, so the answer model also receives explicit uncertainty
    instructions.
    """
    normalized = re.sub(r"\s+", " ", str(text).lower()).strip()

    if not normalized:
        return True

    if any(phrase in normalized for phrase in GARBLED_PHRASES):
        return True

    words = re.findall(r"\b[\w']+\b", normalized)

    if len(words) >= 8:
        unique_ratio = len(set(words)) / len(words)

        # Flag extreme repetition, not ordinary repeated terminology.
        if unique_ratio < 0.30:
            return True

    # Long strings with almost no recognizable word structure are
    # suspicious. Do not reject short technical terms on this basis.
    letters = re.sub(r"[^a-z]", "", normalized)

    if len(letters) >= 20:
        vowel_ratio = sum(
            character in "aeiou"
            for character in letters
        ) / len(letters)

        if vowel_ratio < 0.15:
            return True

    return False


def expand_search_queries(query):
    """Generate alternate retrieval queries, not answers."""
    prompt = f"""
Generate up to three alternative search queries for searching a
personal collection of video transcripts.

Original question:
{query}

Rules:
- Preserve the same subject and intended meaning.
- Use synonyms or alternative descriptions.
- Do not switch to a different topic.
- Do not answer the question.
- Return one short query per line.
- Do not add explanations or numbering.
"""

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )

        raw = response["message"]["content"].strip()
        original = query.strip()
        queries = []
        seen = {original.lower()}

        for line in raw.splitlines():
            candidate = re.sub(
                r"^\s*[-*\d.)]+\s*",
                "",
                line,
            ).strip()

            if not candidate:
                continue

            if len(candidate) > 200:
                continue

            if candidate.lower() in {
                "none",
                "no alternatives",
                original.lower(),
            }:
                continue

            key = candidate.lower()

            if key not in seen:
                seen.add(key)
                queries.append(candidate)

            if len(queries) >= 3:
                break

        return [original] + queries

    except Exception as error:
        print(f"Query expansion failed: {error}")
        return [query]


def retrieve_candidates(query, playlist_id, top_k):
    """
    Retrieve candidates using the original question as the primary
    relevance signal. Alternate queries can help discovery but do not
    automatically make an otherwise weak match relevant.
    """
    if index.ntotal == 0:
        return []

    original_terms = get_query_terms(query)
    query_variants = expand_search_queries(query)

    # Retrieve original-query results separately so we can distinguish
    # direct matches from results found only by alternate queries.
    candidates = {}

    try:
        original_results = search(
            query,
            top_k=min(index.ntotal, max(top_k * 5, 40)),
            max_distance=MAX_SEARCH_DISTANCE,
        )
    except Exception as error:
        print(f"Original-query search failed: {error}")
        original_results = []

    for result in original_results:
        metadata = result.get("metadata", {})

        if metadata.get("playlist_id", "default") != playlist_id:
            continue

        key = get_chunk_key(result)

        try:
            distance = float(result.get("distance", float("inf")))
        except (TypeError, ValueError):
            continue

        candidates[key] = {
            **result,
            "distance": distance,
            "original_distance": distance,
            "matched_queries": [query],
            "direct_match": True,
        }

    # Alternate queries may find paraphrases, but are not allowed to
    # bypass the relevance checks below.
    for variant in query_variants[1:]:
        try:
            results = search(
                variant,
                top_k=min(index.ntotal, max(top_k * 3, 24)),
                max_distance=MAX_SEARCH_DISTANCE,
            )
        except Exception as error:
            print(f"Alternate search failed: {error}")
            continue

        for result in results:
            metadata = result.get("metadata", {})

            if metadata.get("playlist_id", "default") != playlist_id:
                continue

            key = get_chunk_key(result)

            try:
                distance = float(result.get("distance", float("inf")))
            except (TypeError, ValueError):
                continue

            if key not in candidates:
                candidates[key] = {
                    **result,
                    "distance": distance,
                    "original_distance": float("inf"),
                    "matched_queries": [variant],
                    "direct_match": False,
                }
            else:
                candidate = candidates[key]

                candidate["distance"] = min(
                    candidate["distance"],
                    distance,
                )

                if variant not in candidate["matched_queries"]:
                    candidate["matched_queries"].append(variant)

    ranked = []

    for candidate in candidates.values():
        metadata = candidate.get("metadata", {})
        text = str(candidate.get("text", ""))
        title = get_video_title(metadata)

        text_terms = get_query_terms(text)
        title_terms = get_query_terms(title)

        overlap = len(original_terms & text_terms)
        title_overlap = len(original_terms & title_terms)

        # Original-query semantic similarity carries more weight than
        # alternate-query matches.
        direct = candidate["direct_match"]
        distance = candidate["original_distance"]

        if direct:
            semantic_score = -distance
        else:
            semantic_score = -candidate["distance"] - 0.35

        # Penalize chunks that match only through an alternate query
        # and have no lexical connection to the original question.
        if not direct and overlap == 0 and title_overlap == 0:
            semantic_score -= 0.35

        candidate["rank_score"] = (
            semantic_score
            + 0.12 * overlap
            + 0.18 * title_overlap
            + 0.03 * min(
                len(candidate["matched_queries"]) - 1,
                2,
            )
        )

        candidate["lexical_overlap"] = overlap
        candidate["title_overlap"] = title_overlap

        # A result discovered only by an alternate query must have
        # at least some meaningful lexical connection to the original
        # question. This avoids many topic-drift results.
        if not direct and overlap == 0 and title_overlap == 0:
            continue

        if looks_like_garbled_transcript(text):
            candidate["transcript_quality_warning"] = True
        else:
            candidate["transcript_quality_warning"] = False

        ranked.append(candidate)

    ranked.sort(
        key=lambda item: (
            item["rank_score"],
            item["direct_match"],
        ),
        reverse=True,
    )

    return ranked[:top_k]


def expand_with_context(results, playlist_id, max_chunks):
    """
    Add nearby chunks only around strong direct matches.

    Neighboring chunks provide context but are not treated as independent
    evidence of relevance. Obvious transcript-quality problems are
    excluded from the final answer context.
    """
    if not results:
        return []

    # Avoid expanding weak matches or matches found only through an
    # alternate query.
    expansion_seeds = [
        result for result in results
        if result.get("direct_match")
        and not result.get("transcript_quality_warning", False)
    ]

    try:
        expanded = expand_results(expansion_seeds, window=1)
    except Exception as error:
        print(f"Context expansion failed: {error}")
        expanded = []

    combined = {}

    for result in results:
        metadata = result.get("metadata", {})

        if metadata.get("playlist_id", "default") != playlist_id:
            continue

        if result.get("transcript_quality_warning", False):
            continue

        key = get_chunk_key(result)

        combined[key] = {
            **result,
            "direct_match": result.get("direct_match", False),
        }

    for result in expanded:
        metadata = result.get("metadata", {})

        if metadata.get("playlist_id", "default") != playlist_id:
            continue

        text = str(result.get("text", "")).strip()

        if not text or looks_like_garbled_transcript(text):
            continue

        key = get_chunk_key(result)

        if key not in combined:
            combined[key] = {
                **result,
                "distance": float("inf"),
                "original_distance": float("inf"),
                "rank_score": -999,
                "direct_match": False,
                "transcript_quality_warning": False,
            }

    ordered = sorted(
        combined.values(),
        key=lambda item: (
            not item.get("direct_match", False),
            -item.get("rank_score", -999),
            item.get("metadata", {}).get("start", float("inf")),
        ),
    )

    selected = []
    per_video = {}

    for result in ordered:
        metadata = result.get("metadata", {})
        reel_id = metadata.get("id")

        if not reel_id:
            continue

        if per_video.get(reel_id, 0) >= 4:
            continue

        selected.append(result)
        per_video[reel_id] = per_video.get(reel_id, 0) + 1

        if len(selected) >= max_chunks:
            break

    return selected


def build_context(chunks):
    """Build evidence context and matching source records."""
    context_parts = []
    sources = []
    total_chars = 0

    for result in chunks:
        metadata = result.get("metadata", {})
        reel_id = metadata.get("id")

        if not reel_id:
            continue

        reel = load_reel(reel_id)

        filename = (
            metadata.get("filename")
            or (reel["filename"] if reel else "Unknown video")
        )

        source_url = (
            metadata.get("source_url")
            or (reel["source_url"] if reel else None)
        )

        source_type = (
            metadata.get("source")
            or (reel["source"] if reel else None)
        )

        start = metadata.get("start")
        end = metadata.get("end")

        text = str(result.get("text", "")).strip()

        if not text or looks_like_garbled_transcript(text):
            continue

        if start is None:
            start = 0

        if end is None:
            end = start

        remaining = MAX_CONTEXT_CHARS - total_chars

        if remaining <= 0:
            break

        if len(text) > remaining:
            if remaining < 100:
                break

            text = text[:remaining]

        citation_id = len(sources) + 1

        quality_note = (
            "Transcript quality warning: possible speech-recognition "
            "error. Do not infer an instruction from unclear wording."
            if result.get("transcript_quality_warning")
            else "Transcript text; wording may still contain recognition errors."
        )

        source_record = {
            "citation_id": citation_id,
            "id": reel_id,
            "filename": filename,
            "source": source_type,
            "source_url": source_url,
            "start": start,
            "end": end,
            "start_timestamp": format_timestamp(start),
            "end_timestamp": format_timestamp(end),
            "text": text,
        }

        sources.append(source_record)

        context_parts.append(
            f"[{citation_id}]\n"
            f"Video: {filename}\n"
            f"Source type: {source_type or 'unknown'}\n"
            f"URL: {source_url or 'Unavailable'}\n"
            f"Timestamp: {format_timestamp(start)} - "
            f"{format_timestamp(end)}\n"
            f"Evidence type: transcript excerpt\n"
            f"Quality note: {quality_note}\n"
            f"Evidence:\n{text}"
        )

        total_chars += len(text)

    return "\n\n---\n\n".join(context_parts), sources


def generate_answer(query, context):
    """Generate a grounded answer with inline source citations."""
    prompt = f"""
You are a personal multimedia knowledge assistant. Answer questions
using the supplied saved-video evidence.

QUESTION:
{query}

RETRIEVED EVIDENCE:
{context}

Rules:
1. Answer the exact question. Do not mix unrelated topics.
2. Use only claims supported by the supplied excerpts.
3. Treat excerpts as speech-to-text transcripts, not verified facts.
4. Do not silently repair garbled or ambiguous transcription.
5. If a sentence is unclear, say that its meaning cannot be determined
   from the transcript. Do not turn it into an instruction.
6. Distinguish explicit recommendations from general discussion.
7. For questions asking what exercises, methods, steps, or advice were
   mentioned, list only items that the evidence clearly identifies.
8. Do not present general movement or incidental discussion as a
   specific exercise unless the excerpt supports that interpretation.
9. If the evidence does not answer the question, say exactly:
   "{NO_ANSWER}"
10. Cite every factual item with the specific supporting citation,
    for example [1] or [2].
11. Use only citation IDs shown in the evidence.
12. Do not use citation ranges such as [1-4]. Cite IDs individually.
13. Do not cite a video merely because it was retrieved.
14. Keep the answer concise and use bullets when listing multiple items.
15. Do not provide medical recommendations beyond what the videos say.
16. Do not output JSON.
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )

    return response["message"]["content"].strip()


def clean_citations(answer, sources):
    """
    Normalize citation ranges and remove IDs absent from the supplied
    source list. Ranges are expanded only for IDs that actually exist.
    """
    valid_ids = {
        int(source["citation_id"])
        for source in sources
    }

    def replace_citation(match):
        first = int(match.group(1))
        last_text = match.group(2)

        if last_text is None:
            return f"[{first}]" if first in valid_ids else ""

        last = int(last_text)

        if last < first or last - first > 30:
            return ""

        ids = [
            number for number in range(first, last + 1)
            if number in valid_ids
        ]

        return " ".join(f"[{number}]" for number in ids)

    answer = re.sub(
        r"\[(\d+)(?:\s*[-–]\s*(\d+))?\]",
        replace_citation,
        answer,
    )

    answer = re.sub(r"[ \t]{2,}", " ", answer)
    answer = re.sub(r" *\n *", "\n", answer)

    return answer.strip()


def retain_cited_sources(answer, sources):
    """
    Keep only cited sources and renumber them consistently.

    This verifies citation IDs, not the semantic truth of each claim.
    """
    cited_ids = list(dict.fromkeys(
        int(number)
        for number in re.findall(r"\[(\d+)\]", answer)
    ))

    source_by_id = {
        int(source["citation_id"]): source
        for source in sources
    }

    cited_ids = [
        citation_id for citation_id in cited_ids
        if citation_id in source_by_id
    ]

    if not cited_ids:
        return answer, []

    renumber_map = {
        old_id: new_id
        for new_id, old_id in enumerate(cited_ids, start=1)
    }

    def replace_citation(match):
        old_id = int(match.group(1))

        if old_id not in renumber_map:
            return ""

        return f"[{renumber_map[old_id]}]"

    answer = re.sub(
        r"\[(\d+)\]",
        replace_citation,
        answer,
    )

    retained_sources = []

    for old_id in cited_ids:
        source = dict(source_by_id[old_id])
        source["citation_id"] = renumber_map[old_id]
        retained_sources.append(source)

    return answer.strip(), retained_sources



def answer_topics_from_titles(playlist_id):
    """Answer broad collection-topic questions using saved video titles."""
    videos_by_id = {}

    for document in vector_store.documents:
        metadata = document.get("metadata", {})
        video_id = metadata.get("id")

        if not video_id:
            continue

        if metadata.get("playlist_id", "default") != playlist_id:
            continue

        reel = load_reel(video_id)

        title = (
            metadata.get("filename")
            or metadata.get("title")
            or (reel.get("filename") if reel else None)
            or "Untitled video"
        )

        if video_id not in videos_by_id:
            videos_by_id[video_id] = {
                "id": video_id,
                "filename": title,
                "source": (
                    metadata.get("source")
                    or (reel.get("source") if reel else None)
                ),
                "source_url": (
                    metadata.get("source_url")
                    or (reel.get("source_url") if reel else None)
                ),
            }

    if not videos_by_id:
        return None

    sources = []
    answer_lines = []

    for citation_id, video in enumerate(
        videos_by_id.values(), start=1
    ):
        source = {
            "citation_id": citation_id,
            "id": video["id"],
            "filename": video["filename"],
            "source": video["source"],
            "source_url": video["source_url"],
            "start": 0,
            "end": 0,
            "start_timestamp": "",
            "end_timestamp": "",
            "text": f"Saved video title: {video['filename']}",
        }

        sources.append(source)
        answer_lines.append(
            f"- {video['filename']} [{citation_id}]"
        )

    answer = (
        "Based on the saved video titles, your collection covers "
        "these topics:\n\n"
        + "\n".join(answer_lines)
        + "\n\nThese topics are inferred from video titles; "
          "the titles alone do not verify the videos' full contents."
    )

    return {
        "answer": answer,
        "sources": sources,
    }




def answer_question(
    query: str,
    top_k: int = 20,
    playlist_id: str = "default",
):
    query = (query or "").strip()

    if not query:
        return {
            "query": query,
            "answer": "Enter a question about your saved content.",
            "claims": [],
            "sources": [],
        }

    # Handle broad collection-overview questions using video titles.
    # These questions should not depend on random transcript excerpts.
    broad_topic_question = bool(
        re.search(
            r"\b(topics?|subjects?|themes?)\b",
            query.lower(),
        )
        and re.search(
            r"\b(videos?|collection|saved content)\b",
            query.lower(),
        )
    )

    if broad_topic_question:
        overview = answer_topics_from_titles(playlist_id)

        if overview:
            sources = overview["sources"]

            return {
                "query": query,
                "answer": overview["answer"],
                "claims": [{
                    "text": overview["answer"],
                    "citations": [
                        source["citation_id"]
                        for source in sources
                    ],
                }],
                "sources": sources,
            }

    # Normal transcript-based question answering continues below.
    results = retrieve_candidates(
        query=query,
        playlist_id=playlist_id,
        top_k=max(top_k, 12),
    )

    if not results:
        return {
            "query": query,
            "answer": NO_ANSWER,
            "claims": [],
            "sources": [],
        }

    context_chunks = expand_with_context(
        results,
        playlist_id=playlist_id,
        max_chunks=MAX_CONTEXT_CHUNKS,
    )

    if not context_chunks:
        return {
            "query": query,
            "answer": NO_ANSWER,
            "claims": [],
            "sources": [],
        }

    context, sources = build_context(context_chunks)

    if not context or not sources:
        return {
            "query": query,
            "answer": NO_ANSWER,
            "claims": [],
            "sources": [],
        }

    try:
        answer = generate_answer(query, context)
    except Exception as error:
        print(f"Answer generation failed: {error}")

        return {
            "query": query,
            "answer": (
                "I found saved content, but could not generate an answer. "
                f"Check that Ollama is running and {MODEL_NAME!r} "
                "is available."
            ),
            "claims": [],
            "sources": [],
        }

    if not answer:
        answer = NO_ANSWER

    if (
        answer.rstrip(".!?").strip().lower()
        == query.rstrip(".!?").strip().lower()
    ):
        answer = NO_ANSWER

    if NO_ANSWER.lower() in answer.lower():
        return {
            "query": query,
            "answer": NO_ANSWER,
            "claims": [],
            "sources": [],
        }

    answer = clean_citations(answer, sources)

    if not re.search(r"\[\d+\]", answer):
        print("\n[DEBUG] Answer contained no valid citation pattern.")
        print("[DEBUG] Query:", query)
        print("[DEBUG] Retrieved sources:", len(sources))
        print("[DEBUG] Context preview:", context[:3000])
        print("[DEBUG] Raw model answer:", repr(answer))

        return {
            "query": query,
            "answer": (
                "I found potentially relevant saved content, but the "
                "answer did not include a verifiable source citation. "
                "Check the backend terminal for diagnostic details."
            ),
            "claims": [],
            "sources": [],
        }

    answer, cited_sources = retain_cited_sources(answer, sources)

    if not cited_sources:
        return {
            "query": query,
            "answer": (
                "I could not match the answer to verifiable saved sources."
            ),
            "claims": [],
            "sources": [],
        }

    cited_ids = [
        source["citation_id"]
        for source in cited_sources
    ]

    return {
        "query": query,
        "answer": answer,
        "claims": [{
            "text": answer,
            "citations": cited_ids,
        }],
        "sources": cited_sources,
    }
