from sentence_transformers import CrossEncoder


MODEL = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, results: list, top_k: int = 5):
    if not results:
        return []

    pairs = [
        (
            query,
            result.get("embedding_text", result["text"])
        )
        for result in results
    ]

    scores = MODEL.predict(pairs)

    ranked = []

    for result, score in zip(results, scores):
        result = result.copy()
        result["rerank_score"] = float(score)
        ranked.append(result)

    ranked.sort(
        key=lambda result: result["rerank_score"],
        reverse=True
    )

    return ranked[:top_k]