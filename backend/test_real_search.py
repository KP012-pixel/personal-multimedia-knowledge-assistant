from vector_store import search


query = "What meat dishes did she eat?"

results = search(query, top_k=3)

for result in results:
    print("\n--- RESULT ---")
    print("Distance:", result["distance"])
    print("File:", result["metadata"]["filename"])
    print("Time:", result["metadata"]["start"], "to", result["metadata"]["end"], "seconds")
    print("Chunk:", result["metadata"]["chunk_number"])
    print(result["text"])