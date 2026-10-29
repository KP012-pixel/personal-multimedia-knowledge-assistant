from vector_store import add_document, search


add_document(
    "I ate biryani, fish, mutton, chicken and prawn tempura at a Bengali family function.",
    {
        "filename": "videoplayback.mp4",
        "source": "local"
    }
)

add_document(
    "The family function had a main course with several different dishes.",
    {
        "filename": "videoplayback.mp4",
        "source": "local"
    }
)


results = search("What food was eaten?", top_k=2)

for result in results:
    print("\nRESULT:")
    print(result)