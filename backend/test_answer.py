from answer import answer_question


query = "Did she eat pizza?"

result = answer_question(query)

print("\nQUESTION:")
print(result["query"])

print("\nANSWER:")
print(result["answer"])

print("\nSOURCES:")

for source in result["sources"]:
    print(
        f'{source["filename"]} '
        f'({source["start"]}s - {source["end"]}s)'
    )