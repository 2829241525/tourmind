from mem0 import MemoryClient

client = MemoryClient(api_key="m0-IaS9rTU5d8K4Wy70HxcXcuVhX3CiIo3DizWP4xpb")
messages = [
    {"role": "user", "content": "Thinking of making a sandwich. What do you recommend?"},
    {"role": "assistant", "content": "How about adding some vegetable for extra flavor?"},
    {"role": "user", "content": "Actually, I don't like vegetable."},
    {"role": "assistant",
        "content": "I'll remember that you don't like vegetable for future recommendations."}
]
client.add(messages, user_id="alex")

# Example showing location and preference-aware recommendations
query = "I'm craving some pizza. Any recommendations?"
filters = {
    "AND": [
        {
            "user_id": "alex"
        }
    ]
}
rsp = client.search(query, version="v2", filters=filters)
print(rsp)

filters = {
    "AND": [
        {
            "user_id": "alex"
        }
    ]
}

all_memories = client.get_all(
    version="v2", filters=filters, page=1, page_size=50)

print(all_memories)
