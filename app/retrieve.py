
import chromadb
from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD CHROMADB
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="contextmind")

# -----------------------------
# STRONGER EMBEDDING MODEL
# -----------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER QUERY
# -----------------------------

query = input("\nAsk something: ")

# -----------------------------
# EMBED QUERY
# -----------------------------

query_embedding = model.encode(query).tolist()

# -----------------------------
# DETECT QUERY CATEGORY
# -----------------------------

query_lower = query.lower()

query_filter = None

if any(word in query_lower for word in ["recipe", "food", "pasta", "bread", "coffee"]):

    query_filter = {"type": "recipe"}

elif any(word in query_lower for word in ["chat", "conversation", "discussion"]):

    query_filter = {"type": "chat"}

elif any(word in query_lower for word in ["pdf", "paper", "research"]):

    query_filter = {"type": "pdf"}

# -----------------------------
# QUERY DATABASE
# -----------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where=query_filter,
    include=["documents", "metadatas", "distances"]
)

# -----------------------------
# DISPLAY RESULTS
# -----------------------------

print("\n" + "=" * 80)
print("TOP RETRIEVAL RESULTS")
print("=" * 80)

for i in range(len(results["documents"][0])):

    source = results["metadatas"][0][i]["source"]

    file_type = results["metadatas"][0][i]["type"]

    distance = round(results["distances"][0][i], 4)

    content = results["documents"][0][i][:700]

    print("\n" + "-" * 80)

    print(f"RESULT {i+1}")

    print(f"\nSOURCE FILE:")
    print(source)

    print(f"\nFILE TYPE:")
    print(file_type)

    print(f"\nSIMILARITY DISTANCE:")
    print(distance)

    print(f"\nCONTENT:\n")
    print(content)

    print("\n" + "-" * 80)

print("\nDone.")
