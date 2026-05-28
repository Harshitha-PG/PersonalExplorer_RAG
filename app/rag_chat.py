
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# LOAD CHROMADB
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="contextmind")

# -----------------------------
# EMBEDDING MODEL
# -----------------------------

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER QUERY
# -----------------------------

query = input("\nAsk ContextMind: ")

# -----------------------------
# EMBED QUERY
# -----------------------------

query_embedding = embedding_model.encode(query).tolist()

# -----------------------------
# RETRIEVE DOCUMENTS
# -----------------------------

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    include=["documents", "metadatas", "distances"]
)

# -----------------------------
# BEST DISTANCE
# -----------------------------

best_distance = results["distances"][0][0]

# -----------------------------
# CONFIDENCE LOGIC
# -----------------------------

if best_distance < 0.7:
    confidence = "HIGH"

elif best_distance < 1.1:
    confidence = "MEDIUM"

else:
    confidence = "LOW"

# -----------------------------
# OOD / HALLUCINATION CHECK
# -----------------------------

if confidence == "LOW":

    print("\n" + "=" * 80)
    print("CONTEXTMIND RESPONSE")
    print("=" * 80)

    print("\nI could not find relevant information in your saved data.")

    print("\n" + "=" * 80)
    print("RETRIEVAL CONFIDENCE")
    print("=" * 80)

    print(f"\nConfidence Level: {confidence}")
    print(f"Best Distance Score: {round(best_distance, 4)}")

    exit()

# -----------------------------
# BUILD CONTEXT
# -----------------------------

context = ""

sources = []

for i in range(len(results["documents"][0])):

    source = results["metadatas"][0][i]["source"]

    content = results["documents"][0][i]

    sources.append(source)

    context += f"\nSOURCE: {source}\n"
    context += f"{content}\n"

# Remove duplicate sources
unique_sources = list(set(sources))

# -----------------------------
# PROMPT
# -----------------------------

prompt = f"""
You are ContextMind, a personal AI memory assistant.

Answer ONLY using the retrieved context below.

Rules:
- Do not invent information
- Keep answers concise
- Mention useful details clearly
- If uncertain, say so

Retrieved Context:
{context}

User Question:
{query}

Answer:
"""

# -----------------------------
# GENERATE RESPONSE
# -----------------------------

response = ollama.chat(
    model="mistral",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

# -----------------------------
# PRINT RESPONSE
# -----------------------------

print("\n" + "=" * 80)
print("CONTEXTMIND RESPONSE")
print("=" * 80)

print("\n")
print(response["message"]["content"])

# -----------------------------
# PRINT CONFIDENCE
# -----------------------------

print("\n" + "=" * 80)
print("RETRIEVAL CONFIDENCE")
print("=" * 80)

print(f"\nConfidence Level: {confidence}")

print(f"Best Distance Score: {round(best_distance, 4)}")

# -----------------------------
# PRINT SOURCES
# -----------------------------

print("\n" + "=" * 80)
print("SOURCES")
print("=" * 80)

for src in unique_sources:
    print(f"- {src}")

