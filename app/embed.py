
import chromadb
import re
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ingest import documents

# -----------------------------
# CHROMADB SETUP
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

# Delete old collection
try:
    client.delete_collection("contextmind")
except:
    pass

collection = client.get_or_create_collection(name="contextmind")

# -----------------------------
# STRONGER EMBEDDING MODEL
# -----------------------------

model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# CHUNKING
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=80
)

# -----------------------------
# CLEAN OCR / TEXT NOISE
# -----------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[^a-zA-Z0-9.,!?/:;()\-\n ]", "", text)

    text = text.strip()

    return text

# -----------------------------
# DETECT FILE TYPE
# -----------------------------

def detect_type(filename):

    filename = filename.lower()

    if "recipe" in filename or "food" in filename:
        return "recipe"

    elif "chat" in filename or "text" in filename:
        return "chat"

    elif filename.endswith(".pdf"):
        return "pdf"

    elif filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
        return "screenshot"

    elif filename.endswith(".txt"):
        return "note"

    else:
        return "other"

# -----------------------------
# CREATE CHUNKS
# -----------------------------

all_chunks = []

for doc in documents:

    cleaned = clean_text(doc["content"])

    chunks = splitter.split_text(cleaned)

    for chunk in chunks:

        if len(chunk.strip()) < 40:
            continue

        all_chunks.append({
            "content": chunk,
            "source": doc["source"],
            "type": detect_type(doc["source"])
        })

print(f"\nTotal chunks created: {len(all_chunks)}")

# -----------------------------
# STORE EMBEDDINGS
# -----------------------------

for i, chunk in enumerate(all_chunks):

    embedding = model.encode(chunk["content"]).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[chunk["content"]],
        metadatas=[{
            "source": chunk["source"],
            "type": chunk["type"]
        }]
    )

print("\nEmbeddings stored successfully!")
