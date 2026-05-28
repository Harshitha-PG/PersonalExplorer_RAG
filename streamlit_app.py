
import streamlit as st
import chromadb
import ollama
from sentence_transformers import SentenceTransformer

# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="ContextMind",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------

st.title("🧠 ContextMind")
st.caption("Privacy-First Local AI Memory Assistant")

# -----------------------------
# LOAD CHROMADB
# -----------------------------

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection(name="contextmind")

# -----------------------------
# LOAD EMBEDDING MODEL
# -----------------------------

embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")

# -----------------------------
# USER INPUT
# -----------------------------

query = st.chat_input("Ask ContextMind something...")

# -----------------------------
# PROCESS QUERY
# -----------------------------

if query:

    # Show user message
    with st.chat_message("user"):
        st.markdown(query)

    # Embed query
    query_embedding = embedding_model.encode(query).tolist()

    # Retrieve results
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    # Best retrieval score
    best_distance = results["distances"][0][0]

    # Confidence logic
    if best_distance < 0.7:
        confidence = "HIGH"

    elif best_distance < 1.1:
        confidence = "MEDIUM"

    else:
        confidence = "LOW"

    # OOD detection
    if confidence == "LOW":

        with st.chat_message("assistant"):

            st.error("I could not find relevant information in your saved data.")

            st.markdown(f"### Confidence: {confidence}")

            st.markdown(f"Distance Score: {round(best_distance, 4)}")

    else:

        # Build context
        context = ""

        sources = []

        for i in range(len(results["documents"][0])):

            source = results["metadatas"][0][i]["source"]

            content = results["documents"][0][i]

            sources.append(source)

            context += f"\nSOURCE: {source}\n"
            context += f"{content}\n"

        unique_sources = list(set(sources))

        # Prompt
        prompt = f"""
        You are ContextMind, a personal AI memory assistant.

        Answer ONLY using the retrieved context below.

        Rules:
        - Do not invent information
        - Keep answers concise
        - Mention useful details clearly

        Retrieved Context:
        {context}

        User Question:
        {query}

        Answer:
        """

        # Generate response
        response = ollama.chat(
            model="mistral",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response["message"]["content"]

        # Show assistant response
        with st.chat_message("assistant"):

            st.markdown(answer)

            st.markdown("---")

            st.markdown(f"### Confidence: {confidence}")

            st.markdown(f"Distance Score: {round(best_distance, 4)}")

            st.markdown("### Sources")

            for src in unique_sources:
                st.markdown(f"- {src}")

            # Expandable retrieved chunks
            with st.expander("View Retrieved Chunks"):

                for i in range(len(results["documents"][0])):

                    st.markdown("---")

                    st.markdown(
                        f"### {results['metadatas'][0][i]['source']}"
                    )

                    st.write(results["documents"][0][i][:800])

