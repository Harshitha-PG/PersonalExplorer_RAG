# 🧠 ContextMind

Privacy-First Local AI Memory Assistant powered by RAG.

ContextMind semantically searches across local notes, chats, PDFs, and screenshots using Retrieval-Augmented Generation (RAG), OCR, ChromaDB, and local LLM inference with Ollama.

---

## ✨ Features

* Semantic Search
* OCR-based Screenshot Retrieval
* Local Vector Database (ChromaDB)
* Confidence Scoring
* Hallucination Mitigation
* Source Attribution
* Local LLM Responses using Ollama
* Streamlit Chat Interface

---

## 🛠️ Tech Stack

* Python
* Streamlit
* ChromaDB
* Sentence Transformers
* Ollama
* Mistral
* pytesseract
* LangChain Text Splitters

---

## 📂 Supported Data Types

* TXT Notes
* WhatsApp-style Chats
* PDFs
* Screenshots / Images

---

## 🧠 Architecture

Files → OCR/Text Extraction → Chunking → Embeddings → ChromaDB → Retrieval → Ollama → AI Response

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run embeddings

```bash
python app/embed.py
```

### 3. Launch Streamlit UI

```bash
streamlit run streamlit_app.py
```

---

## 🔒 Privacy

All processing runs locally:

* Local embeddings
* Local vector database
* Local LLM inference
* No cloud APIs required

---

## 📸 Future Improvements

* Real-time indexing
* Voice assistant
* Hybrid search
* Reranking
* Memory graph visualization
* File upload UI

---
