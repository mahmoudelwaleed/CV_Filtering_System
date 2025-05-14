# CV Filtering Chatbot (No Memory RAG)

This is a CV filtering chatbot built with multiple Retrieval-Augmented Generation (RAG) strategies to balance **accuracy**, **cost**, and **latency**. It uses **Qdrant DB** (hosted on cloud) for document storage and retrieval.

The system goes through four evolving stages — from a naive RAG to more advanced, optimized pipelines. Each stage is implemented in a different app file.

---

## 🔧 Indexing Setup

Before running any of the apps, you must index the CVs.

- Upload CVs (PDF format) into `DB_init/cv_pdfs/`
- Run the notebook in `DB_init/` to:
  - Chunk PDFs using **Regex**
  - Embed with **`text-embedding-3-small`** from OpenAI (Azure)
  - Upload vectors with metadata into your **Qdrant Cloud** instance

---

## 📄 `query.py` – RAG Sanity Check

Used to verify that indexing and basic RAG pipeline works.

---

## 🚀 `app.py` – Naive RAG

A simple RAG system:
- Embeds query → retrieves top-k CV chunks → passes to LLM (e.g. `gpt-4o-mini`) → gets answer.

> No query rewriting, no advanced filtering.

---

## 🔄 `app2.py` – Query Rewriting + Fusion

Adds **semantic query rewriting** and **RAG Fusion** for better retrieval.

### Key Features:
- First LLM rewrites query into multiple sub-queries
- Each sub-query retrieves top chunks from Qdrant
- RRF (Reciprocal Rank Fusion) is used to combine results

![app2](app2.png)

---

## 🧠 `app3.py` – Dual-Model Based on Token Length

Adds **dynamic model selection** based on context length.

### Key Features:
- Calculates token count of combined context
- Uses:
  - `gpt-4o-mini` if tokens ≤ 16000
  - `gpt-4o` if tokens > 16000

> Saves cost and latency while maintaining answer quality.

![app3](app3.png)

---

## ✅ `app4.py` – Validation Layer

Adds **early validation** to detect irrelevant queries and skip unnecessary computation.

### Key Features:
- First LLM checks if query is relevant
  - If not → responds directly with "irrelevant"
  - If relevant → continues with app3 pipeline

![app4](app4.png)

---

## 🧪 Models Used

| Task                | Model Used           |
|---------------------|----------------------|
| Embeddings          | `text-embedding-3-small` (Azure) |
| RAG + QA LLM        | `gpt-4o`, `gpt-4o-mini` |
| Query Rewriting     | `gpt-4o-mini`        |

---

## ▶️ How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
Index CVs:

Add your PDFs to DB_init/cv_pdfs/

Run the notebook inside DB_init/ to embed and upload

Run any app:

bash
Copy
Edit
streamlit run app2.py  # or app3.py, app4.py, etc.
📁 Project Structure
graphql
Copy
Edit
.
├── DB_init/
│   ├── cv_pdfs/               # Raw CVs
│   └── indexing_notebook.ipynb
├── query.py                  # Sanity check for RAG
├── app.py                    # Naive RAG
├── app2.py                   # Adds query rewriting + RRF
├── app3.py                   # Adds dynamic model selection
├── app4.py                   # Adds query validation
├── app2.png
├── app3.png
├── app4.png
└── README.md
📌 Notes
All RAG operations are stateless (no memory).

Designed to be modular — each improvement can be tested independently.

Ideal for production scenarios where cost and speed matter.

vbnet
Copy
Edit

Let me know if you'd like:

- A version with image paths adjusted for online deployment (e.g., hosted GitHub URLs),
- A `requirements.txt` file,
- A quick deployment script or Dockerfile.

Would you like that?