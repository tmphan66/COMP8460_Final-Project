# COMP8460 Final Project — Drug AI Assistant (RAG + Pandas Agent)

An interactive command‑line assistant that answers questions about **drug reviews** using:
- A **RAG pipeline** over a local ChromaDB vector store built from `drugsComTest_raw.csv`.
- A **Pandas analysis agent** for counts, averages, and simple stats from the dataset.
- A local **Ollama** LLM (`gemma3:4b`) and **HuggingFace** sentence embeddings (`all-MiniLM-L6-v2`).

> ⚙️ First run automatically builds a persistent ChromaDB index in `./chroma_db` from the CSV, then reuses it on subsequent runs.

---

## ✨ Features
- **Two-tool agent** with ReAct reasoning:
  - `chroma_search` — retrieves semantically relevant user reviews with rich metadata (drug name, condition, rating).
  - `pandas_analysis` — answers analytic questions (e.g., averages, counts, “top N”).
- **Local-first** setup: no external API keys needed once Ollama + model are installed.
- **Reproducible**: `requirements.txt` pins major LangChain components and ChromaDB version.

---

## 📦 Repository layout
```
.
├─ main.py                 # CLI app: builds/loads ChromaDB, sets up tools & agent, runs loop
├─ requirements.txt        # Python dependencies
├─ drugsComTest_raw.csv    # Dataset of user reviews (drugName, condition, rating, review, usefulCount)
└─ chroma_db/              # Auto-generated on first run (vector index; do not commit large contents)
```

---

## 🚀 Quickstart

### 0) Clone the repo
```bash
gh repo clone tmphan66/COMP8460_Final-Project
cd COMP8460_Final-Project
```

(Alternatively, you can use `git clone` if you prefer.)

---

## 📦 Setup & Installation

From inside the project directory:

### 1. Create and Activate a Virtual Environment

**Windows (PowerShell)**

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
```

**macOS / Linux (bash)**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Make sure that:

- `drugsComTest_raw.csv` is present in the project root.
- Ollama is running and the required model (e.g. `gemma3:4b`) is available.

---

## 🚀 Running the Streamlit App

From the project root **with the virtual environment activated**, run:

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH (common on Windows), you can also do:

```bash
py -m streamlit run app.py
```

Streamlit will print a local URL such as:

```text
Local URL: http://localhost:8501
```

Open that URL in your browser to use the web app.

---

## 🧑‍💻 Using the App

### Image-Based Questions

1. Upload a photo of a medicine box or blister pack.
2. Type a question, for example:
   - `What is the drug in this image?`
   - `What are the common side effects of this medicine?`
3. The agent will:
   - Call `process_image` to read the text.
   - Either:
     - Answer directly (for identification), or
     - Use `rag_search` and other tools to summarise reviews and side-effects.

### Agent + Tools
- **`chroma_search(query)`**: Use for *semantic, qualitative* questions (e.g., “What side effects do people report for X?”). Returns raw review snippets for summarization.
- **`pandas_analysis(query)`**: Use for *quantitative* questions (e.g., “average rating for Y”, “how many reviews for Z?”). Executes a DF-aware agent with safe code execution enabled.

---

## 💡 Example prompts

Qualitative (RAG):
- “What are people saying about the side effects of taking amoxicillin?”
- “Summarize experiences for patients with **anxiety** taking **buspirone** (rating ≥ 8).”

Quantitative (Pandas):
- “What is the **average rating** for **ibuprofen**?”
- “Top 5 most common **conditions** in the dataset.”

> Tip: If your question is about **numbers, averages, counts, or top lists**, prefer `pandas_analysis`. Otherwise, ask for experiences or side effects and let the agent use `chroma_search`.

---

## 🛠 Configuration

You can change these constants in `main.py`:
```python
CSV_PATH = "drugsComTest_raw.csv"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# LLM: ChatOllama(model="gemma3:4b")
```

If you move the dataset or want a new index location, update `CSV_PATH` and `DB_PATH` accordingly.

---

## 🧪 Troubleshooting

- **Slow first run**
  - The first run may be slow while:
    - Loading the LLM.
    - Building the Chroma index from the CSV.
- **Ollama connection/model errors**
  - Ensure `ollama serve` is running.
  - Ensure the model (e.g. `gemma3:4b`) is downloaded.
- **Chroma issues**
  - Delete `chroma_db/` if the index becomes corrupted and re-run the app.
- **Import errors**
  - Double-check `requirements.txt`.
  - Recreate the virtual environment if needed.

---

## 🙏 Acknowledgements

- [Ollama](https://ollama.com/) for local LLM hosting  
- [LangChain](https://python.langchain.com/) for agents and tools  
- [ChromaDB](https://www.trychroma.com/) for the vector database  
- [Hugging Face](https://huggingface.co/) for embeddings  
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for OCR  

