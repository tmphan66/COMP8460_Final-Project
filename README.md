# COMP8460 Final Project — Drug AI Assistant (RAG + Pandas Agent + OCR)

An interactive command‑line assistant that answers questions about **drug reviews** using:
- A **RAG pipeline** over a local ChromaDB vector store built from `drugsComTest_raw.csv`.
- A **Pandas analysis agent** for counts, averages, and simple stats from the dataset.
- A **Multimodal OCR tool** to extract drug names from image file paths.
- A local **Ollama** LLM (`gemma3:4b`) and **HuggingFace** sentence embeddings (`all-MiniLM-L6-v2`).

> ⚙️ First run automatically builds a persistent ChromaDB index in `./chroma_db` from the CSV, then reuses it on subsequent runs.

---

## ✨ Features
- **Three-tool agent** with ReAct reasoning:
  - `chroma_search` — retrieves semantically relevant user reviews with rich metadata (drug name, condition, rating).
  - `pandas_analysis` — answers analytic questions (e.g., averages, counts, “top N”).
  - `drug_name_from_image` - extracts text (like a drug name) from a local image file path using **EasyOCR**, enabling image-to-text-to-data queries.
- **Local-first** setup: no external API keys needed once Ollama + model are installed.
- **Reproducible**: `requirements.txt` pins major LangChain components and ChromaDB version, and OCR libraries.

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
# Using GitHub CLI
gh repo clone tmphan66/COMP8460_Final-Project
cd COMP8460_Final-Project
```

### 1) Install system prerequisites
- **Python 3.10+**
- **Ollama** (https://ollama.com/download)

Then pull the local LLM used by this project:
```bash
ollama pull gemma3:4b
```

### 2) Create a virtual environment and install deps
**Windows (PowerShell):**
```powershell
py -m venv .venv
. .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

**macOS/Linux (bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3) Make sure the dataset is present
Confirm `drugsComTest_raw.csv` exists at the project root. If you have a different path, update `CSV_PATH` in `main.py`.

### 4) Run the assistant
```bash
python main.py
```
On first run, you’ll see logs for:
- Loading the LLM and embedding model
- Building the ChromaDB index (only once, persisted at `./chroma_db`)

Type your questions at the prompt. Type `exit` to quit.

---

## 🧠 How it works

### Models
- **LLM**: `gemma3:4b` via Ollama
- **Embeddings**: `all-MiniLM-L6-v2` via `langchain-huggingface`

### RAG (ChromaDB)
- The CSV is cleaned and converted into `Document` objects with metadata (`drugName`, `condition`, `rating`, `usefulCount`).
- Documents are embedded and stored in a local **Chroma** vector DB in `./chroma_db`.
- A **SelfQueryRetriever** lets the agent translate natural-language filters into structured metadata queries (e.g., filter by condition or rating).

### Agent + Tools
- **`drug_name_from_image(image_path)`**: **NEW TOOL.** Must be used first when the query includes a file path (e.g., `images/Tylenol.jpg`). It uses EasyOCR to extract text, which the agent then uses to inform the next tool call (e.g., `chroma_search`).
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

New Multimodal Example:
- “Tell me about user experiences and reviews for the drug whose name is in the picture located at **images/my_pill_label.jpg**.” 
  - *This will trigger the agent to use `drug_name_from_image` first, then `chroma_search`.*

> Tip: If your question is about **numbers, av
erages, counts, or top lists**, prefer `pandas_analysis`. Otherwise, ask for experiences or side effects and let the agent use `chroma_search`.

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

- **Ollama model not found**: Run `ollama pull gemma3:4b` and ensure the Ollama server is running.
- **Import errors / version mismatches**: Reinstall from `requirements.txt` inside a fresh venv.
- **Index rebuild**: Delete the `chroma_db/` directory to force a rebuild on next run.
- **Empty/whitespace query**: The agent deliberately returns “Please enter a query” without calling tools.
- **CUDA**: The default embedding device is CPU. If you have a GPU, change `model_kwargs={"device": "cuda"}` in `main.py` for embeddings.

---

## 🔍 Dataset schema (expected columns)
- `drugName` (str)
- `condition` (str)
- `rating` (int)
- `review` (str)
- `usefulCount` (int)

If the CSV file uses different column names, adjust the DataFrame cleaning logic in `main.py` accordingly.

---

## 📜 License
Add a license of your choice (e.g., MIT) at the repo root as `LICENSE`. If you’re unsure, you can start with MIT and update later.

---

## 🙏 Acknowledgements
- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Ollama](https://ollama.com/)
- [Hugging Face](https://huggingface.co/)
