# COMP8460 Final Project — Drug AI Assistant (Streamlit + RAG + Tools)

An interactive **Streamlit web app** and **CLI assistant** that answers questions about **drug reviews** using:

- A **RAG (Retrieval-Augmented Generation) pipeline** over a local ChromaDB vector store built from `drugsComTest_raw.csv`.
- A set of **tool-calling agents** for:
  - OCR on uploaded medicine images (EasyOCR)
  - Searching reviews and side effects (RAG over ChromaDB)
  - Basic analytics like average rating, review counts, and Top-5 lists (Pandas)
- A local **Ollama** LLM (e.g. `gemma3:4b`) and **HuggingFace** sentence embeddings (e.g. `all-MiniLM-L6-v2`).

You can use:

- `app.py` → **Streamlit UI** (main way to demo the project)  
- `main.py` → **Backend / CLI assistant** for debugging and experimentation  

---

## ✨ Key Components

### `app.py` — Streamlit Web App

`app.py` provides the browser-based interface for the assistant. It:

- Sets up the **Streamlit layout**:
  - Sidebar for configuration and instructions
  - Main area for:
    - Image upload
    - Text input for user questions
    - Display of the agent’s final answer (and optionally tool traces / debug logs)
- Uses `st.cache_*` (or equivalent) to **cache heavy objects** so they only load once per session:
  - LLM (via Ollama)
  - Embedding model
  - ChromaDB client
  - Pandas DataFrame with drug reviews
  - EasyOCR reader
- Creates an instance of the **agent executor** defined in `main.py` and calls it whenever the user submits a question.
- Handles:
  - **Image questions**: pass the uploaded image + user query into the agent context.
  - **Text-only questions**: pass just the query.
- Renders the **final answer** clearly (and can optionally show intermediate thoughts / tool calls for debugging).

In short: `app.py` = **UI layer** (Streamlit) + **glue** to call the agent from `main.py`.

---

### `main.py` — Backend, Tools, and Agent

`main.py` contains the **core logic** of the system:

1. **Initialisation**
   - Loads the **Ollama LLM** (e.g. `gemma3:4b`).
   - Loads the **HuggingFaceEmbeddings** model (e.g. `all-MiniLM-L6-v2`).
   - Loads `drugsComTest_raw.csv` into a **Pandas DataFrame**.
   - Creates or connects to a **ChromaDB** collection for semantic search over reviews.
   - Initialises an **EasyOCR** reader for image text extraction.

2. **Tool Definitions**

   Tools are defined with `@tool` decorators and used by the ReAct-style agent:

   - `process_image(query_about_image: str) -> str`  
     - Extracts text from an uploaded image using OCR.  
     - Called **at most once** per user question involving an image.  
     - Returns the raw recognized text (e.g. drug name, strength, etc.).

   - `rag_search(query: str) -> str`  
     - Searches the ChromaDB vector store of reviews.  
     - Use for qualitative questions:
       - side effects
       - experiences
       - “drugs for a specific condition”  
     - Returns a short summary plus snippets and ratings.

   - `get_average_rating(drug_name: str) -> str`  
     - Uses the Pandas DataFrame to compute the average rating for a given drug.  

   - `get_review_count(drug_name: str) -> str`  
     - Returns the total number of reviews for a **single** drug.  

   - `get_top_most(query: str) -> str`  
     - Returns the Top-5 most reviewed drugs or drugs for a condition, with counts.

3. **Agent Prompt & Behaviour**

   `main.py` also defines a **ReAct-style prompt** that instructs the model to:

   - Think in steps: `Thought → Action → Action Input → Observation → ... → Final Answer`.
   - Follow strict **image rules**:
     - If (and only if) there is an image:
       - Call `process_image` **once** to read the text.
       - For “What is the drug in the image?” → answer directly from OCR.
       - For image + reviews/side-effects → use `rag_search` and/or other tools afterwards.
   - Follow strict **RAG rules**:
     - Use `rag_search` only for reviews, experiences, or drugs for a condition.
     - Avoid using it for simple identification or pure numeric questions.
   - **Avoid hallucinations**:
     - Only use drug names from the user’s input or tool Observations.
     - If information is missing, respond that it is not available rather than inventing it.
   - **Handle PHI / redacted tags** safely:
     - Never try to guess `[REDACTED_NAME]`, `[REDACTED_EMAIL]`, etc.
     - Focus only on the medical content of the question.

4. **Optional CLI Loop**

   `main.py` can also be run directly to start a simple **command-line interface**:

   - Prompts you to type a query.
   - Calls the same agent and tools.
   - Prints the final answer (and often some debug logs).

---

## 📂 Project Structure

A typical layout:

```text
.
├─ app.py                 # Streamlit web application (UI and glue)
├─ main.py                # Backend logic, tools, RAG, OCR, and CLI
├─ drugsComTest_raw.csv   # Dataset of drug reviews (drugName, condition, rating, review, etc.)
├─ chroma_db/             # ChromaDB collection (auto-created on first run)
├─ requirements.txt       # Python dependencies
└─ README.md              # This file
```

> `chroma_db/` is created automatically on first run and reused afterwards.  
> Deleting it will force a rebuild of the vector index.

---

## 🛠 Prerequisites

- **Python 3.10+**
- **Ollama** installed and running:
  - https://ollama.com/
- An Ollama model compatible with the code (e.g. `gemma3:4b`) pulled locally:
  ```bash
  ollama pull gemma3:4b
  ```
- (Optional but recommended) a Python **virtual environment**.

---

## 🧬 Getting the Code

Using GitHub CLI (your setup):

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

### Text-Only Questions

You can also ask questions without an image, such as:

- `What are people saying about the side effects of Valtrex?`
- `How many reviews are there for Panadol?`
- `What are the top 5 drugs for treating depression according to the reviews?`

The agent will pick the right tools:

- `rag_search` for qualitative questions.
- `get_average_rating` for numeric rating summaries.
- `get_review_count` for counts.
- `get_top_most` for Top-5 lists.

---

## 🖥 Running the CLI Assistant (Optional)

To use the agent from the command line instead of the browser, run:

```bash
python main.py
```

You’ll be able to type questions into the terminal, and the agent will answer using the same tools as the Streamlit app.

---

## 🧠 High-Level Architecture

1. **Data Layer**
   - `drugsComTest_raw.csv` → Pandas DataFrame.
   - Each review row is converted into a document with text + metadata.

2. **Vector Store**
   - Documents are embedded with a HuggingFace model (e.g. `all-MiniLM-L6-v2`).
   - Stored in ChromaDB for fast similarity search.

3. **LLM & Tools**
   - LLM (via Ollama) drives reasoning and decision-making.
   - Tools wrap OCR, RAG retrieval, and Pandas-based analytics.

4. **Agent**
   - ReAct-style agent orchestrates:
     - When to call each tool,
     - How to combine tool outputs,
     - How to produce a safe, grounded final answer.

5. **Interface**
   - `app.py` → Streamlit front-end.
   - `main.py` → Backend and optional CLI.

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

