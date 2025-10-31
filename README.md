# Meditron App

Privacy‑preserving clinical text analysis with **Streamlit + FAISS + Ollama (meditron:7b)** in one folder.

## Quickstart
```bash
ollama pull meditron:7b
ollama pull nomic-embed-text

python -m venv .venv && source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

streamlit run app.py
```
The app auto-builds a FAISS index from the `*.md` guidance files in this folder the first time it runs.

## Files
- `app.py` – Streamlit UI (Reset, Sources toggle, PHI badge, Section selector, UMLS toggle)
- `engine.py` – De-ID, NER (+optional UMLS link), simple relations, RAG (FAISS + Ollama)
- `config.yaml` – basic knobs
- `*.md` – small guidance snippets for RAG

## Notes
- UMLS linking via SciSpaCy’s EntityLinker is optional and large; keep toggle off if you haven’t installed resources.
- This is **educational only**, not medical advice.
