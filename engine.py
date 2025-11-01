import os, glob, re, yaml

def _engine_log(msg: str):
    print(f"[ENGINE] {msg}", flush=True)


# --- Section detection (HPI/Assessment/Plan + synonyms) ---
SECTION_PATTERNS = {
    "HPI": re.compile(r"\b(HPI|History of Present Illness|Subjective)\s*:\s*", re.I),
    "Assessment": re.compile(r"\b(Assessment|Impression)\s*:\s*", re.I),
    "Plan": re.compile(r"\b(Plan)\s*:\s*", re.I),
    # Optional extras you can enable later:
    # "Objective": re.compile(r"\b(Objective)\s*:\s*", re.I),
    # "ROS": re.compile(r"\b(Review of Systems|ROS)\s*:\s*", re.I),
}

def split_sections(text: str):
    """Return dict of detected sections -> text slices.
    If no sections detected, returns an empty dict.
    """
    if not text:
        return {}
    hits = []
    for name, pat in SECTION_PATTERNS.items():
        for m in pat.finditer(text):
            hits.append((m.start(), m.end(), name))
    if not hits:
        return {}

    # sort by start index
    hits.sort(key=lambda x: x[0])
    sections = {}
    for i, (s, e, name) in enumerate(hits):
        start = e
        end = hits[i+1][0] if i+1 < len(hits) else len(text)
        chunk = text[start:end].strip()
        # merge if repeated headers occur
        if name in sections and chunk:
            sections[name] += "\n\n" + chunk
        elif chunk:
            sections[name] = chunk
    return sections


from typing import List, Dict, Any, Optional
# De-ID
try:
    from philter import Philter
    PHILTER_OK = True
except Exception:
    PHILTER_OK = False

# spaCy / SciSpaCy
try:
    import spacy
    SPACY_OK = True
except Exception:
    SPACY_OK = False

_SCISPACY_LINKER_OK = False
if SPACY_OK:
    try:
        from scispacy.linking import EntityLinker  # type: ignore
        _SCISPACY_LINKER_OK = True
    except Exception:
        _SCISPACY_LINKER_OK = False

# LangChain / FAISS / Ollama
from langchain_community.vectorstores import FAISS
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
    NEW_OLLAMA = True
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM
    NEW_OLLAMA = False
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate

# ---------- De-ID ----------
class DeIdentifier:
    def __init__(self, block_on_phi: bool = True):
        self.block_on_phi = block_on_phi
        self.philter = Philter() if PHILTER_OK else None
        self.patterns = [
            (re.compile(r"\b(MRN|Medical Record Number)[:\s]*\d[\d\-]*\b", re.I), "[REDACTED_MRN]"),
            (re.compile(r"\b(\d{1,2}/\d{1,2}/\d{2,4})\b"), "[REDACTED_DATE]"),
            (re.compile(r"\b\d{3}[-.\s]?\d{2,3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
            (re.compile(r"\b(\d{1,5}\s+[A-Za-z0-9.\s]+(Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln))\b", re.I), "[REDACTED_ADDRESS]"),
            (re.compile(r"\b(Hospital|Clinic|Medical Center)[:\s]*[A-Za-z0-9 ,.-]+\b", re.I), "[REDACTED_FACILITY]"),
            (re.compile(r"\b([A-Z][a-z]+)\s+([A-Z]\.)?\s?([A-Z][a-z]+)\b"), "[REDACTED_NAME]"),
        ]

    def redact(self, text: str) -> Dict[str, Any]:
        _engine_log("DeIdentifier.redact: starting redaction")
        had_phi = False
        out = text or ""
        if self.philter:
            try:
                out2 = self.philter.redact(out)
                had_phi = had_phi or (out2 != out)
                out = out2
            except Exception:
                pass
        for pat, repl in self.patterns:
            new_out = pat.sub(repl, out)
            had_phi = had_phi or (new_out != out)
            out = new_out
        return {"text": out, "had_phi": had_phi}

# ---------- NER + optional UMLS linking ----------
class ClinicalNER:
    def __init__(self, enable_linking: bool = True, threshold: float = 0.75, k: int = 1, model: str = "en_core_sci_sm"):
        self.enable_linking = enable_linking
        self.threshold = threshold
        self.k = k
        self.model_name = model
        self.nlp = None
        if SPACY_OK:
            try:
                self.nlp = spacy.load(model)
            except Exception:
                self.nlp = spacy.blank("en")
            if self.enable_linking and _SCISPACY_LINKER_OK:
                try:
                    if "scispacy_linker" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                except Exception:
                    pass

    def _heuristic_terms(self):
        return ["aspirin", "nitroglycerin", "troponin", "ecg", "chest pain", "dyspnea", "shortness of breath"]

    def extract(self, text: str) -> List[Dict[str, Any]]:
        _engine_log("ClinicalNER.extract: extracting entities")
        ents: List[Dict[str, Any]] = []
        if self.nlp and self.nlp.has_pipe("ner"):
            doc = self.nlp(text)
            linker = self.nlp.get_pipe("scispacy_linker") if "scispacy_linker" in self.nlp.pipe_names else None
            for e in doc.ents:
                item = {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
                if self.enable_linking and linker is not None and hasattr(e._, "kb_ents"):
                    kb_ents = e._.kb_ents[: self.k] if e._.kb_ents else []
                    if kb_ents:
                        cui, score = kb_ents[0]
                        if score >= self.threshold:
                            kb = linker.kb
                            canonical = kb.cui_to_entity[cui].canonical_name if cui in kb.cui_to_entity else None
                            item.update({"cui": cui, "link_score": float(score), "canonical_name": canonical})
                ents.append(item)
            return ents
        # Fallback
        low = text.lower()
        for term in self._heuristic_terms():
            idx = low.find(term.lower())
            if idx >= 0:
                ents.append({"text": text[idx: idx+len(term)], "label": "MEDICAL_TERM", "start": idx, "end": idx+len(term)})
        return ents

# ---------- Simple relation patterns ----------
class RelationExtractor:
    def __init__(self):
        self.patterns = [
            ("TREATMENT_FOR", re.compile(r"(treated with|started on|administered)\s+([A-Za-z0-9\-]+)", re.I)),
            ("SUSPECTED_DIAGNOSIS", re.compile(r"(suspected|impression|assessment:?)\s+([A-Za-z ,\-]+)", re.I)),
            ("TEST_ORDERED", re.compile(r"(obtain|order)\s+(ECG|troponins?)", re.I)),
        ]

    def extract(self, text: str, ents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rels = []
        for label, pat in self.patterns:
            for m in pat.finditer(text):
                rels.append({"type": label, "match": m.group(0), "span": [m.start(), m.end()]})
        return rels

# ---------- RAG pipeline (FAISS + Ollama) ----------
class RagPipeline:
    _BUILDING = False

    def __init__(self, cfg: Dict[str, Any]):
        _engine_log("RagPipeline.__init__: initializing pipeline")
        self.cfg = cfg
        # Embeddings with robust fallback (Fix #1)
        ollama_embed_model = cfg["models"]["embeddings"]["name"]  
        self.embedder = OllamaEmbeddings(model=ollama_embed_model)

        # LLM (modern adapter or legacy)
        llm_cfg = cfg["models"]["llm"]
        self.llm = OllamaLLM(model=llm_cfg["model"], temperature=llm_cfg.get("temperature", 0.2))

        # Prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical information assistant. Provide concise, non-diagnostic, educational information "
             "grounded in provided sources. Always cite sources by number. If unsure, say so. Do not include or infer PHI."),
            ("human",
             "Context:\n{context}\n\nEntities: {entities}\nRelations: {relations}\n\nText:\n{query}\n\n"
             "Give 2–4 key points with bracketed [#] citations and end with a short disclaimer.")
        ])

        # Absolute paths + single KB dir (Fix #4)
        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.index_path = os.path.join(base_dir, "faiss_index")
        self.kb_dir = base_dir

        self.vs = None
        self.build_or_load_index()

        # Runnable chain (Fix #2 Option B)
        try:
            self.chain = self.prompt | self.llm
        except Exception:
            self.chain = None

    def build_or_load_index(self):
        _engine_log("RagPipeline.build_or_load_index: preparing FAISS index")

        os.makedirs(self.index_path, exist_ok=True)

        faiss_file = os.path.join(self.index_path, "index.faiss")
        marker_file = os.path.join(self.index_path, ".building")

        # Fast path: index already exists
        if os.path.exists(faiss_file):
            try:
                self.vs = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                return
            except Exception:
                pass  # fall through and rebuild

        # Another process might be building
        if os.path.exists(marker_file):
            if os.path.exists(faiss_file):
                try:
                    self.vs = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                    return
                except Exception:
                    pass  # rebuild below

        # In-process guard to avoid re-entrant builds
        if getattr(RagPipeline, "_BUILDING", False):
            if os.path.exists(faiss_file):
                try:
                    self.vs = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                    return
                except Exception:
                    pass
            # Last-resort minimal index to avoid crashing
            self.vs = FAISS.from_texts(["Temporary empty index"], self.embedder)
            return

        # Try to create marker atomically
        created_marker = False
        try:
            fd = os.open(marker_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            created_marker = True
        except FileExistsError:
            pass  # someone else created it

        RagPipeline._BUILDING = True
        try:
            # Build from local *.md files (ignore README)
            files = [f for f in glob.glob(os.path.join(self.kb_dir, "*.md")) if os.path.basename(f).lower() != "readme.md"]
            docs = []
            for fpath in files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        docs.append(Document(page_content=fh.read(), metadata={"source": os.path.basename(fpath)}))
                except Exception:
                    pass
            if not docs:
                docs = [Document(page_content="Seed guidance snippet. Add *.md docs to this folder.", metadata={"source": "seed_doc.txt"})]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.cfg["retrieval"]["chunk_size"],
                chunk_overlap=self.cfg["retrieval"]["chunk_overlap"]
            )
            chunks = splitter.split_documents(docs) if docs else []
            self.vs = FAISS.from_documents(chunks, self.embedder)
            self.vs.save_local(self.index_path)
        finally:
            RagPipeline._BUILDING = False
            if created_marker and os.path.exists(marker_file):
                try:
                    os.remove(marker_file)
                except Exception:
                    pass

    def retrieve(self, query: str, k: int):
        _engine_log("RagPipeline.retrieve: similarity_search")
        docs = self.vs.similarity_search(query, k=k)
        return [{"text": d.page_content, "doc_id": d.metadata.get("source", "doc")} for d in docs]

    def answer(self, query: str, entities, relations):
        _engine_log("RagPipeline.answer: assembling context and invoking LLM")
        k = self.cfg["retrieval"]["k"]
        k_final = max(1, min(self.cfg["retrieval"]["k_final"], 5))  # clamp (Fix #5)
        retrieved = self.retrieve(query, k)[:k_final]

        # Truncate long chunks to reduce context overflow 
        MAX_CHARS = 900
        context = "\\n\\n".join([f"[{i+1}] {r['doc_id']}: {r['text'][:MAX_CHARS]}" for i, r in enumerate(retrieved)])

        # --- Derive mtsamples-style keywords from entities (SYMPTOM/DRUG/CONDITION) ---
        kw_items = []
        for e in entities or []:
            lab = (e.get("label") or "").upper()
            if lab in {"SYMPTOM", "DRUG", "CONDITION"}:
                val = e.get("canonical_name") or e.get("text") or ""
                val = val.strip()
                if val:
                    kw_items.append(val)
        # Deduplicate, title-case like mtsamples style
        keywords = []
        seen = set()
        for k in kw_items:
            kk = k.title()
            if kk not in seen:
                seen.add(kk); keywords.append(kk)
        kw_str = ", ".join(keywords) if keywords else ""
        _engine_log(f"Keywords: {kw_str}")

        payload = {"context": context, "entities": entities, "relations": relations, "query": query}
        try:
            if self.chain is not None:
                out = self.chain.invoke(payload)
            else:
                prompt_str = self.prompt.format(**payload).to_string()
                out = self.llm.invoke(prompt_str)
        except Exception as e:
            raise RuntimeError("LLM call failed. Ensure Ollama is running and meditron:7b is pulled.") from e

        return {"answer": out, "sources": retrieved, "keywords": keywords}