import os, glob, re, yaml
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
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_community.llms import Ollama
from langchain.prompts import ChatPromptTemplate

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
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.embedder = OllamaEmbeddings(model=cfg["models"]["embeddings"]["name"])
        llm_cfg = cfg["models"]["llm"]
        self.llm = Ollama(model=llm_cfg["model"], temperature=llm_cfg["temperature"])
        self.index_path = os.path.join(os.getcwd(), "faiss_index")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a clinical information assistant. Provide concise, non-diagnostic, educational information "
             "grounded in provided sources. Always cite sources by number. If unsure, say so. Do not include or infer PHI."),
            ("human",
             "Context:\n{context}\n\nEntities: {entities}\nRelations: {relations}\n\nText:\n{query}\n\n"
             "Give 2–4 key points with bracketed [#] citations and end with a short disclaimer.")
        ])
        self.vs = None
        self.build_or_load_index()

    def build_or_load_index(self):
        faiss_file = os.path.join(self.index_path, "index.faiss")
        os.makedirs(self.index_path, exist_ok=True)
        if os.path.exists(faiss_file):
            self.vs = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
            return
        # Build from local *.md files in this folder
        files = [f for f in glob.glob(os.path.join(os.getcwd(), "*.md")) if os.path.basename(f).lower() != "readme.md"]
        docs = []
        for f in files:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                docs.append(Document(page_content=fh.read(), metadata={"source": os.path.basename(f)}))
        if not docs:
            docs = [Document(page_content="Seed guidance snippet. Add *.md docs to this folder.", metadata={"source": "seed_doc.txt"})]
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.cfg["retrieval"]["chunk_size"],
                                                  chunk_overlap=self.cfg["retrieval"]["chunk_overlap"])
        chunks = splitter.split_documents(docs) if docs else []
        self.vs = FAISS.from_documents(chunks, self.embedder)
        self.vs.save_local(self.index_path)

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        docs = self.vs.similarity_search(query, k=k)
        return [{"text": d.page_content, "doc_id": d.metadata.get("source", "doc") } for d in docs]

    def answer(self, query: str, entities: List[Dict[str, Any]], relations: List[Dict[str, Any]]) -> Dict[str, Any]:
        k = self.cfg["retrieval"]["k"]
        k_final = self.cfg["retrieval"]["k_final"]
        retrieved = self.retrieve(query, k)[:k_final]
        context = "\n\n".join([f"[{i+1}] {r['doc_id']}: {r['text'][:1000]}" for i, r in enumerate(retrieved)])
        msgs = self.prompt.format_messages(context=context, entities=entities, relations=relations, query=query)
        out = self.llm.invoke(msgs)
        return {"answer": out, "sources": retrieved}

# ---------- Utilities ----------
SECTION_PATTERNS = {
    "HPI": re.compile(r"\b(HPI|History of Present Illness)\s*:\s*", re.I),
    "Assessment": re.compile(r"\b(Assessment|Impression)\s*:\s*", re.I),
    "Plan": re.compile(r"\b(Plan)\s*:\s*", re.I),
}

def split_sections(text: str) -> Dict[str, str]:
    sections = {"HPI": "", "Assessment": "", "Plan": ""}
    anchors = []
    for name, pat in SECTION_PATTERNS.items():
        for m in pat.finditer(text):
            anchors.append((m.start(), m.end(), name))
    anchors.sort()
    if not anchors:
        sections["HPI"] = text
        return sections
    for i, (s, e, name) in enumerate(anchors):
        end = anchors[i+1][0] if i+1 < len(anchors) else len(text)
        sections[name] = text[e:end].strip()
    return sections
