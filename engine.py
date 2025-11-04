import os, glob, re, yaml, json
from typing import List, Dict, Any, Optional

def _engine_log(msg: str):
    print(f"[ENGINE] {msg}", flush=True)

# ---------- De-ID (Privacy) ----------
# (This class is unchanged)
try:
    from philter import Philter
    PHILTER_OK = True
except Exception:
    PHILTER_OK = False

class DeIdentifier:
    def __init__(self, block_on_phi: bool = True):
        self.block_on_phi = block_on_phi
        self.philter = Philter() if PHILTER_OK else None
        self.patterns = [
            (re.compile(r"\b(MRN|Medical Record Number|SSN|Social Security Number)[:\s]*[\w\-]+\b", re.I), "[REDACTED_ID]"),
            (re.compile(r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"), "[REDACTED_DATE]"),
            (re.compile(r"\b\d{3}[-.\s]?\d{2,3}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
            (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
            (re.compile(r"\b(\d{1,5}\s+[A-Za-z0-9.\s]+(Street|St|Ave|Avenue|Rd|Road|Blvd|Boulevard|Lane|Ln|Drive|Dr))\b", re.I), "[REDACTED_ADDRESS]"),
            (re.compile(r"\b(Hospital|Clinic|Medical Center|University of|Community Health)[\w\s,.-]+\b", re.I), "[REDACTED_FACILITY]"),
            (re.compile(r"\b(Dr\.|Mr\.|Mrs\.|Ms\.)\s+([A-Z][a-z]+)\s+([A-Z][a-z]+)\b"), "[REDACTED_NAME]"),
        ]

    def redact(self, text: str) -> Dict[str, Any]:
        _engine_log("DeIdentifier.redact: starting redaction")
        had_phi = False
        out = text or ""
        if self.philter:
            try:
                philter_out = self.philter.redact(out)
                if philter_out != out:
                    had_phi = True
                out = philter_out
            except Exception as e:
                _engine_log(f"Philter call failed: {e}")
        
        for pat, repl in self.patterns:
            new_out = pat.sub(repl, out)
            if new_out != out:
                had_phi = True
            out = new_out
        
        return {"text": out, "had_phi": had_phi}

# ---------- NER (Entity Extraction) ----------
try:
    import spacy
    SPACY_OK = True
except Exception:
    SPACY_OK = False

_SCISPACY_LINKER_OK = False
if SPACY_OK:
    try:
        from scispacy.linking import EntityLinker  
        _SCISPACY_LINKER_OK = True
    except Exception:
        _SCISPACY_LINKER_OK = False

class ClinicalNER:
    def __init__(self, enable_linking: bool = True, threshold: float = 0.75, k: int = 1, model: str = "en_core_sci_sm"):
        self.enable_linking = enable_linking
        self.threshold = threshold
        self.k = k
        self.model_name = model
        self.nlp = None
        if SPACY_OK:
            spacy.require_cpu()
            try:
                self.nlp = spacy.load(model)
                _engine_log(f"Loaded spaCy model '{model}'")
            except Exception:
                _engine_log(f"Failed to load spaCy model '{model}', using blank 'en'")
                _engine_log("Try: pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v1.0.0/en_core_sci_sm-1.0.0.tar.gz")
                self.nlp = spacy.blank("en")
                if "ner" not in self.nlp.pipe_names:
                    try:
                        self.nlp.add_pipe("ner", source=spacy.load("en_core_web_sm").get_pipe("ner"))
                    except Exception:
                        _engine_log("Could not add NER pipe to blank model.")

            if self.enable_linking and _SCISPACY_LINKER_OK:
                try:
                    if "scispacy_linker" not in self.nlp.pipe_names:
                        self.nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
                        _engine_log("Enabled scispaCy UMLS linker")
                except Exception as e:
                    _engine_log(f"Failed to add scispacy_linker: {e}")
            else:
                _engine_log("UMLS linking is DISABLED.") 

    def extract(self, text: str) -> List[Dict[str, Any]]:
        _engine_log("ClinicalNER.extract: extracting entities")
        ents: List[Dict[str, Any]] = []
        if not self.nlp or not self.nlp.has_pipe("ner"):
            _engine_log("ClinicalNER.extract: spaCy model not available or 'ner' pipe missing. Returning no entities.")
            return ents

        doc = self.nlp(text)
        linker = self.nlp.get_pipe("scispacy_linker") if "scispacy_linker" in self.nlp.pipe_names else None
        
        for e in doc.ents:
            item = {"text": e.text, "label": e.label_, "start": e.start_char, "end": e.end_char}
            if self.enable_linking and linker is not None and hasattr(e._, "kb_ents"):
                kb_ents = e._.kb_ents[: self.k] if e._.kb_ents else []
                if kb_ents:
                    cui, score = kb_ents[0]
                    if score >= self.threshold:
                        try:
                            kb = linker.kb
                            canonical = kb.cui_to_entity[cui].canonical_name if cui in kb.cui_to_entity else None
                            item.update({"cui": cui, "link_score": float(score), "canonical_name": canonical})
                        except Exception as e:
                            _engine_log(f"Error getting UMLS entity info: {e}")
            ents.append(item)
        
        entity_texts = [entity["text"] for entity in ents]
        _engine_log(f"Named Entities Recognized: {entity_texts}")
        return ents

# ---------- RAG Pipeline (FAISS + Ollama) ----------
from langchain_community.vectorstores import FAISS
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama as OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

class RagPipeline:
    _BUILDING = False

    def __init__(self, cfg: Dict[str, Any], ner: ClinicalNER):
        _engine_log("RagPipeline.__init__: initializing pipeline")
        self.cfg = cfg
        self.ner = ner 
        
        embed_cfg = cfg["models"]["embeddings"]
        self.embedder = OllamaEmbeddings(model=embed_cfg["name"])

        llm_cfg = cfg["models"]["llm"]
        self.llm = OllamaLLM(
            model=llm_cfg["model"], 
            temperature=llm_cfg.get("temperature", 0.1)
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an expert Clinical Text Analyst. Your job is to provide insightful, detailed answers to the user's 'Question' by analyzing the provided 'Clinical Text (Patient Note)' and 'Knowledge Base Context'.\n\n"

             "**Example Question:** What do you recommend?\n\n"

             "**Example Knowledge Base Context:**\n"
             "Osteoarthritis (OA) is a common joint disorder. Management includes pain relief (NSAIDs), physical therapy, and joint replacement surgery for severe cases.\n\n"
             "**Example Clinical Text (Patient Note):**\n"
             "ASSESSMENT: 65 y/o female with severe right knee pain. X-ray confirms advanced degenerative arthritis.\n"
             "PLAN: Discuss Total Knee Arthroplasty (TKA).\n\n"

             "**Example Extracted Entities (for reference):**\n"
             "[{{\"text\": \"severe right knee pain\", \"label\": \"SYMPTOM\"}}, {{\"text\": \"degenerative arthritis\", \"label\": \"DIAGNOSIS\"}}, {{\"text\": \"Total Knee Arthroplasty\", \"label\": \"TREATMENT\"}}]\n\n"
             
             "**Example AI Answer (This is the quality I expect):**\n"
             "Thought:\n"
             "1.  **Analyze Question:** The user is asking for *my* recommendations, not just a summary of the plan.\n"
             "2.  **Check for Recommendation Request:** This is a clear recommendation request.\n"
             "3.  **Formulate Answer:**\n"
             "    * `Patient Diagnosis:` The 'Clinical Text' (ASSESSMENT) identifies \"advanced degenerative arthritis\".\n"
             "    * `Patient Plan:` The 'Clinical Text' (PLAN) suggests \"Total Knee Arthroplasty (TKA)\".\n"
             "    * `Knowledge Base Context:` The 'Knowledge Base Context' mentions that management for OA (arthritis) includes \"pain relief (NSAIDs), physical therapy, and joint replacement surgery\".\n"
             "    * `Synthesize Insight:` The patient is already on the path for surgery, which aligns with the KB. My recommendation should provide this context. I will explain *why* surgery is the next step by contrasting it with the general non-surgical options mentioned in the KB. This provides more value than just repeating \"Discuss TKA\".\n"
             "4.  **Synthesize Final Answer:** (Drafting the detailed answer...)\n\n"
             "Based on the patient's diagnosis of \"advanced degenerative arthritis,\" general medical guidelines often recommend a multi-step approach.\n\n"
             "According to the knowledge base, management for osteoarthritis typically begins with conservative measures such as:\n"
             "* **Pain Relief:** Using medications like non-steroidal anti-inflammatory drugs (NSAIDs).\n"
             "* **Physical Therapy:** To strengthen the muscles around the joint and improve mobility.\n\n"
             "Since the clinical note indicates the patient has \"advanced\" arthritis and is already discussing \"Total Knee Arthroplasty (TKA),\" this suggests the condition is severe. This escalation to surgery aligns with the standard treatment path mentioned in the knowledge base for \"severe cases\" where conservative measures may no longer be sufficient.\n\n"
             "*Disclaimer: This is AI-generated information and not medical advice. Please consult a healthcare professional for diagnosis or treatment. Dial 000 if this is an emergency.*\n\n"

             "You MUST now answer the user's *new* question by following this Chain of Thought:\n\n"
             "1.  **Analyze Question:** What is the user asking for? (e.g., summary, diagnosis, patient's plan, *your recommendations*).\n"
             "2.  **Check for Recommendation Request:** Did the user ask for *your* recommendations (e.g., 'What do you recommend?', 'What are your recommendations?')?\n"
             "3.  **Formulate Answer:**\n"
             "    * **IF a Recommendation Request:** Answer by using the 'Knowledge Base Context' to provide general, insightful recommendations, **similar in style and depth to the example above**. You MUST be detailed and synthesize information. **DO NOT** just repeat the plan from the 'Clinical Text'.\n"
             "    * **IF NOT a Recommendation Request:** Answer by analyzing the 'Clinical Text'. Find symptoms (HPI), treatments (PLAN), or diagnosis (ASSESSMENT) *directly from the note*.\n"
             "4.  **Synthesize Final Answer:** Combine these findings into a concise, detailed answer.\n\n"
             
             "--- RULES ---\n"
             "* **Patient-Specifics:** Use the 'Clinical Text' *only* for questions about the patient's specific history, symptoms, or *prescribed plan*.\n"
             "* **General Recommendations:** Use the 'Knowledge Base Context' for general questions AND for *all* recommendation requests.\n"
             "* **Privacy:** DO NOT guess or infer any [REDACTED] information.\n"
             "* **Disclaimer:** You MUST end every single response with the *exact* disclaimer below.\n\n"
             "*Disclaimer: This is AI-generated information and not medical advice. Please consult a healthcare professional for diagnosis or treatment. Dial 000 if this is an emergency.*"
             ),
            
            ("human",
             "**Knowledge Base Context:**\n{context}\n\n"
             "**Clinical Text (Patient Note):**\n{query}\n\n"
             "**Extracted Entities (for reference):**\n{entities}\n\n"
             "**Question:** {question}\n\n"
             "**Answer:**")
        ])

        base_dir = os.path.abspath(os.path.dirname(__file__))
        self.index_path = os.path.join(base_dir, "faiss_index")
        self.kb_dir = base_dir

        self.vs = None
        self.build_or_load_index()

        try:
            self.chain = self.prompt | self.llm
        except Exception as e:
            _engine_log(f"Failed to create LLM chain: {e}")
            self.chain = None

    def build_or_load_index(self):
        # (This function is unchanged)
        _engine_log("RagPipeline.build_or_load_index: preparing FAISS index")
        os.makedirs(self.index_path, exist_ok=True)
        faiss_file = os.path.join(self.index_path, "index.faiss")
        marker_file = os.path.join(self.index_path, ".building")

        if os.path.exists(faiss_file):
            try:
                self.vs = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                _engine_log("Loaded existing FAISS index from disk.")
                return
            except Exception:
                pass
        if os.path.exists(marker_file):
            _engine_log("Index build in progress, using temporary index.")
            self.vs = FAISS.from_texts(["Temporary empty index"], self.embedder)
            return
        if getattr(RagPipeline, "_BUILDING", False):
            _engine_log("Index build in progress, using temporary index.")
            self.vs = FAISS.from_texts(["Temporary empty index"], self.embedder)
            return

        created_marker = False
        try:
            fd = os.open(marker_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            created_marker = True
            _engine_log("Set build marker, starting index creation...")
        except FileExistsError:
            pass

        RagPipeline._BUILDING = True
        try:
            files = [f for f in glob.glob(os.path.join(self.kb_dir, "*.md")) if os.path.basename(f).lower() != "readme.md"]
            docs = []
            for fpath in files:
                try:
                    with open(fpath, "r", encoding="utf-8", errors="ignore") as fh:
                        docs.append(Document(page_content=fh.read(), metadata={"source": os.path.basename(fpath)}))
                except Exception:
                    pass
            if not docs:
                _engine_log("No *.md files found. Creating seed doc.")
                docs = [Document(page_content="Seed guidance snippet. Add *.md docs to this folder.", metadata={"source": "seed_doc.txt"})]

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.cfg["retrieval"]["chunk_size"],
                chunk_overlap=self.cfg["retrieval"]["chunk_overlap"]
            )
            chunks = splitter.split_documents(docs) if docs else []
            if chunks:
                self.vs = FAISS.from_documents(chunks, self.embedder)
                self.vs.save_local(self.index_path)
                _engine_log(f"Created and saved new FAISS index from {len(docs)} documents.")
            else:
                _engine_log("No chunks generated, creating empty index.")
                self.vs = FAISS.from_texts(["Empty index"], self.embedder)
        finally:
            RagPipeline._BUILDING = False
            if created_marker and os.path.exists(marker_file):
                try:
                    os.remove(marker_file)
                except Exception:
                    pass

    def retrieve(self, query: str, k: int):
        # (This function is unchanged)
        _engine_log("RagPipeline.retrieve: similarity_search")
        if not self.vs:
             _engine_log("RagPipeline.retrieve: Vector store not initialized.")
             return []
        docs = self.vs.similarity_search(query, k=k)
        return [{"text": d.page_content, "doc_id": d.metadata.get("source", "doc")} for d in docs]

    def answer(self, query: str, question: str):
        # (This function is unchanged)
        _engine_log("RagPipeline.answer: starting...")

        entities = self.ner.extract(query)
        
        _engine_log("Retrieving from Knowledge Base based on the question.")
        
        retrieval_query = question 
        k = self.cfg["retrieval"]["k"]
        k_final = max(1, min(self.cfg["retrieval"]["k_final"], 5))
        
        retrieved = self.retrieve(retrieval_query, k)[:k_final]

        MAX_CHARS = 900
        context_str = "\n\n".join([f"[{i+1}] {r['doc_id']}: {r['text'][:MAX_CHARS]}" for i, r in enumerate(retrieved)])
        
        if not context_str:
            _engine_log("No relevant documents found in Knowledge Base.")
            context_str = "No relevant information found in the knowledge base."
        else:
            _engine_log(f"Found {len(retrieved)} relevant context chunks.")

        payload = {
            "context": context_str, 
            "entities": json.dumps(entities, indent=2) if entities else "[]", 
            "query": query, 
            "question": question 
        }
        
        answer: str

        try:
            if self.chain is not None:
                out_str = self.chain.invoke(payload)
            else:
                prompt_str = self.prompt.format(**payload).to_string()
                out_str = self.llm.invoke(prompt_str)

            answer = out_str.strip()

        except Exception as e:
            _engine_log(f"LLM call failed: {e}")
            raise RuntimeError("LLM call failed. Ensure Ollama is running and the model is pulled.") from e

        return {"answer": answer, "sources": retrieved}