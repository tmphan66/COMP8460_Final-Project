import os
import yaml
import streamlit as st
from engine import DeIdentifier, ClinicalNER, RelationExtractor, RagPipeline, split_sections

st.set_page_config(page_title="Clinical Text Assistant", layout="wide")

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.rerun()

enable_umls = st.sidebar.checkbox("Enable UMLS linking (if available)", value=True)
use_hpi = st.sidebar.checkbox("HPI", value=True)
use_assessment = st.sidebar.checkbox("Assessment", value=True)
use_plan = st.sidebar.checkbox("Plan", value=True)
strict_phi_block = st.sidebar.checkbox("Strict PHI block (stop on PHI)", value=False)

st.title("Clinical Text Assistant (Ollama)")

# Load config (safe fallback if file missing)
cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
if os.path.exists(cfg_path):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
else:
    cfg = {
        "models": {
            "llm": {"provider": "ollama", "model": "llama3.2:3b", "temperature": 0.2, "max_tokens": 700},
            "embeddings": {"name": "nomic-embed-text"}
        },
        "retrieval": {"chunk_size": 800, "chunk_overlap": 160, "k": 20, "k_final": 5},
        "privacy": {"block_on_phi_leak": False}
    }

# Core components
deid = DeIdentifier(block_on_phi=cfg["privacy"].get("block_on_phi_leak", False))
ner = ClinicalNER(enable_linking=enable_umls)
rel = RelationExtractor()
rag = RagPipeline(cfg)

st.subheader("Clinical Note")
note = st.text_area("Paste a note. PHI will be redacted automatically.", height=260, key="note_text")

if st.button("Analyze"):
    if not note or not note.strip():
        st.warning("Please provide text first.")
        st.stop()

    # De-identification
    deid_out = deid.redact(note)
    redacted = deid_out["text"]
    phi_detected = deid_out.get("had_phi", False)

    phi_col, red_col = st.columns([1,3])
    with phi_col:
        if phi_detected:
            st.markdown(":red_circle: **PHI detected — proceeding with redacted text**" if not strict_phi_block else ":red_circle: **PHI detected**")
        else:
            st.markdown(":green_circle: **No PHI detected**")
    with red_col:
        st.markdown("**Redacted Input (preview)**")
        st.code(redacted[:1500])

    if phi_detected and strict_phi_block:
        st.info("Strict PHI block is ON. Stopping after redaction preview.")
        st.stop()

    # Sections
    sections = split_sections(redacted)
    selected = []
    if use_hpi and sections.get("HPI"): selected.append(("HPI", sections["HPI"]))
    if use_assessment and sections.get("Assessment"): selected.append(("Assessment", sections["Assessment"]))
    if use_plan and sections.get("Plan"): selected.append(("Plan", sections["Plan"]))
    if not selected:
        selected = [("All", redacted)]

    st.markdown("**Sections used for retrieval**")
    for name, seg in selected:
        st.markdown(f"- {name} ({len(seg.split())} words)")

    combined = "\n\n".join(seg for _, seg in selected)

    # NER & Relations
    ents = ner.extract(combined)
    rels = rel.extract(combined, ents)

    with st.spinner("Retrieving and generating..."):
        result = rag.answer(query=combined, entities=ents, relations=rels)

    st.subheader("Answer")
    st.write(result["answer"])

    # Keywords (safe even if none)
    if isinstance(result, dict) and result.get("keywords"):
        st.subheader("Keywords")
        st.code(", ".join(result["keywords"]))