import streamlit as st
import yaml
from engine import DeIdentifier, ClinicalNER, RelationExtractor, RagPipeline, split_sections

st.set_page_config(page_title="Meditron Simple – Clinical Text", layout="wide")

# Sidebar controls
st.sidebar.header("Controls")
if st.sidebar.button("Reset"):
    st.session_state.clear()
    st.rerun()

show_sources = st.sidebar.checkbox("Show sources (retrieved chunks + CUIs)", value=True)
enable_umls = st.sidebar.checkbox("Enable UMLS linking (if available)", value=True)
use_hpi = st.sidebar.checkbox("HPI", value=True)
use_assessment = st.sidebar.checkbox("Assessment", value=True)
use_plan = st.sidebar.checkbox("Plan", value=True)

st.title("Meditron Simple – Privacy-Preserving Clinical Text (meditron:7b)")

# Load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# Boot core
deid = DeIdentifier(block_on_phi=cfg["privacy"]["block_on_phi_leak"])
ner = ClinicalNER(enable_linking=enable_umls)
rel = RelationExtractor()
rag = RagPipeline(cfg)

# Input
st.subheader("Clinical Note (raw or de-identified)")
note = st.text_area("Paste a note. PHI will be redacted automatically.", height=240, key="note_text")
uploaded = st.file_uploader("...or upload a .txt file", type=["txt"])
if uploaded:
    note = uploaded.read().decode("utf-8")
    st.session_state["note_text"] = note
    st.experimental_rerun()

if st.button("Analyze"):
    if not note or not note.strip():
        st.warning("Please provide text first.")
        st.stop()

    # De-ID
    deid_out = deid.redact(note)
    redacted = deid_out["text"]
    phi_detected = deid_out.get("had_phi", False)

    phi_col, red_col = st.columns([1,3])
    with phi_col:
        if phi_detected:
            st.markdown(":red_circle: **PHI detected and redacted**")
        else:
            st.markdown(":green_circle: **No PHI detected**")
    with red_col:
        st.markdown("**Redacted Input (preview)**")
        st.code(redacted[:1500])

    if phi_detected and cfg["privacy"].get("block_on_phi_leak", True):
        st.info("Processing blocked because PHI was detected. Please review and re-run.")
        st.stop()

    # Section selection
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

    ents = ner.extract(combined)
    rels = rel.extract(combined, ents)

    with st.spinner("Retrieving and generating..."):
        result = rag.answer(query=combined, entities=ents, relations=rels)

    st.subheader("Answer")
    st.write(result["answer"])

    if show_sources:
        st.subheader("Sources & Entities")
        for i, src in enumerate(result.get("sources", []), 1):
            st.markdown(f"**[{i}] {src.get('doc_id','doc')}**")
            st.code(src.get('text', '')[:800])

        if ents:
            pretty = []
            for e in ents:
                item = {"text": e.get("text"), "label": e.get("label")}
                if "cui" in e: item["cui"] = e["cui"]
                if "canonical_name" in e: item["canonical_name"] = e["canonical_name"]
                pretty.append(item)
            st.json({"entities": pretty, "relations": rels})
        else:
            st.write("_No entities detected._")
