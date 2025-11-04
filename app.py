import os
import yaml
import streamlit as st
import json
from engine import DeIdentifier, ClinicalNER, RagPipeline

# --- Page Config ---
st.set_page_config(
    page_title="Meditron",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Config ---
@st.cache_data
def load_config():
    """Loads config.yaml safely."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Fallback config (unchanged)
    return {
        "models": {
            "llm": {"provider": "ollama", "model": "llama3:8b", "temperature": 0.1, "max_tokens": 1024},
            "embeddings": {"name": "nomic-embed-text"}
        },
        "retrieval": {"chunk_size": 800, "chunk_overlap": 160, "k": 5, "k_final": 4},
        "privacy": {"block_on_phi_leak": False},
        "umls_linking": {"enable": False, "k": 1, "threshold": 0.75} 
    }

cfg = load_config()

# --- Engine Instantiation ---
@st.cache_resource
def get_engine(app_cfg):
    """
    Creates and caches all the engine components.
    """
    _deid = DeIdentifier(block_on_phi=app_cfg["privacy"].get("block_on_phi_leak", False))
    
    umls_cfg = app_cfg.get("umls_linking", {"enable": False, "k": 1, "threshold": 0.75})
    
    # We still create the NER engine, but force linking off
    _ner = ClinicalNER(
        enable_linking=False, 
        k=umls_cfg.get("k", 1),
        threshold=umls_cfg.get("threshold", 0.75)
    )
    
    _rag = RagPipeline(cfg=app_cfg, ner=_ner)
    
    return _deid, _ner, _rag

# --- Sidebar Design ---
with st.sidebar:
    # (This section is unchanged, the UMLS toggle was already removed)
    st.title("🩺 Meditron")
    st.write("A privacy-focused clinical text analyzer.")
    st.write("---")

    st.subheader("Controls")
    
    if st.button("Reset Chat & Note"):
        st.session_state.messages = []
        st.session_state.note_context = ""
        st.rerun()
    
    st.session_state.strict_phi_block = st.toggle(
        "Strict PHI block (stop on PHI)", 
        value=st.session_state.get('strict_phi_block', False)
    )
    
    st.write("---")
    st.caption("NER: `scispaCy`")
    st.caption(f"LLM: `{cfg['models']['llm']['model']}`")
    st.caption(f"Embed: `{cfg['models']['embeddings']['name']}`")


# --- Initialize Engine ---
try:
    # (This section is unchanged)
    deid, ner, rag = get_engine(cfg)
except Exception as e:
    st.error(f"Fatal Error: Failed to initialize analysis engine: {e}")
    st.error("Please ensure Ollama is running and you have pulled the required models.")
    st.code(f"ollama pull {cfg['models']['llm']['model']}\nollama pull {cfg['models']['embeddings']['name']}")
    st.error("You may also need to install the scispaCy model:")
    st.code("pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v1.0.0/en_core_sci_sm-1.0.0.tar.gz")
    st.stop()


# --- Main Page Layout ---
# (This section is unchanged)
st.title("🩺 Meditron")
st.caption("Ask questions about the clinical note provided below. The model can summarize, extract information, and answer general questions.")

st.subheader("Clinical Note")
st.caption("Paste the patient note here. This is the primary source of truth for the AI.")

st.session_state.note_context = st.text_area(
    "Your Clinical Note", 
    height=250, 
    key="note_context_input",
    value=st.session_state.get('note_context', ''),
    label_visibility="collapsed"
)
st.write("---")

# --- Chat History ---
st.subheader("Analysis Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display sources if they exist
        if "sources" in message and message["sources"]:
            with st.expander("Sources Consulted (from Knowledge Base)"):
                for i, src in enumerate(message["sources"]):
                    doc_id = src.get('doc_id', 'Unknown')
                    snippet = src.get('text', '...').split('\n')[0]
                    st.caption(f"**[{i+1}] {doc_id}**: *{snippet}...*")
        
        # --- MODIFICATION: Removed the 'keywords' expander ---
        # The 'if "keywords" in message...' block has been deleted.


# Handle user input
if prompt := st.chat_input("Ask a question about your clinical note..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Assistant Response ---
    with st.chat_message("assistant"):
        with st.spinner("Meditron is thinking..."):
            
            note = st.session_state.get('note_context', '')
            if not note or not note.strip():
                st.warning("Please provide a clinical note in the text area above first.")
                st.session_state.messages.pop() 
                st.stop()

            # 2. De-identification (unchanged)
            try:
                deid_out = deid.redact(note)
                redacted_note = deid_out["text"]
                phi_detected = deid_out.get("had_phi", False)
            except Exception as e:
                st.error(f"Error during de-identification: {e}")
                st.stop()

            if phi_detected:
                st.warning(":red_circle: **PHI detected in note** — proceeding with redacted text for analysis.")

            # 3. Strict PHI block (unchanged)
            if phi_detected and st.session_state.strict_phi_block:
                st.error("Strict PHI block is ON. Stopping analysis.")
                response_content = "Error: Strict PHI block is enabled and PHI was detected. Cannot proceed."
                st.session_state.messages.append({"role": "assistant", "content": response_content})
                st.stop()

            # 4. Call RAG Pipeline
            try:
                result = rag.answer(query=redacted_note, question=prompt)
                
                response_text = result.get("answer", "No answer found.")
                sources = result.get("sources", [])
                # --- MODIFICATION: 'keywords' variable removed ---
                # keywords = result.get("keywords", []) # This line is deleted

                st.markdown(response_text)
                
                if sources:
                    with st.expander("Sources Consulted (from Knowledge Base)"):
                        for i, src in enumerate(sources):
                            doc_id = src.get('doc_id', 'Unknown')
                            snippet = src.get('text', '...').split('\n')[0]
                            st.caption(f"**[{i+1}] {doc_id}**: *{snippet}...*")
                
                # --- MODIFICATION: Removed the 'keywords' expander ---
                # The 'if keywords:' block has been deleted.

                # --- MODIFICATION: Removed 'keywords' from session state ---
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": sources
                    # "keywords": keywords # This key is removed
                })

            except Exception as e:
                st.error(f"Error during analysis: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, an error occurred: {e}"})