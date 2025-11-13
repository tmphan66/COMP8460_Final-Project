import os
import re
import pandas as pd
import easyocr
import streamlit as st
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor, tool
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# --- 1. Configurations ---
CSV_PATH = "drugs_cleaned.csv"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LLM_MODEL = "gemma3:4b"

print("Initializing EasyOCR reader...")
reader = easyocr.Reader(['en'])
print("EasyOCR reader initialized.")

# --- 2. Build ChromaDB function ---
def build_chroma_db():
    """
    Build a new ChromaDB from the CSV ONCE. Persists to DB_PATH.
    """
    print(f"--- Building new ChromaDB index at {DB_PATH} ---")
    print(f"Loading dataset from '{CSV_PATH}'...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['review', 'drugName', 'condition', 'rating'])
    print(f"Loaded with {len(df)} records.")

    print("Converting data to LangChain Documents...")
    documents = []
    for _, row in df.iterrows():
        page_content = str(row["review"])
        metadata = {
            "drugName": str(row['drugName']).strip(),
            "condition": str(row['condition']).strip(),
            "rating": int(row['rating']),
            "sentiment_label": str(row['sentiment_label']).strip() 
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
    )
    print("Embedding model loaded.")

    print("Building ChromaDB index... This may take a moment...")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=DB_PATH
    )
    
    print("ChromaDB index built successfully.")
    print(f"Chroma vector database saved to {DB_PATH} folder.")
    print("--- Building ChromaDB completed. ---")

# --- 3. Agent Loading function ---
def load_agent_system():
    """
    Check for DB, load models, and create the agent executor.
    """
    if not os.path.isdir(DB_PATH):
        print(f"ChromaDB not found at '{DB_PATH}'. Building new database...")
        build_chroma_db()

    # Load LLM
    print(f"Loading LLM model ({LLM_MODEL})...")
    llm = ChatOllama(model=LLM_MODEL, temperature=0.1, max_tokens=1024)

    # Load embedding model
    print(f"Loading embedding model ({EMBEDDING_MODEL})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}  # Use "cuda" if you have a GPU
    )

    # Load ChromaDB
    print("Loading Chroma DB...")
    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embeddings
    )

    # Load CSV into Pandas (for tools that compute over the table)
    print("Loading CSV into Pandas...")
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['review', 'drugName', 'condition', 'rating'])
    df['rating'] = df['rating'].astype(int)
    print("--- All models and data loaded successfully. ---")

    # --- 4. Define Tools ---
    simple_retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k":5}
    )

    @tool
    def process_image(query_about_image: str) -> str:
        """
        Use this tool to extract text from the image via OCR.
        This tool MUST be called FIRST if the user's query is about an uploaded image.
        It takes the user's question about the image.
        It returns recognized text from the image.
        After you get the text, you MUST use another tool (like `rag_search` or `get_average_rating`) to find information about the drug name you found.
        """

        print(f"\n>> Calling OCR Tool for uploaded image. Query: {query_about_image}")
        
        # Check for a NEW image FIRST.
        if "pending_image_bytes" in st.session_state and st.session_state.pending_image_bytes is not None:
            try:
                image_bytes = st.session_state.pending_image_bytes
                print(f"Processing new image with {len(image_bytes)} bytes...")
                results = reader.readtext(image_bytes)
                recognized_text = " ".join([text for (bbox, text, prob) in results]).strip()

                # Cache the new text
                st.session_state.last_ocr_text = recognized_text or ""
                # CRITICAL: Clear the pending bytes *after* processing
                st.session_state.pending_image_bytes = None

                if not recognized_text:
                    return "Could not extract any text from the new image."
                return f"Recognized text from image: {recognized_text}"
            except Exception as e:
                # Clear the bytes on error to prevent loops
                st.session_state.pending_image_bytes = None
                return f"Error: Could not process new image or extract text. {e}"

        # If no new image, THEN check the cache.
        cached = st.session_state.get("last_ocr_text")
        if cached:
            print("-> Using cached OCR text.")
            return f"Recognized text from image: {cached}"

        # If no new image AND no cache, then there's nothing to process.
        return "Error: I can't find an image to process. Please upload an image first."
        
    @tool
    def rag_search(query: str) -> str:
        """
        Use this tool for two primary purposes:
        1. To answer QUALITATIVE questions about user experiences, reviews, side effects, or feelings for a *known* drug.
        2. To find drugs for a specific condition or symptom (e.g., 'I have a cold', 'drugs for depression').
        Do NOT use this for math, counts, averages, or top lists.
        It takes plain text input.
        If a drug name was extracted from an image, you can use that name in the query (e.g., 'reviews for <drug_name>').
        It returns a summary of matching reviews, including drug names, ratings, snippets, and sentiment.
        """
        
        print(f"\n>> Calling RAG Tool with query: {query}")
        
        processed_query = query
        match = re.search(r"i have (a|an)\s+(.+)", query.lower().strip())
        if match:
            symptom = match.group(2).strip(" .?!")
            processed_query = f"drugs for {symptom}"
            print(f"-> Query re-written to: '{processed_query}'")
        
        print("-> Using Simple Similarity Retriever...")
        docs = simple_retriever.invoke(processed_query)

        if not docs:
            print("-> Simple Similarity Retriever found 0 documents.")
            return "No relevant reviews found for that query."
        
        print(f"-> Retrieved {len(docs)} documents. Summarizing for agent.")

        summary_lines = [
            f"Found {len(docs)} reviews related to '{processed_query}'. Here is a summary of the retrieved reviews:"
        ]
        
        for i, doc in enumerate(docs):
            drug_name = doc.metadata.get('drugName', 'N/A')
            if drug_name == 'N/A':
                continue 
                
            snippet = doc.page_content.split('.')[0].strip() + "."
            rating = doc.metadata.get('rating', 'N/A')
            sentiment = doc.metadata.get('sentiment_label', 'N/A')
            
            summary_lines.append(
                f"{i+1}. Drug: {drug_name.capitalize()} (Rating: {rating}/10, Sentiment: {sentiment}). Snippet: \"{snippet}\""
            )

        if len(summary_lines) <= 1:
             return "I found reviews, but could not extract their details."

        return "\n".join(summary_lines)

    @tool
    def get_average_rating(drug_name: str) -> str:
        """
        Use this tool to get the average rating (1-10) for ONE drug.
        Call this tool when user asks for "average rating", "effectiveness", "how effective" of a drug.
        It takes drug name ONLY (string).
        It returns one sentence with the average rating and the effectiveness.
        Do NOT use this tool for counts, side effects, or top lists.
        """

        print(f"\n>> Calling get_average_rating tool for: {drug_name}")
        try:
            clean_drug_name = drug_name.lower().strip()
            mask = (
                df["drugName"].astype(str).str.strip().str.lower() == clean_drug_name
            )
            filtered = df.loc[mask, "rating"].dropna()
            n = int(filtered.shape[0])
            if n == 0:
                return f"I'm sorry, I couldn't find any reviews for '{drug_name}' to calculate an average rating."

            mean_rating = float(filtered.mean())
            if 1 <= mean_rating <= 4:
                interpretation = "low effectiveness"
            elif 5 <= mean_rating <= 7:
                interpretation = "average effectiveness"
            else:
                interpretation = "high effectiveness"

            return (
                f"The average rating for {drug_name} (based on {n} reviews) is "
                f"{mean_rating:.1f} out of 10, which indicates {interpretation}."
            )
        except Exception as e:
            return f"Error calculating rating: {e}"

    @tool
    def get_review_count(drug_name: str) -> str:
        """
        Use this tool to count the total reviews for ONE drug.
        Call this tool when user asks "how many reviews" or "total reviews" for a drug.
        It takes drug name ONLY (string).
        It returns one sentence with the total review count.
        Do NOT use this tool for averages, side effects, or top lists.
        """

        print(f"\n>> Calling get_review_count tool for: {drug_name}")
        try:
            clean_drug_name = drug_name.lower().strip()
            mask = (
                df["drugName"].astype(str).str.strip().str.lower() == clean_drug_name
            )
            n = int(df.loc[mask].shape[0])
            if n == 0:
                return f"I'm sorry, I couldn't find any reviews for '{drug_name}'."
            return f"I found a total of {n} reviews for {drug_name}."
        except Exception as e:
            return f"Error counting reviews: {e}"


    @tool
    def get_top_most(query: str) -> str:
        """
        Use this tool to get Top-5 lists of drugs or conditions.
        Call this tool when user's question includes "top 5 drugs", "top 5 conditions", or "top 5 drugs for <condition>".
        It takes plain text input.
        It returns a short labeled Top-5 list (one line per item with counts).
        Do NOT use this tool for averages, single-drug counts, or qualitative summaries.
        """
        
        print(f"\n>> Calling get_top_most tool with query: {query}")
        q = (query or "").strip().lower()

        # Intent:
        m_for = re.search(r"top\s*5\s*drugs?\s*for\s+(.+)", q) 
        has_conditions = ("top 5" in q) and ("condition" in q or "conditions" in q) and not m_for
        has_drugs = ("top 5" in q) and ("drug" in q or "drugs" in q) and not m_for

        try:
            cond_col = df["condition"].astype(str).str.strip()
            drug_col = df["drugName"].astype(str).str.strip()

            if m_for:  # top 5 drugs for <condition>
                condition_name = m_for.group(1).strip(" ?.!").strip()
                if not condition_name:
                    return "Please specify a condition, e.g., 'top 5 drugs for Depression'."

                mask = cond_col.str.contains(re.escape(condition_name), case=False, na=False)
                sub = df.loc[mask]
                if sub.empty:
                    return f"I couldn't find any data for the condition: {condition_name}"

                counts = (
                    sub.assign(drug=drug_col.loc[mask])
                    .groupby("drug", dropna=True)
                    .size()
                    .sort_values(ascending=False)
                    .head(5)
                )
                if counts.empty:
                    return f"I couldn't find any drugs for {condition_name}."

                lines = [f"Top 5 most reviewed drugs for {condition_name}:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                return "\n".join(lines)

            if has_conditions:  # top 5 conditions
                counts = cond_col[cond_col.ne("")].value_counts().head(5)
                if counts.empty:
                    return "I couldn't find any conditions."
                lines = ["Top 5 most common conditions:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                return "\n".join(lines)

            if has_drugs:  # top 5 drugs overall
                counts = drug_col[drug_col.ne("")].value_counts().head(5)
                if counts.empty:
                    return "I couldn't find any drugs."
                lines = ["Top 5 most reviewed drugs overall:"]
                for i, (name, cnt) in enumerate(counts.items(), 1):
                    lines.append(f"{i}. {name}: {cnt}")
                return "\n".join(lines)

            return "This tool supports: 'top 5 drugs', 'top 5 conditions', or 'top 5 drugs for <condition>'."
        
        except Exception as e:
            return f"Error finding top list: {e}"
        

    tools = [process_image, rag_search, get_average_rating, get_review_count, get_top_most]

    # --- 5. Create Agent Executor ---
    agent_prompt_template = """
        You are a helpful pharmacy assistant.
        Answer the following questions as best you can. You have access to the following tools:
        {tools}

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat N times)
        Thought: I now know the final answer
        Final Answer: the final answer to the original input question

        --- RULES (Follow Strictly) ---

        1. **IMAGE QUESTIONS**: If the user uploaded an image:
           - Step 1: You MUST use the `process_image` tool first.
           - Step 2: Read the text from the `Observation`. Find the drug name in that text.
           - Step 3: You MUST then use the `rag_search` tool with the found drug name to get reviews and information about it.
           - Step 4: Use the output of `rag_search` to give a helpful final answer.

        2. **NORMAL QUESTIONS**: If there is no image, just use the best tool for the question.
        
        3. **RAG_SEARCH RULE (IMPORTANT!)**:
           - After you use `rag_search` and get an `Observation` with reviews:
           - Your *next* `Thought` MUST be "I now know the final answer."
           - You MUST then provide a `Final Answer`.
           - Your `Final Answer` should summarize the reviews and mention the drug names found.
        
        4. **ALWAYS**: Answer the user's question in a helpful sentence. Do not just repeat the tool's output.
        
        5. **NEVER HALLUCINATE (CRITICAL!)**:
           - You MUST ONLY use drug names that are explicitly provided in a tool's 'Observation'.
           - DO NOT make up drug names (like 'Ibotragomine') or any other information.
           - Stick *only* to the facts from the 'Observation'.
           - If the tools don't provide an answer, just say you don't know.

        6. **PHI / REDACTED TEXT**:
           - The user's query may contain tags like [REDACTED_NAME] or [REDACTED_EMAIL].
           - You MUST NOT try to guess or infer what the redacted text was.
           - Answer the question as if the text was never there.
           - Example: If the query is "My name is [REDACTED_NAME]. I have a cold", just answer the "I have a cold" part.
        
        Begin!
        Question: {input}
        {agent_scratchpad}
        """
    prompt = PromptTemplate.from_template(agent_prompt_template)

    agent = create_react_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )

    print("--- Agent Executor created successfully. ---")
    return agent_executor
