import pandas as pd
from langchain_ollama import ChatOllama
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_chroma import Chroma
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain.chains.query_constructor.schema import AttributeInfo
from langchain_core.documents import Document
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_huggingface import HuggingFaceEmbeddings
from langchain import hub
import os, sys

CSV_PATH = "drugsComTest_raw.csv"
DB_PATH = "chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  

# --- 1. Check for ChromaDB and load models ---

# function to build ChromaDB
def build_chroma_db():
    print(f"Building new ChromaDB index at {DB_PATH}.")

    print("Converting data to LangChain documents...")
    documents = []
    for _, row in df.iterrows():
        page_content =  row['review']
        metadata = {
            "drugName": str(row['drugName']).strip(),
            "condition": str(row['condition']).strip(),
            "rating": int(row['rating']),
            "usefulCount": int(row['usefulCount'])
        }
        doc = Document(page_content=page_content, metadata=metadata)
        documents.append(doc)

    print("Building ChromaDB index...")
    db = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    persist_directory=DB_PATH
    )

    print("ChromaDB index built successfully.")
    print(f"Chroma vector database saved to {DB_PATH} folder.")
    print("Indexing completed.")

# load models and data
try:
    print("Loading models and data...")

    # load llm
    print("Loading LLM model...")
    llm = ChatOllama(model="gemma3:4b")
    print("LLM model loaded.")

    # Load embedding model
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL, 
        model_kwargs={"device": "cpu"}) # Use "cuda" if you guys have GPU
    print("Embedding model loaded.")

    # load pandas dataframe
    print(f"Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    print("Cleaning dataset...")
    df = df.dropna(subset=['review', 'drugName', 'condition', 'rating'])
    df['rating'] = df['rating'].astype(int)
    print(f"Data cleaned. {len(df)} records remaining after cleaning.")

    print("All models and data loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure installing all requirements by running: pip install -r requirements.txt")
    print("Also ensure pulling the LLM model locally by running: ollama pull gemma3:4b")
    sys.exit(1)

# check if chroma db exists, if not build it
if not os.path.isdir(DB_PATH):
    print(f"ChromaDB not found at {DB_PATH}. Building new ChromaDB...")
    build_chroma_db()
else:
    print(f"ChromaDB found at {DB_PATH}. Using existing database.")

# load chroma db
print("Loading ChromaDB vector store...")
vector_db = Chroma(
    persist_directory=DB_PATH,
    embedding_function=embeddings
)

# --- 2. Define Tools ---

# Tool 1: RAG Search Tool
metadata_field_info = [
    AttributeInfo(name="drugName",  description="the name of the drug", type="string"),
    AttributeInfo(name="condition", description="the medical condition being treated", type="string"),
    AttributeInfo(name="rating",    description="the rating given to the drug (1-10)", type="integer"),
]

document_contents = "A user review of a drug, including their experience, side effects, and feelings."

# create self-query retriever
rag_retriever = SelfQueryRetriever.from_llm(
    llm=llm,
    vectorstore=vector_db,
    document_contents=document_contents,
    metadata_field_info=metadata_field_info,
    structured_query_translator=ChromaTranslator(),
    verbose=True
)

@tool
def chroma_search(query:str) -> str:
    """
    Use this tool to answer questions about user experiences, reviews, side effects, feelings, or semantic meaning. 
    Do not call this tool if the user's query is empty. 
    This is NOT for math, averages, or counts. It returns raw review text as context for you to summarize.
    """
    print(f"\n>> Calling RAG Tool with query: {query}")
    docs = rag_retriever.invoke(query)
    if not docs:
        return "No relevant reviews found."
    return "\n".join([f"Review (Rating: {doc.metadata.get('rating', 'N/A')} for {doc.metadata.get('drugName', 'N/A')}): {doc.page_content}" for doc in docs])

# Tool 2: Pandas Analysis Tool
pandas_agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_executor_kwargs={"handle_parsing_errors": True}
)

@tool
def pandas_analysis(query:str) -> str:
    """
    Use this tool for analytical questions about numbers, averages, counts, totals, or 'top' lists. For example: 'what is the average rating', 'how many reviews', 'top 5 most common'. 
    It returns a final answer as a string. 
    Do not call this tool if the user's query is empty
    """
    print(f"\n>> Calling Pandas Tool with query: {query}")
    try:
        response = pandas_agent.invoke(query)
        return response.get('output', 'Error: Could not get pandas output.')
    except Exception as e:
        return f"Pandas agent error: {e}"
    
tools = [chroma_search, pandas_analysis]

# Define agent prompt template
agent_prompt_template = """
    You are  a helpful pharmacy assistant. You have access to two tools.
    Use these tools to answer the user's query as best as you can.
    Tools:{tools}
    To use a tool, use the following format:
    Thought: Do I need to use a tool? Yes
    Action: The name of the tool to use (from {tool_names})
    Action Input: The input to the tool. This should be the user's query.
    Observation: The result of the tool use
    When you have the final answer, use this format:
    Thought: Do I need to use a tool? No
    Final Answer: The final answer to the user's query.
    If the user's Query is empty or whitespace, reply:
    "Please enter a query" and do not use any tools.
    Begin!
    Query: {input}
    {agent_scratchpad}
    """
prompt = PromptTemplate.from_template(agent_prompt_template)

# --- 3. Create Agent ---
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)
        
# --- 4. Main Interaction Loop ---
def main():
    print("\n--- Drug AI Assistant (v5 - ChromaDB + Agent) ---")
    print("Ready for your query.")
    print("Type 'exit' to quit.")
    
    while True:
        query = input("\n\nYour Query: ")
        if query.lower().strip() == 'exit':
            break
        
        if not query.strip():
            print("Please enter a query or type 'exit'.")
            continue
            # Run the query through our agent
        response = agent_executor.invoke({"input": query})
        print("\nFinal Answer:")
        print(response['output'])
            
if __name__ == "__main__":
    main()