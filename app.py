import streamlit as st
from main import load_agent_system 
import re

# PHI filter function for Privacy Protection
def filter_phi(query: str) -> str:
    """
    Filters a query for common PII/PHI patterns using regex.
    """
    # 1. Redact names after common triggers (case-insensitive)
    name_pattern = r'(\b(my name is|i am|i\'m)\s+)([\w\s]+?)([\.,?!]|$)'
    query = re.sub(name_pattern, r'\1[REDACTED_NAME]\4', query, flags=re.IGNORECASE)

    # 2. Redact email addresses
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    query = re.sub(email_pattern, '[REDACTED_EMAIL]', query)

    # 3. Redact phone numbers (basic North American)
    phone_pattern = r'(\+?\d{1,2}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'
    query = re.sub(phone_pattern, '[REDACTED_PHONE]', query)
    
    # 4. Redact SSN-like numbers
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    query = re.sub(ssn_pattern, '[REDACTED_ID]', query)

    return query

# --- 1. App Configuration ---
st.set_page_config(
    page_title="Drug AI Assistant",
    layout="centered"
)
st.title("🩺 Drug AI Assistant")

def reset_chat_history():
    """
    Clears all messages and image data from the session state
    and re-initializes the welcome message.
    """
    st.session_state.messages = []
    
    # Clear any pending image data
    if "last_uploaded_file_name" in st.session_state:
        del st.session_state.last_uploaded_file_name
    if "pending_image_bytes" in st.session_state:
        del st.session_state.pending_image_bytes
        
    # Add the "system ready" message back
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "✅ System Ready! Ask me anything about the drug reviews."
    })

# --- 2. Sidebar ---
with st.sidebar:
    st.header("Controls")
    st.button(
        "Reset Chat", 
        on_click=reset_chat_history, 
        use_container_width=True,
        type="primary" 
    )
    
    st.header("Image Upload")
    uploaded_file = st.file_uploader(
        "Upload prescription/label",
        type=["png", "jpg", "jpeg"]
    )

# --- 3. Agent Loading --- 
@st.cache_resource(show_spinner=False)
def get_agent_executor():
    """
    Loads and caches the agent executor using Streamlit's cache.
    This function will only run ONCE per session.
    """
    # This print will show in your console, not the app
    print("--- Caching: Initializing agent system... ---")
    return load_agent_system()

# --- 4. Initialization --- 
try:
    # Show a spinner *only* while the agent is first loading
    with st.spinner("Initializing Drug AI Assistant... This can take a moment..."):
        agent_executor = get_agent_executor()
    
    # Initialize chat history in session state if it doesn't exist
    if "messages" not in st.session_state:
        # This will now only run the very first time the app is loaded
        reset_chat_history()

except Exception as e:
    # Display a fatal error if the system couldn't load
    st.error(f"❌ Could not initialize system: {e}")
    st.stop() # Stop the app if the agent fails to load

# --- 5. Image Upload Logic --- 
if uploaded_file is not None:
    # To prevent re-processing the same file on every script rerun, check if it's a new upload.
    if "last_uploaded_file_name" not in st.session_state or st.session_state.last_uploaded_file_name != uploaded_file.name:
        
        # Store the file info in session state
        st.session_state.last_uploaded_file_name = uploaded_file.name
        image_bytes = uploaded_file.getvalue()
        
        # Store image bytes for later processing by the agent
        st.session_state.pending_image_bytes = image_bytes  
        
        # Add the user's upload message to history
        st.session_state.messages.append({
            "role": "user", 
            "content": f"I've uploaded an image: {uploaded_file.name}",
            "image": image_bytes  # Store the image bytes
        })
        
        # Add a placeholder assistant response
        st.session_state.messages.append({
            "role": "assistant",
            "content": "Thanks! I've received the image. **Please ask me a question about it.**"
        })

# --- 6. Display Chat History --- 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        
        if "content" in message:
            st.markdown(message["content"])

        if "image" in message:
            st.image(message["image"], caption="Uploaded Image", width=300) 

# --- 7. Handle New User Input --- 
if prompt := st.chat_input("Ask me anything about the drug reviews..."):
    
    filtered_prompt = filter_phi(prompt)

    # 1. Add user's message to history and display it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Get and display AI's response
    with st.chat_message("assistant"):
        try:
            # Show a thinking spinner *while* the agent is working
            with st.spinner("Thinking..."):
                # We call invoke directly (no async needed)
                response = agent_executor.invoke({"input": filtered_prompt})
                ai_response = response.get("output", "No output produced.")
            
            st.markdown(ai_response)
            
        except Exception as e:
            ai_response = f"❌ Error while handling your request: {e}"
            st.error(ai_response)
    
    # 3. Add AI's response to history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})