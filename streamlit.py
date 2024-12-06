import streamlit as st
from graph import agent
from langchain_ollama import ChatOllama

# Page configuration
st.set_page_config(
    page_title="Network Manuals Expert",
    page_icon="🌐",
    layout="wide",
    menu_items={},
)

# Custom CSS for hiding default elements and styling
st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    div[data-testid="stVerticalBlock"] {
        padding: 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize model
@st.cache_resource
def init_model():
    return ChatOllama(
        model="llama3.2",
        temperature=0,
    )

model = init_model()
graph = agent()

# Title and subtitle
st.markdown("""
    <div style='text-align: center; background: linear-gradient(90deg, #1E88E5 0%, #1565C0 100%); 
                color: white; padding: 2rem 0; border-radius: 10px; margin-bottom: 2rem;'>
        <h1>Network Manuals Expert</h1>
        <p>Compare responses with and without Retrieval Augmented Generation (RAG)</p>
    </div>
    """, unsafe_allow_html=True)

# Create two columns for responses
col1, col2 = st.columns(2)

# Create placeholder containers for responses
with col1:
    st.markdown("#### 🔍 With RAG")
    rag_placeholder = st.empty()
    rag_placeholder.caption("Response will appear here")

with col2:
    st.markdown("#### 💭 Without RAG")
    no_rag_placeholder = st.empty()
    no_rag_placeholder.caption("Response will appear here")

# Create the chat input area
user_input = st.text_area(
    "Enter your question:",
    height=100,
    placeholder="Type your question here...",
    key="user_input"
)

# Add a submit button
if st.button("Ask Question", type="primary"):
    if user_input:
        # Process RAG response
        with col1:
            rag_placeholder.empty()  
            with st.spinner("Generating response ..."):
                inputs = {"messages": [("user", user_input)]}
                value = graph.invoke(inputs)
                rag_placeholder.markdown(value['messages'][-1].content)

        # Process non-RAG response
        with col2:
            no_rag_placeholder.empty() 
            with st.spinner("Generating response ..."):
                simple_response = model.invoke(user_input)
                no_rag_placeholder.markdown(simple_response.content)