import streamlit as st
import os
import tempfile
import re
import time
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import plotly.graph_objects as go
import chat_page
import analytics_page

def ensure_complete_sentence(text):
    # Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # If the last sentence doesn't end with a period, question mark, or exclamation point,
    # it's likely incomplete, so we remove it
    if not sentences[-1].strip().endswith(('.', '!', '?')):
        sentences = sentences[:-1]
    
    # Join the sentences back together
    return ' '.join(sentences).strip()

def sanitize_text(text):
    text = ''.join(char for char in text if char.isprintable())
    text = re.sub(r'\s+', ' ', text).strip()
    print(text)
    return text[:1000]

@st.cache_resource
def load_embeddings(model_name):
    return OllamaEmbeddings(model=model_name, base_url ='http://127.0.0.1:11434')

@st.cache_resource
def load_model(model_name):
    return Ollama(model=model_name, base_url ='http://127.0.0.1:11434')

@st.cache_data
def process_document(_file_path, file_type, _embeddings):
    if file_type == 'txt':
        loader = TextLoader(_file_path)
    elif file_type == 'pdf':
        loader = PyPDFLoader(_file_path)
    elif file_type in ['doc', 'docx']:
        loader = Docx2txtLoader(_file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    documents = loader.load()
    
    for doc in documents:
        doc.page_content = sanitize_text(doc.page_content)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    return texts

@st.cache_resource
def create_vector_store(_texts, _embeddings):
    return FAISS.from_documents(_texts, _embeddings)


def reset_chat():
    st.session_state.messages = []
    st.session_state.model_chains = {}
    st.session_state.analytics = {model: [] for model in st.session_state.selected_llms}
    st.session_state.interaction_count = 0

def calculate_metrics(answer, relevant_docs, response_time, embeddings):
    # Relevance score (cosine similarity between query and answer embeddings)
    query_embedding = embeddings.embed_query(answer)
    doc_embeddings = [embeddings.embed_query(doc.page_content) for doc in relevant_docs]
    relevance_scores = [cosine_similarity([query_embedding], [doc_embedding])[0][0] for doc_embedding in doc_embeddings]
    avg_relevance = np.mean(relevance_scores)
    
    # Consistency score (similarity between answer and top relevant document)
    consistency_score = cosine_similarity([query_embedding], [doc_embeddings[0]])[0][0]
    
    return {
        "avg_relevance": avg_relevance,
        "consistency": consistency_score,
        "response_time": response_time
    }

def visualize_performance_trend():
    fig = go.Figure()
    metrics = ['avg_relevance', 'consistency', 'response_time']
    
    for model, data in st.session_state.analytics.items():
        df = pd.DataFrame(data)
        for metric in metrics:
            fig.add_trace(go.Scatter(
                x=df['interaction'],
                y=df[metric],
                mode='lines+markers',
                name=f'{model} - {metric}',
                visible='legendonly' if metric != 'avg_relevance' else None
            ))
    
    fig.update_layout(
        title="Model Performance Trend",
        xaxis_title="Interaction",
        yaxis_title="Score / Time",
        height=400,
        width=600,
        margin=dict(l=10, r=10, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_conversational_chain(llm, retriever):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )
    return chain

def chat_with_doc(chain, query):
    start_time = time.time()
    result = chain({"question": query})
    end_time = time.time()
    response_time = end_time - start_time
    return result['answer'], response_time

def reset_chat():
    st.session_state.messages = []
    st.session_state.model_chains = {}
    st.session_state.analytics = {model: [] for model in st.session_state.selected_llms}
    st.session_state.interaction_count = 0

def initialize_embeddings(model_name):
    base_url = 'http://127.0.0.1:11434'  # Adjust this URL if your Ollama server is running elsewhere
    return OllamaEmbeddings(model=model_name, base_url=base_url)

def main():
    st.set_page_config(layout="wide", page_title="Document Q&A System")
    
    # Initialize session state variables
    if 'page' not in st.session_state:
        st.session_state.page = 'chat'
    if 'selected_llms' not in st.session_state:
        st.session_state.selected_llms = []
    if 'analytics' not in st.session_state:
        st.session_state.analytics = {}
    if 'interaction_count' not in st.session_state:
        st.session_state.interaction_count = 0
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    if 'embeddings_model' not in st.session_state:
        st.session_state.embeddings_model = None

    # Add functions to session state so they can be accessed from other pages
    st.session_state.create_conversational_chain = create_conversational_chain
    st.session_state.chat_with_doc = chat_with_doc
    st.session_state.calculate_metrics = calculate_metrics
    st.session_state.load_embeddings = load_embeddings
    st.session_state.load_model = load_model
    st.session_state.process_document = process_document
    st.session_state.create_vector_store = create_vector_store
    st.session_state.start_processing = False

    # Sidebar for navigation and model selection
    with st.sidebar:
        st.title("Navigation")
        if st.button("Chat Interface"):
            st.session_state.page = 'chat'
        if st.button("Analytics"):
            st.session_state.page = 'analytics'

        st.subheader("Model and Embedding Selection")
        llm_models = ["mistral", "llama3:8b", "gemma2:2b"]
        embedding_models = ["nomic-embed-text", "mxbai-embed-large"]
        
        st.session_state.selected_llms = st.multiselect("Select LLM models", llm_models, default=["mistral"])
        selected_embedding = st.selectbox("Select embedding model", embedding_models)
        start_processing = st.checkbox('Chat with this configuration')
        
        
        # Initialize or update embeddings if needed
        if st.session_state.embeddings is None or st.session_state.embeddings_model != selected_embedding:
            with st.spinner(f"Initializing {selected_embedding} embedding model..."):
                st.session_state.embeddings = initialize_embeddings(selected_embedding)
                st.session_state.embeddings_model = selected_embedding
            st.success(f"{selected_embedding} embedding model initialized successfully!")

        if st.button("New Chat"):
            reset_chat()
            st.rerun()

    # Main content
    if st.session_state.page == 'chat' and start_processing:
        chat_page.show()
    elif st.session_state.page == 'analytics':
        analytics_page.show()

if __name__ == "__main__":
    main()