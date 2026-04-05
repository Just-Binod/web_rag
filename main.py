import streamlit as st
import os
import re
from typing import List
import tempfile

# Fix for protobuf
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Use ChromaDB's built-in embedding function (no torchvision needed!)
from chromadb.utils import embedding_functions

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Web RAG Chat - Binod Raj Pant",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Web-Based RAG Chatbot")
st.caption("Upload any URLs (single or multiple) and ask questions based on their content!")

# ====================== API KEY ======================
def get_api_key():
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return api_key
    except Exception:
        st.error("❌ API Key not found. Please add GROQ_API_KEY to Streamlit secrets.")
        st.stop()

GROQ_API_KEY = get_api_key()

# ====================== HELPER FUNCTIONS ======================
def validate_url(url: str) -> bool:
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def fetch_and_process_urls(urls: List[str], progress_callback=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    documents = []
    failed_urls = []
    
    for i, url in enumerate(urls):
        try:
            if progress_callback:
                progress_callback(i, len(urls), url)
            
            loader = WebBaseLoader(url, header_template=headers)
            docs = loader.load()
            
            for doc in docs:
                doc.metadata['source'] = url
                documents.append(doc)
                
        except Exception as e:
            failed_urls.append({"url": url, "error": str(e)[:100]})
    
    return documents, failed_urls

def create_vectorstore_from_docs(documents: List[Document]):
    """Create vector store using ChromaDB with built-in embeddings"""
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Use ChromaDB's built-in sentence transformer (no torchvision issues)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    
    # Create Chroma vectorstore
    persist_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=persist_dir,
        collection_name="web_rag_collection"
    )
    
    return vectorstore, chunks

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("📝 URL Input")
    
    input_method = st.radio(
        "Choose input method:",
        ["Text Area (Multiple URLs)", "Upload Text File"]
    )
    
    urls_text = ""
    
    if input_method == "Text Area (Multiple URLs)":
        urls_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning",
            height=150
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt file with URLs",
            type=['txt']
        )
        if uploaded_file:
            urls_text = uploaded_file.read().decode('utf-8')
    
    urls = []
    if urls_text:
        urls = [url.strip() for url in urls_text.split('\n') if url.strip() and validate_url(url.strip())]
    
    process_button = st.button("🚀 Process URLs", type="primary", use_container_width=True, disabled=len(urls) == 0)
    
    st.markdown("---")
    
    st.subheader("🎛️ Model Parameters")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)
    k_value = st.slider("Number of chunks (k)", 1, 10, 4)
    
    st.markdown("---")
    
    st.subheader("👨‍💻 Author")
    st.markdown("""
    **Binod Raj Pant**
    
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/binod-raj-pant-303767330/)
    """)

# ====================== PROCESS URLS ======================
if process_button and urls:
    with st.spinner("Processing URLs..."):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = (current + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"📥 Fetching: {url[:50]}... ({current + 1}/{total})")
        
        documents, failed_urls = fetch_and_process_urls(urls, update_progress)
        
        if documents:
            status_text.text("🔨 Creating embeddings and building vector database...")
            vectorstore, chunks = create_vectorstore_from_docs(documents)
            
            st.session_state.vectorstore = vectorstore
            st.session_state.current_urls = urls
            
            progress_bar.empty()
            status_text.empty()
            
            st.success(f"✅ Successfully processed {len(documents)} page(s) from {len(urls)} URL(s)!")
            st.info(f"📊 Created {len(chunks)} chunks for retrieval")
            
            if failed_urls:
                with st.expander(f"⚠️ {len(failed_urls)} URL(s) failed"):
                    for failed in failed_urls:
                        st.write(f"• {failed['url']}: {failed['error']}")
            
            st.session_state.messages = [
                {"role": "assistant", "content": f"✅ Ready! I've loaded {len(urls)} URL(s) with {len(chunks)} chunks. Ask me anything!"}
            ]
            st.rerun()
        else:
            st.error("❌ No documents could be loaded.")
            progress_bar.empty()
            status_text.empty()

# ====================== CHAT INTERFACE ======================
if "vectorstore" in st.session_state and st.session_state.vectorstore:
    
    @st.cache_resource
    def get_llm(temperature):
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=temperature
        )
    
    llm = get_llm(temperature)
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"✅ Ready! I've loaded {len(st.session_state.current_urls)} URL(s). Ask me anything!"}
        ]
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask questions about the loaded content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.spinner("🔍 Searching for relevant information..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_value})
            docs = retriever.invoke(prompt)
        
        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                st.markdown(f"**Source {i+1}:** `{source}`")
                st.markdown(f"**Content:** {doc.page_content[:300]}...")
                st.divider()
        
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        system_prompt = """You are a helpful assistant. Answer the question based ONLY on the following context.
If you don't know the answer, say "I don't have enough information from the provided sources."
Be concise and accurate.

Context:
{context}
"""
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        
        chain = prompt_template | llm
        
        with st.chat_message("assistant"):
            with st.spinner("💭 Generating answer..."):
                try:
                    response = chain.invoke({"context": context, "question": prompt})
                    answer = response.content
                    st.markdown(answer)
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👋 **Welcome!** To get started:\n\n"
            "1. Enter one or more URLs in the sidebar\n"
            "2. Click 'Process URLs' to load the content\n"
            "3. Start asking questions about the loaded content!\n\n"
            "📝 **Example URLs:**\n"
            "- https://en.wikipedia.org/wiki/Artificial_intelligence\n"
            "- https://en.wikipedia.org/wiki/Machine_learning")

# ====================== FOOTER ======================
st.divider()
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Built with ❤️ by <strong>Binod Raj Pant</strong></p>
            <a href='https://www.linkedin.com/in/binod-raj-pant-303767330/' target='_blank'>
                <img src='https://img.shields.io/badge/Connect_on_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white' alt='LinkedIn'>
            </a>
            <br>
            <br>
            <p style='font-size: 12px;'>
                Tech Stack: LangChain | ChromaDB | Groq LLM | Streamlit Cloud
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )