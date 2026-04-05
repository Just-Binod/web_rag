import streamlit as st
import os
import re
from typing import List

# LangChain imports
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="Web RAG Chat - Binod Raj Pant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================== HEADER ======================
st.title("🌐 Web-Based RAG Chatbot")
st.caption("Upload any URLs (single or multiple) and ask questions based on their content!")

# ====================== API KEY FROM STREAMLIT SECRETS ======================
def get_api_key():
    """Get API key from Streamlit Cloud secrets"""
    try:
        api_key = st.secrets["GROQ_API_KEY"]
        return api_key
    except Exception as e:
        st.error(f"❌ API Key not found. Please add GROQ_API_KEY to Streamlit secrets.")
        st.info("📖 Go to: App Settings → Secrets → Add GROQ_API_KEY")
        st.stop()

# Get API key from secrets
GROQ_API_KEY = get_api_key()

# ====================== HELPER FUNCTIONS ======================
def validate_url(url: str) -> bool:
    """Validate URL format"""
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def fetch_and_process_urls(urls: List[str], progress_callback=None) -> List[Document]:
    """Fetch and process multiple URLs"""
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

def create_vectorstore_from_docs(documents: List[Document]) -> Chroma:
    """Create vector store from documents"""
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)
    
    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create vectorstore
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./custom_chroma_db"
    )
    
    return vectorstore, chunks

# ====================== SIDEBAR ======================
with st.sidebar:
    st.image("https://img.shields.io/badge/RAG-Chatbot-blue", use_container_width=True)
    
    st.header("📝 URL Input")
    
    # URL input method
    input_method = st.radio(
        "Choose input method:",
        ["Text Area (Multiple URLs)", "Upload Text File"],
        help="Paste URLs manually or upload a .txt file with one URL per line"
    )
    
    urls_text = ""
    
    if input_method == "Text Area (Multiple URLs)":
        urls_text = st.text_area(
            "Enter URLs (one per line):",
            placeholder="https://en.wikipedia.org/wiki/Artificial_intelligence\nhttps://en.wikipedia.org/wiki/Machine_learning\nhttps://example.com/article",
            height=150,
            help="Paste your URLs here, one per line"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a .txt file with URLs",
            type=['txt'],
            help="File should have one URL per line"
        )
        if uploaded_file:
            urls_text = uploaded_file.read().decode('utf-8')
            st.text_area("URLs to process:", urls_text, height=150, disabled=True)
    
    # Parse URLs
    urls = []
    if urls_text:
        urls = [url.strip() for url in urls_text.split('\n') if url.strip() and validate_url(url.strip())]
        invalid_urls = [url for url in urls_text.split('\n') if url.strip() and not validate_url(url.strip())]
        
        if invalid_urls:
            st.warning(f"⚠️ {len(invalid_urls)} invalid URL(s) found and will be skipped")
    
    # Process button
    process_button = st.button("🚀 Process URLs", type="primary", use_container_width=True, disabled=len(urls) == 0)
    
    st.markdown("---")
    
    # Model settings
    st.subheader("🎛️ Model Parameters")
    temperature = st.slider(
        "Temperature",
        0.0, 1.0, 0.0, 0.1,
        help="Lower = more focused answers, Higher = more creative"
    )
    k_value = st.slider(
        "Number of chunks (k)",
        1, 10, 4,
        help="More chunks = more context but slower response"
    )
    
    st.markdown("---")
    
    # Database info
    st.subheader("🗄️ Current Database")
    if "current_urls" in st.session_state and st.session_state.current_urls:
        st.info(f"📊 {len(st.session_state.current_urls)} URL(s) loaded")
        with st.expander("View loaded URLs"):
            for url in st.session_state.current_urls:
                st.write(f"• {url}")
        
        if st.button("🗑️ Clear Database", use_container_width=True):
            st.session_state.vectorstore = None
            st.session_state.current_urls = []
            st.session_state.messages = []
            st.success("✅ Database cleared!")
            st.rerun()
    else:
        st.info("No URLs loaded yet. Add URLs above and click 'Process URLs'")
    
    st.markdown("---")
    
    # Author section
    st.subheader("👨‍💻 Author")
    st.markdown("""
    ### **Binod Raj Pant**
    
    [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/binod-raj-pant-303767330/)
    
    **AI/ML Enthusiast | RAG Developer**
    
    Building intelligent applications with LangChain & Streamlit
    """)

# ====================== PROCESS URLS ======================
if process_button and urls:
    with st.spinner("Processing URLs..."):
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total, url):
            progress = (current + 1) / total
            progress_bar.progress(progress)
            status_text.text(f"Fetching: {url[:50]}... ({current + 1}/{total})")
        
        # Fetch and process URLs
        documents, failed_urls = fetch_and_process_urls(urls, update_progress)
        
        if documents:
            # Create vectorstore
            status_text.text("Creating embeddings and building vector database...")
            vectorstore, chunks = create_vectorstore_from_docs(documents)
            
            # Store in session state
            st.session_state.vectorstore = vectorstore
            st.session_state.current_urls = urls
            st.session_state.documents = documents
            st.session_state.chunks = chunks
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Show success message
            st.success(f"✅ Successfully processed {len(documents)} page(s) from {len(urls)} URL(s)!")
            
            if failed_urls:
                with st.expander(f"⚠️ {len(failed_urls)} URL(s) failed to load"):
                    for failed in failed_urls:
                        st.write(f"• {failed['url']}")
                        st.write(f"  Error: {failed['error']}")
            
            # Initialize chat
            st.session_state.messages = [
                {"role": "assistant", "content": f"Hello! I've loaded {len(urls)} URL(s). Ask me anything about the content!"}
            ]
            st.rerun()
        else:
            st.error("❌ No documents could be loaded. Please check your URLs and try again.")
            progress_bar.empty()
            status_text.empty()

# ====================== CHAT INTERFACE ======================
# Check if vectorstore exists
if "vectorstore" in st.session_state and st.session_state.vectorstore:
    
    # Initialize LLM
    @st.cache_resource
    def get_llm(temperature):
        return ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=temperature
        )
    
    llm = get_llm(temperature)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": f"Hello! I've loaded {len(st.session_state.current_urls)} URL(s). Ask me anything about the content!"}
        ]
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Ask questions about the loaded content..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Retrieve context
        with st.spinner("🔍 Searching for relevant information..."):
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": k_value})
            docs = retriever.invoke(prompt)
        
        # Display retrieved sources in expander
        with st.expander("📚 Retrieved Sources", expanded=False):
            for i, doc in enumerate(docs):
                source = doc.metadata.get('source', 'Unknown')
                st.markdown(f"**Source {i+1}:** `{source}`")
                st.markdown(f"**Content:** {doc.page_content[:300]}...")
                st.divider()
        
        # Prepare context
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt template
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
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("💭 Generating answer..."):
                try:
                    response = chain.invoke({"context": context, "question": prompt})
                    answer = response.content
                    st.markdown(answer)
                except Exception as e:
                    answer = f"Error: {str(e)}"
                    st.error(answer)
        
        # Add assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    # Show welcome message when no URLs are loaded
    st.info("👋 **Welcome!** To get started:\n\n"
            "1. Enter one or more URLs in the sidebar\n"
            "2. Click 'Process URLs' to load the content\n"
            "3. Start asking questions about the loaded content!\n\n"
            "📝 **Example URLs:**\n"
            "- https://en.wikipedia.org/wiki/Artificial_intelligence\n"
            "- https://en.wikipedia.org/wiki/Machine_learning\n"
            "- Any news article, blog post, or documentation page")

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
            <p style='font-size: 12px;'>
                💡 Pro tip: You can load multiple URLs at once for comprehensive Q&A!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )