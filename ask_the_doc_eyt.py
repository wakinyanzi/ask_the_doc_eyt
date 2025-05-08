import streamlit as st
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, OpenAI

# Set up Streamlit app
st.title("Ask the Doc 📄")
st.write("Upload a document and ask questions about its content")

# Sidebar for API key input
with st.sidebar:
    st.header("Settings")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    st.markdown("[Get an OpenAI API key](https://platform.openai.com/account/api-keys)")

# File uploader
uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx", "md"])

# Question input
question = st.text_input("Ask a question about the document")

if uploaded_file and question and openai_api_key:
    try:
        # Read file content
        if uploaded_file.name.endswith('.pdf'):
            from PyPDF2 import PdfReader
            pdf_reader = PdfReader(uploaded_file)
            documents = [page.extract_text() for page in pdf_reader.pages]
        else:
            documents = [uploaded_file.read().decode()]
        
        # Split documents into chunks
        text_splitter = CharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200
        )
        texts = text_splitter.create_documents(documents)
        
        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings)
        
        # Create retriever
        retriever = db.as_retriever()
        
        # Create QA chain
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key, temperature=0),
            chain_type="stuff",
            retriever=retriever
        )
        
        # Get answer
        answer = qa.run(question)
        
        # Display answer
        st.subheader("Answer")
        st.write(answer)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
elif question and not uploaded_file:
    st.warning("Please upload a document first")
elif uploaded_file and not question:
    st.warning("Please enter a question")
elif question and uploaded_file and not openai_api_key:
    st.warning("Please enter your OpenAI API key")