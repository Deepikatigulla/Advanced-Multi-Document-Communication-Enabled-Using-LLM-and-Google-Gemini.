import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from deep_translator import GoogleTranslator
from docx import Document  # Import for handling .docx files

# Load environment variables
load_dotenv()

# Configure Google Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Function to extract text from PDF or Word documents
def get_document_text(documents):
    text = ""
    for doc_file in documents:
        _, file_extension = os.path.splitext(doc_file.name)
        if file_extension.lower() == ".pdf":
            pdf_reader = PdfReader(doc_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
        elif file_extension.lower() == ".docx":
            doc = Document(doc_file)
            for para in doc.paragraphs:
                text += para.text
        else:
            st.warning(f"Unsupported file format: {file_extension}. Only PDF and DOCX are supported.")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save a FAISS vector store from text chunks
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create a conversational chain for QA
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to generate a summary for a given text
def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    summary_prompt = f"Summarize the following document text:\n\n{text}"
    response = model.predict(text=summary_prompt)
    return response

# Function to handle user input and generate a response
def user_input(user_question, conversation_history):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    # Update context with conversation history
    context = "\n".join([f"Q: {q}\nA: {a}" for q, a in conversation_history])
    context += f"\n\nQ: {user_question}"
    response = chain({"input_documents": docs, "context": context, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to display the text input box for the user's question
def display_query_input():
    user_question = st.text_input("Ask a Question from the PDF Files", value=st.session_state.get("user_question", ""))
    return user_question

# Translation function
def translate_text(text, target_language):
    translator = GoogleTranslator(source='auto', target=target_language)
    translation = translator.translate(text)
    return translation

languages = {"French": "fr", "Spanish": "es", "German": "de", "Chinese (Simplified)": "zh-CN",
             "Telugu": "te", "Hindi": "hi", "Tamil": "ta"}

# Main function to set up the Streamlit app
def main():
    st.set_page_config("Chat PDF")
    st.header("Multi-Document Answering")

    # Initialize session state variables
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    if "user_question" not in st.session_state:
        st.session_state.user_question = ""

    if "document_summaries" not in st.session_state:
        st.session_state.document_summaries = []

    with st.sidebar:
        st.title("Menu:")
        st.subheader("Upload PDF or DOCX Files:")
        documents = st.file_uploader("Upload your PDF or DOCX Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_document_text(documents)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                
                # Generate summaries for each document and store them in session state
                st.session_state.document_summaries = []
                for doc_file in documents:
                    doc_text = get_document_text([doc_file])
                    summary = generate_summary(doc_text)
                    st.session_state.document_summaries.append((doc_file.name, summary))
                
            st.success("Done")

    # Display document summaries
    st.subheader("Document Summaries:")
    for doc_name, summary in st.session_state.document_summaries:
        st.write(f"**Summary of {doc_name}:**")
        st.write(summary)
    

    user_question = display_query_input()

    if user_question:
        response = user_input(user_question, st.session_state.conversation_history)
        st.session_state.conversation_history.append((user_question, response))
        st.write("Reply: ", response)

        # Add language selection and translation
        selected_language = st.selectbox("Translate to", list(languages.keys()), index=0)
        if selected_language:
            translated_response = translate_text(str(response), languages[selected_language])
            st.write("Translated Reply: ", translated_response)

        st.session_state.user_question = ""  # Reset the user_question after processing

if __name__ == "__main__":
    main()
    



