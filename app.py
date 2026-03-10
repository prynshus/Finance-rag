import streamlit as st
from document_loader import load_pdf
from rag_pipeline import build_vector_store, retrieve_answer


st.title("Financial Report Q&A (RAG System)")

uploaded_file = st.file_uploader("Upload Financial Report PDF", type="pdf")

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file:

    st.write("Processing document...")

    pages = load_pdf(uploaded_file)

    vector_store = build_vector_store(pages)

    st.success("Document processed!")

    query = st.text_input("Ask a question about the report")

    if query:

        answer = retrieve_answer(query, vector_store, st.session_state.history)

        st.session_state.history.append(f"User: {query}")
        st.session_state.history.append(f"Assistant: {answer}")

    for message in st.session_state.history:
        st.write(message)
