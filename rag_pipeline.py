from embeddings import get_embedding
from vector_store import VectorStore
from transformers import pipeline
from document_loader import is_table

# load LLM
generator = pipeline("text2text-generation", model="google/flan-t5-large")

def chunk_text(text, chunk_size=150, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def build_vector_store(pages):

    embeddings = []
    texts = []
    page_refs = []

    for page in pages:

        text = page["text"]
        page_number = page["page"]
        chunks = chunk_text(text)

        for chunk in chunks:
            if is_table(chunk):
                continue

            emb = get_embedding(chunk)
            embeddings.append(emb)
            texts.append(chunk)
            page_refs.append(page_number)

    dimension = len(embeddings[0])

    store = VectorStore(dimension)

    store.add_embeddings(embeddings, texts, page_refs)

    return store


def retrieve_answer(query, vector_store, history):

    query_embedding = get_embedding(query)

    results = vector_store.search(query_embedding)

    top_chunks = results[:3]

    context = "\n\n---\n\n".join([r["text"] for r in top_chunks])

    pages = list(set([r["page"] for r in top_chunks]))

    history_text = "\n".join(history[-4:])

    prompt = f"""
You are answering questions about a financial report.

Conversation history:
{history_text}

Information from the report:
{context}

User question:
{query}

Explain clearly and simply.

Answer:
"""

    response = generator(
        prompt,
        max_length=200,
        do_sample=True,
        temperature=0.7
    )

    answer = response[0]["generated_text"]

    citation = f"\n\nSources: pages {pages}"

    return answer + citation