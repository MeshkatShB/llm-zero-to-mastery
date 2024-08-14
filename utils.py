import os
from PyPDF2 import PdfReader
import subprocess


def extract_text_from_pdfs(pdf_directory):
    documents = []
    for filename in os.listdir(pdf_directory):
        if filename.lower().endswith('.pdf'):
            pdf_path = os.path.join(pdf_directory, filename)
            reader = PdfReader(pdf_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append({"text": text, "source": filename})
    return documents


def retrieve_relevant_documents(query, collection, model, k=3):
    query_embedding = model.encode(query)
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    retrieved_texts = [doc for doc in results['documents'][0]]
    return retrieved_texts


def generate_response_with_ollama(context, query):
    prompt = f"""
    Context: {context}

    Question: {query}

    Answer:
    """
    # Use Ollama's CLI to generate the response
    result = subprocess.run(
        ["ollama", "generate", "--prompt", prompt, "llama2"],  # Replace 'llama2' with the model you've set up
        capture_output=True,
        text=True
    )
    return result.stdout.strip()
