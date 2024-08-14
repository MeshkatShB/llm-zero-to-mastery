from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions


def create_vector_database(documents):
    # Initialize ChromaDB client`
    client = chromadb.Client()

    # Create a collection
    collection = client.create_collection("pdf_documents")

    # Initialize the SentenceTransformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Add documents to the collection
    for idx, doc in enumerate(documents):
        embedding = model.encode(doc['text'])
        collection.add(
            documents=[doc['text']],
            embeddings=[embedding],
            metadatas=[{"source": doc['source']}],
            ids=[str(idx)]
        )
    return client, collection, model
