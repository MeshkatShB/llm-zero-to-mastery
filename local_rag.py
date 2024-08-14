from utils import extract_text_from_pdfs, retrieve_relevant_documents, generate_response_with_ollama
from embeddings import create_vector_database


def main():
    pdf_directory = 'data'  # Replace with your PDF directory path
    documents = extract_text_from_pdfs(pdf_directory)
    client, collection, model = create_vector_database(documents)

    print("PDF documents have been processed and stored.")

    while True:
        query = input("\nEnter your question (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        retrieved_docs = retrieve_relevant_documents(query, collection, model)
        context = "\n".join(retrieved_docs)
        response = generate_response_with_ollama(context, query)
        print(f"\nResponse: {response}")


if __name__ == "__main__":
    main()
