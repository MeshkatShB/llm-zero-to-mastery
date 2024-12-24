# llm-zero-to-mastery

## About

In this repo...

## Structure

### 1. local-rag [`local_rag.py`]

In this part, I implemented a simple local-rag which will get the data from the ./data directory and then implements the following:

1.  It sets the `pdf_directory` variable to `data`. You should replace this with the actual path to your PDF directory.
2.  The `extract_text_from_pdfs()` function is called with the `pdf_directory` as an argument. This function extracts text from all the PDF files in the specified directory and returns a list of documents.
3.  The `create_vector_database()` function is called with the `documents` list as an argument. This function creates a vector database using the extracted documents and returns the client, collection, and model objects.
4.  A message is printed to indicate that the PDF documents have been processed and stored.
5.  An infinite loop is started using the `while True` statement. Inside the loop:

    - The user is prompted to enter a question. If the user types `exit`, the loop breaks, and the program terminates.

    - The `retrieve_relevant_documents()` function is called with the user's query, collection, and model objects as arguments. This function retrieves relevant documents from the vector database based on the query.
    - The retrieved documents are joined into a single string using the `join()` method with a newline character as the separator.
    - The `generate_response_with_ollama()` function is called with the context (joined retrieved documents) and the user's query as arguments. This function generates a response using the OpenAI's Ollama model.
    - The generated response is printed to the console.

### 2. fine-tuning gpt2

- **Imports**  
   Libraries required for fine-tuning `GPT-2`.
- **Check GPU Availability**  
   Ensures GPU is used if available.
- **Define Model Name**  
   Specifies the base model (`GPT-2`).
- **Load the Tokenizer and Model**  
   Loads `GPT-2` and its tokenizer from `Hugging Face`.
- **Add Special Tokens**  
   Adds padding and resizes token embeddings.
- **Prepare the Dataset**
  - **Dataset Loading**: Loads text datasets for training and testing.
  - **Tokenization**: Prepares and tokenizes data for training.
  - **Data Collator**: Formats batches for training.
- **Define Training Arguments**  
   Configures hyperparameters for training.
- **Define the Trainer**  
   Initializes the `Trainer` for training and evaluation.
- **Fine-Tune the Model**  
   Trains the model on the provided dataset.
- **Save the Fine-Tuned Model**  
   Saves the trained model and tokenizer locally.
- **Test the Fine-Tuned Model**  
   Generates text from the fine-tuned model using custom prompts.
- **Evaluate the Model**  
   Tests the model with multiple prompts to assess its performance.

### 3. MORE WILL COME OUT SOON
