# PDF Chat with AWS Bedrock and Multiple LLMs

This project implements a PDF question-answering system using AWS Bedrock, Langchain, and Streamlit. It allows users to upload PDF documents, process them, and ask questions about their content using different large language models.

## Features

- PDF Processing: Ingests and processes PDF files from a specified directory
- Vector Store Creation: Generates and stores document embeddings using FAISS
- Multiple LLM Support: Integrates Mistral-7B and LLaMA3-70B models via AWS Bedrock
- Interactive UI: Provides a user-friendly interface with Streamlit
- Question Answering: Retrieves relevant information and generates responses to user queries

## How It Works

1. Users can update the vector store with new PDF documents
2. The system processes PDFs, splits them into chunks, and creates embeddings
3. Users can input questions about the PDF content
4. The system retrieves relevant context using FAISS and generates responses using the chosen LLM
5. Responses are displayed in the Streamlit interface

## Tech Stack Used

- AWS Bedrock
- Langchain
- Streamlit
- FAISS
- Boto3
- PyPDF
- Python

## Note

This project requires proper AWS Bedrock setup and credentials. Ensure you have the necessary permissions and configuration before running the application.

**Sample Output with Mistral Model:**

![sample output 1](https://github.com/user-attachments/assets/03498921-5f0b-4571-b635-2ba278ab72d6)

**Sample Output with Llama3 Model:**

![sample output 2](https://github.com/user-attachments/assets/d11c4bae-13a1-4fb2-809b-f8a9ee672082)
