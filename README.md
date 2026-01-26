# Chat With Your PDF - RAG Based

A Retrieval-Augmented Generation (RAG) based chatbot that allows you to have conversational interactions with PDF documents using Google's Gemini 2.0 Flash model.

**Link of Project** => https://colab.research.google.com/drive/18YUJnRuVxsBYdLUUj5hEfruaB_8aCPzR?usp=sharing

![image](https://github.com/user-attachments/assets/da1c666a-1d20-48de-8daa-fe0f415591b9)

## Setup Instructions

### 1. Install Dependencies

```bash
pip install langchain-openai
pip install openai
pip install langchain-community
pip install pymupdf
pip install sentence-transformers
pip install google-generativeai
pip install numpy
```

### 2. Get Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Set Environment Variable

**Linux/Mac:**
```bash
export GOOGLE_API_KEY='your-api-key-here'
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY='your-api-key-here'
```

**Using .env file (optional):**
```bash
cp .env.example .env
# Edit .env file and add your API key
```

### 4. Run the Application

```bash
python Chat_with_your_pdf.py
```

The application will prompt you to enter the path to your PDF file, then you can start asking questions!

## How It Works
The core functionality of this tool revolves around enabling conversational interactions with any PDF by leveraging a series of natural language processing steps powered by Gemini 2.0 Flash. Here’s a breakdown of how the system works under the hood:

### 1. Text Extraction:
The first step involves extracting the raw text content from the uploaded PDF file. This provides the foundational data for all subsequent processing.

### 2. Chunking the Content:
Once the text is extracted, it is segmented into smaller, manageable chunks. This is essential to ensure context relevance and to optimize the processing load for downstream embedding and retrieval operations.

### 3. Vectorization (Embeddings):
Each of the text chunks is then converted into a list format and transformed into numerical vector representations — commonly known as embeddings. These embeddings capture the semantic meaning of the text and enable efficient comparison with user queries.

### 4. Query Processing and Embedding:
When a user submits a query, it is similarly converted into an embedding. This allows the system to compare the user’s query against the preprocessed text chunks using vector similarity.
### 5. Similarity Calculation (Cosine Similarity):
To find the most relevant content, the cosine similarity between the query embedding and each of the text chunk embeddings is computed. This measures how close each chunk is to the user’s intent.
### 6. Top-k Chunk Selection:
The system then selects the top k chunks (e.g., top 3) with the highest similarity scores. These chunks are considered the most relevant for answering the user’s query.
### 7. Response Generation Using Gemini 2.0 Flash:
Finally, the selected chunks are passed to the Gemini 2.0 Flash model along with the user’s query. The model uses this focused context to generate a coherent, accurate, and relevant response.
