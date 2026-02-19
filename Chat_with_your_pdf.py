# # Install necessary langchain components
# !pip install langchain-openai
# !pip install openai
# !pip install langchain-community
# !pip install pymupdf


# import necessary library 
from langchain_openai import ChatOpenAI    # open ai LLM model
from langchain import PromptTemplate       # Prompt template for the llm model
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun   # tool for searching the web 
from langchain.agents import create_react_agent , AgentExecutor     # agents for the llm 
from langchain_core.tools import tool
from sentence_transformers import SentenceTransformer   # used for sentence embeddings
import fitz  # PyMuPDF  # for extracting text from pdf files 
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for splitting text into chunks 
import numpy as np
import google.generativeai as genai   
import os
import heapq  # for efficient top-k selection



# 1 extract text from a pdf file 
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()
    return text

# 2 split the text into chunks 
def split_text_into_chunks(text, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_text(text)


# 3 encode the sentence using sentence transformers 
# OPTIMIZATION: Initialize model once globally to avoid reloading on every call
_embedding_model = None

def get_embedding_model():
    """Get or initialize the embedding model (singleton pattern)"""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _embedding_model

def sentence_encode(sentences):
    model = get_embedding_model()
    embeddings = model.encode(sentences)
    return embeddings


# 4. calculate cosine similarity between vectors
# OPTIMIZATION: Vectorized computation for batch similarity calculation
def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def batch_cosine_similarity(query_vector, chunk_vectors):
    """Calculate cosine similarity between query and all chunks efficiently"""
    query_vector = np.array(query_vector).reshape(1, -1)
    chunk_vectors = np.array(chunk_vectors)
    
    # Vectorized dot product
    dot_products = np.dot(chunk_vectors, query_vector.T).flatten()
    
    # Calculate norms
    query_norm = np.linalg.norm(query_vector)
    chunk_norms = np.linalg.norm(chunk_vectors, axis=1)
    
    # Calculate similarities
    similarities = dot_products / (chunk_norms * query_norm)
    return similarities



if __name__ == "__main__":
    pdf_path = "/content/Data_Science_Saurabh_Resume (1) (5).pdf"
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    chunk_vectors = []
    chunk_vectors = sentence_encode(chunks)

    # OPTIMIZATION: Initialize API and model once outside the loop
    # Use environment variable for API key (security best practice)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyABtGiltCFuqqdh6Wbcl3MVVVoVu2ZCKyU")
    
    # Configure the API once
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Initialize the model once
    model = genai.GenerativeModel('gemini-2.0-flash')

    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ")

        if query.lower() == 'quit':
            break

        # Encode query once
        query_vector = sentence_encode([query])[0]
        top_k = 3

        # OPTIMIZATION: Use vectorized similarity calculation instead of loop
        similarities = batch_cosine_similarity(query_vector, chunk_vectors)
        
        # Convert to list of tuples (similarity, index) for compatibility
        similarity_tuples = [(sim, idx) for idx, sim in enumerate(similarities)]

        print("Similarities:", similarity_tuples)

        print("==" * 20)

        # OPTIMIZATION: Use heapq.nlargest for efficient top-k selection
        top_chunks = heapq.nlargest(top_k, similarity_tuples)
        top_indices = [idx for _, idx in top_chunks]

        print("Top chunk indices:", top_indices)

        # OPTIMIZATION: Use list comprehension and join instead of string concatenation
        new_context = "\n".join([chunks[i] for i in top_indices])

        prompt_template = f"""You are a helpful assistant. Answer the question based on the context provided.
        Context: {new_context}
        Question: {query}"""

        try:
                # Generate response with the actual prompt
                response = model.generate_content(prompt_template)
                print("\nResponse:")
                print(response.text)
        except Exception as e:
                print(f"Error generating response: {str(e)}")