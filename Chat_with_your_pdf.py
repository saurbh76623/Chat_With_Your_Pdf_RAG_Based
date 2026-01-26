# # Install necessary dependencies
# !pip install langchain
# !pip install pymupdf
# !pip install sentence-transformers
# !pip install google-generativeai
# !pip install numpy


# import necessary library 
from sentence_transformers import SentenceTransformer   # used for sentence embeddings
import fitz  # PyMuPDF  # for extracting text from pdf files 
from langchain.text_splitter import RecursiveCharacterTextSplitter  # for splitting text into chunks 
import numpy as np
import google.generativeai as genai   
import os



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
def sentence_encode(sentences):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = model.encode(sentences)
    return embeddings


# 4. cal cosine similarity between two vectors
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))



if __name__ == "__main__":
    # Get API key from environment variable
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY environment variable is not set.")
        print("Please set it before running the script:")
        print("  export GOOGLE_API_KEY='your-api-key-here'")
        exit(1)
    
    # Configure the API once at the start
    genai.configure(api_key=GOOGLE_API_KEY)
    
    # Get PDF path from user or environment variable
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        pdf_path = input("Enter the path to your PDF file: ")
    
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        exit(1)
    
    print(f"Loading PDF: {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    chunks = split_text_into_chunks(text)
    chunk_vectors = sentence_encode(chunks)
    print(f"PDF loaded successfully! Created {len(chunks)} chunks.\n")

    while True:
        # Get user input
        query = input("\nEnter your question (or 'quit' to exit): ")

        if query.lower() == 'quit':
            break

        query_vector = sentence_encode([query])
        top_k = 3

        similarities = []
        for idx, chunk_vec in enumerate(chunk_vectors):
            sim = cosine_similarity(chunk_vec, query_vector[0])
            similarities.append((sim, idx))

        print("Similarities:", similarities)

        print("==" * 20)

        # Sort by similarity descending and get top_k indices
        top_chunks = sorted(similarities, reverse=True)[:top_k]
        top_indices = [idx for _, idx in top_chunks]

        print("Top chunk indices:", top_indices)

        new_context = ""
        for i in top_indices:
            new_context += chunks[i] + "\n"

        prompt_template = f"""You are a helpful assistant. Answer the question based on the context provided.
        Context: {new_context}
        Question: {query}"""

        try:
            # Initialize the model correctly
            model = genai.GenerativeModel('gemini-2.0-flash')

            # Generate response with the actual prompt
            response = model.generate_content(prompt_template)
            print("\nResponse:")
            print(response.text)
        except Exception as e:
            print(f"Error generating response: {str(e)}")