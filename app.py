from flask import Flask, request
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Read keys from .env
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_KEY)

index_name = "pdf-embeddings"
embedding_dimension = 1536

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embedding_dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )
    print(f"✅ Index '{index_name}' created!")
else:
    print(f"ℹ️ Index '{index_name}' already exists.")

index = pc.Index(index_name)

# OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'pdf' not in request.files:
        return '❌ No file uploaded.', 400

    file = request.files['pdf']
    if file.filename.endswith('.pdf'):
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = get_openai_embeddings(chunks)

        vectors = [
            (f"{file.filename}-chunk-{i}", embeddings[i], {"text": chunks[i]})
            for i in range(len(embeddings))
        ]
        index.upsert(vectors=vectors)

        return {
            "message": f"✅ '{file.filename}' uploaded & stored in Pinecone.",
            "text_length": len(text),
            "chunks": len(chunks),
            "pinecone_index": index_name
        }
    return '❌ Invalid file. Only PDF allowed.', 400

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, max_tokens=800):
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

def get_openai_embeddings(texts):
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts)
    return [res.embedding for res in response.data]

@app.route('/ask', methods=['POST'])
def ask_question():
    question = request.data.decode('utf-8').strip()
    if not question:
        return {"error": "No question provided."}, 400

    question_embedding = get_openai_embeddings([question])[0]
    search_response = index.query(vector=question_embedding, top_k=3, include_metadata=True)
    top_chunks = [match.metadata['text'] for match in search_response.matches]

    if not top_chunks:
        return {"error": "No relevant data found in Pinecone."}, 404

    context = "\n\n---\n\n".join(top_chunks)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer the question using only the given context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"}
        ]
    )
    return {"question": question, "answer": response.choices[0].message.content.strip()}

@app.route('/')
def home():
    return "✅ API is running. Use /upload to store PDFs and /ask to query them."

if __name__ == '__main__':
    app.run(debug=True)
