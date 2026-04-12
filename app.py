from flask_cors import CORS
from flask import Flask, request, jsonify
import bcrypt
from supabase import create_client
import os
import uuid
from dotenv import load_dotenv
import fitz  # PyMuPDF
import tiktoken
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt )


# Load environment variables
load_dotenv()

app = Flask(__name__)


# Read keys from .env
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
FRONTEND_URL = os.getenv("FRONTEND_URL")
CORS(app, origins=[FRONTEND_URL, "http://localhost:5173"], supports_credentials=True)

app.config["JWT_SECRET_KEY"] = os.getenv("JWT_SECRET_KEY")
app.config["JWT_ACCESS_TOKEN_EXPIRES"] = 3600       # 1 hour
app.config["JWT_REFRESH_TOKEN_EXPIRES"] = 604800   # 7 days
jwt = JWTManager(app)
BLOCKLIST = set()


# Create supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

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
    print(f" Index '{index_name}' created!")
else:
    print(f" Index '{index_name}' already exists.")

client = OpenAI(api_key=OPENAI_KEY)
index = pc.Index(index_name)

def extract_text_from_pdf(pdf_path):
     doc = fitz.open(pdf_path) 
     return "\n".join([page.get_text() for page in doc]) 

def chunk_text(text, max_tokens=800): 
    tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002") 
    tokens = tokenizer.encode(text) 
    return [tokenizer.decode(tokens[i:i+max_tokens]) 
            for i in range(0, len(tokens), max_tokens)] 
    
def get_openai_embeddings(texts): 
    response = client.embeddings.create(model="text-embedding-ada-002", input=texts) 
    return [res.embedding for res in response.data]



@app.route('/upload', methods=['POST'])
@jwt_required()
def upload_file():
    
    email = get_jwt_identity()
    user = supabase.table("users").select("user_id").eq("email", email).execute()

    if not user.data:
        return {"error": "User not found"}, 404

    user_id = user.data[0]["user_id"]

    if 'pdf' not in request.files:
        return 'No file uploaded.', 400

    file = request.files['pdf']

    if file.filename.endswith('.pdf'):
        file_path = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(file_path)

        doc = supabase.table("user_pdfs").insert({
            "user_id": user_id,
            "file_name": file.filename
        }).execute()

        doc_id = doc.data[0]["doc_id"]   
        text = extract_text_from_pdf(file_path)
        chunks = chunk_text(text)
        embeddings = get_openai_embeddings(chunks)
        vectors = [
            (
                f"{doc_id}-chunk-{i}",  
                embeddings[i],
                {
                    "text": chunks[i],
                    "user_id": user_id,
                    "doc_id": doc_id
                }
            )
            for i in range(len(embeddings))
        ]

        index.upsert(
            vectors=vectors,
            namespace=email
        )

        return {
            "message": f"{file.filename} uploaded successfully",
            "doc_id": doc_id,  
            "chunks": len(chunks)
        }

    return "Invalid file. Only PDF allowed.", 400



@app.route('/ask', methods=['POST'])
@jwt_required()
def ask_question():

    email = get_jwt_identity()
    data = request.get_json()
    question = data.get("question")
    doc_id = data.get("doc_id")
    conversation_id = data.get("conversation_id")

    if not question:
        return {"error": "No question provided."}, 400

    if not doc_id:
        return {"error": "doc_id is required"}, 400

    if not conversation_id:
        return {"error": "conversation_id is required"}, 400

    user = supabase.table("users").select("user_id").eq("email", email).execute()
    user_id = user.data[0]["user_id"]
    question_embedding = get_openai_embeddings([question])[0]

    search_response = index.query(
        vector=question_embedding,
        top_k=3,
        include_metadata=True,
        namespace=email,
        filter={
            "doc_id": doc_id,
            "user_id": user_id
        }
    )

    top_chunks = [
        match["metadata"]["text"]
        for match in search_response["matches"]
    ]

    if not top_chunks:
        return {"error": "No relevant data found in this PDF."}, 404

    context = "\n\n---\n\n".join(top_chunks)

    history = supabase.table("chat_history") \
        .select("question, answer") \
        .eq("conversation_id", conversation_id) \
        .order("created_at", desc=False) \
        .execute()

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer only from given context."
        }
    ]

    for chat in history.data:
        messages.append({
            "role": "user",
            "content": chat["question"]
        })
        messages.append({
            "role": "assistant",
            "content": chat["answer"]
        })

    messages.append({
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion: {question}"
    })

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    answer = response.choices[0].message.content.strip()

    if len(history.data) == 0:
        supabase.table("conversations") \
            .update({"title": question[:30]}) \
            .eq("id", conversation_id) \
            .execute()


    supabase.table("chat_history").insert({
        "doc_id": doc_id,
        "conversation_id": conversation_id,
        "question": question,
        "answer": answer
    }).execute()

    return {
        "question": question,
        "answer": answer
    }



@app.route('/history', methods=['GET'])
@jwt_required()
def history():

    email = get_jwt_identity()

    response = supabase.table("chat_history") \
        .select("*") \
        .eq("user_email", email) \
        .order("created_at", desc=True) \
        .execute()

    return jsonify(response.data)



@app.route('/signup', methods=['POST'])
def signup():
    data = request.json  
    f_name = data.get("f_name")
    l_name = data.get("l_name")
    email = data.get("email")
    password = data.get("password")

    if not f_name or not l_name or not email or not password :
        return jsonify({"error" : "all fields are required"}) , 400
    
    hashed_pw = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    
         
    try:
        response = supabase.table("users").insert({
         "f_name": f_name,
         "l_name": l_name,
         "email": email,
         "password": hashed_pw
        }).execute()

    
        return jsonify({"message": "User signed up successfully", "data": response.data}), 201

    except Exception as e:
        if "duplicate key value violates unique constraint" in str(e):
           return jsonify({"error": "Email already exists"}), 400
        return jsonify({"error": str(e)}), 500
    


@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()

    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    try:
        # 1. Get user by email
        response = supabase.table("users").select("*").eq("email", email).execute()

        if not response.data:  # Agar user nahi mila
            return jsonify({"error": "User not found"}), 404

        user_data = response.data[0]
        stored_hashed_pw = user_data["password"].encode("utf-8")

        # 2. Verify password
        if bcrypt.checkpw(password.encode("utf-8"), stored_hashed_pw):
            access_token = create_access_token(identity=email)
            refresh_token = create_refresh_token(identity=email)

            return jsonify({
                "message": "Login successful",
                "access_token": access_token,
                "refresh_token": refresh_token
            }), 200
        else:
            return jsonify({"error": "Invalid password"}), 401

    except Exception as e:
        print("LOGIN ERROR:", str(e))
        return jsonify({"error": str(e)}), 500
    


@app.route('/profile', methods=['GET'])
@jwt_required()  
def profile():
    current_user = get_jwt_identity()
    return jsonify({
        "message": f"Welcome {current_user}! Access token is valid."
    }), 200



@app.route('/refresh', methods=['POST'])
@jwt_required(refresh=True)  
def refresh():
    current_user = get_jwt_identity()
    new_access_token = create_access_token(identity=current_user)
    return jsonify({
        "message": "New access token generated successfully!",
        "access_token": new_access_token
    }), 200



@app.route('/forgot-password' , methods=['POST'])
def forgot_password():
    data = request.get_json()
    email = data.get ("email")

    if not email:
        return jsonify({"error" : "email required"}), 400
    
    try:
        response = supabase.table("users").select("*").eq("email",email).execute()
        if not response.data:
          return jsonify({"error" : "user not found"}),404
        
        import uuid
        token = str(uuid.uuid4())

        from datetime import datetime, timedelta, timezone
        expiry = datetime.now(timezone.utc) + timedelta(minutes=15)

        supabase.table("users").update({ "reset_token" : token, "token_expiry" : expiry.isoformat()}).eq("email",email).execute()
        
        reset_link = f"http://localhost:5173/reset-password?token={token}"
        print(" RESET LINK:",reset_link)

        return jsonify({"message": "reset link sent", "reset_link":reset_link }), 200
    
    except Exception as e:
        return jsonify({"error" : str(e)}), 500   



@app.route('/reset-password' , methods=['POST'])
def reset_password():
    data=request.get_json()
    token = data.get("token")
    new_password = data.get("new_password")

    if not token or not new_password:
        return jsonify({"error": "all fields required"}), 400
    
    try:
        response = supabase.table("users").select("*").eq("reset_token", token).execute()

        if not response.data:
            return jsonify({"error":"invalid token"}), 400
        
        user = response.data[0]
        from datetime import datetime, timezone

        expiry_time = user["token_expiry"]

        if not expiry_time:
           return jsonify({"error": "invalid token or used token"}), 400

        if datetime.now(timezone.utc) > datetime.fromisoformat(expiry_time):
           return jsonify({"error": "token expired"}), 400
        
        import bcrypt 
        hashed_pw = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        supabase.table("users").update({"password" : hashed_pw, "reset_token": None, "token_expiry": None}).eq("reset_token", token).execute()
        return jsonify({"message" : "password reset successfully"}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@jwt.token_in_blocklist_loader
def check_if_token_revoked(jwt_header, jwt_payload):
    return jwt_payload["jti"] in BLOCKLIST



@app.route('/logout', methods=['POST'])
@jwt_required()
def logout():
    current_user = get_jwt_identity()
    jti= get_jwt()["jti"]
    BLOCKLIST.add(jti)
    return jsonify({"message" : "successfully logout"}), 200



@app.route("/conversation/start", methods=["POST"])
@jwt_required()
def start_conversation():

    email = get_jwt_identity()
    data = request.get_json()
    doc_id = data.get("doc_id")

    user = supabase.table("users") \
        .select("user_id") \
        .eq("email", email) \
        .execute()

    user_id = user.data[0]["user_id"]

    response = supabase.table("conversations").insert({
        "user_id": user_id,
        "doc_id": doc_id,
        "title": "New Chat"
    }).execute()

    return {
        "message": "Conversation started",
        "conversation_id": response.data[0]["id"]
    }



@app.route("/pdfs", methods=["GET"])
@jwt_required()
def get_pdfs():
    email = get_jwt_identity()
    user = supabase.table("users").select("user_id").eq("email", email).execute()
    user_id = user.data[0]["user_id"]

    docs = supabase.table("user_pdfs") \
        .select("*") \
        .eq("user_id", user_id) \
        .execute()

    return jsonify(docs.data), 200



@app.route("/pdfs/<doc_id>", methods=["DELETE"])
@jwt_required()
def delete_pdf(doc_id):
    email = get_jwt_identity()
    user = supabase.table("users").select("user_id").eq("email", email).execute()
    user_id = user.data[0]["user_id"]

    supabase.table("user_pdfs") \
        .delete() \
        .eq("doc_id", doc_id) \
        .eq("user_id", user_id) \
        .execute()

    index.delete(
        filter={
            "doc_id": doc_id,
            "user_id": user_id
        }
    )

    return jsonify({"message": "PDF deleted"}), 200



@app.route("/pdfs/<doc_id>", methods=["PUT"])
@jwt_required()
def rename_pdf(doc_id):
    email = get_jwt_identity()
    user = supabase.table("users").select("user_id").eq("email", email).execute()
    user_id = user.data[0]["user_id"]
    data = request.get_json()
    new_name = data.get("file_name")

    supabase.table("user_pdfs") \
        .update({"file_name": new_name}) \
        .eq("doc_id", doc_id) \
        .eq("user_id", user_id) \
        .execute()

    return jsonify({"message": "Renamed successfully"}), 200



@app.route("/conversations", methods=["GET"])
@jwt_required()
def get_conversations():

    email = get_jwt_identity()

    user = supabase.table("users") \
        .select("user_id") \
        .eq("email", email) \
        .execute()

    user_id = user.data[0]["user_id"]

    response = supabase.table("conversations") \
        .select("*") \
        .eq("user_id", user_id) \
        .order("created_at", desc=True) \
        .execute()

    return jsonify(response.data)



@app.route("/conversations/<conversation_id>", methods=["GET"])
@jwt_required()
def get_conversation_messages(conversation_id):
    response = supabase.table("chat_history") \
        .select("*") \
        .eq("conversation_id", conversation_id) \
        .order("created_at", desc=False) \
        .execute()

    return jsonify(response.data)



@app.route('/')
def home():
    return " API is running. Use /upload to store PDFs and /ask to query them."



if __name__ == '__main__':
    app.run(debug=True)

