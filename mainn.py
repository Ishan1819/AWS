# import os
# import pickle
# import hashlib
# import PyPDF2
# import chromadb
# from chromadb.utils import embedding_functions
# from dotenv import load_dotenv
# import google.generativeai as genai
# from fastapi import FastAPI, File, UploadFile, Form
# from fastapi.responses import JSONResponse
# from fastapi.middleware.cors import CORSMiddleware

# # -------------------- Load Environment --------------------
# load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# if not GEMINI_API_KEY:
#     raise ValueError("❌ Please set GEMINI_API_KEY in your .env file")

# genai.configure(api_key=GEMINI_API_KEY)

# # -------------------- FastAPI Setup --------------------
# app = FastAPI(title="Research Paper Assistant API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -------------------- Initialize ChromaDB --------------------
# chroma_client = chromadb.Client(
#     settings=chromadb.config.Settings(anonymized_telemetry=False)
# )

# # Create or fetch collection
# collection_name = "research_paper_collection"
# try:
#     collection = chroma_client.get_collection(name=collection_name)
# except:
#     collection = chroma_client.create_collection(
#         name=collection_name,
#         embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="all-MiniLM-L6-v2"
#         )
#     )

# # -------------------- Utility Functions --------------------

# def read_pdf(pdf_file_path):
#     """Extract text from PDF file."""
#     pdf_text = []
#     with open(pdf_file_path, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         for i, page in enumerate(reader.pages):
#             text = page.extract_text()
#             if text:
#                 pdf_text.append({"page": i + 1, "content": text})
#     return pdf_text


# def chunk_text(text, chunk_size=300, overlap=30):
#     """Split text into overlapping chunks."""
#     words = text.split()
#     chunks = []
#     start = 0
#     while start < len(words):
#         end = start + chunk_size
#         chunk = " ".join(words[start:end])
#         chunks.append(chunk)
#         start += chunk_size - overlap
#     return chunks


# def get_pdf_hash(pdf_bytes):
#     """Generate a unique hash for caching."""
#     return hashlib.md5(pdf_bytes).hexdigest()


# def reset_collection():
#     """Delete all previous embeddings before new upload."""
#     global collection
#     try:
#         chroma_client.delete_collection(name=collection_name)
#     except:
#         pass

#     collection = chroma_client.create_collection(
#         name=collection_name,
#         embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
#             model_name="all-MiniLM-L6-v2"
#         )
#     )


# def store_pdf(file_path, file_name, pdf_bytes):
#     """Store or load cached PDF embeddings."""
#     pdf_hash = get_pdf_hash(pdf_bytes)
#     cache_path = f"cache_{pdf_hash}.pkl"

#     # Load from cache if available
#     if os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             chunks, metadatas, ids = pickle.load(f)
#         collection.add(documents=chunks, metadatas=metadatas, ids=ids)
#         return len(chunks)

#     # Read PDF and chunk it
#     pdf_text = read_pdf(file_path)
#     all_chunks, metadatas, ids = [], [], []

#     for page_data in pdf_text:
#         page_chunks = chunk_text(page_data["content"])
#         for idx, chunk in enumerate(page_chunks):
#             all_chunks.append(chunk)
#             metadatas.append({"page": page_data["page"]})
#             ids.append(f"{file_name}_p{page_data['page']}_{idx}")

#     # Store in Chroma and cache
#     collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
#     with open(cache_path, "wb") as f:
#         pickle.dump((all_chunks, metadatas, ids), f)

#     return len(all_chunks)


# def query_gemini(question, n_results=2):
#     """Query ChromaDB and get Gemini-based answer."""
#     results = collection.query(query_texts=[question], n_results=n_results)
#     contexts = results["documents"][0]
#     metadatas = results["metadatas"][0]

#     context_text = ""
#     for i, ctx in enumerate(contexts):
#         context_text += f"\n[Source p.{metadatas[i]['page']}] {ctx}"

#     prompt = f"""
# You are a research paper assistant.
# Answer the user question using only the context below. 
# If user asks to summarize, then summarize every key point in the paper.
# If relevant context is insufficient, add your own general knowledge.
# Include citations as [Source p.X].

# Context:
# {context_text}

# Question: {question}
# Answer:
# """
#     model = genai.GenerativeModel("gemini-2.0-flash")
#     response = model.generate_content(prompt)
#     return response.text


# # -------------------- ROUTERS --------------------

# @app.post("/upload")
# async def upload_pdf(file: UploadFile = File(...)):
#     """Upload and store PDF embeddings (resets old ones)."""
#     try:
#         file_bytes = await file.read()

#         # Delete previous embeddings
#         reset_collection()

#         file_path = f"temp_{file.filename}"
#         with open(file_path, "wb") as f:
#             f.write(file_bytes)

#         total_chunks = store_pdf(file_path, file.filename, file_bytes)
#         os.remove(file_path)

#         return JSONResponse({"message": f"✅ Stored {total_chunks} chunks from '{file.filename}'"})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


# @app.post("/summarize")
# async def summarize_paper():
#     """Summarize the paper contents using Gemini."""
#     try:
#         question = "Summarize the entire paper in detailed key points."
#         summary = query_gemini(question)
#         return JSONResponse({"summary": summary})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


# @app.post("/chat")
# async def chat_with_paper(question: str = Form(...)):
#     """Chatbot interaction for user questions."""
#     try:
#         answer = query_gemini(question)
#         return JSONResponse({"answer": answer})
#     except Exception as e:
#         return JSONResponse({"error": str(e)}, status_code=500)


# # -------------------- MAIN ENTRY --------------------
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("mainn:app", host="127.0.0.1", port=8000, reload=True)
import os
import re
import pickle
import hashlib
import fitz  # PyMuPDF
import requests
from bs4 import BeautifulSoup
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
from collections import defaultdict, Counter
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Disable telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "false"

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in your .env file")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# Download NLTK resources
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("stopwords", quiet=True)

# -------------------------------------------------------------
# FastAPI App Setup
# -------------------------------------------------------------
app = FastAPI(title="Privacy Policy Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------
# ChromaDB Setup
# -------------------------------------------------------------
chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)
collection_name = "privacy_policy_collection"

try:
    collection = chroma_client.get_collection(name=collection_name)
except:
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )

conversation_sessions = defaultdict(list)

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------

def extract_text_from_pdf(pdf_file_path):
    """Extract text from PDF using PyMuPDF (fitz)."""
    text_data = []
    with fitz.open(pdf_file_path) as doc:
        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_data.append({"page": page_num + 1, "content": text})
    return text_data


def extract_text_from_website(url):
    """Extract privacy policy text from a website using BeautifulSoup."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Remove scripts, styles, and navigation elements
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Extract paragraph text
        paragraphs = [p.get_text(separator=" ", strip=True) for p in soup.find_all("p")]
        text = " ".join(paragraphs)
        return [{"page": 1, "content": text}]
    except Exception as e:
        raise ValueError(f"Failed to extract text from website: {e}")


def chunk_text(text, chunk_size=300, overlap=30):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def get_file_hash(data_bytes):
    """Generate a unique hash for caching."""
    return hashlib.md5(data_bytes).hexdigest()


def reset_collection():
    """Reset ChromaDB collection."""
    global collection
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass
    collection = chroma_client.create_collection(
        name=collection_name,
        embedding_function=embedding_functions.DefaultEmbeddingFunction()
    )


def store_text_data(file_path, file_name, data_bytes, source_type="pdf"):
    """Store or load cached embeddings for PDF or website."""
    file_hash = get_file_hash(data_bytes)
    cache_path = f"cache_{file_hash}.pkl"

    # Load from cache if available
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            chunks, metadatas, ids = pickle.load(f)
        collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        return len(chunks)

    # Extract text
    if source_type == "pdf":
        text_data = extract_text_from_pdf(file_path)
    elif source_type == "web":
        text_data = extract_text_from_website(file_name)  # here file_name is URL
    else:
        raise ValueError("Unsupported source type")

    # Chunk and store
    all_chunks, metadatas, ids = [], [], []
    for page_data in text_data:
        page_chunks = chunk_text(page_data["content"])
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            metadatas.append({"page": page_data["page"], "source": source_type})
            ids.append(f"{file_name}_p{page_data['page']}_{idx}")

    collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
    with open(cache_path, "wb") as f:
        pickle.dump((all_chunks, metadatas, ids), f)
    return len(all_chunks)


def query_gemini(question, chat_history, n_results=2):
    """Query ChromaDB + Gemini for privacy policy context."""
    results = collection.query(query_texts=[question], n_results=n_results)
    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_text = ""
    for i, ctx in enumerate(contexts):
        context_text += f"\n[Source p.{metadatas[i]['page']}] {ctx}"

    recent_history = "\n".join(
        [f"User: {turn['user']}\nAssistant: {turn['assistant']}" for turn in chat_history[-3:]]
    )

    prompt = f"""
You are an expert assistant specialized in analyzing Privacy Policies.
Answer questions based strictly on the provided policy context.
Use a formal and legally clear tone — no assumptions or filler text.
Include the reference page number (e.g., Source p.X) when possible.
Never ask the user questions or use casual phrasing.
Write coherent paragraphs — no bullet points or markdown formatting.

--- Previous Conversation ---
{recent_history}

--- Relevant Policy Context ---
{context_text}

User Question: {question}
Answer:
"""

    model = genai.GenerativeModel("gemini-2.0-flash")
    response = model.generate_content(prompt)
    return response.text


# TECH_KEYWORDS = [
#     "data", "collection", "processing", "storage", "consent",
#     "third-party", "cookies", "retention", "security", "encryption",
#     "rights", "GDPR", "CCPA", "personal information", "user data",
#     "policy", "privacy", "access", "sharing", "deletion"
# ]


# def extract_technical_phrases(text, top_n=25):
#     """Extract key technical/legal phrases from privacy policy."""
#     stop_words = set(stopwords.words("english"))
#     sentences = nltk.sent_tokenize(text)
#     candidate_phrases = []

#     for sent in sentences:
#         tokens = [t for t in word_tokenize(sent) if t.isalnum()]
#         tagged = pos_tag(tokens)

#         phrase = []
#         for word, pos in tagged:
#             if pos in ("JJ", "NN", "NNS", "NNP", "NNPS"):
#                 phrase.append(word)
#             else:
#                 if phrase:
#                     candidate_phrases.append(" ".join(phrase))
#                     phrase = []
#         if phrase:
#             candidate_phrases.append(" ".join(phrase))

#     filtered_phrases = []
#     for ph in candidate_phrases:
#         ph_lower = ph.lower()
#         if any(kw in ph_lower for kw in TECH_KEYWORDS):
#             words = [w for w in ph.split() if w.lower() not in stop_words]
#             if len(words) >= 2:
#                 filtered_phrases.append(" ".join(words))

#     freq = Counter(filtered_phrases)
#     return [p for p, _ in freq.most_common(top_n)]



def extract_keypoints_with_gemini(full_text, top_n=30):
    """Use Gemini to extract key legal/technical points from the privacy policy."""
    model = genai.GenerativeModel("gemini-2.0-flash")

    prompt = f"""
You are an expert in data privacy compliance and legal policy analysis.
Analyze the following Privacy Policy text and extract the most important {top_n} key points or clauses.
Focus on topics like data collection, consent, retention, sharing, third-party access, security, user rights, and compliance (GDPR, CCPA, etc.).
Return the key points as a simple numbered list, without explanations or markdown formatting.

--- Privacy Policy Content ---
{full_text}

Answer:
"""

    response = model.generate_content(prompt)
    keypoints_text = response.text.strip()

    # Convert response into a clean list
    key_points = [
        re.sub(r"^\d+[\).]?\s*", "", line.strip())
        for line in keypoints_text.split("\n") if line.strip()
    ]
    return key_points[:top_n]

# -------------------------------------------------------------
# Routes
# -------------------------------------------------------------

@app.get("/")
async def root():
    return HTMLResponse("<h1>Privacy Policy Assistant is Running </h1>")


@app.post("/upload_pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload PDF Privacy Policy."""
    try:
        file_bytes = await file.read()
        reset_collection()

        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        total_chunks = store_text_data(file_path, file.filename, file_bytes, source_type="pdf")
        os.remove(file_path)

        return JSONResponse({"message": f"Stored {total_chunks} chunks from '{file.filename}'"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload_url")
async def upload_website(url: str = Form(...)):
    """Upload Website Privacy Policy URL."""
    try:
        reset_collection()
        dummy_bytes = url.encode("utf-8")  
        total_chunks = store_text_data("", url, dummy_bytes, source_type="web")

        return JSONResponse({"message": f"Stored {total_chunks} chunks from website '{url}'"})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/summarize")
async def summarize_policy():
    """Summarize the Privacy Policy."""
    try:
        question = "Provide a detailed summary of this Privacy Policy, covering data collection, storage, rights, and compliance points."
        summary = query_gemini(question, [])
        return JSONResponse({"summary": summary})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/extract_keypoints")
async def extract_keypoints():
    """Extract important policy clauses using Gemini model."""
    try:
        results = collection.get()
        if not results or not results.get("documents"):
            return JSONResponse({"error": "No privacy policy uploaded yet."}, status_code=400)

        full_text = " ".join(results["documents"])
        key_points = extract_keypoints_with_gemini(full_text, top_n=30)

        return JSONResponse({
            "message": "Extracted key policy points successfully using Gemini.",
            "key_points": key_points
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat_with_policy(question: str = Form(...), session_id: str = Form("default")):
    """Chatbot Q&A for Privacy Policy understanding."""
    try:
        chat_history = conversation_sessions[session_id]
        answer = query_gemini(question, chat_history)
        conversation_sessions[session_id].append({
            "user": question,
            "assistant": answer
        })
        return JSONResponse({"answer": answer})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("mainn:app", host="127.0.0.1", port=8001, reload=True)

