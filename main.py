import os
import PyPDF2
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import google.generativeai as genai


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


genai.configure(api_key=GEMINI_API_KEY)

def read_pdf(pdf_path):
    pdf_text = []
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                pdf_text.append({"page": i + 1, "content": text})
    return pdf_text


def chunk_text(text, chunk_size=500, overlap=50):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


chroma_client = chromadb.Client(
    settings=chromadb.config.Settings(anonymized_telemetry=False)
)

try:
    chroma_client.delete_collection(name="privacy_policies")
except:
    pass

collection = chroma_client.create_collection(
    name="resrach_paper_collection",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
)

def store_pdf(pdf_path):
    pdf_text = read_pdf(pdf_path)
    all_chunks, metadatas, ids = [], [], []
    for page_data in pdf_text:
        page_chunks = chunk_text(page_data["content"])
        for idx, chunk in enumerate(page_chunks):
            all_chunks.append(chunk)
            metadatas.append({"page": page_data["page"]})
            ids.append(f"{pdf_path}_p{page_data['page']}_{idx}")
    collection.add(documents=all_chunks, metadatas=metadatas, ids=ids)
    print(f"Stored {len(all_chunks)} chunks from {pdf_path}")

def query_policy(question, n_results=3):
    results = collection.query(query_texts=[question], n_results=n_results)
    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    context_text = ""
    for i, ctx in enumerate(contexts):
        context_text += f"\n[Source p.{metadatas[i]['page']}] {ctx}"

    prompt = f"""You are a research paper assistant.
Answer the user question using only the context below. If user asks to summarize then from every point in the resrach paper summarize the point and give as the answer.
If you didn't get the very relatable response from the model then add your own knowledge to answer the question.
Include citations as [Source p.X].

Context:
{context_text}

Question: {question}
Answer:"""

    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(prompt)
    return response.text

if __name__ == "__main__":
    store_pdf("research paper.pdf")

    ans = query_policy("Tell about the autonomy and new technologies.")
    print("\nAnswer:\n", ans)