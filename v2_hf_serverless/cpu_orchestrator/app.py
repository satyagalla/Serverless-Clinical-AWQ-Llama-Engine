import os
import time
import torch
import requests
import gradio as gr
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# --- 1. CONFIGURATION & SECRETS ---
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
MICROSERVICE_SECRET_KEY = os.environ.get("MICROSERVICE_SECRET_KEY", "dev_override_key")

MODAL_API_URL = "https://akhilgalla41--medical-llama-3-1-8b-instruct-lora-awq-inf-f9aee9.modal.run" 
INDEX_NAME = "medical-usmle-index" # 🚨 WARNING: THIS MUST BE A 384-DIMENSION INDEX

print("🌐 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- 2. LOAD THE BGE ECOSYSTEM (LIGHTWEIGHT CPU VERSIONS) ---
print("🧠 Loading Dense Embedder (bge-small - 33M Params)...")
dense_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
dense_model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")
dense_model.eval()

print("⚖️ Loading Cross-Encoder (MiniLM - 22M Params)...")
rerank_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
rerank_model.eval()

# --- 3. LOAD THE BM25 SPARSE ENCODER ---
print("🧮 Loading BM25 Medical Vocabulary...")
bm25 = BM25Encoder()
bm25.load("bm25_medical_vocab.json")

# --- 4. CORE PIPELINE FUNCTIONS ---
def get_dense_vector(text):
    inputs = dense_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = dense_model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

def retrieve_and_rerank(user_query):
    t_start = time.perf_counter()
    
    # Checkpoint 1: Vectorization Math
    dense_vec = get_dense_vector(user_query)
    sparse_vec = bm25.encode_queries(user_query)
    t_embed = time.perf_counter()
    
    # Checkpoint 2: Pinecone Network Call
    raw_results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=10, 
        include_metadata=True
    )
    t_pinecone = time.perf_counter()
    
    retrieved_docs = [match['metadata']['text'] for match in raw_results['matches']]
    if not retrieved_docs:
        print("🚨 Pinecone returned 0 results.")
        return "No relevant medical context found in the USMLE database."

    # Checkpoint 3: CPU Cross-Encoder Math
    pairs = [[user_query, doc] for doc in retrieved_docs]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1,).float()
    
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    t_rerank = time.perf_counter()
    
    # 📊 TELEMETRY OUTPUT TO HF LOGS
    print("\n--- RAG TELEMETRY SPLITS ---")
    print(f"├─ Embedding Math:    {t_embed - t_start:.3f}s")
    print(f"├─ Pinecone Fetch:    {t_pinecone - t_embed:.3f}s")
    print(f"└─ Cross-Encoder:     {t_rerank - t_pinecone:.3f}s")
    print(f"Total Retrieval Time: {t_rerank - t_start:.3f}s")
    print("----------------------------\n")
    
    top_3_contexts = [doc for score, doc in scored_docs[:3]]
    return "\n\n---\n\n".join(top_3_contexts)

# --- 5. THE API HAND-OFF TO MODAL ---
def clinical_chat_agent(user_message, history):
    # Added: Query Expansion for Semantic Search
    search_query = user_message
    if len(user_message.split()) < 8:
        search_query = f"Clinical definition, differential diagnosis, and treatment guidelines for: {user_message}"
        
    print(f"🔍 PHASE 1: Hybrid RAG Pipeline Started for query: '{search_query}'")
    context = retrieve_and_rerank(user_message)

    # Added: History Serialization for the LLM Payload (OpenAI Dict Format)
    conversation_block = ""
    if history:
        conversation_block = "Previous Conversation History:\n"
        for message in history:
            # Map Gradio's standard roles to our clinical prompt roles
            speaker = "Patient" if message.get("role") == "user" else "System"
            conversation_block += f"{speaker}: {message.get('content', '')}\n"
        conversation_block += "\nCurrent Patient Query: "
    full_question_payload = f"{conversation_block}{user_message}"

    print(f"🚀 PHASE 2: Pinging Modal API...")
    t_api_start = time.perf_counter()
    
    try:
        payload = {
            "question": str(full_question_payload),
            "context": str(context),
            "api_key": str(MICROSERVICE_SECRET_KEY)
        }
        
        response = requests.post(MODAL_API_URL, json=payload)
        t_api_end = time.perf_counter()
        
        print(f"✅ Modal Round-Trip Time: {t_api_end - t_api_start:.3f}s")
        
        if response.status_code != 200:
            return f"Modal Gateway Error ({response.status_code}). Check Hugging Face Logs."
            
        response_data = response.json()
        
        if "error" in response_data:
            return str(response_data["error"])
            
        return str(response_data.get("answer", "🚨 GPU returned an empty answer."))
        
    except Exception as e:
        return f"🚨 Network Crash: {str(e)}"

# --- 6. GRADIO WEB UI ---
demo = gr.ChatInterface(
    fn=clinical_chat_agent,
    title="USMLE Clinical Inference Engine (Llama-3.1-8B-Instruct)",
    description=(
        "**Retrieval Architecture:** Hybrid Sparse/Dense (BM25 + BGE-Small) → MiniLM Cross-Encoder Reranking.\n"
        "**Inference Backend:** USMLE-finetuned LoRA adapter merged into Llama-3.1, compressed via W4A16 AWQ. Deployed on a serverless A10G GPU.\n"
        "**Execution:** vLLM continuous batching via forced eager execution."
    ),
)

if __name__ == "__main__":
    demo.launch()