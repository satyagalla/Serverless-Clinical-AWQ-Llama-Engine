import os
import torch
import requests
import gradio as gr
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

# --- 1. CONFIGURATION & SECRETS ---
# Pulled securely from Hugging Face Space Settings
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
MICROSERVICE_SECRET_KEY = os.environ.get("MICROSERVICE_SECRET_KEY", "dev_override_key")

# PASTE YOUR MODAL URL HERE
MODAL_API_URL = "https://akhilgalla41--llama3-8b-lora-medical-inference-api-endpoint.modal.run" 

INDEX_NAME = "medical-usmle-index"

print("🌐 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# --- 2. LOAD THE BGE ECOSYSTEM ---
print("🧠 Loading BGE Dense Embedder...")
dense_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")
dense_model = AutoModel.from_pretrained("BAAI/bge-base-en-v1.5")
dense_model.eval()

print("⚖️ Loading BGE Cross-Encoder...")
rerank_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
rerank_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
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
    dense_vec = get_dense_vector(user_query)
    sparse_vec = bm25.encode_queries(user_query)
    
    raw_results = index.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=50,
        include_metadata=True
    )
    
    retrieved_docs = [match['metadata']['text'] for match in raw_results['matches']]
    if not retrieved_docs:
        return "No relevant medical context found in the USMLE database."

    pairs = [[user_query, doc] for doc in retrieved_docs]
    inputs = rerank_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
    
    with torch.no_grad():
        scores = rerank_model(**inputs, return_dict=True).logits.view(-1,).float()
    
    scored_docs = list(zip(scores, retrieved_docs))
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    top_3_contexts = [doc for score, doc in scored_docs[:3]]
    return "\n\n---\n\n".join(top_3_contexts)

# --- 5. THE API HAND-OFF TO MODAL ---
def clinical_chat_agent(user_message, history):
    print("Searching USMLE database...")
    context = retrieve_and_rerank(user_message)
    
    print("Pinging Modal GPU Space for Inference...")
    print(f"Pinging Modal API at: {MODAL_API_URL}")
    try:
        # Explicitly enforce string types for the payload
        payload = {
            "question": str(user_message),
            "context": str(context),
            "api_key": str(MICROSERVICE_SECRET_KEY)
        }
        
        response = requests.post(MODAL_API_URL, json=payload)
        
        # --- THE DIAGNOSTIC SHIELD ---
        # If Modal rejects the request before hitting the GPU, catch it here
        if response.status_code != 200:
            print(f"🚨 MODAL REJECTION: {response.status_code} - {response.text}")
            return f"Modal Gateway Error ({response.status_code}). Check Hugging Face Logs."
            
        response_data = response.json()
        
        # Safely extract and force string type to prevent Gradio UI crashes
        if "error" in response_data:
            return str(response_data["error"])
            
        return str(response_data.get("answer", "🚨 GPU returned an empty answer."))
        
    except Exception as e:
        return f"🚨 Network Crash: {str(e)}"

# --- 6. GRADIO WEB UI ---
demo = gr.ChatInterface(
    fn=clinical_chat_agent,
    title="Clinical Diagnostic Assistant",
    description="A Hybrid RAG pipeline powered by BGE-v1.5, Pinecone, and Llama-3 running on Modal Serverless GPUs.",
)

if __name__ == "__main__":
    demo.launch()