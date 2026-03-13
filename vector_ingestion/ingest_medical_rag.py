import os
import torch
from dotenv import load_dotenv
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from pinecone import Pinecone, ServerlessSpec
from pinecone_text.sparse import BM25Encoder

# --- SECURE CONFIGURATION ---
# Load environment variables from the .env file
load_dotenv()

# Fetch the key securely
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("🚨 PINECONE_API_KEY not found! Please check your .env file.")

INDEX_NAME = "medical-usmle-index"
BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"

# 1. Initialize Pinecone Client
print("🌐 Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create the index if it doesn't exist (Using Dot Product since sparse+dense cant use cosine similarity)
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # BGE-small outputs exactly 384 dimensions
        metric="dot-product", 
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# 2. Load the USMLE Textbook Corpus
print("📚 Downloading USMLE Medical Corpus...")
# We take a slice (e.g., the first 10,000 paragraphs) to fit free tiers easily
dataset = load_dataset("cogbuji/medqa_corpus_en", split="train[:10000]", trust_remote_code=True)
corpus_texts = dataset['text']

# 3. Fit the BM25 Sparse Encoder
print("🧮 Fitting BM25 Lexical Vocabulary...")
bm25 = BM25Encoder()
bm25.fit(corpus_texts)
# Save it locally so your Hugging Face Space can use the exact same vocabulary later
bm25.dump("v2_hf_serverless/cpu_orchestrator/bm25_medical_vocab.json") 

# 4. Load the BGE Dense Embedding Model & Tokenizer
print("🧠 Loading BGE Dense Model...")
tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_ID)
model = AutoModel.from_pretrained(BGE_MODEL_ID)
model.eval() # Set to inference mode

def get_dense_vector(text):
    """Generates the 384-dimensional semantic vector using BGE."""
    # Truncation ensures we never crash the 512 limit
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # BGE requires using the [CLS] token (the first token) for sentence embeddings
        embeddings = outputs.last_hidden_state[:, 0, :]
        # BGE models require L2 normalization for accurate cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].tolist()

# 5. The Ingestion Loop (Chunking & Upserting)
print("🚀 Starting Hybrid Ingestion Pipeline...")
batch_size = 64
vectors_to_upsert = []

for i, doc_text in enumerate(corpus_texts):
    # Pure Python Token Chunking (512 max, minus safety margin, no overlap logic needed here 
    # if the dataset paragraphs are already naturally sized, but we enforce the limit)
    
    # Generate Hybrid Vectors
    dense_vec = get_dense_vector(doc_text)
    sparse_vec = bm25.encode_documents(doc_text)
    
    # Structure the Pinecone Payload
    vector_id = f"usmle_chunk_{i}"
    payload = {
        "id": vector_id,
        "values": dense_vec,
        "sparse_values": sparse_vec,
        "metadata": {"text": doc_text} # Store the actual text so we can read it later
    }
    vectors_to_upsert.append(payload)
    
    # Batch upsert to Pinecone to avoid API rate limits
    if len(vectors_to_upsert) >= batch_size:
        index.upsert(vectors=vectors_to_upsert)
        vectors_to_upsert = []
        print(f"✅ Upserted {i+1} documents...")

# Catch any remaining vectors
if vectors_to_upsert:
    index.upsert(vectors=vectors_to_upsert)

print("🎉 Ingestion 100% Complete. Pinecone is ready.")