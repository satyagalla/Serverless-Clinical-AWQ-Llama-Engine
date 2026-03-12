import os
import time
import json
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
import cohere

# Load API keys from .env
load_dotenv()

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
CSV_FILE = "data/questions_answers_large.csv"  # <-- Change this to your exact CSV file name
BATCH_SIZE = 96                # Cohere's max inputs per single API call
TEST_MODE = False               # Set to False ONLY when ready to ingest a lot of data
MAX_ROWS = 5000                # Safe limit to prevent exhausting Cohere's 1k monthly free API calls

def run_cloud_etl():
    print("🚀 INITIALIZING CLOUD RAG ETL PIPELINE...")
    
    # 1. Initialize Cloud Clients
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index("lora-finetune-llama-3-8b-medical") # Must match your exact Pinecone Index Name
        co = cohere.Client(os.getenv("COHERE_API_KEY"))
    except Exception as e:
        print(f"❌ API Authentication Error. Check your .env file. Details: {e}")
        return

    # Check current database size
    stats = index.describe_index_stats()
    current_db_size = stats.total_vector_count
    print(f"📊 Current Pinecone Database Size: {current_db_size} vectors.")

    # ==========================================
    # 🛠️ EXTRACTION & TRANSFORMATION
    # ==========================================
    print(f"\n📂 Extracting data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Could not find {CSV_FILE}. Check the file name.")
        return

    # Clean data & reset index to prevent silent deletion bugs
    initial_rows = len(df)
    df = df.dropna(subset=['Question', 'Answer']).reset_index(drop=True)
    print(f"🧹 Data Cleaned: Removed {initial_rows - len(df)} empty rows.")

    # Apply limits to protect your free tier API key
    if TEST_MODE:
        print("\n⚠️ TEST MODE ACTIVE: Truncating dataset to 500 rows.")
        df = df.head(500)
    else:
        print(f"\n⚠️ PROD MODE ACTIVE: Limiting to {MAX_ROWS} rows to protect Free Tier limits.")
        df = df.head(MAX_ROWS)

    # Transform into semantic strings
    print("🧬 Transforming rows into semantic strings...")
    documents = []
    
    for _, row in df.iterrows():
        merged_text = f"Question: {str(row['Question']).strip()}\nAnswer: {str(row['Answer']).strip()}"
        documents.append(merged_text)

    # ==========================================
    # 💾 SAVE LOCAL BM25 CORPUS
    # ==========================================
    print("\n📝 Saving local text corpus for BM25...")
    with open("cloud_corpus.json", "w") as f:
        json.dump(documents, f)
    print("✅ Saved 'cloud_corpus.json'. Local Sparse Engine is ready.")

    # ==========================================
    # ☁️ CLOUD INGESTION (COHERE + PINECONE)
    # ==========================================
    total_target_rows = len(documents)
    
    if current_db_size >= total_target_rows:
        print("\n✅ Cloud Database is already fully populated. No ingestion needed.")
        return
        
    docs_to_process = documents[current_db_size:total_target_rows]
    print(f"\n⏳ Starting Cloud Ingestion: Processing {len(docs_to_process)} documents...")
    
    start_time = time.time()

    # Process in batches of 96 to respect Cohere's strict limits
    for i in tqdm(range(0, len(docs_to_process), BATCH_SIZE), desc="Uploading to Cloud"):
        batch_end = min(i + BATCH_SIZE, len(docs_to_process))
        batch_docs = docs_to_process[i:batch_end]
        
        # We need the absolute IDs for Pinecone to match the local BM25 JSON index
        batch_ids = [str(current_db_size + idx) for idx in range(i, batch_end)]
        
        try:
            # 1. Call Cohere to get vectors
            embed_response = co.embed(
                texts=batch_docs,
                model='embed-english-v3.0',
                input_type='search_document'
            )
            embeddings = embed_response.embeddings
            
            # 2. Format for Pinecone: [("id_0", [0.1, 0.2...]), ("id_1", [0.4...])]
            vectors_to_upsert = list(zip(batch_ids, embeddings))
            
            # 3. Upsert to Pinecone
            index.upsert(vectors=vectors_to_upsert)
            
            # Rate limit safety sleep
            time.sleep(15)
            
        except Exception as e:
            print(f"\n❌ API Error during batch {i} to {batch_end}. Script Paused. Error: {e}")
            print("   You can restart the script later and it will resume from the last successful batch.")
            break

    end_time = time.time()
    
    print("\n" + "="*40)
    print(f"🎉 CLOUD INGESTION COMPLETE")
    print("="*40)
    print(f"Time Elapsed: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    run_cloud_etl()