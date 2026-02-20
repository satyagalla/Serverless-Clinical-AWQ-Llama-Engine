import os
import sys
import time
import pandas as pd
from tqdm.auto import tqdm

# Ensure Python can find your rag_engine.py
sys.path.append(os.getcwd())
from rag_engine import HybridRetriever

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
CSV_FILE = "data/questions_answers_large.csv" 
BATCH_SIZE = 5000                 # How many rows to process before saving a checkpoint
TEST_MODE = False                  # Set to False ONLY when you are ready for the 3-hour run

def run_etl_pipeline():
    print("🚀 INITIALIZING RAG ETL PIPELINE...")
    
    # 1. Initialize the Database Connection
    # We do NOT want to wipe the database here. We want to connect to it
    # so we can check if there is already data inside (for resuming).
    retriever = HybridRetriever(persist_dir="./chroma_db_production")
    collection = retriever.collection
    
    # Check current database size
    current_db_size = collection.count()
    print(f"📊 Current Database Size: {current_db_size} documents.")

    # ==========================================
    # 🛠️ EXTRACTION & TRANSFORMATION
    # ==========================================
    print(f"\n📂 Extracting data from {CSV_FILE}...")
    try:
        df = pd.read_csv(CSV_FILE, low_memory=False)
    except FileNotFoundError:
        print(f"❌ CRITICAL ERROR: Could not find {CSV_FILE}.")
        return

    # Clean the data: Drop rows where Question or Answer is missing
    initial_rows = len(df)
    df = df.dropna(subset=['Question', 'Answer']).reset_index(drop=True)
    cleaned_rows = len(df)
    print(f"🧹 Data Cleaned: Removed {initial_rows - cleaned_rows} empty rows.")

    # Apply TEST_MODE constraint
    if TEST_MODE:
        print("\n⚠️ TEST MODE ACTIVE: Truncating dataset to 500 rows.")
        df = df.head(500)
        total_target_rows = 500
    else:
        total_target_rows = cleaned_rows

    # Semantic Transformation: Merge Q & A for maximum vector surface area
    print("🧬 Transforming rows into semantic strings...")
    documents = []
    doc_ids = []
    
    for index, row in df.iterrows():
        # Format: "Question: [Text] \n Answer: [Text]"
        merged_text = f"Question: {str(row['Question']).strip()}\nAnswer: {str(row['Answer']).strip()}"
        documents.append(merged_text)
        # We use the literal row index as the ID so it's consistent
        doc_ids.append(str(index))

    # ==========================================
    # 💾 LOADING (WITH RESILIENCE)
    # ==========================================
    # Calculate how much work is left. 
    # If DB has 10,000 rows, and we want 207,000, we only process the remaining 197,000.
    if current_db_size >= total_target_rows:
        print("\n✅ Database is already fully populated. No ingestion needed.")
        return
        
    # Slice the arrays to skip the documents we've already processed
    docs_to_process = documents[current_db_size:total_target_rows]
    ids_to_process = doc_ids[current_db_size:total_target_rows]
    
    print(f"\n⏳ Resuming Ingestion: Processing {len(docs_to_process)} remaining documents...")
    print(f"   (Batch Size: {BATCH_SIZE})")
    
    start_time = time.time()

    # The Checkpoint Loop
    # We step through the remaining data in chunks (e.g., 0 to 5000, 5000 to 10000)
    for i in tqdm(range(0, len(docs_to_process), BATCH_SIZE), desc="Ingesting Batches"):
        
        # Calculate the end of the current slice
        batch_end = min(i + BATCH_SIZE, len(docs_to_process))
        
        batch_docs = docs_to_process[i:batch_end]
        batch_ids = ids_to_process[i:batch_end]
        
        # Upsert pushes the data to ChromaDB and automatically saves to disk.
        # If the script crashes during a batch, at most you lose this specific batch.
        collection.upsert(
            documents=batch_docs,
            ids=batch_ids
        )

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60

    # ==========================================
    # 🏁 FINAL VERIFICATION
    # ==========================================
    final_count = collection.count()
    print("\n" + "="*40)
    print(f"🎉 INGESTION COMPLETE")
    print("="*40)
    print(f"Final Database Size : {final_count} documents")
    print(f"Time Elapsed        : {elapsed_minutes:.2f} minutes")
    print("You can now safely run your 'agent.py' script.")

if __name__ == "__main__":
    run_etl_pipeline()