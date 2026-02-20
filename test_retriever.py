import os
import time
from rag_engine import HybridRetriever

def run_retrieval_test():
    print("🔍 INITIALIZING HYBRID RETRIEVAL TEST...")
    
    # 1. Load the Engine (This will automatically rebuild BM25 in RAM)
    retriever = HybridRetriever(persist_dir="./chroma_db_production", test_mode=True)
    
    count = retriever.collection.count()
    print(f"\n📊 Current Database Size: {count} documents.")
    
    # 2. Safety Check
    if count == 0:
        print("❌ CRITICAL: Database is empty. Please run your ETL script first.")
        return

    print("\n" + "="*60)
    print("🧪 ISOLATED RETRIEVAL TEST ENVIRONMENT")
    print("   Testing: BM25 (Sparse) + BGE-Large (Dense) + Reranker")
    print("   Type 'q' to quit.")
    print("="*60)

    # 3. The Test Loop
    while True:
        query = input("\n🔎 Enter a test query: ")
        if query.lower() == 'q':
            print("Exiting test environment.")
            break
            
        print(f"⚙️  Executing Hybrid Search for: '{query}'...")
        start_time = time.time()
        
        # We fetch the top 3 results to inspect accuracy
        results = retriever.search(query, top_k=3)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # 4. Display Results
        if not results:
            print("⚠️ No matching documents found.")
            continue
            
        print(f"\n✅ Retrieved Top {len(results)} Results in {elapsed:.3f} seconds:\n")
        
        for i, doc in enumerate(results, 1):
            print(f"--- 🏆 RANK {i} ---")
            # We strip extra whitespace just for clean terminal printing
            print(doc.strip())
            print("-" * 20)

if __name__ == "__main__":
    run_retrieval_test()