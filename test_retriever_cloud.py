import os
from dotenv import load_dotenv
from rag_engine_cloud import CloudHybridRetriever

# Load API keys from .env
load_dotenv()

def run_test():
    print("🚀 INITIALIZING CLOUD RETRIEVER TEST...")
    
    try:
        # Initialize the retriever 
        # (This connects to Pinecone, Cohere, and loads the local cloud_corpus.json for BM25)
        retriever = CloudHybridRetriever()
        print("✅ Retriever initialized successfully.")
    except Exception as e:
        print(f"❌ Failed to initialize retriever. Error: {e}")
        return

    # ==========================================
    # 🧪 SIMULATED INPUTS (Bypassing LangGraph)
    # ==========================================
    # 1. The natural language question (Routed to Pinecone/Cohere)
    original_query = "What are the common treatments for a severe migraine?"
    
    # 2. The simulated LLM keyword expansion (Routed to local BM25)
    expanded_query = "severe migraine headache treatment medication therapy neurological pain relief triptans"

    print(f"\n🔍 Original Query (Dense Search): '{original_query}'")
    print(f"🔑 Expanded Query (Sparse Search): '{expanded_query}'\n")

    print("⏳ Retrieving and Reranking documents... (Calling APIs)\n")
    
    try:
        # Call the dual-routing retrieval method
        results = retriever.search(
            original_query=original_query, 
            expanded_query=expanded_query, 
            top_k=3
        )
        
        # Display results
        print("="*60)
        print("🏆 TOP RERANKED RESULTS")
        print("="*60)
        
        if not results:
            print("⚠️ No results found. Are you sure vectors are uploaded to Pinecone?")
            
        for i, doc in enumerate(results, 1):
            print(f"\n[{i}] 📄 Document Snippet:")
            # Print first 400 chars to keep the terminal clean but show enough context
            print(f"{doc[:400]}...") 
            print("-" * 60)
            
    except Exception as e:
        print(f"❌ Retrieval failed. Error: {e}")

if __name__ == "__main__":
    run_test()