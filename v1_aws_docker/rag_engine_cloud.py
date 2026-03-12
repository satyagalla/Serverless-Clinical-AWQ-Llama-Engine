import os
import re
import json
import numpy as np
from pinecone import Pinecone
import cohere
from rank_bm25 import BM25Okapi

class CloudHybridRetriever:
    def __init__(self, index_name="lora-finetune-llama-3-8b-medical", test_mode=False):
        self.test_mode = test_mode
        
        # 1. Initialize Cloud API Clients
        print("☁️ Connecting to Cloud Infrastructure...")
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name) 
        
        self.co = cohere.Client(os.getenv("COHERE_API_KEY"))
        
        # 2. Load Local Text Corpus for BM25 (Sparse Search)
        self.documents = []
        self.bm25 = None
        self._load_local_corpus()

    def _normalize(self, text):
        return re.sub(r'[^\w\s]', '', text.lower())

    def _load_local_corpus(self):
        """Loads the lightweight JSON text array so BM25 can run locally on the free tier."""
        corpus_path = "cloud_corpus.json"
        
        if os.path.exists(corpus_path):
            print("♻️ Loading text corpus into memory for BM25...")
            with open(corpus_path, "r") as f:
                self.documents = json.load(f)
            
            tokenized_docs = [self._normalize(doc).split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print(f"✅ BM25 initialized with {len(self.documents)} documents.")
        else:
            print(f"⚠️ {corpus_path} not found. BM25 is offline. Please run the Cloud ETL script.")

    def search(self, original_query, expanded_query, top_k=5, fetch_k=20):
        if not self.bm25:
            print("⚠️ Engine not fully loaded. Cannot search.")
            return []

        # ==========================================
        # 🧠 1. DENSE SEARCH (Cohere + Pinecone)
        # ==========================================
        # Ask Cohere to embed the natural language query
        embed_response = self.co.embed(
            texts=[original_query],
            model='embed-english-v3.0',
            input_type='search_query'
        )
        query_vector = embed_response.embeddings[0]

        # Send that vector to Pinecone to find the closest matches
        pinecone_results = self.index.query(
            vector=query_vector,
            top_k=fetch_k,
            include_values=False
        )
        # Extract the integer IDs
        vector_indices = [int(match['id']) for match in pinecone_results['matches'] if int(match['id']) < len(self.documents)]

        # ==========================================
        # 🔍 2. SPARSE SEARCH (BM25 Local)
        # ==========================================
        # Use the Expanded "Keyword Salad" query here
        clean_expanded = self._normalize(expanded_query)
        bm25_scores = self.bm25.get_scores(clean_expanded.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]

        # ==========================================
        # 🤝 3. MERGE & FETCH TEXT
        # ==========================================
        all_indices = set(bm25_indices).union(set(vector_indices))
        candidates = [self.documents[idx] for idx in all_indices]

        if not candidates: 
            return []

        # ==========================================
        # 🏆 4. RERANK (Cohere API)
        # ==========================================
        # Send the original query and the merged documents to Cohere's Reranker
        rerank_response = self.co.rerank(
            query=original_query,
            documents=candidates,
            top_n=top_k,
            model='rerank-english-v3.0'
        )

        # Cohere returns objects containing the index of the winning documents
        final_docs = [candidates[result.index] for result in rerank_response.results]

        return final_docs