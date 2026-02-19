import os
import re
import shutil
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tqdm import tqdm

class HybridRetriever:
    def __init__(self, persist_dir="./chroma_db_production"):
        """
        Initializes the database and loads the heavy AI models.
        """
        # 1. Setup Persistent Storage
        # We use 'PersistentClient' so data is saved to disk (./chroma_db_production)
        # instead of vanishing when the script stops.
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # 2. Load Embeddings (BGE-M3)
        # We choose BGE-large because it is State-of-the-Art (SOTA) for dense retrieval.
        # It handles multi-lingual text and complex sentences better than older models.
        print("⚡ Loading BGE-Large Embeddings...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda" # Change to "cuda" if you have a GPU
        )
        
        # 3. Create/Load Collection
        # A 'collection' in Chroma is like a table in SQL.
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_rag",
            embedding_function=self.embedding_fn
        )
        
        # 4. Load Reranker (BGE-Reranker-V2-M3)
        # This model is the 'Second Opinion'. It's computationally expensive,
        # so we only use it on the small set of results we retrieve first.
        print("⚡ Loading BGE-Reranker...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        
        # placeholders for BM25 (which must be built in-memory)
        self.bm25 = None
        self.documents = []

    def _normalize(self, text):
        # Simple cleaning: lowercase and remove punctuation.
        # This helps BM25 match "Doctor" with "doctor".
        return re.sub(r'[^\w\s]', '', text.lower())

    def index_documents(self, documents):
        """
        Ingests a list of text strings into both BM25 and ChromaDB.
        """
        print(f"🚀 Indexing {len(documents)} documents...")
        
        # 1. Update In-Memory BM25 Index
        # BM25 needs the dataset to be tokenized (split into words).
        self.documents = documents
        tokenized_docs = [self._normalize(doc).split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        # 2. Update Vector DB with Batching
        # Pushing 1,000 docs at once can crash the API. We split them into chunks of 50.
        batch_size = 50
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_end = min(i + batch_size, len(documents))
            batch_docs = documents[i:batch_end]
            # Create simple IDs: "0", "1", "2"...
            batch_ids = [str(k) for k in range(i, batch_end)]
            
            # .upsert() means "Update if exists, Insert if new"
            self.collection.upsert(
                documents=batch_docs,
                ids=batch_ids
            )
        print("✅ Indexing Complete.") 

    def search(self, query, top_k=5, fetch_k=20):
        """
        Retrieves 'fetch_k' candidates, reranks them, and returns 'top_k'.
        """
        if not self.bm25: return []

        clean_query = self._normalize(query)
        
        # --- Step A: Sparse Retrieval (Keywords) ---
        # Get BM25 scores for every doc, sort them, take top 20.
        bm25_scores = self.bm25.get_scores(clean_query.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        
        # --- Step B: Dense Retrieval (Meaning) ---
        # Query ChromaDB. It converts the query to a vector and finds nearest neighbors.
        vector_results = self.collection.query(
            query_texts=[query],
            n_results=fetch_k
        )
        
        # Convert string IDs back to integers for list indexing
        # Handle empty results gracefully
        if not vector_results['ids'][0]:
             vector_indices = []
        else:
             vector_indices = [int(id) for id in vector_results['ids'][0] if int(id) < len(self.documents)]
        
        # --- Step C: Fusion (The Union) ---
        # We combine the lists. using set() automatically removes duplicates.
        all_indices = set(bm25_indices).union(set(vector_indices))
        
        # Fetch the actual text for these indices
        candidates = []
        for idx in all_indices:
            if idx < len(self.documents):
                candidates.append(self.documents[idx])
        
        if not candidates: return []

        # --- Step D: Reranking (The Judge) ---
        # We create pairs: [[Query, Doc1], [Query, Doc2], ...]
        pairs = [[query, doc] for doc in candidates]
        
        # The Cross-Encoder scores how relevant each pair is.
        scores = self.reranker.predict(pairs)
        
        # Sort based on the new scores (highest first)
        ranked_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        
        # Return only the top K texts
        return [doc for score, doc in ranked_results[:top_k]]