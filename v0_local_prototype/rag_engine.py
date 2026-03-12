import os
import re
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from tqdm import tqdm

class HybridRetriever:
    def __init__(self, persist_dir="./chroma_db_production", test_mode=False):
        self.test_mode = test_mode
        self.persist_dir = persist_dir
        self.chroma_client = chromadb.PersistentClient(path=persist_dir)
        
        # 1. Load BGE-Large (SOTA English Medical)
        print("⚡ Loading BGE-Large-EN Embeddings...")
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="BAAI/bge-large-en-v1.5",
            device="cuda" 
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="medical_rag",
            embedding_function=self.embedding_fn
        )
        
        # 2. Load Reranker
        print("⚡ Loading BGE-Reranker-V2-M3...")
        self.reranker = CrossEncoder('BAAI/bge-reranker-v2-m3')
        
        self.bm25 = None
        self.documents = []
        self.doc_ids = []

        # 3. CRITICAL FIX: The Memory Recovery
        if self.collection.count() > 0:
            print(f"♻️  Found {self.collection.count()} docs on disk. Rebuilding BM25 in RAM...")
            self._load_bm25()

    def _normalize(self, text):
        return re.sub(r'[^\w\s]', '', text.lower())

    def _load_bm25(self):
        """Fetches docs from disk, sorts them by ID, and rebuilds BM25."""
        all_docs = self.collection.get()
        
        # Chroma doesn't guarantee order. We MUST sort by ID so indices match.
        try:
            sorted_pairs = sorted(zip(all_docs['ids'], all_docs['documents']), key=lambda x: int(x[0]))
            self.doc_ids = [pair[0] for pair in sorted_pairs]
            self.documents = [pair[1] for pair in sorted_pairs]
            
            tokenized_docs = [self._normalize(doc).split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            print("✅ RAM Index Rebuilt Successfully.")
        except Exception as e:
            print(f"⚠️ Error rebuilding BM25: {e}. You may need to wipe the database.")

    def index_documents(self, documents):
        print(f"🚀 Indexing {len(documents)} documents...")
        self.documents = documents
        self.doc_ids = [str(i) for i in range(len(documents))]
        
        tokenized_docs = [self._normalize(doc).split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
        batch_size = 50
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_end = min(i + batch_size, len(documents))
            self.collection.upsert(
                documents=documents[i:batch_end],
                ids=self.doc_ids[i:batch_end]
            )
        print("✅ Indexing Complete.")

    def search(self, original_query, expanded_query, top_k=5, fetch_k=20):
        if not self.bm25:
            print("⚠️ BM25 is empty. Cannot search.")
            return []

        # 1. SPARSE SEARCH gets the EXPANDED query (Keyword Salad)
        clean_expanded = self._normalize(expanded_query)
        bm25_scores = self.bm25.get_scores(clean_expanded.split())
        bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_k]
        
        # 2. DENSE SEARCH gets the ORIGINAL query (Natural Language)
        vector_results = self.collection.query(query_texts=[original_query], n_results=fetch_k)
        
        if not vector_results['ids'][0]:
             vector_indices = []
        else:
             vector_indices = [int(id) for id in vector_results['ids'][0] if int(id) < len(self.documents)]
        
        # Merge
        all_indices = set(bm25_indices).union(set(vector_indices))
        candidates = [self.documents[idx] for idx in all_indices if idx < len(self.documents)]
        
        if not candidates: 
            return []

        # 3. RERANKER gets the ORIGINAL query (to judge the final results accurately)
        pairs = [[original_query, doc] for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked_results = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

        if self.test_mode:
            # BM25 is SPARSE (Keyword)
            sparse_search_results = [self.documents[idx] for idx in bm25_indices if idx < len(self.documents)]
            # Embeddings are DENSE (Semantic Vector)
            dense_search_results = [self.documents[idx] for idx in vector_indices if idx < len(self.documents)]
            
            print('\n--------------------------------------------------\n')
            print(f"🔍 Sparse (BM25) search results: retrieved {len(sparse_search_results)} docs \n\n Docs: {sparse_search_results}")
            print('\n--------------------------------------------------\n')
            print(f"🧠 Dense (Vector) search results: retrieved {len(dense_search_results)} docs \n\n Docs: {dense_search_results}")
            print('\n--------------------------------------------------\n')
            
            # # Extract just the documents from the ranked tuples for cleaner printing
            # final_top_docs = [doc for score, doc in ranked_results[:top_k]]
            final_top_docs = [doc for score, doc in ranked_results]
            print(f"🏆 Final Reranked Top {top_k} results \n\n Docs: {final_top_docs}")
            print('\n--------------------------------------------------\n')
        
        return [doc for score, doc in ranked_results[:top_k]]