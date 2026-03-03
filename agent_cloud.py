import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

# Load environment variables securely from .env
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Import the strictly cloud-based retriever with the updated class name
from rag_engine_cloud import CloudHybridRetriever

# ==========================================
# ⚙️ 1. INITIALIZE CLOUD INFRASTRUCTURE
# ==========================================
print("⚙️  Starting Cloud RAG Agent Initialization...")

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("⚠️ GROQ_API_KEY not found in .env file.")

# Initialize the Retriever using the updated index name
retriever = CloudHybridRetriever(index_name="lora-finetune-llama-3-8b-medical")

# Initialize the Generation Model (Groq API)
print("🧠 Brain: GROQ (Llama-3 via Cloud)")
groq_model = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

# ==========================================
# 🔌 2. THE LLM BRIDGE
# ==========================================
def call_llm(prompt_text: str, system_instruction: str = "") -> str:
    """Routes the prompt to the Groq Cloud API."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_instruction),
        ("user", prompt_text)
    ])
    chain = prompt | groq_model
    return chain.invoke({}).content

# ==========================================
# 📋 3. LANGGRAPH STATE
# ==========================================
class AgentState(TypedDict):
    question: str           # Original user query
    expanded_query: str     # Keyword salad for BM25
    context: List[str]      # Retrieved documents
    answer: str             # Final JSON output

# ==========================================
# 🏭 4. LANGGRAPH NODES
# ==========================================
def query_expansion_node(state: AgentState):
    """[Node 1] The Translator: Generates clinical keywords."""
    print(f"\n🔍 [Step 1] Analyzing Symptom: '{state['question']}'")
    
    system_prompt = (
        "You are a medical search optimizer. "
        "Generate 3 specific medical synonyms or clinical terms for the user's query. "
        "Output ONLY the keywords separated by spaces. Do not write sentences."
    )
    
    new_keywords = call_llm(state['question'], system_prompt)
    expanded = f"{state['question']} {new_keywords}"
    
    print(f"   ↳ Search Strategy: '{expanded}'")
    return {"expanded_query": expanded}

def retrieval_node(state: AgentState):
    """[Node 2] The Librarian: Executes the Dual-Routing Hybrid Search."""
    print("📂 [Step 2] Retrieving Clinical Guidelines (Dual-Routing)...")
    
    # Execute Pinecone + BM25 + Cohere Rerank using the 'search' method
    docs = retriever.search(
        original_query=state["question"], 
        expanded_query=state["expanded_query"], 
        top_k=5,
        fetch_k=20
    )
    
    print(f"   ↳ Found {len(docs)} highly relevant documents.")
    return {"context": docs}

def generation_node(state: AgentState):
    """[Node 3] The Doctor: Synthesizes evidence into a JSON diagnosis."""
    print("🧠 [Step 3] Formulating JSON Diagnosis...")
    
    if not state["context"]:
        empty_resp = json.dumps({"topic": "Unknown", "answer": "No relevant clinical context found.", "confidence": "Low"})
        return {"answer": empty_resp}

    # We format the context to explicitly label it as "OTHER PATIENT HISTORIES"
    context_str = "\n".join([f"Historical Case {i+1}:\n{doc}\n" for i, doc in enumerate(state["context"])])
    
    system_instruction = (
        "You are an expert clinical triage AI. Your job is to answer the CURRENT USER'S question using general medical knowledge extracted from HISTORICAL CASES.\n\n"
        "🛑 CRITICAL SAFETY RULES 🛑\n"
        "1. ENTITY ISOLATION: The Historical Cases belong to OTHER patients. The CURRENT USER is a completely different person.\n"
        "2. NO HALLUCINATIONS: DO NOT attribute any test results or prior medical histories from the Historical Cases to the CURRENT USER.\n"
        "3. EXTRACTION ONLY: Look at the doctor's answers in the Historical Cases. Extract ONLY the general diagnoses and advice that match the CURRENT USER's stated symptoms.\n\n"
        "Output strictly valid JSON with keys: 'topic', 'answer' (your generalized medical advice), and 'confidence' (0.0 to 1.0)."
    )
    
    user_input = f"--- HISTORICAL CASES (DO NOT ASSUME THIS IS THE USER) ---\n{context_str}\n\n--- CURRENT USER SYMPTOMS ---\n{state['question']}"
    
    response = call_llm(user_input, system_instruction)
    
    return {"answer": response}

# ==========================================
# 🛤️ 5. GRAPH ASSEMBLY
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("expand", query_expansion_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)

workflow.set_entry_point("expand")
workflow.add_edge("expand", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ==========================================
# 🚀 6. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🏥 CLOUD-NATIVE RAG PIPELINE ONLINE")
    print("="*50)
    
    # Check if Pinecone database is empty
    stats = retriever.index.describe_index_stats()
    if stats.total_vector_count == 0:
        print("⚠️ CRITICAL: Database is empty. Please run your ETL ingestion script first.")
    else:
        while True:
            query = input("\n🩺 Patient Query (or 'q' to quit): ")
            if query.lower() == 'q': 
                print("Shutting down pipeline.")
                break
            
            # Execute the LangGraph State Machine
            result = app.invoke({"question": query})
            
            print("\n" + "="*50)
            print("📄 FINAL OUTPUT")
            print("="*50)
            
            try:
                # Deterministic JSON Extraction: Rip out everything outside the brackets
                raw_text = result["answer"]
                start_idx = raw_text.find('{')
                end_idx = raw_text.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    clean_json_str = raw_text[start_idx:end_idx]
                    parsed_json = json.loads(clean_json_str)
                    print(json.dumps(parsed_json, indent=2))
                else:
                    # If it completely failed to generate brackets, just print the text
                    print(raw_text)
                    
            except json.JSONDecodeError:
                print("⚠️ [Parser Warning]: LLM generated malformed JSON.")
                print(result["answer"])