import os
import json
from typing import TypedDict, List
from dotenv import load_dotenv

# Load environment variables (e.g., GROQ_API_KEY) securely from .env
load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
# Switch between "groq" (Cloud) or "local" (Unsloth Fine-Tune)
LLM_BACKEND = "groq" 

if LLM_BACKEND == "groq":
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("⚠️ GROQ_API_KEY not found in .env file.")

# ==========================================
# 🧠 2. INITIALIZE MODELS
# ==========================================
print("⚙️  Starting RAG Agent Initialization...")
from local.rag_engine import HybridRetriever
retriever = HybridRetriever(persist_dir="../chroma_db_production", test_mode=False)

# from rag_engine_cloud import CloudHybridRetriever
# retriever = CloudHybridRetriever(index_name="medical-rag")

local_model = None
local_tokenizer = None
groq_model = None

if LLM_BACKEND == "groq":
    print("🧠 Brain: GROQ (Llama-3-70B via Cloud)")
    groq_model = ChatGroq(model_name="llama3-70b-8192", temperature=0)

elif LLM_BACKEND == "local":
    print("🧠 Brain: LOCAL (Unsloth Fine-Tune)")
    from unsloth import FastLanguageModel
    local_model, local_tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(local_model)

# ==========================================
# 🔌 3. THE LLM BRIDGE (Strategy Pattern)
# ==========================================
def call_llm(prompt_text: str, system_instruction: str = "") -> str:
    """Routes the prompt to either the Cloud or Local model seamlessly."""
    if LLM_BACKEND == "groq":
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", prompt_text)
        ])
        chain = prompt | groq_model
        return chain.invoke({}).content

    elif LLM_BACKEND == "local":
        alpaca_prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{system_instruction}

### Input:
{prompt_text}

### Response:
"""
        inputs = local_tokenizer([alpaca_prompt], return_tensors="pt").to("cuda")
        outputs = local_model.generate(**inputs, max_new_tokens=512, use_cache=True)
        decoded = local_tokenizer.batch_decode(outputs)[0]
        return decoded.split("### Response:")[-1].replace("<|end_of_text|>", "").strip()

# ==========================================
# 📋 4. LANGGRAPH STATE
# ==========================================
class AgentState(TypedDict):
    question: str           # Original user query
    expanded_query: str     # Keyword salad for BM25
    context: List[str]      # Retrieved documents
    answer: str             # Final JSON output

# ==========================================
# 🏭 5. LANGGRAPH NODES
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
    
    # DUAL-ROUTING ARCHITECTURE:
    # Dense Search (Vector) gets the original, natural language query.
    # Sparse Search (BM25) gets the expanded keyword salad.
    docs = retriever.search(
        original_query=state["question"], 
        expanded_query=state["expanded_query"], 
        top_k=5
    )
    
    print(f"   ↳ Found {len(docs)} highly relevant documents.")
    return {"context": docs}

def generation_node(state: AgentState):
    """[Node 3] The Doctor: Synthesizes evidence into a JSON diagnosis."""
    print("🧠 [Step 3] Formulating JSON Diagnosis...")
    
    if not state["context"]:
        empty_resp = json.dumps({"topic": "Unknown", "answer": "No relevant clinical context found.", "confidence": "Low"})
        return {"answer": empty_resp}

    context_str = "\n".join([f"- {doc}" for doc in state["context"]])
    
    system_instruction = (
        "You are a clinical diagnostic assistant. Use ONLY the retrieved context to answer the question. "
        "Output strictly valid JSON with keys: 'topic', 'answer', 'confidence'. "
        "Do not use markdown blocks like ```json."
    )
    
    user_input = f"Context:\n{context_str}\n\nQuestion:\n{state['question']}"
    response = call_llm(user_input, system_instruction)
    
    return {"answer": response}

# ==========================================
# 🛤️ 6. GRAPH ASSEMBLY
# ==========================================
workflow = StateGraph(AgentState)

workflow.add_node("expand", query_expansion_node)
workflow.add_node("retrieve", retrieval_node)
workflow.add_node("generate", generation_node)

# Linear Flow
workflow.set_entry_point("expand")
workflow.add_edge("expand", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

app = workflow.compile()

# ==========================================
# 🚀 7. EXECUTION
# ==========================================
if __name__ == "__main__":
    print("\n" + "="*50)
    print("🏥 AGENTIC CLINICAL RAG PIPELINE ONLINE")
    print("="*50)
    
    # Ensure database isn't completely empty before trying to search
    if retriever.collection.count() == 0:
        print("⚠️ CRITICAL: Database is empty. Please run your ETL ingestion script first.")
    else:
        while True:
            query = input("\n🩺 Patient Query (or 'q' to quit): ")
            if query.lower() == 'q': 
                print("Shutting down pipeline.")
                break
            
            # Execute the Graph
            result = app.invoke({"question": query})
            
            print("\n" + "="*50)
            print("📄 FINAL OUTPUT")
            print("="*50)
            
            try:
                # Pretty print if it's valid JSON
                parsed_json = json.loads(result["answer"])
                print(json.dumps(parsed_json, indent=2))
            except json.JSONDecodeError:
                # Fallback if the LLM hallucinated outside the JSON structure
                print(result["answer"])