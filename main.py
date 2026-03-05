from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import json

# Import your compiled LangGraph state machine
from agent_cloud import app as agent_app

# Initialize FastAPI
app = FastAPI(title="Medical RAG API")

@app.get("/")
def read_root():
    return {"status": "Online", "message": "Cloud-Native Medical RAG Agent is running."}

@app.get("/diagnose")
def diagnose(query: str):
    """
    Accepts a 'query' parameter from the URL and runs the RAG pipeline.
    Example: http://<AWS-IP>:8000/diagnose?query=left arm pain
    """
    if not query:
        raise HTTPException(status_code=400, detail="Query parameter is required.")
    
    try:
        # 1. Execute the LangGraph State Machine
        result = agent_app.invoke({"question": query})
        
        # 2. Deterministic JSON Extraction (from our previous fix)
        raw_text = result["answer"]
        start_idx = raw_text.find('{')
        end_idx = raw_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx != 0:
            clean_json_str = raw_text[start_idx:end_idx]
            parsed_json = json.loads(clean_json_str)
            return JSONResponse(content=parsed_json)
        else:
            # Fallback if the LLM completely failed JSON structure
            return JSONResponse(content={"topic": "Error", "answer": raw_text, "confidence": 0.0})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))