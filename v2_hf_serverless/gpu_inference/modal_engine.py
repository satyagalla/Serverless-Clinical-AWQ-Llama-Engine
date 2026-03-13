import os
import modal
from pydantic import BaseModel

# --- 1. MODEL CACHING FUNCTION (Defined first) ---
def download_base_model():
    """Bakes the 15GB model directly into the Docker image."""
    import os
    from huggingface_hub import snapshot_download
    print("📥 Baking Llama-3 Base Model into Docker Image...")
    # Using snapshot_download ensures all shards are present before the image is finalized
    snapshot_download("meta-llama/Meta-Llama-3-8B-Instruct", token=os.environ["HF_TOKEN"])

# --- 2. DEFINE THE ENVIRONMENTS ---
gpu_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("torch", "transformers", "peft", "accelerate", "bitsandbytes", "huggingface_hub", "pydantic")
    .env({"CACHE_BUSTER": "1"})
    .run_function(download_base_model, secrets=[modal.Secret.from_name("hf-secret")])
)

web_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("fastapi[standard]", "pydantic")
)

app = modal.App("llama3-8b_lora_medical_inference")
lora_volume = modal.Volume.from_name("lora-volume")

# --- 3. THE API PAYLOAD SCHEMA ---
class RequestPayload(BaseModel):
    question: str
    context: str
    api_key: str

# --- 4. THE GPU LIFECYCLE CLASS ---
@app.cls(
    gpu="A10G", 
    image=gpu_image, 
    volumes={"/weights": lora_volume}, 
    secrets=[modal.Secret.from_name("hf-secret")],
    scaledown_window=2, # Keep GPU warm
    timeout=900 
)
class MedicalLLM:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        print("🧠 Loading Base Model and Tokenizer...")
        BASE_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=os.environ["HF_TOKEN"])
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            token=os.environ["HF_TOKEN"]
        )
        
        print("🔗 Attaching Medical LoRA Adapter from Volume...")
        self.model = PeftModel.from_pretrained(base_model, "/weights/lora-production")
        self.model.eval() # Ensure inference mode

    @modal.method()
    def generate(self, question: str, context: str):
        import time
        import torch
        
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a highly authoritative clinical diagnostic AI. Use the provided USMLE medical context to answer the user's question.

Context:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs['input_ids'].shape[1]
        
        # --- PRO METRICS INITIALIZATION ---
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        
        with torch.no_grad():
            # First Token (TTFT)
            first_token_output = self.model.generate(
                **inputs, 
                max_new_tokens=1, 
                do_sample=True,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            first_token_event.record()
            
            # Full Response
            full_output = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.1,
                eos_token_id=[self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")],
                pad_token_id=self.tokenizer.eos_token_id
            )
            end_event.record()

        torch.cuda.synchronize()
        ttft = start_event.elapsed_time(first_token_event) / 1000  
        total_time = start_event.elapsed_time(end_event) / 1000
        
        tokens_gen = full_output[0].shape[0] - input_len
        tps = tokens_gen / (total_time - ttft) if (total_time - ttft) > 0 else 0

        print(f"\n--- 🧠 LLM INFRASTRUCTURE METRICS ---")
        print(f"├─ Input Prompt:      {input_len} tokens")
        print(f"├─ TTFT:              {ttft:.3f}s")
        print(f"├─ Throughput:        {tps:.2f} tokens/sec")
        print(f"└─ Total GPU Time:    {total_time:.3f}s")
        print(f"--------------------------------------\n")

        return self.tokenizer.decode(full_output[0][input_len:], skip_special_tokens=True)

# --- 5. SECURE WEB ENDPOINT ---
@app.function(
    image=web_image, 
    secrets=[modal.Secret.from_name("api-secret")]
)
@modal.fastapi_endpoint(method="POST")
def api_endpoint(payload: RequestPayload):
    import time
    
    t0 = time.perf_counter()
    EXPECTED_KEY = os.environ.get("MICROSERVICE_SECRET_KEY")
    
    if payload.api_key != EXPECTED_KEY:
        return {"error": "🚨 Unauthorized Microservice Access"}
    
    model = MedicalLLM()
    t1 = time.perf_counter()
    
    # .remote() hands the execution off to the A10G class
    answer = model.generate.remote(payload.question, payload.context)
    
    t2 = time.perf_counter()
    print(f"🏁 API TOTAL: {t2 - t0:.2f}s (Wait/Exec: {t2 - t1:.2f}s)")
    
    return {"answer": answer}