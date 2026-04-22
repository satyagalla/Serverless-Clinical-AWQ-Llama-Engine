import modal
import os
from pydantic import BaseModel

# --- 1. MISSING DEFINITIONS ---
class RequestPayload(BaseModel):
    question: str
    context: str
    api_key: str

# --- 2. DEFINE THE ENVIRONMENTS ---
# Corrected: Removed "flash-attn" to ensure a strict ablation from bitsandbytes to AWQ
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", 
        "transformers", 
        "accelerate", 
        "autoawq", 
        "fastapi",
        "pydantic",
        "compressed-tensors"
    )
)

app = modal.App("llama3-8b_lora_medical_inference_awq")
awq_volume = modal.Volume.from_name("awq-volume")

@app.cls(
    gpu="A10G", 
    image=image, 
    volumes={"/weights": awq_volume}, 
    scaledown_window=300, 
    timeout=300 
)
class MedicalLLM:
    @modal.enter()
    def load_model(self):
        import time
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM

        start_load = time.perf_counter()

        self.model_dir = "/weights/awq-4bit"
        print(f"🧠 Loading AWQ Model from {self.model_dir}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
            dtype=torch.float16 # Explicitly set for AWQ
        )
        load_time = time.perf_counter() - start_load
        print(f"✅ GPU is ready in {load_time:.2f}s.")

    @modal.method()
    def generate(self, question: str, context: str):
        import time
        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a highly authoritative clinical diagnostic AI. Use the provided USMLE medical context to answer the user's question.

Context:
{context}<|eot_id|><|start_header_id|>user<|end_header_id|>

{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        input_len = inputs['input_ids'].shape[1]
        
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.1,
            streamer=streamer,
            eos_token_id=[
                self.tokenizer.eos_token_id, 
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ],
            pad_token_id=self.tokenizer.eos_token_id
        )

        start_time = time.perf_counter()
        
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        generated_text = ""
        ttft = None

        for new_text in streamer:
            if ttft is None:
                ttft = time.perf_counter() - start_time
            generated_text += new_text

        thread.join()
        total_time = time.perf_counter() - start_time
        
        tokens_gen = len(self.tokenizer.encode(generated_text, add_special_tokens=False))
        
        generation_time = total_time - ttft
        tps = tokens_gen / generation_time if generation_time > 0 else 0

        print(f"\n--- 🧠 LLM INFRASTRUCTURE METRICS ---")
        print(f"├─ Input Prompt:      {input_len} tokens")
        print(f"├─ Generated:         {tokens_gen} tokens")
        print(f"├─ TTFT:              {ttft:.3f}s")
        print(f"├─ Throughput:        {tps:.2f} tokens/sec")
        print(f"└─ Total GPU Time:    {total_time:.3f}s")
        print(f"--------------------------------------\n")

        return generated_text

# --- 5. SECURE WEB ENDPOINT ---
@app.function(
    image=image, 
    secrets=[modal.Secret.from_name("api-secret")]
)
@modal.fastapi_endpoint(method="POST")
def api_endpoint(payload: RequestPayload):
    import time
    import os 
    
    t0 = time.perf_counter()
    EXPECTED_KEY = os.environ.get("MICROSERVICE_SECRET_KEY")
    
    if payload.api_key != EXPECTED_KEY:
        return {"error": "🚨 Unauthorized Microservice Access"}
    
    model = MedicalLLM()
    t1 = time.perf_counter()
    
    answer = model.generate.remote(payload.question, payload.context)
    
    t2 = time.perf_counter()
    print(f"🏁 API TOTAL: {t2 - t0:.2f}s (Wait/Exec: {t2 - t1:.2f}s)")
    
    return {"answer": answer}