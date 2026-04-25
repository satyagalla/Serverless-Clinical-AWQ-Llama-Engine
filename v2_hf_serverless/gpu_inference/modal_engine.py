import modal
import os
import time
from pydantic import BaseModel

class RequestPayload(BaseModel):
    question: str
    context: str
    api_key: str

# 1. Update the image to pull vLLM
image = (
    modal.Image.debian_slim()
    .pip_install(
        "vllm", 
        "transformers",
        "fastapi",
        "pydantic"
    )
    .env({
        # Routes PyTorch's heavy C++ Triton kernels
        "TORCHINDUCTOR_CACHE_DIR": "/weights/compile_cache/torch",
        
        # THE FIX: Routes vLLM's Dynamo bytecode and graph wrappers
        "VLLM_CACHE_ROOT": "/weights/compile_cache/vllm_root" 
    })
)

app = modal.App("medical_llama-3.1-8b-instruct_lora_awq_inference_vllm")
awq_volume = modal.Volume.from_name("awq-volume")

@app.cls(
    gpu="A10G", 
    image=image, 
    volumes={"/weights": awq_volume}, 
    scaledown_window=300, 
    timeout=300,
)
@modal.concurrent(max_inputs=10)
class MedicalLLM:
    @modal.enter()
    def load_model(self):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from transformers import AutoTokenizer

        start_load = time.perf_counter()
        self.model_dir = "/weights/awq-4bit"
        print(f"🧠 Booting vLLM Engine from {self.model_dir}...")
        
        # Load tokenizer purely for chat template formatting
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir, 
            fix_mistral_regex=True
        )

        # Configure the high-performance engine
        engine_args = AsyncEngineArgs(
            model=self.model_dir,
            dtype="bfloat16",
            max_model_len=4096,
            tensor_parallel_size=1,
            gpu_memory_utilization=0.90,
            enforce_eager=False
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        print(f"✅ Engine online in {time.perf_counter() - start_load:.2f}s.")

    @modal.method()
    async def generate(self, question: str, context: str):
        from vllm import SamplingParams
        import uuid

        # 1. Standardize formatting without hardcoding Jinja templates
        messages = [
            {"role": "system", "content": f"You are a highly authoritative clinical diagnostic AI. Use the provided USMLE medical context to answer the user's question.\n\nContext:\n{context}"},
            {"role": "user", "content": question}
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )

        # 2. Strict generation boundaries
        sampling_params = SamplingParams(
            temperature=0.1,
            repetition_penalty=1.15,
            max_tokens=300,
            stop_token_ids=[
                self.tokenizer.eos_token_id, 
                self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
        )

        request_id = str(uuid.uuid4())
        results_generator = self.engine.generate(prompt, sampling_params, request_id)

        start_time = time.perf_counter()
        ttft = None
        generated_text = ""
        
        # 3. Asynchronous metric tracking (Replaces TextIteratorStreamer)
        async for request_output in results_generator:
            if ttft is None:
                ttft = time.perf_counter() - start_time
            generated_text = request_output.outputs[0].text

        total_time = time.perf_counter() - start_time
        
        # 4. Hardware Deployment Metrics
        input_len = len(request_output.prompt_token_ids)
        tokens_gen = len(request_output.outputs[0].token_ids)
        generation_time = total_time - ttft
        tps = tokens_gen / generation_time if generation_time > 0 else 0

        print(f"\n--- 🧠 vLLM INFRASTRUCTURE METRICS ---")
        print(f"├─ Input Prompt:      {input_len} tokens")
        print(f"├─ Generated:         {tokens_gen} tokens")
        print(f"├─ TTFT:              {ttft:.3f}s")
        print(f"├─ Throughput:        {tps:.2f} tokens/sec")
        print(f"└─ Total GPU Time:    {total_time:.3f}s")
        print(f"--------------------------------------\n")

        return generated_text

@app.function(
    image=image, 
    secrets=[modal.Secret.from_name("api-secret")]
)
@modal.fastapi_endpoint(method="POST")
async def api_endpoint(payload: RequestPayload):
    t0 = time.perf_counter()
    EXPECTED_KEY = os.environ.get("MICROSERVICE_SECRET_KEY")
    
    if payload.api_key != EXPECTED_KEY:
        return {"error": "🚨 Unauthorized Microservice Access"}
    
    model = MedicalLLM()
    t1 = time.perf_counter()
    
    # Must explicitly await the remote call due to AsyncLLMEngine
    answer = await model.generate.remote.aio(payload.question, payload.context)
    
    t2 = time.perf_counter()
    print(f"🏁 API TOTAL: {t2 - t0:.2f}s (Wait/Exec: {t2 - t1:.2f}s)")
    
    return {"answer": answer}