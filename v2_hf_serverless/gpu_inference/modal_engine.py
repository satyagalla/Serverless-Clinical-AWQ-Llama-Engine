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
            enforce_eager=True
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        print(f"✅ Engine online in {time.perf_counter() - start_load:.2f}s.")

    @modal.method()
    async def generate(self, question: str, context: str):
        from vllm import SamplingParams
        import uuid

        # The Multi-Turn State Machine Prompt
        messages = [
    {
        "role": "system", 
        "content": (
            "You are an expert, empathetic clinical diagnostic AI trained on USMLE standards. "
            "You must respond directly to the patient in a natural, concise, conversational tone. "
            "CRITICAL: Never narrate your internal instructions, reference rules, or explain your formatting choices to the user. Speak directly as the physician.\n\n"
            "Analyze the conversation and respond using ONLY the appropriate behavior below:\n\n"
            "- IF purely a medical question (no personal symptoms): Provide a direct, authoritative answer using the provided context in under 4 sentences.\n"
            "- IF a new symptom intake (no specific question): Speak as a warm intake physician. Validate their discomfort briefly. Weave EXACTLY 2 to 3 specific differential diagnoses from the context into your response. (CRITICAL: Do NOT list more than 3 possibilities). Conclude with exactly ONE targeted, clinical follow-up question to narrow the scope (e.g., ask about duration, severity, or associated symptoms). Maximum response length: 4 sentences.\n"
            "- IF a new intake WITH symptoms AND a specific question:\n"
            "  1. Write a brief empathetic paragraph (1-2 sentences) validating discomfort and weaving in EXACTLY 2 to 3 differential diagnoses.\n"
            "  2. Write a Markdown bulleted list directly answering their question.\n"
            "  (CRITICAL: Stop generating immediately after the list. No concluding paragraph. No follow-up questions.)\n"
            "- IF it is a conversational follow-up (user is replying, adding history, or using filler): Acknowledge the new information naturally. DO NOT act like a new patient intake. DO NOT repeat differential diagnoses already listed. Move the diagnostic process forward with the next logical clinical follow-up question. Maximum response length: 3 sentences.\n\n"
            f"Context:\n{context}"
        )
    },
    {
        "role": "user", 
        "content": question
    }
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