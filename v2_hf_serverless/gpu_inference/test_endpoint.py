import requests
import time

# ⚠️ Replace with the actual URL printed in your `modal serve` terminal
DEV_URL = "https://akhilgalla41--llama3-8b-lora-medical-inference-awq-a-e9ecd9-dev.modal.run"

payload = {
    "question": "What is the primary mechanism of action for Metformin?",
    "context": "Metformin decreases hepatic glucose production, decreases intestinal absorption of glucose, and improves insulin sensitivity by increasing peripheral glucose uptake and utilization.",
    "api_key": "@khIlgalla41"
}

print("Sending request to Modal...")
start_time = time.time()

response = requests.post(DEV_URL, json=payload)

print(f"Request took: {time.time() - start_time:.2f} seconds")
print("\nResponse:")
print(response.json())