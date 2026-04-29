import requests
import time

# ⚠️ Replace with the actual URL printed in your `modal serve` terminal
DEV_URL = "https://akhilgalla41--medical-llama-3-1-8b-instruct-lora-awq-f9aee9-dev.modal.run"

payload = {
    "question": "I have a fever",
    "context": "",
    "api_key": "@khIlgalla41"
}

print("Sending request to Modal...")
start_time = time.time()

response = requests.post(DEV_URL, json=payload)

print(f"Request took: {time.time() - start_time:.2f} seconds")
print("\nResponse:")
print(response.json())