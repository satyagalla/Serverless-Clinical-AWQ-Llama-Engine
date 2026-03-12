import json
import os
from dotenv import load_dotenv
import instructor
from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from datasets import load_dataset
from tqdm import tqdm

load_dotenv()

# --- CONFIGURATION ---
# Get Key: https://aistudio.google.com/app/apikey
# Ensure you set this in your .env file:
# GOOGLE_API_KEY = "YOUR_KEY_HERE"


# --- 1. DEFINE THE BLUEPRINT ---
class MedicalFlashcard(BaseModel):
    topic: str = Field(description="The primary medical condition, drug, or concept.")
    answer: str = Field(description="The clinical answer derived from the text.")
    confidence: Literal["High", "Medium", "Low"] = Field(description="Reliability.")


# --- 2. SETUP THE TEACHER MODEL (NEW SDK) ---
# We wrap the new 'genai.Client' with instructor
client = instructor.from_genai(
    client=genai.Client(api_key=os.getenv("GOOGLE_API_KEY")),
    mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS,
)

# --- 3. LOAD DATA ---
print("Loading Medical Flashcards...")
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
# Select 500 rows for the demo
subset = dataset.shuffle(seed=42).select(range(500))

training_data = []

# --- 4. THE LOOP ---
print(f"Transforming {len(subset)} rows using Gemini 1.5 Flash...")

for row in tqdm(subset):
    original_text = f"Q: {row['input']} \nA: {row['output']}"
    
    try:
        # The 'response_model' magic forces the specific JSON structure
        resp = client.chat.completions.create(
            model="gemini-1.5-flash", # Use the flash model for speed/free-tier
            response_model=MedicalFlashcard,
            messages=[
                {
                    "role": "user", 
                    "content": f"Extract the medical topic and structure this answer: \n\n{original_text}"
                }
            ],
        )

        # Convert to Unsloth Training Format
        entry = {
            "instruction": "You are a medical diagnostic assistant. Output a valid JSON response.",
            "input": row['input'],
            "output": json.dumps({
                "topic": resp.topic,
                "answer": resp.answer,
                "confidence": resp.confidence
            })
        }
        training_data.append(entry)

    except Exception as e:
        print(f"Error on row: {e}")

# --- 5. SAVE ---
output_file = "data/medical_instruct_train_v2.jsonl"
with open(output_file, "w") as f:
    for entry in training_data:
        f.write(json.dumps(entry) + "\n")

print(f"\nSUCCESS! Saved {output_file}")
print("You are ready for the Training Step.")