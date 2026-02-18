import json
from datasets import load_dataset
from gliner import GLiNER
from tqdm import tqdm

# --- 1. LOAD LOCAL MODEL (Zero API Key) ---
print("Loading GLiNER model (Runs on CPU)...")
# This downloads once (~300MB) and lives on your machine forever.
model = GLiNER.from_pretrained("urchade/gliner_small-v2.1")

# --- 2. LOAD DATA ---
print("Loading Flashcards...")
dataset = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
# We take 500 rows. Quality > Quantity.
subset = dataset.shuffle(seed=42).select(range(500))

training_data = []

print(f"Transforming {len(subset)} rows...")

# --- 3. THE EXTRACTION LOOP ---
for row in tqdm(subset):
    text = row['input']
    
    # We ask the model to find these specific entities in the text
    labels = ["Medical Condition", "Disease", "Symptom", "Treatment", "Drug", "Body Part"]
    
    # Predict (Fast!)
    entities = model.predict_entities(text, labels)
    
    # LOGIC: If we find an entity, that's our 'Topic'. 
    # If not, we fallback to a generic tag.
    if entities:
        # Pick the most relevant entity (usually the first one found)
        topic = entities[0]['text'].title()
    else:
        topic = "General Medical Query"

    # --- 4. CREATE TRAINING ENTRY ---
    # This is the EXACT format Unsloth needs for fine-tuning
    entry = {
        "instruction": "You are a medical diagnostic assistant. Output a valid JSON response.",
        "input": row['input'],
        "output": json.dumps({
            "topic": topic, 
            "answer": row['output'],
            "confidence": "High"
        })
    }
    training_data.append(entry)

# --- 5. SAVE ---
output_file = "data/medical_instruct_train.jsonl"
with open(output_file, "w") as f:
    for entry in training_data:
        f.write(json.dumps(entry) + "\n")

print(f"\nSUCCESS! Data saved to '{output_file}'")
print("You are now ready to train.")