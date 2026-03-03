import os
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("lora-finetune-llama-3-8b-medical")

# Delete all vectors in the default namespace
index.delete(delete_all=True)

print("✅ Pinecone index 'lora-finetune-llama-3-8b-medical' successfully wiped clean.")