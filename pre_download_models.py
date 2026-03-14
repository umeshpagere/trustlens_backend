import os
os.environ["HF_HOME"] = "/app/model_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/app/model_cache"

from sentence_transformers import SentenceTransformer
import spacy

print("Downloading sentence-transformer model (all-MiniLM-L6-v2)...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading spaCy model (en_core_web_sm)...")
spacy.cli.download("en_core_web_sm")

print("All models pre-downloaded successfully!")
