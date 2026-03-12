from sentence_transformers import SentenceTransformer
import spacy

print("Downloading sentence-transformer model (all-MiniLM-L6-v2)...")
SentenceTransformer("all-MiniLM-L6-v2")

print("Downloading spaCy model (en_core_web_sm)...")
spacy.cli.download("en_core_web_sm")

print("All models pre-downloaded successfully!")
