import os
# NOTE: /app/model_cache is the Docker container path.
# On local Mac, we do NOT set HF_HOME so models cache in ~/.cache/huggingface (always writable).
# Uncomment the two lines below only when building the Docker image:
# os.environ["HF_HOME"] = "/app/model_cache"
# os.environ["SENTENCE_TRANSFORMERS_HOME"] = "/app/model_cache"


from sentence_transformers import SentenceTransformer
from transformers import pipeline
import spacy
import nltk

print("Downloading sentence-transformer model (all-MiniLM-L6-v2)...")
SentenceTransformer("all-MiniLM-L6-v2")

# Fix-4: Pre-download NLI model used by evidence_filter.py at runtime
print("Downloading NLI model (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)...")
pipeline("text-classification", model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli", device=-1)

print("Downloading spaCy model (en_core_web_sm)...")
spacy.cli.download("en_core_web_sm")

# Fix-4: Pre-download NLTK punkt so evidence_alignment.py never hits the network at request time
print("Downloading NLTK punkt tokenizer...")
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

print("All models pre-downloaded successfully!")

