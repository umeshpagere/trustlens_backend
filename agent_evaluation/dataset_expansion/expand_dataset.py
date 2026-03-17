import json
import os
import sys
import logging
from openai import AzureOpenAI
from generation_prompts import VARIANT_GENERATION_PROMPT
from dataset_validator import validate_sample

# Ensure the root of the project is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from app.config.settings import Config

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DatasetExpansion")

# For local runtime if venv is used, we might need to load .env manually if not already
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Use the user's requested input path
INPUT_DATASET = "agent_evaluation/dataset/trustlens_eval_dataset_v2.json" 
# Use the user's requested output path
OUTPUT_DATASET = "agent_evaluation/dataset/trustlens_eval_dataset_500.json"

client = AzureOpenAI(
    api_key=Config.AZURE_OPENAI_API_KEY,
    api_version=Config.AZURE_OPENAI_API_VERSION,
    azure_endpoint=Config.AZURE_OPENAI_ENDPOINT
)

def generate_variants(post, claim, verdict):
    """
    Calls Azure OpenAI to generate paraphrased variants.
    """
    prompt = VARIANT_GENERATION_PROMPT.format(
        post=post,
        claim=claim,
        verdict=verdict
    )

    try:
        response = client.chat.completions.create(
            model=Config.AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"} if "gpt-4o" in Config.AZURE_OPENAI_DEPLOYMENT or "gpt-4" in Config.AZURE_OPENAI_DEPLOYMENT else None
        )

        content = response.choices[0].message.content
        logger.debug(f"LLM Response: {content}")
        
        # Parse JSON
        variants_data = json.loads(content)
        
        # If the LLM returned a dict, try to find the list of variants
        if isinstance(variants_data, dict):
            # Try common keys
            for key in ["variants", "posts", "variants_list", "responses", "output"]:
                if key in variants_data and isinstance(variants_data[key], list):
                    return variants_data[key]
            
            # If no obvious key, find any list of strings
            for v in variants_data.values():
                if isinstance(v, list) and len(v) > 0 and isinstance(v[0], str):
                    return v
            
            # If no lists found, maybe the dict itself contains values that are the variants
            if all(isinstance(v, str) for v in variants_data.values()):
                return list(variants_data.values())

        if isinstance(variants_data, list):
            return variants_data
            
        return []
    except Exception as e:
        logger.error(f"Failed to generate variants for post: {post[:50]}... Error: {e}")
        return []

def expand_dataset():
    """
    Main expansion loop.
    """
    input_path = os.path.join(os.getcwd(), INPUT_DATASET)
    output_path = os.path.join(os.getcwd(), OUTPUT_DATASET)

    if not os.path.exists(input_path):
        logger.error(f"Input dataset not found at {input_path}")
        return

    with open(input_path, "r") as f:
        dataset = json.load(f)

    logger.info(f"Loaded dataset: {len(dataset)} samples")
    expanded_dataset = []
    
    # Load existing results if any to resume
    expanded_dataset = []
    processed_ids = set()
    if os.path.exists(output_path):
        try:
            with open(output_path, "r") as f:
                expanded_dataset = json.load(f)
                # Group by original ID to see what's done
                # We know each original sample 1..50 results in 10 records
                # Let's just see which original posts are already in there
                existing_posts = {s["post"] for s in expanded_dataset}
                logger.info(f"Checking existing output... Found {len(expanded_dataset)} records.")
        except Exception:
            pass

    # We'll use a new ID sequence to avoid collisions if any
    current_new_id = len(expanded_dataset) + 1

    for i, sample in enumerate(dataset):
        # Simple resume check: if the original post is already in expanded_dataset, 
        # and it has its variants (id 1..10 for that sample), skip.
        # This is a bit rough but works for this task.
        if any(s["post"] == sample["post"] for s in expanded_dataset):
            logger.info(f"[{i+1}/{len(dataset)}] Skipping Sample ID {sample['id']} (already processed)")
            continue

        logger.info(f"[{i+1}/{len(dataset)}] Expanding Sample ID {sample['id']}...")
        
        # Add original sample with new sequential ID
        orig_sample = sample.copy()
        orig_sample["id"] = current_new_id
        expanded_dataset.append(orig_sample)
        current_new_id += 1

        # Generate 9 variants
        variants = generate_variants(
            sample["post"],
            sample["claims"][0] if sample["claims"] else "",
            sample["ground_truth_verdict"]
        )

        # Limit to 9 as requested
        count = 0
        for variant in variants:
            if count >= 9: break
            try:
                new_sample = sample.copy()
                new_sample["id"] = current_new_id
                new_sample["post"] = variant
                
                if validate_sample(new_sample):
                    expanded_dataset.append(new_sample)
                    current_new_id += 1
                    count += 1
            except Exception as e:
                logger.warning(f"Skipping invalid variant for sample {sample['id']}: {e}")

        # Save every 5 samples
        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
            with open(output_path, "w") as f:
                json.dump(expanded_dataset, f, indent=2)
            logger.info(f"Progress saved: {len(expanded_dataset)} records")

    logger.info(f"Final expanded dataset size: {len(expanded_dataset)}")
    logger.info(f"Saved to {OUTPUT_DATASET}")

if __name__ == "__main__":
    expand_dataset()
