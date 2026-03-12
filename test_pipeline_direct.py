import asyncio
import json
from app.services.evidence_pipeline.pipeline import run_evidence_pipeline

async def main():
    text = "NASA confirmed Earth will experience 6 days of darkness."
    print(f"Testing pipeline with claim: {text}\n")
    
    # run_evidence_pipeline is synchronous in pipeline.py
    results = run_evidence_pipeline(text)
    
    print("--- Pipeline Results ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main())
