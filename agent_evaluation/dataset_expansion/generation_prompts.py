VARIANT_GENERATION_PROMPT = """
You are generating evaluation data for a misinformation detection system.

Original Post:
{post}

Underlying Claim:
{claim}

Ground Truth Label:
{verdict}

Task:
Generate 9 new social media posts that express the SAME claim.

Rules:
- The meaning of the claim must NOT change
- The factual correctness must remain identical
- The tone can vary (headline, tweet, caption, question, breaking news)
- Avoid adding new facts not present in the claim
- Keep each post realistic and natural

Output ONLY a JSON array:

[
"variant 1",
"variant 2",
"variant 3",
...
]
"""
