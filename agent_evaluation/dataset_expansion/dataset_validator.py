def validate_sample(sample):
    """
    Ensures generated samples follow the required schema.
    """
    required_fields = [
        "id",
        "post",
        "claims",
        "reference_evidence",
        "ground_truth_verdict",
        "expected_score_range"
    ]

    for field in required_fields:
        if field not in sample:
            raise ValueError(f"Missing field: {field}")
            
    # Additional semantic checks
    if not isinstance(sample["id"], int):
        raise ValueError(f"ID must be an integer, got {type(sample['id'])}")
    if not isinstance(sample["post"], str) or len(sample["post"]) == 0:
        raise ValueError("Post must be a non-empty string")
    if not isinstance(sample["claims"], list) or len(sample["claims"]) == 0:
        raise ValueError("Claims must be a non-empty list")
    if not isinstance(sample["reference_evidence"], list):
        raise ValueError("Reference evidence must be a list")
    if sample["ground_truth_verdict"] not in ["SUPPORTED", "REFUTED", "MISLEADING", "UNVERIFIED"]:
        raise ValueError(f"Invalid ground truth verdict: {sample['ground_truth_verdict']}")
    if not isinstance(sample["expected_score_range"], list) or len(sample["expected_score_range"]) != 2:
        raise ValueError("Expected score range must be a list of 2 numbers")

    return True

def validate_dataset(dataset):
    """
    Validates the entire dataset.
    """
    for i, sample in enumerate(dataset):
        try:
            validate_sample(sample)
        except Exception as e:
            raise ValueError(f"Sample at index {i} failed validation: {e}")
    return True
