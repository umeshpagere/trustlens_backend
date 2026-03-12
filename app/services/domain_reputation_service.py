async def evaluate_domain(url: str | None) -> dict:
    return {
        "domainTrustScore": 65,
        "domain": url,
        "domainAgeDays": None,
        "httpsSecure": bool(url and url.startswith("https")),
        "isTrustedSource": False,
        "isBlacklisted": False,
        "riskFactors": ["Domain check unavailable; simulated service"]
    }
