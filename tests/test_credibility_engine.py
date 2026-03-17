import unittest
import math
from app.services.credibility_engine import (
    calculate_credibility_score,
    compute_weighted_final_result,
    _classify_claim_type,
    WEIGHTS,
    NEUTRAL_SCORES
)

class TestCredibilityEngine(unittest.TestCase):

    def test_calculate_credibility_score_perfect(self):
        scores = {
            "evidenceSupportScore": 100.0,
            "sourceTrustScore": 100.0,
            "mediaAuthenticityScore": 100.0,
            "semanticRiskScore": 100.0
        }
        # 100*0.5 + 100*0.2 + 100*0.2 + 100*0.1 = 50 + 20 + 20 + 10 = 100
        self.assertEqual(calculate_credibility_score(scores), 100.0)

    def test_calculate_credibility_score_zero(self):
        scores = {
            "evidenceSupportScore": 0.0,
            "sourceTrustScore": 0.0,
            "mediaAuthenticityScore": 0.0,
            "semanticRiskScore": 0.0
        }
        self.assertEqual(calculate_credibility_score(scores), 0.0)

    def test_calculate_credibility_score_neutral(self):
        scores = {
            "evidenceSupportScore": 50.0,
            "sourceTrustScore": 50.0,
            "mediaAuthenticityScore": 75.0,
            "semanticRiskScore": 50.0
        }
        # 50*0.5 + 50*0.2 + 75*0.2 + 50*0.1 = 25 + 10 + 15 + 5 = 55
        self.assertEqual(calculate_credibility_score(scores), 55.0)

    def test_classify_claim_type_static(self):
        self.assertEqual(_classify_claim_type("The capital of France is Paris"), "static")
        self.assertEqual(_classify_claim_type("Water consists of hydrogen and oxygen"), "static")

    def test_classify_claim_type_dynamic(self):
        self.assertEqual(_classify_claim_type("Protests in London today"), "dynamic")
        self.assertEqual(_classify_claim_type("Explosion in Beirut"), "dynamic")
        self.assertEqual(_classify_claim_type("Election results 2024"), "dynamic")

    def test_compute_weighted_final_result_penalties(self):
        # AI Video Penalty: -40
        res = compute_weighted_final_result(
            evidence_verification_score=100.0,
            ai_video_probability=0.8
        )
        # Base score should be around 100 * 0.5 + neutral others
        # Neutral scores: source=50, media=75, risk=100 (exp(0))
        # 100*0.5 + 50*0.2 + 75*0.2 + 100*0.1 = 50 + 10 + 15 + 10 = 85
        # 85 - 40 = 45
        self.assertEqual(res["credibility_score"], 45)

        # Context Reuse Penalty: -25
        res = compute_weighted_final_result(
            evidence_verification_score=100.0,
            context_reuse_detected=True
        )
        # 85 - 25 = 60
        self.assertEqual(res["credibility_score"], 60)

    def test_compute_weighted_final_result_breaking_news(self):
        res = compute_weighted_final_result(
            evidence_verification_score=80.0,
            breaking_news_detected=True,
            breaking_news_confidence=90.0
        )
        # When breaking news, it uses breaking_news_confidence instead of evidenceSupportScore
        # 90*0.5 + 50*0.2 + 75*0.2 + 100*0.1 = 45 + 10 + 15 + 10 = 80
        self.assertEqual(res["credibility_score"], 80)

if __name__ == '__main__':
    unittest.main()
