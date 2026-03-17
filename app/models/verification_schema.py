"""
Pydantic schema for structured claim verification output — Part-10.

Used to validate and normalise the LLM response before passing it
downstream to the credibility engine.
"""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, field_validator


class ReasoningStep(BaseModel):
    """Per-evidence reasoning step produced by the LLM."""
    evidence_id: int = Field(..., description="1-indexed evidence number as presented in the prompt")
    relation: Literal["SUPPORT", "CONTRADICT", "NEUTRAL"] = Field(
        ..., description="Relationship between this evidence and the claim"
    )
    reason: str = Field(..., description="Short explanation of the relation")

    @field_validator("relation", mode="before")
    @classmethod
    def normalise_relation(cls, v: str) -> str:
        mapping = {
            "SUPPORT": "SUPPORT",
            "SUPPORTED": "SUPPORT",
            "SUPPORTS": "SUPPORT",
            "CONTRADICT": "CONTRADICT",
            "CONTRADICTED": "CONTRADICT",
            "CONTRADICTS": "CONTRADICT",
            "NEUTRAL": "NEUTRAL",
            "IRRELEVANT": "NEUTRAL",
        }
        return mapping.get(str(v).upper(), "NEUTRAL")


class VerificationResult(BaseModel):
    """Full structured verification output."""
    reasoning_steps: List[ReasoningStep] = Field(default_factory=list)
    verdict: Literal["SUPPORTED", "CONTRADICTED", "UNVERIFIED"] = Field(
        ..., description="Final verdict aggregated across all reasoning steps"
    )
    credibility_score: int = Field(..., ge=0, le=100)
    confidence: float = Field(..., ge=0.0, le=1.0)
    explanation: str = Field(..., description="Final reasoning summary")

    @field_validator("verdict", mode="before")
    @classmethod
    def normalise_verdict(cls, v: str) -> str:
        mapping = {
            "SUPPORTED": "SUPPORTED",
            "SUPPORT": "SUPPORTED",
            "TRUE": "SUPPORTED",
            "CONTRADICTED": "CONTRADICTED",
            "CONTRADICT": "CONTRADICTED",
            "FALSE": "CONTRADICTED",
            "UNVERIFIED": "UNVERIFIED",
            "INCONCLUSIVE": "UNVERIFIED",
            "MIXED": "UNVERIFIED",
        }
        return mapping.get(str(v).upper(), "UNVERIFIED")

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: float) -> float:
        return max(0.0, min(1.0, float(v)))

    @field_validator("credibility_score", mode="before")
    @classmethod
    def clamp_score(cls, v) -> int:
        return max(0, min(100, int(v)))
