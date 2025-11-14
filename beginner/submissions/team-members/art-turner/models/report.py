"""Data models for research reports."""

from typing import List, Optional
from pydantic import BaseModel, Field, HttpUrl


class Source(BaseModel):
    """Model for a source citation."""

    title: str = Field(..., description="Title of the source")
    url: str = Field(..., description="URL of the source")
    snippet: str = Field(..., description="Relevant excerpt or snippet from the source")
    score: Optional[float] = Field(None, description="Relevance score (0-1)")
    why_matters: Optional[str] = Field(
        None, description="Explanation of why this source is important"
    )


class KeyFinding(BaseModel):
    """Model for a key finding with citation."""

    finding: str = Field(..., description="The key finding or insight")
    citations: List[str] = Field(
        default_factory=list,
        description="List of source URLs supporting this finding"
    )


class ResearchReport(BaseModel):
    """Complete research report model."""

    topic: str = Field(..., description="The research topic or question")

    tldr: str = Field(
        ...,
        description="TL;DR summary (≤120 words)",
        max_length=800  # Roughly 120 words
    )

    key_findings: List[KeyFinding] = Field(
        default_factory=list,
        description="List of key findings with citations"
    )

    conflicts_and_caveats: str = Field(
        default="",
        description="Discussion of conflicts between sources and important caveats"
    )

    top_sources: List[Source] = Field(
        default_factory=list,
        description="Top 5 most relevant sources",
        max_length=5
    )

    metadata: Optional[dict] = Field(
        default_factory=dict,
        description="Additional metadata (timestamp, model used, etc.)"
    )

    def model_dump_summary(self) -> dict:
        """Return a summary version of the report."""
        return {
            "topic": self.topic,
            "tldr": self.tldr,
            "num_findings": len(self.key_findings),
            "num_sources": len(self.top_sources),
        }

    class Config:
        """Pydantic configuration."""
        json_schema_extra = {
            "example": {
                "topic": "Recent advances in large language models",
                "tldr": "Large language models have seen significant advances...",
                "key_findings": [
                    {
                        "finding": "Transformer architecture has become dominant",
                        "citations": ["https://example.com/paper1"]
                    }
                ],
                "conflicts_and_caveats": "Some sources disagree on...",
                "top_sources": [
                    {
                        "title": "Attention Is All You Need",
                        "url": "https://example.com/paper",
                        "snippet": "We propose a new architecture...",
                        "score": 0.95,
                        "why_matters": "Foundational paper introducing transformers"
                    }
                ],
                "metadata": {
                    "timestamp": "2025-01-14T10:30:00Z",
                    "model": "gpt-4-turbo-preview"
                }
            }
        }
