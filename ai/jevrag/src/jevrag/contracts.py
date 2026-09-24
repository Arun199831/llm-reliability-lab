from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NoulQuestion(BaseModel):
    """Ask Jev for the probability that a statement is true."""

    model_config = ConfigDict(extra="forbid")
    type: str = "noul"
    instructions: str
    criteria: dict[str, str] | None = None


class NoulAnswer(BaseModel):
    model_config = ConfigDict(extra="ignore")
    type: str
    noul: float = Field(..., ge=0.0, le=1.0)


class JevUsage(BaseModel):
    model_config = ConfigDict(extra="ignore")
    input_tokens: int = 0
    output_tokens: int = 0


class JevResponse(BaseModel):
    """Jev's raw reply, before we've picked out any one answer."""

    model_config = ConfigDict(extra="ignore")
    model: str
    answers: dict[str, NoulAnswer]
    usage: JevUsage = Field(default_factory=JevUsage)

    def noul(self, key: str) -> float:
        return self.answers[key].noul
