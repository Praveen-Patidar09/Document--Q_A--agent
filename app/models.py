from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentPart:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    chunk: Chunk
    similarity: float
    lexical_score: float
    rank_score: float
