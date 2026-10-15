from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)
    document_id: str = Field(pattern=r"^[a-f0-9]{24}$")

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question cannot be blank.")
        return value


class Source(BaseModel):
    filename: str
    chunk: int | None = None
    page: int | None = None
    slide: int | None = None
    sheet: str | None = None
    section: str | None = None
    title: str | None = None
    headings: list[str] | None = None
    columns: str | None = None
    structure: str | None = None
    similarity: float


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    chunks: int
    created_at: str | None = None
