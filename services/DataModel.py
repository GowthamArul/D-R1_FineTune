from pydantic import BaseModel


class QueryRequestModel(BaseModel):
    """
    Purpose: Query request model
    """
    query: str
    mode: str = "Q&A"


class QueryResponseModel(BaseModel):
    file_name: str | None = None
    simantic_score: float
    response: str | None = None