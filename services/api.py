from .DataModel import QueryRequestModel, QueryResponseModel
from scripts.encoders import get_textbook_embeddings

async def query_func(request:QueryRequestModel, current_path:str) ->QueryResponseModel:
    if request.mode == 'Q&A':
        file_name, simantic_score = await get_textbook_embeddings(request.query, current_path)
        response = {
            "file_name": file_name,
            "simantic_score": simantic_score
        }
        
        return QueryResponseModel(**response)
