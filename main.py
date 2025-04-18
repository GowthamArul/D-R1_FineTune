import uvicorn
from fastapi import FastAPI
from starlette.responses import JSONResponse
from services.api import query_func
from services.DataModel import QueryRequestModel, QueryResponseModel
import os

current_path = os.getcwd()
app = FastAPI(
    title='D-R1 Distilled RAG Application',
    description="",
    version= "1.0.0"
)

@app.get("/")
async def root():
    """
    Welcome message on the application startup
    """
    return {"Welcome to the AE&I applciation"}

@app.get("/get_status")
async def get_status():
    """
    Status of the Application
    """
    return JSONResponse({"status": "Status OK"})


@app.post("/query",
          tags=["Query"],
          summary="Query the text document",
          response_model=QueryResponseModel,)
async def query_api(request: QueryRequestModel):
    """
    Purpose: Query the text document using the retriever and generator
    """
    response = await query_func(request, current_path)
    
    return response

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=8000)