from fastapi import FastAPI
app = FastAPI()

from pydantic import BaseModel
from typing import List

import main

class Preference(BaseModel):
    preferences: List[str]

@app.post("/recommend")
async def recommend():
    output = main.BooksRecommendation()
    return output.get_recommend()

## uvicorn my_api:app --port 8080   
## curl -X POST -H "Content-Type: application/json" http://localhost:8080/recommend
