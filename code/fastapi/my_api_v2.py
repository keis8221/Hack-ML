from fastapi import FastAPI

from pydantic import BaseModel
from typing import List

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir+"/mlike/")
import main

# class BookRecommendation(BaseModel):
#     title: str
#     isbn: str
#     itemUrl: str
#     item_price: int

# class RecommendationRequest(BaseModel):
#     recommendBooks: list[BookRecommendation]

app = FastAPI()

@app.get("/")
async def root():
    output = main.BooksRecommendation()
    recommend_books = output.get_recommend()['recommendBooks']
    return recommend_books
    # rec_lsts = []
    # for book in recommend_books:
    #     rec_lsts.append(
    #         {"res": "ok", 
    #         "title": book.title, 
    #         "isbn": book.isbn,
    #         "itemUrl": book.itemUrl,
    #         "itemUrl": book.item_price
    #         })
    # return rec_lsts

## pip install fastapi, uvicorn
## uvicorn my_api:app --port 8080   
## curl -X POST -H "Content-Type: application/json" http://localhost:8080/recommend

## curl -X POST -H "Content-Type: application/json" http://localhost:5000
