from fastapi import FastAPI
app = FastAPI()

from pydantic import BaseModel
from typing import List

import sys
import os

# 現在のディレクトリを取得
current_dir = os.path.dirname(os.path.abspath(__file__))
# 親ディレクトリのパスを取得
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
# 親ディレクトリをPythonのパスに追加
sys.path.append(parent_dir+"/img2text/")
# モジュールをインポート
import main

class Preference(BaseModel):
    preferences: List[str]

@app.post("/recommend")
async def recommend():
    output = main.BooksRecommendation()
    return output.get_recommend()

## pip install fastapi, uvicorn
## uvicorn my_api:app --port 8080   
## curl -X POST -H "Content-Type: application/json" http://localhost:8080/recommend
