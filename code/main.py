import requests
from predict_category import PredictCategory
import pytorch_lightning as pl
import os
from pytorch_lightning.callbacks import ModelCheckpoint


class BooksRecommendation():
    def __init__(self):
        base_url = "https://app.rakuten.co.jp/services/api/BooksTotal/Search/20170404?applicationId=1028959429215953336"
        output = PredictCategory()
        new_keyword = output.get_predict()

        url = f"{base_url}&keyword={new_keyword}&sort=%2BitemPrice"
        response = requests.get(url)
        self.json_response = response.json()

    def get_recommend(self):
        recommendBooks = {"recommendBooks": []}
        for i in range(30):
            item = self.json_response['Items'][i]['Item']
            title, isbn, item_url, item_price = (item['title'].replace('\u3000', ''), 
                                                 item['isbn'], 
                                                 item['itemUrl'], 
                                                 item['itemPrice'])
            
            recommendBooks["recommendBooks"].append({"title": title, 
                                                     "isbn": isbn, 
                                                     "itemUrl": item_url, 
                                                     "item_price": item_price})
        return recommendBooks

output = BooksRecommendation()
print(output.get_recommend())
