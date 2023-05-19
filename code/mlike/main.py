import numpy as np
import pandas as pd
import os
import random
import json
from PIL import Image
import requests
import time

import spacy
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine
import sister

class PredictCategory():
    def __init__(self):
        self.category = ['Health',
                    'Education', 
                    'Economy', 
                    'Science', 
                    'Art', 
                    'Environment', 
                    'Politics', 
                    'Sports', 
                    'Food', 
                    'Family']
        
    def get_random_images(self):
        imgdata_dir = '/home/tomo/Documents/hack/Hack-ML/hack_data'
        os.chdir(imgdata_dir)
        image_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(('.jpg'))]
        selected_images = random.sample(image_files, 5)
        return selected_images

    def get_img2text(self):
        df = pd.read_csv("./output/img2txt.csv")
        selected_images = self.get_random_images()
        filtered_data = df[df["img_name"].isin(selected_images)]
        img2text_values = filtered_data["img2text"].tolist()
        img2text_lst = []
        for img2text_value in img2text_values:
            img2text_lst.append(img2text_value)
        return img2text_lst
        
    def predict_category(self):
        sents_lst = self.get_img2text()
        embedder = sister.MeanEmbedding(lang="en")
        nlp = spacy.load('en_core_web_sm')

        def extract_nouns(text):
            doc = nlp(text)
            return [token.text for token in doc if token.pos_ == 'NOUN']

        combined_sents = ' '.join(sents_lst)
        nouns_count = Counter(extract_nouns(combined_sents))
        most_common_lst = nouns_count.most_common(3)
        most_common_lst = [t[0] for t in most_common_lst]

        category_similarities = {}
        for word in most_common_lst:
            word_vec = embedder(word)
            for cat in self.category:
                cat_vec = embedder(cat.lower())
                sim = 1 - cosine(word_vec, cat_vec)
                category_similarities[cat]=sim

        max_key = max(category_similarities, key=category_similarities.get)
        return max_key

class BooksRecommendation():
    def __init__(self):
        base_url = "https://app.rakuten.co.jp/services/api/BooksTotal/Search/20170404?applicationId=1028959429215953336"
        output = PredictCategory()
        new_keyword = output.predict_category()
        # new_keyword = "Health"
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
    

if __name__ == "__main__":
    time.sleep(3)
    output = BooksRecommendation()
    print(output.get_recommend())

