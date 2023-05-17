import spacy
import sister
from collections import Counter
from scipy.spatial.distance import cosine
#from img2text_clipreward import Img2TxtCLIPReward
from img2text_clipreward import Img2TxtCLIPReward
from pytorch_lightning.callbacks import ModelCheckpoint


class PredictCategory():
    def __init__(self):
        self.sents_lst = ['a couple of birds flying in the cloudy sky with the clouds behind it', 
                     'a young person jumping on a metal railing with a sign next to the metal railing', 
                     'a guy wearing sunglasses standing on a wooden deck with his cell phone behind him', 
                     'a group of three young women standing together in a mirror with their cell phone behind', 
                     'two young women standing together near a fountain with each other on the concrete surface']
        output = Img2TxtCLIPReward()
        self.sents_lst = output.get_img2txt()
        

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
        
    def get_predict(self):
        embedder = sister.MeanEmbedding(lang="en")

        nlp = spacy.load('en_core_web_sm')

        def extract_nouns(text):
            doc = nlp(text)
            return [token.text for token in doc if token.pos_ == 'NOUN']

        combined_sents = ' '.join(self.sents_lst)
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

#output = PredictCategory()
#category = output.get_predict()
#print(output.get_predict())