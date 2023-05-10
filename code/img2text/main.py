import torch
import torch.nn as nn
import numpy as np
import os
import random
import gdown
import json
import captioning.utils.opts as opts
import captioning.models as models
import captioning.utils.misc as utils
import clip as clip
import pytorch_lightning as pl
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.models.vision_transformer import resize_pos_embed
import requests

import spacy
from collections import Counter
import numpy as np
from scipy.spatial.distance import cosine
import sister

class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.prefix + 'interrupt.ckpt')
        self._save_model(filepath)

class Img2TxtCLIPReward():
    def __init__(self):
        self.device = 'cuda'
        reward = 'clips_grammar'

        cfg = f'./configs/phase2/clipRN50_{reward}.yml'
        self.opt = opts.parse_opt(parse=False, cfg=cfg)

        url = "https://drive.google.com/drive/folders/1nSX9aS7pPK4-OTHYtsUD_uEkwIQVIV7W"
        gdown.download_folder(url, quiet=True, use_cookies=False, output="./save/")
        url = "https://drive.google.com/uc?id=1HNRE1MYO9wxmtMHLC8zURraoNFu157Dp"
        gdown.download(url, quiet=True, use_cookies=False, output="./data/")

        dict_json = json.load(open('./data/cocotalk.json'))

        self.ix_to_word = dict_json['ix_to_word']
        vocab_size = len(self.ix_to_word)

        seq_length = 1
        self.opt.vocab_size = vocab_size
        self.opt.seq_length = seq_length

    def load_model_checkpoint(self):
        self.opt.batch_size = 1
        self.opt.vocab = self.ix_to_word

        model = models.setup(self.opt)
        del self.opt.vocab
        ckpt_path = self.opt.checkpoint_path + '-last.ckpt'

        raw_state_dict = torch.load(
            ckpt_path,
            map_location=self.device)
        
        strict = True
        state_dict = raw_state_dict['state_dict']

        if '_vocab' in state_dict:
            model.vocab = utils.deserialize(state_dict['_vocab'])
            del state_dict['_vocab']
        elif strict:
            raise KeyError
        
        if '_opt' in state_dict:
            saved_model_opt = utils.deserialize(state_dict['_opt'])
            del state_dict['_opt']
            # Make sure the saved opt is compatible with the curren topt
            need_be_same = ["caption_model", "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and getattr(self.opt, checkme) in ['updown', 'topdown']:
                    continue
                assert getattr(saved_model_opt, checkme) == getattr(
                    self.opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
        elif strict:
            raise KeyError
        
        res = model.load_state_dict(state_dict, strict)

        model = model.to(self.device)
        model.eval()
        self.model = model

        self.clip_model, self.clip_transform = clip.load("RN50", jit=False, device=self.device)

        self.preprocess = Compose([
            Resize((448, 448), interpolation=Image.BICUBIC),
            CenterCrop((448, 448)),
            ToTensor()
        ])
        self.image_mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(self.device).reshape(3, 1, 1)
        self.image_std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(self.device).reshape(3, 1, 1)

        num_patches = 196
        pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.clip_model.visual.attnpool.positional_embedding.shape[-1],  device=self.device),)
        pos_embed.weight = resize_pos_embed(self.clip_model.visual.attnpool.positional_embedding.unsqueeze(0), pos_embed)
        self.clip_model.visual.attnpool.positional_embedding = pos_embed
    
    def get_imgpath_lst(self):
        dir_path = '/home/tomo/Documents/hack/Hack-ML/hack_data'
        os.chdir(dir_path)
        image_files = [f for f in os.listdir('.') if os.path.isfile(f) and f.lower().endswith(('.jpg'))]
        selected_images = random.sample(image_files, 5)
        imgs_path = [os.path.join(dir_path, name)for name in selected_images]
        return imgs_path

    def get_img2txt(self):
        self.load_model_checkpoint()
        imgpath_lst = self.get_imgpath_lst()
        sents_lst = []
        for img_path in imgpath_lst:
            with torch.no_grad():
                image = self.preprocess(Image.open(img_path).convert("RGB"))
                image = torch.tensor(np.stack([image])).to(self.device)
                image -= self.image_mean
                image /= self.image_std

                tmp_att, tmp_fc = self.clip_model.encode_image(image)
                tmp_att = tmp_att[0].permute(1, 2, 0)
                tmp_fc = tmp_fc[0]

                att_feat = tmp_att

                eval_kwargs = {}
                eval_kwargs.update(vars(self.opt))

                with torch.no_grad():
                    fc_feats = torch.zeros((1,0)).to(self.device)
                    att_feats = att_feat.view(1, 196, 2048).float().to(self.device)
                    att_masks = None

                    tmp_eval_kwargs = eval_kwargs.copy()
                    tmp_eval_kwargs.update({'sample_n': 1})
                    seq, seq_logprobs = self.model(
                    fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                    seq = seq.data

                    sents = utils.decode_sequence(self.model.vocab, seq)
                sents_lst.append(sents[0])
        return sents_lst

# output = Img2TxtCLIPReward()
# sents_lst = output.get_img2txt()

class PredictCategory():
    def __init__(self):
        self.sents_lst = ['a couple of birds flying in the cloudy sky with the clouds behind it', 
                     'a young person jumping on a metal railing with a sign next to the metal railing', 
                     'a guy wearing sunglasses standing on a wooden deck with his cell phone behind him', 
                     'a group of three young women standing together in a mirror with their cell phone behind', 
                     'two young women standing together near a fountain with each other on the concrete surface']
        # output = Img2TxtCLIPReward()
        # self.sents_lst = output.get_img2txt()

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

# output = PredictCategory()
# print(output.get_predict())

class BooksRecommendation():
    def __init__(self):
        base_url = "https://app.rakuten.co.jp/services/api/BooksTotal/Search/20170404?applicationId=1028959429215953336"
        output = PredictCategory()
        new_keyword = output.get_predict()
        new_keyword = "Health"
        url = f"{base_url}&keyword={new_keyword}&sort=%2BitemPrice"

        response = requests.get(url)
        self.json_response = response.json()

    def get_recommend(self):
        recommendBooks = {"recommendBooks": []}
        for i in range(30):
            title = self.json_response['Items'][i]['Item']['title'].replace('\u3000', '')
            isbn = self.json_response['Items'][i]['Item']['isbn']
            item_url = self.json_response['Items'][i]['Item']['itemUrl']
            item_price = self.json_response['Items'][i]['Item']['itemPrice']
            recommendBooks["recommendBooks"].append({"title": title, "isbn": isbn, "itemUrl": item_url, "item_price": item_price})
        return recommendBooks

output = BooksRecommendation()
print(output.get_recommend())