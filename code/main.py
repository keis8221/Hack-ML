# Import library
import torch
import torch.nn as nn
import numpy as np
import json
import os
import captioning.utils.opts as opts
import captioning.models as models
import captioning.utils.misc as utils
import pytorch_lightning as pl
import gdown
import clip
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
from timm.models.vision_transformer import resize_pos_embed
import path
import os
 
# Checkpoint class
class ModelCheckpoint(pl.callbacks.ModelCheckpoint):
    def on_keyboard_interrupt(self, trainer, pl_module):
        # Save model when keyboard interrupt
        filepath = os.path.join(self.dirpath, self.prefix + 'interrupt.ckpt')
        self._save_model(filepath)

class Img2TxtCLIPReward:
    def __init__(self):
        # Device and model configurations
        self.device = 'cuda'
        reward = 'clips_grammar'

        cfg = f'./configs/phase2/clipRN50_{reward}.yml'
        self.opt = opts.parse_opt(parse=False, cfg=cfg)

        # Doenload pretraied checkpoint
        url = "https://drive.google.com/drive/folders/1nSX9aS7pPK4-OTHYtsUD_uEkwIQVIV7W"
        gdown.download_folder(url, quiet=True, use_cookies=False, output="save/")
        # --- Load vocabulary ---
        url = "https://drive.google.com/uc?id=1HNRE1MYO9wxmtMHLC8zURraoNFu157Dp"
        gdown.download(url, quiet=True, use_cookies=False, output="data/")

        dict_json = json.load(open('./data/cocotalk.json'))

        ix_to_word = dict_json['ix_to_word']
        vocab_size = len(ix_to_word)

        seq_length = 1
        self.opt.vocab_size = vocab_size
        self.opt.seq_length = seq_length

        # --- Load Model checkpoint ---
        self.opt.batch_size = 1
        self.opt.vocab = ix_to_word

        self.model = models.setup(self.opt)
        del self.opt.vocab

        ckpt_path = self.opt.checkpoint_path + '-last.ckpt'

        raw_state_dict = torch.load(
            ckpt_path,
            map_location=self.device)

        strict = True

        state_dict = raw_state_dict['state_dict']

        if '_vocab' in state_dict:
            self.model.vocab = utils.deserialize(state_dict['_vocab'])
            del state_dict['_vocab']
        elif strict:
            raise KeyError
        if '_opt' in state_dict:
            saved_model_opt = utils.deserialize(state_dict['_opt'])
            del state_dict['_opt']
            # Make sure the saved opt is compatible with the curren topt
            need_be_same = ["caption_model",
                            "rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                if getattr(saved_model_opt, checkme) in ['updown', 'topdown'] and \
                        getattr(self.opt, checkme) in ['updown', 'topdown']:
                    continue
                assert getattr(saved_model_opt, checkme) == getattr(
                    self.opt, checkme), "Command line argument and saved model disagree on '%s' " % checkme
        elif strict:
            raise KeyError
        self.res = self.model.load_state_dict(state_dict, strict)

        self.model = self.model.to(self.device)
        self.model.eval()

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
    
    def get_img2txt(self):
        for path_list in path.get_path():
            img_path = path_list[1]
            r = os.path.exists(img_path)
            if r == True:
                with torch.no_grad():
                    image = self.preprocess(Image.open( img_path ).convert("RGB"))
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

                        # forward the model to also get generated samples for each image
                        # Only leave one feature for each image, in case duplicate sample
                        tmp_eval_kwargs = eval_kwargs.copy()
                        tmp_eval_kwargs.update({'sample_n': 1})
                        seq, seq_logprobs = self.model(
                        fc_feats, att_feats, att_masks, opt=tmp_eval_kwargs, mode='sample')
                        seq = seq.data

                        sents = utils.decode_sequence(self.model.vocab, seq)

                    sents.append(path_list[0])
                    print(sents)

output = Img2TxtCLIPReward()
print(output.get_img2txt())