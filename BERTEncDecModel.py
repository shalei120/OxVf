import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import datetime
from transformers import EncoderDecoderModel, BertTokenizer
from Hyperparameters import args

class BERTEncDecModel(nn.Module):
    def __init__(self):
        super(BERTEncDecModel, self).__init__()
        print("Model creation...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')

    def forward(self, x):
        input_sentences = x['enc_input_raw']
        input_ids = torch.tensor(self.tokenizer(input_sentences, padding=True, truncation=True, return_tensors="pt", add_special_tokens=True)['input_ids']).to(args['device'])  # Batch size 1
        # print(len(input_sentences), self.tokenizer(input_sentences, add_special_tokens=True)['input_ids'])
        # exit()
        # outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
        output_sentences = x['dec_input_raw']
        output_ids = torch.tensor(self.tokenizer(output_sentences,padding=True, truncation=True,return_tensors="pt", add_special_tokens=True)['input_ids']).to(args['device'])
        output_ids_in = output_ids[:, :-1].contiguous()
        output_ids_in = output_ids_in - (output_ids_in == 2).long()  # EOS -> PAD
        output_ids_tar = output_ids[:, 1:].contiguous()
        loss, outputs = self.model(input_ids=input_ids, decoder_input_ids=output_ids_in, labels=output_ids_tar)[:2]
        return loss

    def predict(self, x):
        input_sentences = x['enc_input_raw']
        input_ids = torch.tensor(self.tokenizer(input_sentences, padding=True, truncation=True,return_tensors="pt", add_special_tokens=True)['input_ids']).to(args['device'])
        generated = self.model.generate(input_ids, decoder_start_token_id=self.model.config.decoder.pad_token_id)
        res = [self.tokenizer.decode(gen).split(' ') for gen in generated]
        def remove(r):
            rr = [w for w in r if '[' not in w]
            return rr
        res = [remove(r) for r in res]
        return res