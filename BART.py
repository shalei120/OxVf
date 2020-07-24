import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np
import datetime
from transformers import BartTokenizer, BartForConditionalGeneration
# from transformers import BartTokenizer,BartModel
from Hyperparameters import args

class BARTModel(nn.Module):
    def __init__(self):
        super(BARTModel, self).__init__()
        print("Model creation...")
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
        # self.tokenizer.save_vocabulary(args['rootDir'])
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        # self.model = BartModel.from_pretrained('facebook/bart-large')
        # self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    def forward(self, x):
        # print(x['enc_input_raw'])
        input_sentences = x['enc_input_raw']
        batchsize = len(input_sentences)
        input_ids = self.tokenizer(input_sentences, padding=True, truncation=True, return_tensors="pt")['input_ids'].clone().detach()
        input_ids=input_ids.to(args['device'])  # Batch size 1
        # print(len(input_sentences), self.tokenizer(input_sentences, add_special_tokens=True)['input_ids'])
        # exit()
        # outputs = model(input_ids=input_ids, decoder_input_ids=input_ids)
        output_sentences =  x['dec_input_raw']
        output_ids = self.tokenizer(output_sentences,padding=True, truncation=True,return_tensors="pt")['input_ids'].clone().detach()
        output_ids = output_ids.to(args['device'])
        # output_ids_in = output_ids[:, :-1].contiguous()
        # output_ids_in = output_ids_in - (output_ids_in == 2).long()  # EOS -> PAD
        # output_ids_tar = output_ids[:, 1:].contiguous()
        # assert output_ids_in.size()==output_ids_tar.size()
        # try:
        loss, logits, hidden= self.model(input_ids=input_ids, decoder_input_ids=output_ids, labels=output_ids)#, use_cache=False)

        # outputs = self.model(
        #     input_ids,
        #     decoder_input_ids=output_ids_in,
        #     use_cache=False,
        # )
        # lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
        # # outputs = (lm_logits,) + outputs[1:]  # Add cache, hidden states and attention if they are here
        #
        # loss_fct = nn.CrossEntropyLoss(ignore_index=1)
        #
        # # TODO(SS): do we need to ignore pad tokens in labels?
        # loss = loss_fct(lm_logits.view(-1, self.model.config.vocab_size), output_ids_tar.view(-1))

        return loss

    def predict(self, x):
        input_sentences =x['enc_input_raw']
        input_ids = torch.tensor(self.tokenizer(input_sentences, padding=True, truncation=True,return_tensors="pt")['input_ids']).to(args['device'])

        summary_ids = self.model.generate(input_ids, num_beams=4,  decoder_start_token_id = 0, early_stopping=True)
        print(summary_ids,summary_ids.size())
        res = [self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False).split(' ') for g in summary_ids]
        print(res)
        return res