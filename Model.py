import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

import numpy as np

import datetime


from Encoder import Encoder
from Decoder import Decoder


from Hyperparameters import args

class Model(nn.Module):
    """
    Implementation of a seq2seq model.
    Architecture:
        Encoder/decoder
        2 LTSM layers
    """

    def __init__(self, w2i, i2w):
        """
        Args:
            args: parameters of the model
            textData: the dataset object
        """
        super(Model, self).__init__()
        print("Model creation...")

        self.word2index = w2i
        self.index2word = i2w
        self.max_length = args['maxLengthDeco']

        self.dtype = 'float32'
        self.NLLloss = torch.nn.NLLLoss(reduction = 'none')
        self.CEloss =  torch.nn.CrossEntropyLoss(reduction = 'none')

        self.embedding = nn.Embedding(args['vocabularySize'], args['embeddingSize'])
        self.emo_embedding = nn.Embedding(args['emo_labelSize'], args['embeddingSize'])
        self.encoder = Encoder(w2i, i2w, self.embedding)
        self.decoder = Decoder(w2i, i2w, self.embedding)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim = -1)

    def buildmodel(self, x):
        '''
        :param encoderInputs: [batch, enc_len]
        :param decoderInputs: [batch, dec_len]
        :param decoderTargets: [batch, dec_len]
        :return:
        '''

        # print(x['enc_input'])
        self.encoderInputs = x['enc_input']
        self.encoder_lengths = x['enc_len']
        self.decoderInputs = x['dec_input']
        self.decoder_lengths = x['dec_len']
        self.decoderTargets = x['dec_target']
        self.emo_label = x['emo_label']
        self.batch_size = self.encoderInputs.size()[0]

        _, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)
        emo_vector = self.embedding(self.emo_label) # batch * hid

        de_outputs = self.decoder(en_state, emo_vector, self.decoderInputs, self.decoder_lengths, self.decoderTargets)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        mask = torch.sign(self.decoderTargets.float())
        recon_loss = torch.squeeze(recon_loss) * mask
        recon_loss_mean = torch.mean(recon_loss)

        return recon_loss_mean, en_state

    def forward(self, x):
        loss, _ = self.buildmodel(x)
        return loss

    def predict(self, x):
        _, en_state = self.buildmodel(x)
        de_words = self.decoder.generate(en_state)
        return de_words