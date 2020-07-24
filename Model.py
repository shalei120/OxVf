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

from transformers import AlbertTokenizer, AlbertConfig, AlbertModel
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
        # self.BERTtokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        # albert_base_configuration = AlbertConfig(
        #       hidden_size=args['ALBERT_hidden_size'],
        #       num_attention_heads=12,
        #       intermediate_size=3072,
        #   )
        # self.Albert_model = AlbertModel(albert_base_configuration)

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

        '''
        ALBERT: https://huggingface.co/transformers/model_doc/albert.html
        
        last_hidden_state (torch.FloatTensor of shape (batch_size, sequence_length, hidden_size)):
        
        Sequence of hidden-states at the output of the last layer of the model.
        
        pooler_output (torch.FloatTensor: of shape (batch_size, hidden_size)):
        
        Last layer hidden-state of the first token of the sequence (classification token) further processed by a Linear 
        layer and a Tanh activation function. The Linear layer weights are trained from the next sentence prediction 
        (classification) objective during pre-training.
        
        This output is usually not a good summary of the semantic content of the input, you’re often better with averaging 
        or pooling the sequence of hidden-states for the whole input sequence.
        
        hidden_states (tuple(torch.FloatTensor), optional, returned when output_hidden_states=True is passed or when 
        config.output_hidden_states=True):
        
        Tuple of torch.FloatTensor (one for the output of the embeddings + one for the output of each layer) of shape 
        (batch_size, sequence_length, hidden_size).
        
        Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        
        attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True is passed or when 
        config.output_attentions=True):
        
        Tuple of torch.FloatTensor (one for each layer) of shape (batch_size, num_heads, sequence_length, sequence_length).
        
        Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
        '''

        # ALBERT_input_sentences = x['enc_input_raw']
        # ALBERT_encoded_inputs = self.BERTtokenizer(ALBERT_input_sentences,padding=True, truncation=True,return_tensors="pt")
        # ALBERT_encoded_inputs['input_ids'] = ALBERT_encoded_inputs['input_ids'].to(args['device'])
        # ALBERT_encoded_inputs['token_type_ids'] = ALBERT_encoded_inputs['token_type_ids'].to(args['device'])
        # ALBERT_encoded_inputs['attention_mask'] = ALBERT_encoded_inputs['attention_mask'].to(args['device'])
        # last_hidden_state, pooler_output = self.Albert_model(**ALBERT_encoded_inputs)

        _, en_state = self.encoder(self.encoderInputs, self.encoder_lengths)
        emo_vector = self.embedding(self.emo_label) # batch * hid
        # info_vector = torch.cat([emo_vector, pooler_output], dim = 1)
        info_vector = emo_vector
        de_outputs = self.decoder(en_state, info_vector, self.decoderInputs, self.decoder_lengths, self.decoderTargets)

        recon_loss = self.CEloss(torch.transpose(de_outputs, 1, 2), self.decoderTargets)
        mask = torch.sign(self.decoderTargets.float())
        recon_loss = torch.squeeze(recon_loss) * mask
        recon_loss_mean = torch.mean(recon_loss)

        return recon_loss_mean, en_state, info_vector

    def forward(self, x):
        loss, _, _ = self.buildmodel(x)
        return loss

    def predict(self, x):
        _, en_state, info = self.buildmodel(x)
        de_words = self.decoder.generate(en_state, info)
        return de_words