# Copyright 2020 . All Rights Reserved.
# Author : Lei Sha

import functools
print = functools.partial(print, flush=True)
import argparse
import os

from dataloader import Dataloader
import time, sys
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import time, datetime
import math, random
import nltk
import pickle
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import numpy as np
import copy
from Hyperparameters import args

from Model import Model
from BERTEncDecModel import BERTEncDecModel
from BART import BARTModel
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g')
parser.add_argument('--modelarch', '-m')
cmdargs = parser.parse_args()

usegpu = True

if cmdargs.gpu is None:
    usegpu = False
else:
    usegpu = True
    args['device'] = 'cuda:' + str(cmdargs.gpu)

if cmdargs.modelarch is None:
    args['model_arch'] = 'None'
else:
    args['model_arch'] = cmdargs.modelarch



def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (%s)' % (asMinutes(s), datetime.datetime.now())


class Task:
    def __init__(self):
        self.model_path = args['rootDir'] + '/saved_model.mdl'

    def main(self):
        if  args['model_arch'] == 'li2016':
            args['batchSize'] = 16
        elif args['model_arch'] == 'bert2bert':
            args['batchSize'] = 8
        elif args['model_arch'] == 'bart':
            args['batchSize'] = 8

        self.textData = Dataloader('fb')
        self.start_token = self.textData.word2index['START_TOKEN']
        self.end_token = self.textData.word2index['END_TOKEN']
        args['vocabularySize'] = self.textData.getVocabularySize()
        args['emo_labelSize'] = len(self.textData.index2emotion)
        print(self.textData.getVocabularySize())

        print('Using ',args['model_arch'] ,' model.')
        if  args['model_arch'] == 'li2016':
            self.model = Model(self.textData.word2index, self.textData.index2word)
        elif args['model_arch'] == 'bert2bert':
            self.model = BERTEncDecModel()
            self.model.train()
        elif args['model_arch'] == 'bart':
            self.model = BARTModel()
            self.model.train()

        self.model = self.model.to(args['device'])
        self.train()

    def train(self, print_every=1000, plot_every=10, learning_rate=8e-4):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        print(type(self.textData.word2index))

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, eps = 0.001,amsgrad=True)

        iter = 1
        batches = self.textData.getBatches()
        n_iters = len(batches)
        print('niters ', n_iters)

        max_BLEU = -1
        # accuracy = self.test('test', max_BLEU)
        for epoch in range(args['numEpochs']):
            losses = []

            for batch in batches:
                optimizer.zero_grad()
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.contextSeqs)).to(args['device'])
                x['enc_len'] = batch.context_lens
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.senSeqs)).to(args['device'])
                x['dec_len'] = batch.senSeqs
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.senSeqs_target)).to(args['device'])
                x['emo_label'] = autograd.Variable(torch.LongTensor(batch.emo_label)).to(args['device'])
                x['enc_input_raw'] = [' '.join(r) for r in batch.context_raw]
                x['dec_input_raw'] = [' '.join(r) for r in batch.sen_raw]

                loss = self.model(x)  # batch seq_len outsize

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), args['clip'])
                optimizer.step()

                print_loss_total += loss.data
                plot_loss_total += loss.data

                losses.append(loss.data)

                if iter % print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0

                    print('%s (%d %d%%) %.4f' % (timeSince(start, iter / (n_iters * args['numEpochs'])),
                                                     iter, iter / n_iters * 100, print_loss_avg))

                if iter % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

                iter += 1

            BLEU = self.test('test')
            if BLEU > max_BLEU or max_BLEU == -1:
                print('BLEU = ', BLEU, '>= max_BLEU(', max_BLEU, '), saving model...')
                # torch.save(self.model, self.model_path)
                max_BLEU = BLEU

            print('Epoch ', epoch, 'loss = ', sum(losses) / len(losses), 'Valid BLEU = ', BLEU,
                  'max_BLEU=', max_BLEU)

        # self.test()
        # showPlot(plot_losses)

    def test(self, datasetname, eps=1e-20):

        pred_ans = []
        gold_ans = []
        with torch.no_grad():
            for batch in self.textData.getBatches(datasetname):
                x = {}
                x['enc_input'] = autograd.Variable(torch.LongTensor(batch.contextSeqs)).to(args['device'])
                x['enc_len'] = batch.context_lens
                x['dec_input'] = autograd.Variable(torch.LongTensor(batch.senSeqs)).to(args['device'])
                x['dec_len'] = batch.sen_lens
                x['dec_target'] = autograd.Variable(torch.LongTensor(batch.senSeqs_target)).to(args['device'])
                x['emo_label'] = autograd.Variable(torch.LongTensor(batch.emo_label)).to(args['device'])
                x['enc_input_raw'] = [' '.join(r) for r in batch.context_raw]
                x['dec_input_raw'] = [' '.join(r) for r in batch.sen_raw]

                decoded_words= self.model.predict(x)
                print(decoded_words[0])
                print(batch.sen_raw[0])
                pred_ans.extend(decoded_words)
                gold_ans.extend([[r] for r in batch.sen_raw])

        # bleu = corpus_bleu(gold_ans, pred_ans)
        corpusbleu = corpus_bleu(list_of_references=gold_ans, hypotheses=pred_ans)

        return corpusbleu

    def indexesFromSentence(self, sentence):
        return [self.textData.word2index[word] if word in self.textData.word2index else self.textData.word2index['UNK']
                for word in sentence]

    def tensorFromSentence(self, sentence):
        indexes = self.indexesFromSentence(sentence)
        # indexes.append(self.textData.word2index['END_TOKEN'])
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def evaluate(self, sentence, correctlabel, max_length=20):
        with torch.no_grad():
            input_tensor = self.tensorFromSentence(sentence)
            input_length = input_tensor.size()[0]
            # encoder_hidden = encoder.initHidden()

            # encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

            x = {}
            # print(input_tensor)
            x['enc_input'] = torch.transpose(input_tensor, 0, 1)
            x['enc_len'] = [input_length]
            x['labels'] = [correctlabel]
            # print(x['enc_input'], x['enc_len'])
            # print(x['enc_input'].shape)
            decoded_words, label, _ = self.model.predict(x, True)

            return decoded_words, label

    def evaluateRandomly(self, n=10):
        for i in range(n):
            sample = random.choice(self.textData.datasets['train'])
            print('>', sample)
            output_words, label = self.evaluate(sample[2], sample[1])
            output_sentence = ' '.join(output_words[0])  # batch=1
            print('<', output_sentence, label)
            print('')

    def get_sentence_BLEU(self, actual_word_lists, generated_word_lists):
        bleu_scores = self.get_corpus_bleu_scores([actual_word_lists], [generated_word_lists])
        sumss = 0
        for s in bleu_scores:
            sumss += 0.25 * bleu_scores[s]
        return sumss

    def get_corpus_BLEU(self, actual_word_lists, generated_word_lists):
        bleu_scores = self.get_corpus_bleu_scores(actual_word_lists, generated_word_lists)
        sumss = 0
        for s in bleu_scores:
            sumss += 0.25 * bleu_scores[s]
        return sumss

    def get_corpus_bleu_scores(self, actual_word_lists, generated_word_lists):
        bleu_scores = dict()
        for i in range(len(bleu_score_weights)):
            bleu_scores[i + 1] = round(
                corpus_bleu(
                    list_of_references=actual_word_lists,
                    hypotheses=generated_word_lists,
                    weights=bleu_score_weights[i + 1]), 4)

        return bleu_scores

if __name__ == '__main__':
    r = Task()
    r.main()