
import numpy as np
import nltk  # For tokenize

from tqdm import tqdm  # Progress bar
import pickle  # Saving the data
import math  # For float comparison
import os  # Checking file existance
import random
import string, copy
import json
from nltk.tokenize import word_tokenize

from Hyperparameters import args

class Batch:
    """Struct containing batches info
    """
    def __init__(self):
        self.contextSeqs = []
        self.context_lens = []
        self.emo_label = []
        self.senSeqs = []
        self.sen_lens = []


class Dataloader:
    """Dataset class
    Warning: No vocabulary limit
    """


    def __init__(self, corpusname, embfile = None):
        """Load all conversations
        Args:
            args: parameters of the model
        """

        # Path variables
        if corpusname == 'fb':
            self.tokenizer = word_tokenize

        self.trainingSamples = []  # 2d array containing each question and his answer [[input,target]]

        self.embfile = embfile
        self.datasets = self.loadCorpus()

        self._printStats(corpusname)

        if args['playDataset']:
            self.playDataset()

        self.batches = {}

    def _printStats(self, corpusname):
        print('Loaded {}: {} words, {} '.format(corpusname, len(self.word2index), len(self.trainingSamples)))


    def shuffle(self):
        """Shuffle the training samples
        """
        print('Shuffling the dataset...')
        random.shuffle(self.datasets['train'])

    def _createBatch(self, samples):
        """Create a single batch from the list of sample. The batch size is automatically defined by the number of
        samples given.
        The inputs should already be inverted. The target should already have <go> and <eos>
        Warning: This function should not make direct calls to args['batchSize'] !!!
        Args:
            samples (list<Obj>): a list of samples, each sample being on the form [input, target]
        Return:
            Batch: a batch object en
        """

        batch = Batch()
        batchSize = len(samples)

        # Create the batch tensor
        for i in range(batchSize):
            # Unpack the sample
            context_ids, sen_ids, emotion, raw_context, raw_sentence = samples[i]

            if len(sen_ids) > args['maxLengthEnco']:
                sen_ids = sen_ids[:args['maxLengthEnco']]

            batch.contextSeqs.append(context_ids)
            batch.senSeqs.append(sen_ids)
            batch.context_lens.append(len(batch.contextSeqs[i]))
            batch.sen_lens.append(len(batch.senSeqs[i]))
            batch.emo_label.append(emotion)

        maxlen_context = max(batch.context_lens)
        maxlen_sen = max(batch.sen_lens)


        for i in range(batchSize):
            batch.contextSeqs[i] = batch.contextSeqs[i] + [self.word2index['PAD']] * (maxlen_context - len(batch.contextSeqs[i]))
            batch.senSeqs[i] = batch.senSeqs[i] + [self.word2index['PAD']] * (maxlen_sen - len(batch.senSeqs[i]))

        return batch

    def getBatches(self, setname = 'train'):
        """Prepare the batches for the current epoch
        Return:
            list<Batch>: Get a list of the batches for the next epoch
        """
        if setname not in self.batches:
            self.shuffle()

            batches = []
            print(len(self.datasets[setname]))
            def genNextSamples():
                """ Generator over the mini-batch training samples
                """
                for i in range(0, self.getSampleSize(setname), args['batchSize']):
                    yield self.datasets[setname][i:min(i + args['batchSize'], self.getSampleSize(setname))]

            # TODO: Should replace that by generator (better: by tf.queue)

            for index, samples in enumerate(genNextSamples()):
                # print([self.index2word[id] for id in samples[5][0]], samples[5][2])
                batch = self._createBatch(samples)
                batches.append(batch)

            self.batches[setname] = batches

        return self.batches[setname]

    def getSampleSize(self, setname = 'train'):
        """Return the size of the dataset
        Return:
            int: Number of training samples
        """
        return len(self.datasets[setname])

    def getVocabularySize(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.word2index)

    def getChargeNum(self):
        """Return the number of words present in the dataset
        Return:
            int: Number of word on the loader corpus
        """
        return len(self.lawinfo['c2i'])

    def loadCorpus(self):
        """Load/create the conversations data
        """
        self.basedir = '../empatheticdialogues_data/'
        self.corpus_file_train = self.basedir + 'train.csv'
        self.corpus_file_valid = self.basedir + 'valid.csv'
        self.corpus_file_test  = self.basedir + 'test.csv'

        self.data_dump_path = args['rootDir'] + '/Facebookdata.pkl'
        if not os.path.exists(args['rootDir']):
            os.mkdir(args['rootDir'])

        print(self.data_dump_path)
        datasetExist = os.path.isfile(self.data_dump_path)

        max_hist_len = args['max_history_length']
        maxlen = args['maxLength']
        newmaxlen = args['maxLength'] *2

        words = []
        emotions = set()

        def load_data_from_file(filename, reactonly = False):
            with open(filename, 'r') as rhandle:
                df = rhandle.readlines()
                history = []
                data = []
                ids = []
                for i in range(1, len(df)):
                    cparts = df[i - 1].strip().split(",")
                    sparts = df[i].strip().split(",")
                    if cparts[0] == sparts[0]:
                        prevsent = cparts[5].replace("_comma_", ",")
                        prevsent = self.tokenizer(prevsent.lower())
                        words.extend(prevsent)
                        history.append(' '.join(prevsent))
                        idx = int(sparts[1])
                        if not reactonly or ((idx % 2) == 0):
                            prev_str = " SOC ".join(history[-max_hist_len:])
                            contextt = prev_str.split(' ')[:newmaxlen]
                            sent = sparts[5].replace("_comma_", ",")
                            label = sent.split(' ')[:maxlen]
                            lbl_min = sparts[2]
                            emotions.add(lbl_min)
                            data.append((contextt, label, lbl_min))
                            # self.ids.append((sparts[0], sparts[1]))
                    else:
                        history = []

            return data, words, emotions

        if not datasetExist:  # First time we load the database: creating all files
            print('Training data not found. Creating dataset...')

            dataset = {'train': [], 'valid':[], 'test':[]}

            dataset['train'], train_words, train_emotions = load_data_from_file(self.corpus_file_train)
            dataset['valid'], _, _ = load_data_from_file(self.corpus_file_valid)
            dataset['test'], _, _ = load_data_from_file(self.corpus_file_test)

            print(len(dataset['train']), len(dataset['valid']), len(dataset['test']))

            self.index2emotion = list(train_emotions)
            self.emotion2index = {emo:ind for ind, emo in enumerate(self.index2emotion)}

            fdist = nltk.FreqDist(train_words)
            sort_count = fdist.most_common(30000)
            print('sort_count: ', len(sort_count))

            # nnn=8
            with open(self.basedir + "/voc.txt", "w") as v:
                for w, c in tqdm(sort_count):
                    # if nnn > 0:
                    #     print([(ord(w1),w1) for w1 in w])
                    #     nnn-= 1
                    if w not in [' ', '', '\n', '\r', '\r\n']:
                        v.write(w)
                        v.write(' ')
                        v.write(str(c))
                        v.write('\n')

                v.close()
            if not self.embfile:
                self.word2index = self.read_word2vec(self.basedir + '/voc.txt')
                sorted_word_index = sorted(self.word2index.items(), key=lambda item: item[1])
                print('sorted')
                self.index2word = [w for w, n in sorted_word_index]
                print('index2word')
            else:
                self.word2index, self.index2word, self.index2vector = self.read_word2vec_from_pretrained(self.embfile)


            self.index2word_set = set(self.index2word)

            for setname in ['train', 'valid', 'test']:
                dataset[setname] = [(self.TurnWordID(context), self.TurnWordID(sen), self.emotion2index[emotion], context, sen)
                                    for context, sen, emotion in tqdm(dataset[setname])]
            # Saving
            print('Saving dataset...')
            self.saveDataset(self.data_dump_path, dataset)  # Saving  samples
        else:
            dataset = self.loadDataset(self.data_dump_path)
            print('loaded')

        return  dataset

    def saveDataset(self, filename, datasets):
        """Save samples to file
        Args:
            filename (str): pickle filename
        """

        with open(os.path.join(filename), 'wb') as handle:
            data = {  # Warning: If adding something here, also modifying loadDataset
                'word2index': self.word2index,
                'index2word': self.index2word,
                'datasets': datasets,
                'emotion2index' : self.emotion2index,
                'index2emotion' : self.index2emotion
            }
            pickle.dump(data, handle, -1)  # Using the highest protocol available

    def loadDataset(self, filename):
        """Load samples from file
        Args:
            filename (str): pickle filename
        """
        dataset_path = os.path.join(filename)
        print('Loading dataset from {}'.format(dataset_path))
        with open(dataset_path, 'rb') as handle:
            data = pickle.load(handle)  # Warning: If adding something here, also modifying saveDataset
            self.word2index = data['word2index']
            self.index2word = data['index2word']
            datasets = data['datasets']
            self.emotion2index = data['emotion2index']
            self.index2emotion = data['index2emotion']

        self.index2word_set = set(self.index2word)
        return  datasets


    def read_word2vec(self, vocfile ):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        word2index['SOC'] = 4
        cnt = 5
        with open(vocfile, "r") as v:

            for line in v:
                word = line.strip().split()[0]
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        print(len(word2index),cnt)
        # dic = {w:numpy.random.normal(size=[int(sys.argv[1])]).astype('float32') for w in word2index}
        print ('Dictionary Got!')
        return word2index

    def read_word2vec_from_pretrained(self, embfile, topk_word_num= 30000 ):
        word2index = dict()
        word2index['PAD'] = 0
        word2index['START_TOKEN'] = 1
        word2index['END_TOKEN'] = 2
        word2index['UNK'] = 3
        word2index['SOC'] = 4
        cnt = 5
        vectordim = -1
        index2vector = []
        with open(embfile, "r") as v:
            lines = v.readlines()
            lines = lines[:topk_word_num]
            for line in tqdm(lines):
                word_vec = line.strip().split()
                word = word_vec[0]
                vector = np.asarray([float(value) for value in word_vec[1:]])
                if vectordim == -1:
                    vectordim = len(vector)
                index2vector.append(vector)
                word2index[word] = cnt
                print(word,cnt)
                cnt += 1

        index2vector = [np.random.normal(size=[vectordim]).astype('float32') for _ in range(cnt)] + index2vector
        index2vector = np.asarray(index2vector)
        index2word = [w for w, n in word2index]
        print(len(word2index),cnt)
        print ('Dictionary Got!')
        return word2index, index2word, index2vector

    def TurnWordID(self, words):
        res = []
        for w in words:
            w = w.lower()
            if w in self.index2word_set:
                id = self.word2index[w]
                res.append(id)
            else:
                res.append(self.word2index['UNK'])
        return res



    def playDataset(self):
        """Print a random dialogue from the dataset
        """
        print('Randomly play samples:')
        print(len(self.datasets['train']))
        for i in range(args['playDataset']):
            idSample = random.randint(0, len(self.datasets['train']) - 1)
            print('sen: {} {} {}'.format(self.index2emotion[self.datasets['train'][idSample][2]], self.datasets['train'][idSample][3], self.datasets['train'][idSample][4]))
            print()
        pass


if __name__ == '__main__':
    # textdata = Dataloader('fb')
    textdata = Dataloader('fb', embfile = '../glove.6B.50d.txt')