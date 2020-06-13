# OxVf
The code base of empathetic dialogue generation

## dataloader

the dataloader can automatically read Facebook empathetic data

Assume empathetic data is at last file level:

```buildoutcfg
../empatheticdialogues_data/train.csv
../empatheticdialogues_data/valid.csv
../empatheticdialogues_data/test.csv
```

Then at anywhere of python file, call:
```buildoutcfg
textdata = Dataloader('fb')
```

Then, the data is loaded to the instance `textdata`  
There are 5 data structures in the `textdata`

1, The train, valid, test data is stored in `textdata.dataset`

`textdata.dataset['train']` is a list, each element is a 5-element tuple:

```buildoutcfg
(context_id, sentence_id, emotion_id, raw_context, raw_sentence)
```
`context_id` is a list of integers, each of them is a word index. context here means the previous `k`
sentences in the dialogue.  
`sentence_id` is a list of word indexes, which is the current utterance  
`emotion_id` is the emotion's id  
`raw_context, raw_sentence` are the original words of the context and sentence

2, `textdata.word2index` is a mapping between vocabulary to index,

Example:
```buildoutcfg
index = textdata.word2index['hello']
```

3, `textdata.index2word`  

Example:
```buildoutcfg
word = textdata.index2word[23]
```

4, `textdata.emotion2index`

5, `textdata.index2emotion`

Similar to above


