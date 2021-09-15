# Name Entity Recognition

Named Entity Recognition (NER) is a process of identifying and recognizing entities through text.

The goal of the project is to create a model that is able to find an entity from the raw data and can determine the category to which the element belongs. There are four categories: names of people, organizations, places and more. Identified by the labels PER, ORG, LOC, and O respectively. This article will discuss several techniques to solve this problem and the results obtained.

## Install 

```
conda create --name ner python=3.7.11
conda activate ner
pip install -r requirements.txt
```

## Download the glove data

```
$ cd dataset
$ python download_glove.py
```

## Run 

```
python main.py --char-embedding-dim CHAR-EMB-DIM --char-len -- CHAR-LEN --hidden-dim HIDDEN-DIM --embedding-dim EMB-DIM --epochs EPOCHS --batch-size BATCH-SIZE --lr LEARNING-RATE --dropout DROPOUT --bidirectional BIDIRECTIONAL --num-layers NUM-LAYERS --only-test ONLY-TEST
```
where

- `CHAR-EMB-DIM` is the dimension of the char embedding, default is 10
- `CHAR-LEN` is the maximum length of the char sequence, default is 8
- `HIDDEN-DIM` is the dimension of the hidden layer, default is 256
- `EMB-DIM` is the dimension of the word embedding, default is 300
- `EPOCHS` is the number of epochs, default is 50
- `BATCH-SIZE` is the batch size, default is 64
- `LEARNING-RATE` is the learning rate, default is 0.001
- `DROPOUT` is the dropout rate, default is 0.5
- `BIDIRECTIONAL` is the bidirectional flag, default is True
- `NUM-LAYERS` is the number of layers, default is 2
- `ONLY-TEST` is the only test flag, default is False


for example, for training:

```
python main.py
```

for testing
```
python main.py --only-test True
```
