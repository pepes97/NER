import argparse

import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.dataset import SvevaDataset 
from lib.pretrained import PreTrainedEmbedding
from lib.model import HParams, SvevaModel
from lib.trainer import Trainer
from lib.utils import compute_metrics

def main(char_embedding_dim, char_len, hidden_dim, embedding_dim, bidirectional, num_layers, dropout, learning_rate, batch_size, epochs):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\033[1mDevice \033[0m: {device} \033[0m")

    training_file = 'dataset/train.tsv'
    dev_file = 'dataset/dev.tsv'
    test_file = 'dataset/test.tsv'
    fast_text_file = 'dataset/glove.6B.300d.txt'

    print(f"\033[1mTrain file \033[0m: {training_file} \033[0m")
    print(f"\033[1mDev file \033[0m: {dev_file} \033[0m")
    print(f"\033[1mTest file \033[0m: {test_file} \033[0m")

    print(f"\033[1mCreating dataset... \033[0m")
    trainset = SvevaDataset(training_file,char_len)
    devset = SvevaDataset(dev_file,char_len,trainset.vocab,trainset.vocab_label, trainset.vocab_char)
    testset = SvevaDataset(test_file,char_len,trainset.vocab,trainset.vocab_label, trainset.vocab_char)

    print(f"\033[1mVocab size \033[0m: {len(trainset.vocab)} \033[0m")
    print(f"\033[1mVocab label size \033[0m: {len(trainset.vocab_label)} \033[0m")


    embedding = PreTrainedEmbedding(fast_text_file, 300, trainset.vocab)

    params = HParams(trainset.vocab, trainset.vocab_label, trainset.vocab_char, 
                     embedding, char_embedding_dim, char_len,
                     hidden_dim, embedding_dim, bidirectional, num_layers, dropout)

    model = SvevaModel(params).to(device)

    trainer = Trainer(
        model = model,
        loss_function = nn.CrossEntropyLoss(ignore_index=trainset.vocab_label['<pad>']),
        optimizer = optim.Adam(model.parameters(),lr=learning_rate),
        device = device,
        label_vocab=trainset.vocab_label
    )

    train_dataset = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_dataset = DataLoader(devset, batch_size=batch_size)
    test_dataset = DataLoader(testset, batch_size=batch_size)

    print(f"\033[1mTraining... \033[0m")
    trainer.train(train_dataset, valid_dataset, epochs)

    print(f"\033[1mTesting... \033[0m")
    precisions = compute_metrics(model, test_dataset, trainset.vocab)
    p = precisions["precision"]
    mp = precisions["macro_precision"]
    r = precisions["recall"]
    f1= precisions["f1"]
    print(f"\033[1mMicro Precision: {p}, Macro Precision: {mp}, Recall: {r}, F1_score: {f1}\033[0m")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--char-embedding-dim', type=int, default=10, help='dimension char embedding')
    parser.add_argument('--char-len', type=int, default=8, help='len chars')
    parser.add_argument('--hidden-dim', type=int, default=256, help='dimension hidden layer LSTM')
    parser.add_argument('--embedding-dim', type=int, default=300, help='dimension word embedding')
    parser.add_argument('--bidirectional', type=bool, default=True, help='bidirectional LSTM')
    parser.add_argument('--num-layers', type=int, default=2, help='number of layers LSTM')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs')

    args = parser.parse_args()
    char_embedding_dim = args.char_embedding_dim
    char_len = args.char_len
    hidden_dim = args.hidden_dim
    embedding_dim = args.embedding_dim
    bidirectional = args.bidirectional
    num_layers = args.num_layers
    dropout = args.dropout
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs

    main(char_embedding_dim, char_len, hidden_dim, embedding_dim, bidirectional, num_layers, dropout, learning_rate, batch_size, epochs)