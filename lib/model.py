import torch
from torch import nn

class SvevaModel(nn.Module):
    def __init__(self, hparams):
        super(SvevaModel, self).__init__()

        # Embedding layer for char: a matrix vocab_char x char_embedding_dim where each index 
        # correspond to a char in the vocabulary and the i-th row corresponds to 
        # a latent representation of the i-th char in the vocabulary.
        self.char_embedding = nn.Embedding(hparams.vocab_char,hparams.char_embedding_dim)

        # Embedding layer for words: a matrix vocab_size x embedding_dim where each index 
        # correspond to a word in the vocabulary and the i-th row corresponds to 
        # a latent representation of the i-th word in the vocabulary.
        self.word_embedding = nn.Embedding(hparams.vocab_size, hparams.embedding_dim)
        
        if hparams.embeddings is not None:
            print("initializing embeddings from pretrained")
            self.word_embedding.weight.data.copy_(hparams.embeddings.weights)

        # LSTM layer: an LSTM neural network that process the input text
        # (encoded with word embeddings) from left to right and outputs 
        # a new **contextual** representation of each word that depend
        # on the preciding words. 
        # As we can see there is a concatenation of two embeddings
        self.lstm = nn.LSTM(hparams.embedding_dim + hparams.char_embedding_dim*hparams.char_len, hparams.hidden_dim, 
                            bidirectional=hparams.bidirectional,
                            num_layers=hparams.num_layers, 
                            batch_first=True,
                            dropout = hparams.dropout)
        

        # Hidden layer: transforms the input value/scalar into
        # a hidden vector representation.
        lstm_output_dim = hparams.hidden_dim if hparams.bidirectional is False else hparams.hidden_dim * 2


        # During training, randomly zeroes some of the elements of the 
        # input tensor with probability hparams.dropout using samples 
        # from a Bernoulli distribution. Each channel will be zeroed out 
        # independently on every forward call.
        # This has proven to be an effective technique for regularization and 
        # preventing the co-adaptation of neurons, so to avoid overfitting
        self.dropout = nn.Dropout(hparams.dropout)
        self.classifier = nn.Linear(lstm_output_dim, hparams.num_classes)

    
    def forward(self, x, y):
        char_embedding = self.char_embedding(y)
        chars = char_embedding.reshape(char_embedding.size()[0], char_embedding.size()[1], char_embedding.size()[2]*char_embedding.size()[3])
        embeddings = self.word_embedding(x)
        embeddings = self.dropout(embeddings)      
        embeddings = torch.cat((embeddings, chars), dim=2)
        o, (h, c) = self.lstm(embeddings)
        o = self.dropout(o)
        output = self.classifier(o)
        return output


class HParams():
    def __init__(self,vocabulary,label_vocabulary,vocabulary_char,
                      embedding, char_embedding_dim, char_len,
                      hidden_dim, embedding_dim,
                      bidirectional, num_layers, dropout):

        self.vocab_size = len(vocabulary)
        self.vocab_char = len(vocabulary_char)
        self.char_embedding_dim = char_embedding_dim
        self.char_len = char_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = len(label_vocabulary)
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.embeddings = embedding