import torch
from torch.utils.data import Dataset
from collections import Counter
from tqdm import tqdm

class SvevaDataset(Dataset):
  
    def __init__(self, 
                input_file:str, 
                char_len:int,
                vocabulary:dict=None, 
                vocabulary_label:dict=None, 
                vocabulary_char:dict=None, 
                min_freq:int=1):
        
        """
        Args: 
            - input_file (string): the path to the dataset to be loaded
            - char_len (int): identifies the maximum size of each word
            - vocabulary (dictionary): vocabulary of words
            - vocabulary_label (dictionary): vocabulary of labels
            - vocabulary_char (dictionary): vocabulary of chars
            - min_freq (int): minimum frequency of words and char that you want to consider when the vocabulary is created
        """

        self.input_file = input_file
        self.char_len = char_len
        self.tokens, self.labels = self.sentences_of_file(self.input_file)
        self.max_len=self.find_max_len(self.tokens)


        if vocabulary is None:
            self.vocab = self.build_vocabulary(self.tokens,min_freq)
        else:
            self.vocab = vocabulary
        
        if vocabulary_label is None:
            self.vocab_label = self.build_vocabulary_label(self.labels)
        else:
            self.vocab_label = vocabulary_label
        
        if vocabulary_char is None:
            self.vocab_char = self.build_vocabulary_char(self.tokens,min_freq)
        else:
            self.vocab_char = vocabulary_char

        self.char2idx = self.char2idx()

    def __len__(self):
            return len(self.char2idx)

    def __getitem__(self, idx):
            return self.char2idx[idx]
  
    def find_max_len(self,sentences):

        """
        Args:
        - sentences (list of lists of words): where each list represent a sentence of the file
        
        Return:
        - max_len :the length of the longest sentence
        """

        max_len = 0
        for sentence in sentences:
            if len(sentence)>max_len:
                max_len=len(sentence)
        return max_len


    def char2idx(self):

        """
        Return:
        A list with elements inside that are dictionaries. 
        Each element of dictionary is composed of 3 subelements: 
            - inputs: encoding a sentence
            - outputs: encoding of the labels corresponding to that sentence
            - char: encoding of the characters of the words of that sentence
        """

        vector = []
        for t,l in zip(self.tokens,self.labels):
            data_t =[]
            data_l = []
            window_t = self.create_fix_dim_sentence(t)
            window_l = self.create_fix_dim_sentence(l)
            chars = self.create_fix_dim_char(window_t)
            encode_char = self.encode_char(chars)
            encode_sentence= self.encode_input(window_t)
            encode_label = self.encode_output(window_l)
            
            item = {"inputs":torch.tensor(encode_sentence), "outputs": torch.tensor(encode_label), "char":torch.tensor(encode_char)}
            vector.append(item)
        return vector

    def create_fix_dim_char(self,sentence):

        """
        Args:
        - sentence: list of words

        Return:
        - data: list of lists of characters. 
                Each list identifies the characters of that word in the sentence, 
                all lists have a fixed length. 
                If the word has a length smaller than the fixed length then I add None, 
                otherwise I cut it to the chosen size
        """

        data =[]
        for word in sentence:
            build_char = []
            if word is None:
                for i in range(0,self.char_len):
                    build_char.append(None)
            else:
                if len(word)>self.char_len:
                    for i in range(self.char_len):
                        build_char.append(word[i])
                else:
                    if len(word) < self.char_len:
                        for i in range(self.char_len):
                            if i>=len(word):
                                build_char.append(None)
                            else:
                                build_char.append(word[i])
                    else:
                        for i in range(self.char_len):
                            build_char.append(word[i])
            assert len(build_char) == self.char_len
            data.append(build_char)
        return data



    def create_fix_dim_sentence(self,sentence):
        
        """
        Args:
        - sentence: list of words

        Return:
        - data: list of words. 
                list identifies the words in the sentence, 
                but list have a fixed length. 
                If the sentence has a length smaller than the fixed length then I add None, 
                otherwise I cut it to the chosen size
        """

        
        for i in range(0,len(sentence)):
            w = sentence[:self.max_len]
            if len(w) < self.max_len:
                w = w + [None]*(self.max_len - len(w))
        assert len(w) == self.max_len
        return w
  
    def encode_char(self,sentences):
        
        """
        Args:
        - sentences: list of lists of sentences

        Return:
        - all_elements: list of lists of the encoding of the chars of words in a sentence. 
                        If the char is in the vocabulary of chars, I associate the corresponding value to it, 
                        if the char is None I take the PADDING value, 
                        otherwise I take the value of UNK
        """

        all_elements = []
        for sentence in sentences:
            data = []
            for c in sentence:
                if c is None:
                    data.append(self.vocab_char['<pad>'])
                elif c in self.vocab_char:
                    data.append(self.vocab_char[c])
                else:
                    data.append(self.vocab_char['<unk>'])
            all_elements.append(data)
        return all_elements
      
    def encode_input(self,sentence):
        
        """
        Args:
        - sentences: list of lists of sentences

        Return:
        - data: list of the encoding of the words in the sentence. 
                        If the word is in the vocabulary, I associate the corresponding value to it, 
                        if the word is None I take the PADDING value, 
                        otherwise I take the value of UNK
        """
    
        data =[]
        for word in sentence:
            if word is None:
                data.append(self.vocab["<pad>"])
            elif word in self.vocab:
                data.append(self.vocab[word])
            else:
                data.append(self.vocab["<unk>"])
        return data

    def encode_output(self,sentence):
        
        """
        Args:
        - sentences: list of lists of sentences, each sentence contains labels

        Return:
        - data: list of the encoding of the labels in the sentence. 
                        If the label is in the vocabulary of the labels, I take the corresponding value to it, 
                        otherwise the label is None I take the PADDING value
        """
    
        data =[]
        for word in sentence:
            if word in self.vocab_label:
                data.append(self.vocab_label[word])
            else:
                data.append(self.vocab_label["<pad>"])
    
        return data


    def sentences_of_file(self,path):
        
        """
        Args:
        - path: path of file 

        Return:
        - tokens: list of list of sentences of the file
        - labels: list of list of labels of sentences of the file
        """
    
        tokens = []
        labels = []
        all_tokens = []
        all_labels = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line.startswith('# '):
                    all_tokens = []
                    all_labels = []
                elif line == '':
                    tokens.append(all_tokens)
                    labels.append(all_labels)
                else:
                    _, token, label = line.split('\t')
                    all_tokens.append(token)
                    all_labels.append(label)
        return tokens,labels


    def build_vocabulary (self,tokens,min_freq=1):
        
        """
        Args:
        - tokens: list of lists of sentences
        - min_freq: minimum frequency threshold to take a certain word
        Return:
        
        - dictionary_tok: dictionary containing all the words with minimum frequency specified before. 
                            In which the key is the word and the value is the integer that represents it.
        """
    
        dictionary_tok={}
        counter = Counter()
        for i in tqdm(range(len(tokens))):
            for word in tokens[i]:
                if word is not None:
                    counter[word]+=1
        dictionary_tok.update({'<pad>': 0})
        dictionary_tok.update({'<unk>': 1})
        for index, (key,value) in enumerate(counter.most_common()):
            if value >= min_freq:
                dictionary_tok.update({key: index+2})
        
        return dictionary_tok

    def build_vocabulary_label(self,labels):
        
        """
        Args:
        - labels: list of lists of labels

        Return:
        - dictionary_lab: dictionary containing all the labels. 
                            In which the key is the label and the value is the integer that represents it.
        """
        
        dictionary_lab={}
        counter_lab = Counter()
        for i in tqdm(range(len(labels))):
            for word in labels[i]:
                if word is not None:
                    counter_lab[word]+=1

        dictionary_lab['<pad>'] = 0
        for index, (key,_) in enumerate(counter_lab.most_common()):
            dictionary_lab[key] = index+1
        
        return dictionary_lab

    def build_vocabulary_char(self,tokens,min_freq=1):
        
        """
        Args:
        - tokens: list of lists of sentences
        - min_freq: minimum frequency threshold to take a certain char

        Return:
        - dictionary_char: dictionary containing all the chars with minimum frequency specified before. 
                            In which the key is the char and the value is the integer that represents it.
        """
        
        dictionary_char={}
        counter_char = Counter()
        for i in tqdm(range(len(tokens))):
            for word  in tokens[i]:
                if word is not None:
                    for c in word:
                        counter_char[c]+=1

        dictionary_char.update({'<pad>': 0})
        dictionary_char.update({'<unk>': 1})
        for index, (key,value) in enumerate(counter_char.most_common()):
            if value >= min_freq:
                dictionary_char.update({key: index+2})
    
        return dictionary_char