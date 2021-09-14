import torch
from tqdm import tqdm

class PreTrainedEmbedding():
    def __init__(self,
                path:str, 
                dim:int, 
                vocab:dict):
        
        """
        Args:
        - path (string): The path to the file of embeddings to be loaded
        - dim (integer): dimension of embedding od this file
        - vocab (dictionary): vocabulary of words of sentence of training file, created before when we create the dataset 
        """
        
        self.path = path
        self.dim = dim
        self.vocab = vocab

        self.dictionary_emb = self.build_dictionary_fast()
        self.weights = self.embedding_weights() 
  
    def build_dictionary_fast(self):
        
        """
        Return:
        - dictionary: dictionary containing the words of the pre-trained file, in particular this for fast_file, 
                        with the corresponding weight as associated value
        """
        
        dictionary = {}
        fin = open(self.path, 'r', encoding='utf-8', newline='\n', errors='ignore')
        
        for idx, line in enumerate(tqdm(fin)):
            if idx==0:
              continue
            tokens = line.strip().split()
            data = []
            for i in range(self.dim):
              data.append(float(tokens[i+1]))
            dictionary[tokens[0]]= data
      
        return dictionary

    def embedding_weights(self):
        
        """
        Return:
        - embeddings_weights: size N_words of self.vocab x dim.
                                If the word in self.vocab is also in the embedding dictionary 
                                then I will associate those weights with the corresponding value of the word, 
                                otherwise a random value.
        """
        
        embeddings_weights = torch.zeros([len(self.vocab),self.dim])
        random = torch.rand(1,self.dim)[0]

        for elem in self.vocab:
            if elem not in self.dictionary_emb:
                embeddings_weights[self.vocab[elem]] = random
            else:
                
                embeddings_weights[self.vocab[elem]] = torch.tensor(self.dictionary_emb[elem])
        
        return embeddings_weights
