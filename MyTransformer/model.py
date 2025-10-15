import torch
import torch.nn as nn
import math

# Initialy we have a sentence such as "The cat is sleeping on the couch", a computer can't use directly that kind of data
# So we will associate to each word an ID, such as : cat -> 127
# Then we will associate to each ID an embedding (a vector with numbers that represents the word) to work with it.
# It's the first step presented in the paper "Attention is all you need".
class InputEmbeddings(nn.Module):   
    # d_model : size of the embedding
    # vocab_size : number of words in the vocabulary
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        # We will use the model provided by Torch that already associates to each "word" an embedding
        self.embedding = nn.Embedding(vocab_size,d_model)
        
    def forward(self, x):
        # In the paper "Attention is all you need", they say that they multiply the embedding obtained by sqrt(size of the embedding)
        return self.embedding(x) * math.sqrt(self.d_model)
    
# Now we have the model that give for each word an embedding
# BUT we still have to work a bit more on the representation of the words to give an order to the words.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model) = (length of the sentence, size of one embedding)
        positional_encoding = torch.zeros(seq_len,d_model)
        
        # Create a vector of shape (number of words, 1)
        position = torch.arange(0,seq_len, d_type=torch.float()).unsqueeze(1)
        # The original formula in the paper calculates : 1/(10000^(2i/d_model)), but we will use log space for efficiency and stability
        div_term = torch.exp(torch.arange(0,d_model,2).float() * (-math.log(10000.0) / d_model))
        
        # We have to multiply position * div_term and then apply sin or cos
        # ADD MORE DETAILS HERE
        positional_encoding[:,0::2] = math.sin(position * div_term)
        positional_encoding[:,1::2] = math.cos(position * div_term)
        
        # Position Encoding will become a tensor of (1,number of words, size of embedding)
        positional_encoding = positional_encoding.unsqueeze(0)
        
        # Allows to save the positional encoding when saving the model
        self.register_buffer('positional_encoding',positional_encoding)
        
    def forward(self,x):
        # Here x should be (length of the sentence, size of embedding)
        # So we recover the length of the sentence first positional encoding, and we add them to x
        x = x + (self.positional_encoding[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
        
        