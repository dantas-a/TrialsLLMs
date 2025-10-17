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
    
# Now we will have to code the encoder and the decoder.
# However, if we look closely at the architecture of the transformer, we can see that they introduce a normalization layer.
class LayerNormalization(nn.Module): 
    def __init__(self, epsilon: float = 10**(-6)) -> None :
        super().__init__()
        # Epsilon is used for numerical stability
        self.epsilon = epsilon
        # Alpha and Bias will allows to scale and shift
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self,x):
        # In x, we have for each word its embedding
        # So we calculate the mean for each word and we substract the mean of each word to its embedding
        x_centered = x - torch.mean(x, dim=-1,keepdim=True)
        # Then we will divide by its standard deviation (epsilon is added so that we don't divide by a number close to 0)
        div_term = 1 / torch.sqrt(torch.var(x,dim=-1,keepdim=True) + self.epsilon)
        
        return (x_centered * div_term) * self.alpha + self.bias

# After the First Normalization Layer in the encoder, we can see that there is a Layer Named Feed Forward layer  
# We will use the parameters from the paper to code it   
class FeedForwardLayer(nn.Module):
    def __init__(self, d_model: int, inner_layer_size: int, dropout: float) -> None:
        super().__init__()
        # The layer maps input vectors of size d_model through a hidden layer of size inner_layer_size, then back to d_model.
        self.input_layer = nn.Linear(d_model,inner_layer_size)
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(inner_layer_size,d_model)
        
    def forward(self,x):
        # Here we go through the first layer, then we use the relu function to replace all the negative values by 0.
        # We use the dropout to prevent overfitting by randomly nullifying outputs from neurons during the training.
        x = self.dropout(torch.relu(self.input_layer(x)))
        # Finally we go to the output layer to return to the original dimension
        x = self.output_layer(x)
        return x
        
# Now that we have the normalization and feed-forward layers, we can focus on the multi-head attention mechanism.
# Multi-head attention is a key component of the Transformer architecture.
# For example, in the sentence "The cat is sleeping on the couch", the model does not inherently understand the relationships between the words.
# With (non-masked) attention, the word "cat" can attend to other words in the sentence and adjust its embedding to better reflect its contextual meaning.
# With masked attention, however, when processing the word "on", the model can only attend to the words that come before it, 
# ensuring that it does not look ahead at future words (as needed in language generation tasks).
class MultiHeadAttentionLayer(nn.Module):
    
    def __init__(self, d_model: int, nb_heads: int, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.nb_heads = nb_heads
        assert d_model % nb_head == 0, "d_model has to be divisible by nb_heads"
        
        self.proj_size = d_model // nb_heads
        self.query_layer = nn.Linear(d_model,d_model)
        self.key_layer = nn.Linear(d_model,d_model)
        self.value_layer = nn.Linear(d_model,d_model)
        
        
    # How do we calculate attention?
    # In the case of Single-Head Attention, we start by computing the query for each embedding: embedding * W_Q
    # Then, we compute the key for each embedding: embedding * W_K
    # For non-masked attention, we calculate the dot product between every query and every key.
    # This gives us a matrix of shape (sequence_length, sequence_length), which we usually scale by 1 / sqrt(d_k),
    # where d_k is the dimensionality of the key and query vectors (i.e., the projection size in W_Q and W_K).
    # Next, we apply the softmax function to each row of this matrix (i.e., across all keys for a given query)
    # to obtain the Attention Matrix.
    # Then, we compute the value for each embedding: embedding * W_V
    # Each row of the Attention_Matrix weights a linear combination of the value vectors,
    # producing context-dependent representations for each token.
    
    # Obviously, here we are coding the Multi-Head Attention. So, we will have to adapt some elements.
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        attention_matrix = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        # NOT FINISHED
        
        
    
    
    
    
    
    def forward(self,q,k,v,mask):
        query = self.query_layer(q) 
        key = self.key_layer(k)
        value = self.value_layer(v)
        
        # We want multi-head attention, and since we defined the query_layer as a Linear(d_model, d_model)
        # We have to split the output into nb_heads vectors, so it will be equivalent to calculating embedding * W_Q with a matrix of size (d_model, proj_size) nb_head times
        # the output query has the shape (Batch, nb_heads, seq_len, d_k)
        query = query.view(query.shape[0],query.shape[1],self.nb_heads,self.proj_size).transpose(1,2)
        # We use the same logic for the keys
        key = key.view(key.shape[0], key.shape[1], self.nb_heads, self.proj_size).transpose(1,2)
        # FINISH EXPLAINING THIS PART
        value = value.view(value.shape[0], value.shape[1], self.nb_heads, self.proj_size).transpose(1,2)
        
        # NOT FINISHED
        
        
        
        
        
        
        
        
        