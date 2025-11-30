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
        assert d_model % nb_heads == 0, "d_model has to be divisible by nb_heads"
        
        self.proj_size = d_model // nb_heads
        self.query_layer = nn.Linear(d_model,d_model)
        self.key_layer = nn.Linear(d_model,d_model)
        self.value_layer = nn.Linear(d_model,d_model)
        
        self.w_o = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
        
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
        
        # Here we calculate the attention matrix before the softmax 
        attention_matrix = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        # Each row in `attention_matrix` corresponds to a query position, and each column corresponds to a key position.
        # In causal (decoder) self-attention, a token should not attend to future tokens.
        # Therefore, we mask out (set to -inf) the upper triangular part of the matrix,
        # where the column index > row index, to prevent attention to future positions.
        if mask is not None:
            attention_matrix.masked_fill_(mask == 0, -1e9)
        
        # Apply softmax over the key dimension to obtain attention weights
        attention_matrix = attention_matrix.softmax(dim=-1)
        
        if dropout is not None:
            attention_matrix = dropout(attention_matrix)
              
        return (attention_matrix @ value), attention_matrix
            

    def forward(self,q,k,v,mask):
        query = self.query_layer(q) 
        key = self.key_layer(k)
        value = self.value_layer(v)
        
        # We want multi-head attention, and since we defined the query_layer as a Linear(d_model, d_model)
        # We have to split the output into nb_heads vectors, so it will be equivalent to calculating embedding * W_Q with a matrix of size (d_model, proj_size) nb_head times
        # the output query has the shape (Batch, nb_heads, seq_len, proj_size)
        query = query.view(query.shape[0],query.shape[1],self.nb_heads,self.proj_size).transpose(1,2)
        # We use the same logic for the keys
        key = key.view(key.shape[0], key.shape[1], self.nb_heads, self.proj_size).transpose(1,2)
        # Here we follow what the paper recommended.
        value = value.view(value.shape[0], value.shape[1], self.nb_heads, self.proj_size).transpose(1,2)
        
        x, self.attention_matrix = MultiHeadAttentionLayer.attention(query, key, value, mask, self.dropout)
        
        # Here we first transpose to go back to this shape : (Batch, seq_len, nb_heads, proj_size)
        # Then we want to concatenate for each word the vectors obtained by the different heads
        # Finally, we have the shape (Batch, seq_len, nb_heads * proj_size) = (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.nb_heads * self.proj_size)
        
        # Finally, we use a last layer to compute for each word the concatenated vector and return for each word a context-aware vector.
        return self.w_o(x)
    
    
# We will now code the Residual Connections which is the part where we add the result from the attention to the embeddings and norm
class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
        
    # sublayer will take either the multihead attention or the feed forward network
    def forward(self,x, sublayer):
        # In the paper, they do self.norm(x + sublayer(x))
        # But for efficiency, we can do it like that
        return x + self.dropout(sublayer(self.norm(x)))
    
# Now it's time to build the encoder blocks by using everything we coded before
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_layer : MultiHeadAttentionLayer, feed_forward_layer : FeedForwardLayer, dropout : float) -> None :
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.feed_forward_layer = feed_forward_layer
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        # Here, we create the first part of one encoder which is the multi-head attention and the residual connection
        # So lambda x: self.self_attention_layer(x, x, x, src_mask) means that instead of sublayer it will use this function with self.norm(x) as x
        x = self.residual_connections[0](x, lambda x: self.self_attention_layer(x, x, x, src_mask))
        # Here we create the second part of one encoder which is the feed forward network and the residual connection
        x = self.residual_connections[1](x, self.feed_forward_layer)
        return x
        
# Now that we have built one encoder block, we just need to use multiple blocks
class Encoder(nn.Module):
    
    def __init__(self, layers: nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
        
    def forward(self, x, mask):
        # Each layer will be an encoder block taking x and the mask as a value
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)
    
# The next step is to focus on the decoder
class DecoderBlock(nn.Module):
    
    def __init__(self, self_attention_layer : MultiHeadAttentionLayer, cross_attention_layer: MultiHeadAttentionLayer, feed_forward_layer : FeedForwardLayer, dropout : float ) -> None :
        super().__init__()
        self.self_attention_layer = self_attention_layer
        self.cross_attention_layer = cross_attention_layer
        self.feed_forward_layer = feed_forward_layer
        # We need 3 residual connections : one for the masked self attention, one for the cross attention and one for the feed forward layer
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        # Step 1 : Masked Multi Head Attention
        # Here we take the output sentence and we use masked multi head attention because we don't know the future so each word can only look at the ones before
        # Then we will have context aware embeddings of the output sentence
        x = self.residual_connections[0](x, lambda x: self.self_attention_layer(x,x,x,decoder_mask))
        # Step 2 : Cross Multi Head Attention
        # Now we will use for the query the output sentence and for the key and values we use the encoder output
        # Cross multi head attention will allow to look at the encoder output to make the traduction
        x = self.residual_connections[1](x, lambda x: self.cross_attention_layer(x, encoder_output, encoder_output, encoder_mask))
        # Step 3 : Feed Forward Layer
        x = self.residual_connections[2](x,self.feed_forward_layer)
        return x
      
# Now we will use multiple Decoder Blocks to build the Decoder  
class Decoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList) -> None :
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization
        
    def forward(self,x, encoder_output, encoder_mask, decoder_mask) :
        for layer in self.layers :
            x = layer(x, encoder_output, encoder_mask, decoder_mask)
        return self.norm(x)

# Once we get out of the transformer, we have the modified embeddings of the output sentence
# But we need to predict what is the next word. 
# The last step is to build a linear layer and use a softmax to find the correct word in our vocabulary.
class ProjectionLayer(nn.Module):
    
    def __init__(self, d_model : int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self,x):
        return torch.log_softmax(self.proj(x),dim=-1)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed : InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, proj_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.proj_layer = proj_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)
    
    def decode(self, tgt, encoder_output , src_mask, tgt_mask) :
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        
    def project(self,x):
        return self.proj_layer(x)
    
    def build_transformer(src_vocab_size :int, tgt_vocab_size :int, src_seq_len: int, tgt_seq_len: int, d_model :int =512, N :int =6, h :int = 8, dropout :float =0.1, d_ff: int = 2048) :
        # 1st Step : Create the embedding layers (src and tgt)
        src_embed = InputEmbeddings(d_model=d_model,vocab_size=src_vocab_size)
        tgt_embed = InputEmbeddings(d_model=d_model,vocab_size=tgt_vocab_size)
        
        # 2nd Step : Create the positional encoding layers (src and tgt)
        src_pos = PositionalEncoding(d_model=d_model,seq_len=src_seq_len,dropout=dropout)
        tgt_pos = PositionalEncoding(d_model=d_model,seq_len=tgt_seq_len,dropout=dropout)
        
        # 3rd Step : Create the encoder layers
        encoder_blocks = []
        for _ in range(N) :
            encoder_self_attention_block = MultiHeadAttentionLayer(d_model=d_model,nb_heads=h,dropout=dropout)
            encoder_feed_forward = FeedForwardLayer(d_model=d_model,inner_layer_size=d_ff,dropout=dropout)
            encoder_block = EncoderBlock(encoder_self_attention_block,encoder_feed_forward,dropout)
            encoder_blocks.append(encoder_block)
        
        # 4th Step : Create the decoder layers
        decoder_blocks = []
        for _ in range(N):
            decoder_masked_self_attention_block = MultiHeadAttentionLayer(d_model,h,dropout)
            decoder_cross_attention_block = MultiHeadAttentionLayer(d_model,h, dropout)
            decoder_feed_forward = FeedForwardLayer(d_model,d_ff,dropout)
            decoder_block = DecoderBlock(decoder_masked_self_attention_block, decoder_cross_attention_block, decoder_feed_forward, dropout)
            decoder_blocks.append(decoder_block)
            
        # 5th Step : Create the Encoder and the Decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))
        
        # 6th Step : Create the projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
        
        # 7th Step : Build the Transformer
        transformer = Transformer(encoder,decoder, src_embed,tgt_embed, src_pos, tgt_pos, projection_layer)
        
        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)
                
        return transformer
            
        
        