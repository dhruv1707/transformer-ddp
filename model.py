import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model=d_model
        self.vocab_size=vocab_size
        self.embedding=nn.Embedding(vocab_size, d_model)
    
    def forward(self, x):
        # print(f"X: {x}")
        return self.embedding(x)*math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model=d_model
        self.seq_len=seq_len
        self.dropout=nn.Dropout(dropout)

        #Create a matrix of (seq_len, d_model)

        pe = torch.zeros(seq_len, d_model)

        # Creating the numerator and denominator of positional encoding

        pos = torch.arange(seq_len).unsqueeze(1) #(seq_len, 1)

        div_term = torch.exp((torch.arange(0, d_model, 2).float() * (-math.log(10000/d_model)))).unsqueeze(0) # (1, d_model)
        # print(f"Pos: {pos.shape}, Div_term:{div_term.shape}")
        pe[:, 0::2] = torch.sin(pos * div_term)

        pe[:, 1::2] = torch.cos(pos*div_term)
        # batch of sentences
        pe = pe.unsqueeze(0) # (1, seq_len, d_model) 
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(f"PE Shape: {self.pe.shape}")
        x = x + (self.pe[:, :x.shape[1], :])
        return self.dropout(x)

class LayerNormalization(nn.Module):
    def __init__(self, eps: float=10**-6):
        super().__init__()
        self.eps = eps
        # adding fluctuation in the data to make the model not too restrictive between 0 and 1 always. Tune these parameters so the model can learn to amplify values when necessary
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) # Added
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim =-1, keepdim=True)
        return self.alpha*((x - mean) / (std + self.eps)) + self.bias

class FeedForwardBlock(nn.Module):

    def __init__(self, d_ff: int, d_model: int, dropout: float):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.linear_2(self.dropout(self.relu(self.linear_1(x))))

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, h: int, dropout: float, d_model: int):
        super().__init__()
        self.h = h
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        assert d_model % h == 0, "d_model is not divisible by 0"
        self.w_q = nn.Linear(d_model, d_model) # (Batch, seq_len, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.d_k = d_model // h
    
    @staticmethod
    def attention(query, key, value, d_k, mask, dropout):
        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        # print(f"Query:{query.shape}, Key: {key.shape}, Value: {value.shape}")
        attention_output = (query @ key.transpose(2, 3))/(math.sqrt(d_k))
        if mask is not None:
            attention_output.masked_fill(mask == 0, 1e-9)
        attention_scores = attention_output.softmax(dim = -1) # (Batch, h, seq_len, seq_len)
        # print(f"Attention output: {attention_output.shape}, Attention score:{attention_scores.shape}")
        attention_scores = dropout(attention_scores)
        return (attention_scores @ value),attention_scores
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q).view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --> (Batch, h, seq_len, d_k)
        key = self.w_k(k).view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2)
        x, attention_scores = MultiHeadAttentionBlock.attention(query=query, key=key, value=value, d_k=self.d_k, mask=mask, dropout=self.dropout) # (Batch, h, seq_len, d_k)
        # print(f"Input shape:{x.shape}")
        x = x.transpose(1, 2)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.h * self.d_k)
        return self.w_o(x)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: int):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()
    
    def forward(self, x, prev_output):
        return x + self.dropout(prev_output(self.norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, multi_head_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float):
        super().__init__()
        self.multi_head_attention_block = multi_head_attention_block
        self.feed_forward_block =  feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for i in range(2)])
        self.dropout = dropout
    
    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.multi_head_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, lambda x: self.feed_forward_block(x))
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, dropout: float, feed_forward_block:  FeedForwardBlock):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.dropout = dropout
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, target_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, lambda x: self.feed_forward_block(x))
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, encoder_output, src_mask, target_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, target_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)
    

class Transformer(nn.Module):

    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, target_embed: InputEmbedding, src_pos: PositionalEncoding, target_pos: PositionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.target_embed = target_embed
        self.src_pos = src_pos
        self.target_pos = target_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src_embed = self.src_embed(src)
        src_final = self.src_pos(src_embed)
        return self.encoder(src_final, src_mask)
    
    def decode(self, encoder_output, target, src_mask, target_mask):
        target_embed = self.target_embed(target)
        target_final = self.target_pos(target_embed)
        return self.decoder(target_final, encoder_output, src_mask, target_mask)
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int, N: int=6, h: int=8, d_model: int=512, d_ff: int=2048, dropout: float=0.1):
    # Create the input embedding layers
    src_embed = InputEmbedding(d_model, src_vocab_size)
    target_embed = InputEmbedding(d_model, target_vocab_size)

    # Create the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    target_pos = PositionalEncoding(d_model, target_seq_len, dropout)

    # Create the Encoder
    encoder_blocks = []
    for _ in range(N):
        enc_self_attention = MultiHeadAttentionBlock(h, dropout, d_model)
        enc_feed_forward = FeedForwardBlock(d_ff, d_model, dropout)
        encoder_block = EncoderBlock(enc_self_attention, enc_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # Create the Decoder
    decoder_blocks = []
    for _ in range(N):
        dec_self_attention = MultiHeadAttentionBlock(h, dropout, d_model)
        dec_cross_attention = MultiHeadAttentionBlock(h, dropout, d_model)
        dec_feed_forward = FeedForwardBlock(d_ff, d_model, dropout)
        decoder_block = DecoderBlock(dec_self_attention, dec_cross_attention, dropout, dec_feed_forward)
        decoder_blocks.append(decoder_block)
    
    encoder = Encoder(nn.ModuleList(encoder_blocks))

    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    # Create the project layer
    projection_layer = ProjectionLayer(d_model, target_vocab_size)

    transformer = Transformer(encoder, decoder, src_embed, target_embed, src_pos, target_pos, projection_layer)

    # Initialize the params
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return transformer



