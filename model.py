from config import *
from utils import *


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels):
        super(TimeEncoder, self).__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t):
        return self.lin(t.view(-1, 1)).cos()
    
class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(
            in_channels, out_channels, heads=8, dropout=0.0, edge_dim=edge_dim
        )
        self.conv2 = TransformerConv(
            out_channels * 8,
            out_channels,
            heads=1,
            concat=False,
            dropout=0.0,
            edge_dim=edge_dim,
        )

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)       
        x = F.relu(self.conv(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        return x


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels * 2)
        self.lin_dst = Linear(in_channels, in_channels * 2)

        self.lin_seq = nn.Sequential(
            Linear(in_channels * 4, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )

    def forward(self, z_src, z_dst):
        h = torch.cat([self.lin_src(z_src), self.lin_dst(z_dst)], dim=-1)

        h = self.lin_seq(h)

        return h


class SparseAttentionTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, device):
        super(SparseAttentionTransformer, self).__init__()
        self.embedding = nn.Linear(input_dim, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim, 
                nhead=num_heads, 
                dim_feedforward=embed_dim * 4, 
                activation='relu',
                batch_first = True),
            num_layers=6
        )

    def forward(self, x, mask):
        # 将节点输入映射到嵌入空间
        x = self.embedding(x)  # (batch_size, seq_len, hidden_dim)
        # 应用mask
        x = self.transformer_encoder(x, mask=mask)
        # print(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        # self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.sep_embedding = nn.Embedding(1, d_model)
        self.dropout = nn.Dropout(dropout)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)  # 自注意力层权重
            nn.init.xavier_uniform_(layer.linear1.weight)   

    def forward(self, src1, src2, mask1, mask2):
        # x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embeddings
        # src1.shape = (batch, padding_len, 100)
        # src2.shape = (batch, padding_len, 100)
        # mask1.shape = (batch, padding_len)
        # mask2.shape = (batch, padding_len)
        sep = self.sep_embedding(torch.tensor([0], device=src1.device)).repeat(src1.shape[0], 1, 1)
        # sep.shape = (batch, 1, 100)
        sep_mask = torch.zeros(src1.shape[:1], dtype=torch.long, device=mask1.device).reshape(-1, 1)
        # sep_mask.shape = (batch, 1)
        x = torch.cat((src1, sep, src2), dim=1)
        mask = torch.cat([mask1, sep_mask, mask2], dim=1)
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        return x

class TransformerEncoder2(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dropout=0.1):
        super(TransformerEncoder2, self).__init__()
        # self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.sep_embedding = nn.Embedding(1, d_model)
        self.dropout = nn.Dropout(dropout)
        for layer in self.transformer_encoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)  # 自注意力层权重
            nn.init.xavier_uniform_(layer.linear1.weight)   

    def forward(self, src, mask, sep_len):
        # x = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)  # Scale embeddings
        # src.shape = (n_id, 100)
        # sep.shape = (batch_size, 100)
        sep = self.sep_embedding(torch.tensor([0], device=src.device)).repeat(sep_len, 1)
        # sep.shape = (batch, 1, 100)
        # sep_mask.shape = (batch, 1)
        x = torch.cat((src, sep), dim=0).unsqueeze(0)
        mask = torch.tensor(mask, device=src.device, dtype=float)
        
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x, mask=mask)
        return x

class EdgePredictor(nn.Module):
    def __init__(self,in_channels, out_channels):
        super(EdgePredictor, self).__init__()
        self.lin_seq = nn.Sequential(
            Linear(in_channels, in_channels * 8),
            torch.nn.BatchNorm1d(in_channels * 8),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 8, in_channels * 2),
            torch.nn.BatchNorm1d(in_channels * 2),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(in_channels * 2, int(in_channels // 2)),
            torch.nn.BatchNorm1d(int(in_channels // 2)),
            torch.nn.Dropout(0.5),
            nn.Tanh(),
            Linear(int(in_channels // 2), out_channels),
        )
        # nn.init.xavier_uniform_(self.lin_seq.weight)
    
    def forward(self,x):
        x = self.lin_seq(x)
        return x
