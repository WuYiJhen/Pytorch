import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform

class TextCNN(nn.Module):
    
    def __init__(self, vocab_size, embedding_dim, embed_drop, kernel, num_filter, num_class, attention):
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embed_drop = embed_drop
        self.kernel = kernel # [2, 3, 5]
        self.num_filter = num_filter
        self.Y = num_class
        self.attention = attention
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed_drop = nn.Dropout(p=self.embed_drop)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filter, (k, self.embedding_dim)) for k in self.kernel])
        self.ensemble_weight = nn.Linear(len(self.kernel), 1, bias=False)
        
        for conv in self.convs:
            xavier_uniform(conv.weight)
        xavier_uniform(self.ensemble_weight.weight)
        
        if self.attention:
            self.U = nn.ModuleList([nn.Linear(self.Y, self.num_filter, bias=False) for k in self.kernel]) # [C, Y]
            self.B = nn.ModuleList([nn.Linear(self.num_filter, self.Y) for k in self.kernel]) # [Y, C]
            for u in self.U:
                xavier_uniform(u.weight)
            for b in self.B:
                xavier_uniform(b.weight)   
        else:
            self.fc = nn.Linear(self.num_filter * len(self.kernel), self.Y)
                                       
    def forward(self, x):
        
        # B: Batch Size
        # N: Text Length
        # D: Embedding Dimension
        # C: Number of filters
        # M: Number of region
        # text_input: [B, N, D]
        
        x = self.embedding(x)  
        x = self.embed_drop(x)
        # [B, N, D]
        x = x.unsqueeze(1)  
        # [B, 1, N, D]
        
        if self.attention:
            x = [F.tanh(conv(x).squeeze(3)) for conv in self.convs]
            # [B, C, M] x kernels
            atten = [F.softmax(torch.matmul(x[i].transpose(1, 2), self.U[i].weight), dim=1) for i in range(len(self.convs))]
            # [B, M, Y] x kernels
            x = [torch.matmul(x[i], atten[i]).transpose(1, 2) for i in range(len(self.convs))]
            # [B, Y, C] x kernels
            x = [torch.add(torch.matmul(x[i], self.B[i].weight.transpose(0, 1)).sum(dim=2), self.B[i].bias) for i in range(len(self.convs))]
            # [B, Y] x kernels
            x = torch.cat([(w * x[i]).unsqueeze(2) for i, w in enumerate(F.softmax(self.ensemble_weight.weight).squeeze(0))], 2).sum(dim=2)
            # [B, Y]
        else:
            x = [F.max_pool1d(conv(x).squeeze(-1), conv(x).size()[2]) for conv in self.convs]
            # [B, C, 1] x kernels
            x = torch.cat(x, 1).squeeze(-1)
            # [B, C * len(kernel)]
            x = self.fc(x)
            # [B, Y]
            
        if self.attention: return x, atten
        return  x, _