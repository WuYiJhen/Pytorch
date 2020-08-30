import torch
import torchvision
import torch.nn as nn
from torchvision import transforms, datasets
import torch.nn.functional as F
import torch.optim as optim
import time
from torch.nn.init import xavier_uniform

class TextCNN(nn.Module):
    
    def __init__(self, params):
        super(TextCNN, self).__init__()
        
        self.vocab_size = params['vocab_size']
        self.embedding_dim = params['embedding_dim']
        self.embed_drop = params['embed_drop']
        self.kernel = params['kernel'] # [2, 3, 5]
        self.num_filter = params['num_filter']
        self.Y = params['Y']
        self.Attention = params['Attention']
        
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.embed_drop = nn.Dropout(p = self.embed_drop)
        self.convs = nn.ModuleList([nn.Conv2d(1, self.num_filter, (k, self.embedding_dim)) for k in self.kernel])
        
        for conv in self.convs:
            xavier_uniform(conv.weight)
        
        if self.Attention:
            self.U = nn.ModuleList([nn.Linear(self.Y, self.num_filter) for k in self.kernel]) # [C, Y]
            for u in self.U:
                xavier_uniform(u.weight)
            self.l1 = nn.Linear(self.num_filter, self.num_filter//2)
            self.l2 = nn.Linear(self.num_filter//2, 1)
            self.fc = nn.Linear(self.Y * len(self.kernel), self.Y)
        else:
            self.fc = nn.Linear(self.num_filter * len(self.kernel), self.Y)
                                       
    def forward(self, x):
        
        # B: Batch Size
        # N: Text Length
        # D: Embedding Dimension
        # C: Number of filters
        # M: Text Length After Convolution
        # text_input: [B, N, D]
        
        x = self.embedding(x)  
        x = self.embed_drop(x)
        # [B, N, D]
        x = x.unsqueeze(1)  
        # [B, 1, N, D]
        
        if self.Attention:
            x = [F.tanh(conv(x).squeeze(3)) for conv in self.convs]
            # [B, C, M]

            ##### Then attention ####

            alphas = [F.softmax(torch.matmul(x[i].transpose(1, 2), self.U[i].weight), dim = 1) for i in range(len(self.convs))]
            # [B, M, Y]
            x = torch.cat([torch.matmul(x[i], alphas[i]) for i in range(len(self.convs))], 2)
            # [B, C, len(kernel) * Y]

            x = F.relu(self.l1(x.transpose(1, 2)))
            # [B, len(kernel) * Y, C//2]
            
            x = self.l2(x)
            # [B, len(kernel) * Y, 1]

            x = self.fc(x.squeeze(2))
            # [B, Y]
            
        else:
            x = [F.max_pool1d(conv(x).squeeze(-1), conv(x).size()[2]) for conv in self.convs]
            # [B, C, 1]
            x = torch.cat(x, 1).squeeze(-1)
            # [B, C * len(kernel)]
            x = self.fc(x)
            # [B, Y]
            
        if self.Attention: return x, alphas
        return  x, x
