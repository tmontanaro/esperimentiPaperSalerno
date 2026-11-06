import torch, torch.nn as nn, torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); self.b2 = nn.BatchNorm1d(cout)
        self.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout)) if (stride != 1 or cin != cout) else nn.Identity()
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h)); return F.relu(h + self.sk(x))

class ECGNet(nn.Module):
    def __init__(self, n_out, n_tab, use_tab):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, 15, 2, 7, bias=False), nn.BatchNorm1d(32), nn.ReLU())
        self.body = nn.Sequential(Block(32, 32, 2), Block(32, 64, 2), Block(64, 128, 2), Block(128, 128, 2))
        self.use_tab = use_tab; emb = 256 + (32 if use_tab else 0)
        self.tab = nn.Sequential(nn.Linear(n_tab, 32), nn.ReLU()) if use_tab else None
        self.drop = nn.Dropout(0.3); self.head = nn.Linear(emb, n_out)
    def embed(self, x, t):
        h = self.body(self.stem(x)); h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return torch.cat([h, self.tab(t)], 1) if self.use_tab else h
    def forward(self, x, t, return_emb=False):
        e = self.embed(x, t); o = self.head(self.drop(e)); return (o, e) if return_emb else o
