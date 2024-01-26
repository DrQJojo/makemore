import torch
import torch.nn as nn
import torch.nn.functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300  # use to evaluate the loss once a while
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200  # evaluate the model with several batches then take the mean
n_embd = 32

# scale-up hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200  # evaluate the model with several batches then take the mean
n_embd = 384
n_head = 6      # number of heads in multihead attention
n_layer = 6     # number of blocks
dropout = 0.2   # since the model is scaled up, we use drop out to prevent overfitting
# ---------------

torch.manual_seed(1337)

with open(r'input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(''.join(text))))
vocab_size = len(chars)
stoi = {s: i for i, s in enumerate(chars)}
itos = {i: s for i, s in enumerate(chars)}
encode = lambda e: [stoi[c] for c in e]
decode = lambda l: ''.join(itos[i] for i in l)

data = torch.tensor(encode(text), dtype=torch.long)
n = int(len(data) * 0.9)
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
# when we don't do back propagation, it would be better to tell pytorch that, so it can run more efficienty
def evaluate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iter)
        for k in range(eval_iter):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x)  # B,T,head_size
        k = self.key(x)
        v = self.value(x)
        # I guess Andrej assumes that head_size = n_embd
        wei = q @ k.transpose(-1, -2) * C ** -0.5  # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v  # B,T,head_size
        return out


# multi-head attention
class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# forward layer
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # this 4*n_embd comes from the original paper
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# block
# this block is the decoder part except for the cross-attention part from the original paper
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        # About layer norm, what we will implement is slightly different from the original paper
        # we implement layer norm before multihead attention or feedforward layer, as this is more common used nowadays
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = self.sa(self.ln1(x)) + x  # residual connection
        x = self.ffw(self.ln2(x)) + x  # residual connection
        return x


# simple bigram model
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def __call__(self, idx, targets=None):
        # idx = (B,T), targets = (B,T)
        logits = self.token_embedding_table(idx)  # B,T,C
        if targets is None:
            loss = None
        else:
            l = torch.transpose(logits, 1, 2)  # B,C,T
            loss = F.cross_entropy(l, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# self-attention model with only one head
class OneHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))  # T,C
        x = tok_emb + pos_emb  # B,T,C
        x = self.sa_head(x)  # B,T,C
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            l = logits.transpose(1, 2)
            loss = F.cross_entropy(l, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logtis, loss = self(idx[:, -block_size:])
            logits = logtis[:, -1, :]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1, replacement=True)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx


# self-attention with multi-head
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding = nn.Embedding(block_size, n_embd)
        self.ma_head = MultiHead(4, n_embd // 4)  # under this setting, # channels will still be n_embd
        self.ffw = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.positional_encoding(torch.arange(T, device=device))  # T,C
        x = tok_emb + pos_emb
        x = self.ma_head(x)
        x = self.ffw(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            l = logits.transpose(1, 2)
            loss = F.cross_entropy(l, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


# decoder except for the cross-attention part
class SimpleDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head=n_head) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        # now as we stack up several blocks, the network gets quite deep
        # there are 2 methods to improve the performance: residual connection and layer normalization
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)  # B,T,C
        pos_emb = self.positional_encoding_table(torch.arange(T, device=device))  # T,T
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # B,T,C
        if targets is None:
            loss = None
        else:
            l = logits.transpose(1, 2)
            loss = F.cross_entropy(l, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -block_size:])
            logits = logits[:, -1, :]
            probs = F.softmax(logits, -1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=-1)
        return idx


# model = BigramLanguageModel()
# model = OneHeadAttention()
# model = MultiHeadAttention()
model = SimpleDecoder()
m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = evaluate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f},val loss {losses['val']:.4f}")

    # minibatch
    xb, yb = get_batch('train')

    # forward
    logits, loss = m(xb, yb)

    # backward
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

'''
performance log:
Bigram:                 train 2.5686, val 2.5781 
OneHeadAttention:       train 2.3838, val 2.4043
MultiHeadAttention:     train 2.2105, val 2.2290
Decoder (without norm): train 1.9852, val 2.0779    it gets overfitting 
Decoder (with norm):    train 1.9750, val 2.0676

scale up the model with dropout
                        train , val                 ehhhh, it's too time consuming, my pc cannot run model this large
'''
