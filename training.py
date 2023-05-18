import torch
import torch.nn as nn
from torch.nn import functional as functional

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
num_emb = 32

torch.manual_seed(32)

#wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# creatig the character list
chars = sorted(set(text))
vocab_size = len(chars)
itos = {i:char for i,char in enumerate(chars)}
stoi = {char:i for i,char in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda i: ''.join([itos[j] for j in i])

data = torch.tensor(encode(text), dtype=torch.long)
# train/val/test split
n1 = int(0.8*len(data))
n2 = int(0.9*len(data))
train_set = data[:n1]
val_set = data[n1:n2]
test_set = data[n2:]

def get_batch(data, batch_size=4, block_size=8): # 4 different chunks with context size of 8
    # specifying the dataset
    if data == 'train':
        data = train_set
    elif data =='val':
        data = val_set
    else:
        data = test_set
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)

    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = loss.mean()
    model.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # each token reads the logit for the next token
        self.token_embedding_table = nn.Embedding(vocab_size, num_emb)
        self.position_embedding_table = nn.Embedding(block_size, num_emb)
        self.lm_head = nn.Linear(num_emb, vocab_size)
        
    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None: # to successfully run the generate func
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = functional.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_tokens):
        # idx is (B,T) array of indices
        for _ in range(max_tokens):
            # getting the predictions
            logits, loss = self(idx)
            # focusing on the last indices
            logits = logits[:, -1, :] # becomes (B,C)
            # calculating the probabilities
            probs = functional.softmax(logits, dim=1) # (B,C)
            # sampling from prob distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
            # append the sampled index to the end of the context
            idx = torch.cat((idx, idx_next), dim=1) # becomes (B,T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}")

    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_tokens=500)[0].tolist()))
