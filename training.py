import torch
import torch.nn as nn
from torch.nn import functional as functional

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

torch.manual_seed(31)

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
    
    x, y = 

    return x, y

