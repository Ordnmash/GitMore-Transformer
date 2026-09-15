import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

file   = "/home/ordn/Documents/ordn_projects/Gitmore-Transformer/commits.txt"
data   = open(file, "r", encoding='utf-16').read().splitlines()
vocabs = ['^'] + list(sorted(set("".join(data)))) + ['~']
#print(f"{len(vocabs)} vocabs = {'|'.join(vocabs)}")

stoi   = {s:i for i,s in enumerate(vocabs)}
itos   = {i:s for i,s in enumerate(vocabs)}

def encode(data: list):

  data = data if isinstance(data, list) else [data]
  ix   = []
  for line in data:
    ix.append([stoi[s] for s in list('^' + line + '~')])
  x    = []
  for dx in ix:
    x += dx
  return x

def decode(x):
  out  = []
  for ix in x:
    out.append(itos[ix])
  return ''.join(out)

encoded_data = encode(data)

# model hyperParameters
block_size = 16
vocab_size = len(vocabs)
n_embed    = 8
n_hidden   = 100

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.emd = nn.Embedding(vocab_size, n_embed)

  def forward(self, x, y=None):
    logits = self.emd(x) # (T, C)
    if y is not None: # (T)
      loss = F.cross_entropy(logits, y)
    else:
      loss = None
    return logits, loss

  def generate(self):
    out = []
    inn = torch.tensor([stoi['^']])
    while True:
      logits, _ = self(inn[-1])
      #print('logits shape = ',logits.shape)
      probs     = F.softmax(logits[-1], dim=0) if logits.ndim > 1 else F.softmax(logits, dim=0)
      #print('probs shape = ', probs.shape)
      ix        = torch.multinomial(probs, num_samples=1).item()
      if ix != stoi['~']:
        out.append(itos[ix])
        lin = inn.view(-1).tolist(); lin.append(ix)
        inn = torch.tensor(lin)
      else:
        break
    print(''.join(out))


model     = BigramLanguageModel()
optimizer = optim.AdamW(model.parameters(),lr=1e-3)
model.generate()

for i in range(20000):

  optimizer.zero_grad(set_to_none=True)

  inx   = torch.randint(0, (len(encoded_data)-(block_size+1)), (1,)).item()
  x     = torch.tensor(encoded_data[inx:inx+block_size], dtype=torch.long)
  y     = torch.tensor(encoded_data[inx+1:(inx+block_size)+1], dtype=torch.long)

  _,loss= model.forward(x, y)
  loss.backward()

  # update
  optimizer.step()
  if (i+1) % 1000 == 0:
    print(f"epoch:{i+1}   | loss={loss.item():.4f}")

model.generate()