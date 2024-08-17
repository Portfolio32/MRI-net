
import torch, torch.nn.functional as F, torchvision, torchvision.transforms as T, copy
dataset = torch.load(r"C:\Users\joeyh\OneDrive\Desktop\Motionclassifier\BETTERDATASETV2")
Xtr,Ytr, XVal, YVal, Xtest, Ytest = dataset

Xtrain, Ytrain = copy.deepcopy(Xtr), copy.deepcopy(Ytr)
Xtr = torch.cat((Xtr, T.RandomRotation((-45, 45))(Xtrain)))
Ytr = torch.cat((Ytr, Ytrain))

Xtr /= 255.0
XVal /= 255.0
Xtest /= 255.0
Xtr = Xtr[:,None, :,:]
XVal = XVal[:,None, :,:]
Xtest = Xtest[:,None, :,:]
Ytr = Ytr[:,None]
YVal = YVal[:,None]
Ytest = Ytest[:,None]

class Linear:
  def __init__(self, fan_in, fan_out, bias = True):
    self.weight = torch.randn((fan_in, fan_out), generator = g) / fan_in**0.5
    self.bias = torch.zeros(fan_out) if bias else None
  def __call__(self, x):
    self.out = x.reshape(x.shape[0], -1) @ self.weight
    if self.bias is not None:
      self.out += self.bias
    return self.out
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])

class Relu:
  def __call__(self, x):
    self.zero = torch.tensor(0).float()
    self.out = torch.maximum(x, self.zero)
    return self.out
  def parameters(self):
    return []

class Dropout:
  def __init__(self, p = 0.5):
    self.p = p
  def __call__(self, x):
    self.mask = (torch.rand(*x.shape) < self.p) / self.p
    self.out = x * self.mask
    return self.out
  def parameters(self):
    return []

class Conv2d:
  # I think this will work, but
  def __init__(self, in_channels, out_channels, kernal_size, stride = 1, padding = 0, bias = True):
    self.weight = torch.randn((out_channels, in_channels, kernal_size, kernal_size), generator=g) / (kernal_size*kernal_size*in_channels)**0.5
    self.bias = torch.zeros((1, out_channels, 1, 1) if bias else None)
    self.kernal_size = kernal_size
    self.padding = padding
    self.stride = stride
    self.out_channels = out_channels
  def __call__(self, x):
    self.H = 1 + (x.shape[2] + 2 * self.padding - self.kernal_size) // self.stride
    self.W = 1 + (x.shape[3] + 2 * self.padding - self.kernal_size) // self.stride
    self.out = ((F.unfold(x, (self.kernal_size, self.kernal_size),  padding=self.padding, stride=self.stride).transpose(1,2)) @ self.weight.view(self.out_channels, -1).t()).transpose(1,2).view(x.shape[0], self.out_channels, self.H, self.W)
    if self.bias is not None:
      self.out += self.bias
    return self.out
  def parameters(self):
    return [self.weight] + ([] if self.bias is None else [self.bias])




g = torch.Generator().manual_seed(31231)

layers = [Conv2d(1, 64, kernal_size = 5, stride=4, padding= 1), Relu(),
          Conv2d(64, 256, kernal_size =5, stride=4, padding= 1), Relu(),
          Conv2d(256, 128, kernal_size = 5, stride=4, padding= 1), Relu(),
          Conv2d(128, 256, kernal_size=5, stride=4, padding= 1), Relu(), Dropout(0.3), 
          Linear(256, 256), Relu(), Dropout(),
          Linear(256, 1),
]


with torch.no_grad():
  layers[-1].weight *=0.1
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 2**0.5
    elif isinstance(layer, Conv2d):
      layer.weight *= 2**0.5

parameters = [p for layer in layers for p in layer.parameters()]
for p in parameters:
  p.requires_grad = True
lb = 1e-3
# so this is the better dataset which I got rid of the intervals
# L2 reg is a good idea
from sklearn.metrics import accuracy_score
for i in range(200):
  ix = torch.randint(0, Xtr.shape[0], (200,), generator=g)
  x, Yb = Xtr[ix], Ytr[ix]
  for layer in layers:
    x = layer(x)
  loss = F.binary_cross_entropy_with_logits(x, Yb.float())

  reg = 0
  for p in parameters[::2]:
    reg += (p ** 2).sum()
  loss += 0.5 * reg * lb

  for layer in layers:
    layer.out.retain_grad()
  for p in parameters:
    p.grad = None
  loss.backward()
  if i % 10 == 0:
    print(i, 'train loss:', loss.item())
  lr = 0.1 #if i < 65 else 0.01
  for p in parameters:
    p.data += -lr * p.grad
print(i, 'train loss:', loss.item())



@torch.no_grad()
def split_loss(split):
  x,y = {
    'train': (Xtr, Ytr),
    'val': (XVal, YVal),
    'test': (Xtest, Ytest),
  }[split]
  for layer in layers:
    if not(isinstance(layer, Dropout)):
      x = layer(x)
  loss = F.binary_cross_entropy_with_logits(x, y.float())
  print(split, loss.item())
split_loss('val')
split_loss('test')

from sklearn.metrics import accuracy_score
with torch.no_grad():
  x = Xtest
  for layer in layers:
    if not(isinstance(layer, Dropout)):
      x = layer(x)
  logits = x
  probabilites = torch.sigmoid(logits)
  predictions = (probabilites > 0.5).int()
  predictions.view(predictions.shape[0])
  guesses = predictions.view(predictions.shape[0])
  accuracy = accuracy_score(guesses, Ytest)
print('accuracy', accuracy)
print(guesses)
print(Ytest.view(Ytest.shape[0]))

