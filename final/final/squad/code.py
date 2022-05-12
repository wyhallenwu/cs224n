import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

window_size = 2
sent_len = 20
embedding_size = 10
batch_embedding = torch.rand([5, sent_len, embedding_size])
print(batch_embedding.shape)
layer = nn.Conv2d(5,
                  100,
                  kernel_size=[window_size, embedding_size],
                  stride=1,
                  bias=True)
y = layer(batch_embedding)
layer2 = nn.Tanh()
y = layer2(y)
y = y.squeeze()
print(y.shape)
layer3 = nn.MaxPool1d(kernel_size=(sent_len - window_size + 1))
y = layer3(y).squeeze()
print(y.shape)

x = [torch.tensor([1, 2]), torch.tensor([3, 4])]
print(torch.stack(x).shape)

print("*" * 20)
filter_shape = torch.randn(100, sent_len, 5)
# x = batch_embedding.view(-1, embedding_size)
# x = x.squeeze()
# print(x.shape)
print(batch_embedding.shape)
z = F.conv1d(batch_embedding, filter_shape, stride=1)
z_ = F.conv2d(batch_embedding.unsqueeze(1), torch.rand(1, 1, 5, 10))
print(z_.shape)
print(z.shape)
mx_layer = nn.MaxPool1d(kernel_size=6)
z = mx_layer(z).squeeze()
print(z.shape)
z = F.linear(z, torch.randn(200, 100))
print(z.shape)

x = torch.rand(5, 3, 2)
y = torch.rand(5, 3, 4)
z = torch.cat((x, y), dim=2)
print(z.shape)
print("--")
z = F.max_pool1d(x, 2)
print(z.shape)
print("--")
l = [x, y]
print(torch.cat(l, dim=2).shape)

print('-' * 20)
# batch_size, sent, sub, emb
x = torch.rand(20, 307, 16, 64)
# outc, inc, window
f = torch.rand(100, 307, 16, 64)
z = F.conv2d(x, f).squeeze()
print(z.shape)
x = torch.rand(20 * 307, 1, 16 * 64)
y = torch.rand(100, 1, 5 * 64)
z = F.conv1d(x, y, stride=64)
print(z.shape)
z = z.view(20 * 307, 100, -1)
z = F.max_pool1d(z, 12)
print(z.shape)
z = z.view(20, 307, -1)
print(z.shape)
# q = z
# o = [z, q]
# print(z.shape)
# print(torch.cat(o, dim=2).shape)

print("test remove dim")
x = torch.ones((10, 10, 10))
r = np.random.choice(10, 7, False)
x = x[:, r, :]
print(x.shape)
x = x.permute([0, 2, 1])
x = F.linear(x, torch.ones(20, 7))

print(x.shape)
print(r)
