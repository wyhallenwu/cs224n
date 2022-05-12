from numpy import outer
import torch

x = torch.rand([1, 2, 3])
print(x.shape)
print(x)
x = x.expand(3, 2, 3)
print(x.shape)
print(x)

lstm = torch.nn.LSTM(input_size=10, hidden_size=1, num_layers=2)
h_0 = torch.rand([2, 5, 1])
c_0 = torch.rand([2, 5, 1])
input = torch.rand([10, 5, 10])
output, (h_n, c_n) = lstm(input, (h_0, c_0))
print(output.shape)
s = torch.functional.F.softmax(output, dim=0)
print(s[:, 0, :])
print(sum(s[:, 0, :]))