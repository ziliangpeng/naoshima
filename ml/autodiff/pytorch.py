import torch

x = torch.tensor([0.0], requires_grad=True)
y = torch.sin(x)
y.backward()
print(x.grad)