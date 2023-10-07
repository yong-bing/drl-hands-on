import torch
import torch.nn as nn

input1 = torch.tensor([1, 2, 3, 4, 5]).float() # convert input1 to a float tensor
Li = nn.Linear(5, 3)
output1 = Li(input1)
print(output1)