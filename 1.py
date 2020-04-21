import torch
import torch.nn as nn
import torch.nn.functional as F

class Ted(nn.Module):
    def __init__(self):
        super(Ted, self).__init__()
        self.fc1 = nn.Linear(4, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(x)
        return x

    
ted = Ted()
for i in ted.parameters():
    print(i)