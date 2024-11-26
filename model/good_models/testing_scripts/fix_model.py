import torch
import torch.nn as nn

class TempModel(nn.Module):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 5, (3, 3))
    def forward(self, inp):
        return self.conv1(inp)

model = TempModel()
model.load_state_dict(torch.load("model/model.pth"))
model.eval()