import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        x = x.permute(0, 2, 1)     # (B, C, T)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)     # (B, T, C)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
