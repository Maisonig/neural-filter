from torch import nn, round, max


class DenoisingLSTM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=False
        )
        self.fc = nn.Linear(config.hidden_size, config.output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return self.sigmoid(out)

