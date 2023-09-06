from torch import nn


class OODDetector(nn.Module):
    def __init__(self, arc='wrn', img_size=32):
        super().__init__()
        
        out_channels = 342 if arc == 'densenet' else 128    
        ratio = 8 * (img_size // 32)
        self.fc_out = nn.Linear(out_channels * ratio * ratio, 2048)
        self.relu = nn.ReLU()
        self.out_layer = nn.Linear(2048, 1)

    def forward(self, x, threshold=1e9):
        x = x.flatten(1)
        x = self.fc_out(x)
        x = self.relu(x)
        x = self.out_layer(x)
        return x

    def get_features(self, x):
        return [self.fc_out(x)]