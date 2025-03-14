import torch
import torch.nn as nn

# Conv + norm + relu
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, norm=True):
        super().__init__()
        if norm:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=True, padding_mode="reflect"),
                nn.LeakyReLU(0.2)
            )
    
    def forward(self, x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()   
        layers = [] 
        for feature in features:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2, norm=False if feature==features[0] else True))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x)) # use sigmoid to make sure output is between 0 and 1
        
        
def test():
    x = torch.randn(5, 3, 256, 256)
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(model)
    print(preds.shape)
    
if __name__ == "__main__":
    test()