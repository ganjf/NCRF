import torch
import torch.nn as nn

class DiscriminatePatch(nn.Module):

    def __init__(self):
        super(DiscriminatePatch, self).__init__()
        self.layer1 = nn.Sequential(
            #   (3, 400, 400)
            nn.Conv2d(in_channels=3, out_channels=80, kernel_size=10, stride=2),     #80 x 196x196
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),
            nn.MaxPool2d(kernel_size=6, stride=4)                          #  48x48
        )
        self.layer2 = nn.Sequential(
            #   (80, 48, 48)
            nn.Conv2d(in_channels=80, out_channels=120, kernel_size=5, stride=1),    #120 x 44x44
            nn.ReLU(),
            nn.LocalResponseNorm(size=2),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.layer3 = nn.Sequential(
            #   (120, 21, 21)
            nn.Conv2d(in_channels=120, out_channels=160, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            #   (160, 19, 19)
            nn.Conv2d(in_channels=160, out_channels=200, kernel_size=3, stride=1),   #200x 17x17
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)                         #200x 8 x8
        )
        self.layer5 = nn.Sequential(
            #   (200 * 8 * 8)
            nn.Linear(in_features=200 * 8 * 8, out_features=320, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.layer6 = nn.Sequential(
            #   (320, )
            nn.Linear(in_features=320, out_features=320, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
        self.layer7 = nn.Sequential(
            #   (320, )
            nn.Linear(in_features=320, out_features=1, bias=True),
            nn.Sigmoid()
        )



    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x.view(-1, 200 * 8 * 8))
        x = self.layer6(x)
        x = self.layer7(x)
        return x

if __name__ == '__main__':
    dsp = DiscriminatePatch()
    print(dsp)
    x = torch.ones((2, 3, 400, 400))
    y = dsp(x)
    print(y.size())