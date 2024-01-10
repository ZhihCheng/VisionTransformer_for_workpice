from torch import nn

class ResNetModified(nn.Module):
    def __init__(self, CNN_model,VIT_model):
        super(ResNetModified, self).__init__()
        self.CNN_model = CNN_model
        self.VIT_model = VIT_model
        self.CNN_model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
    def forward(self, x):
        x = self.CNN_model.conv1(x)
        x = self.CNN_model.bn1(x)
        x = self.CNN_model.relu(x)
        x1 = self.CNN_model.layer1(x)     
        x2 = self.CNN_model.layer2(x1)

        output = self.VIT_model(x2)

        return output