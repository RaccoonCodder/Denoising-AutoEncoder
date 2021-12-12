from torch import nn
from torch.nn import BatchNorm2d as bn
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU
from torch.nn.modules.pooling import MaxPool2d
from torchvision.models import resnet50

# Output Size = (Input size - K + 2P) / S + 1
class SimpleAutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(SimpleAutoEncoder, self).__init__()
        backbone = resnet50(pretrained=True)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])
        
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=3, bias=True),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        #     nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        # )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            # bn(1024),
            nn.Conv2d(512, 128, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=4, mode='bilinear'),
            # bn(512),
            nn.Conv2d(128, 3, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            # bn(256),
            # nn.Conv2d(256, 128, kernel_size=1, padding=0),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
            # bn(128),
            # nn.Conv2d(128, 3, kernel_size=1, padding=0),
            # nn.Upsample(scale_factor=2, mode='bilinear'),
        )

    def forward(self, x):
        # out0 = self.encoder[0](x)
        # print(out0.shape)
        # out1 = self.encoder[1](out0)
        # print(out1.shape)
        # out2 = self.encoder[2](out1)
        # print(out2.shape)
        # out3 = self.encoder[3](out2)
        # print(out3.shape)
        
        # out4 = self.encoder[4](out3)
        # print(out4.shape)
        # out5 = self.encoder[5](out4)
        # print(out5.shape)
        # out6 = self.encoder[6](out5)
        # print(out6.shape)
        
        # out7 = self.encoder[7](out6)
        # print(out7.shape)
        # out8 = self.encoder[8](out7)
        # print(out8.shape)
        # out9 = self.encoder[9](out8)
        # print(out9.shape)
        out = self.encoder(x)
        # recon = self.decoder(out)
        recon = self.decoder(out)
        
        return recon