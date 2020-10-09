import os
import torch.nn.functional as F
import  torch
import torch.nn as nn
from model.Resnet import resnet18, resnet50
from torch.autograd import Variable


def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        out_channels = self.expansion*channels
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        # (x has shape: (batch_size, in_channels, h, w))

        out = F.relu(self.bn1(self.conv1(x))) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)
        out = self.bn2(self.conv2(out)) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = out + self.downsample(x) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        out = F.relu(out) # (shape: (batch_size, channels, h, w) if stride == 1, (batch_size, channels, h/2, w/2) if stride == 2)

        return out

class Auto_encoder(nn.Module):
    def __init__(self):
        super(Auto_encoder, self).__init__()
        ##For img1
        self.Res_Block_1 = self._make_layer(BasicBlock, 3, 64, 2, stride=2)
        self.Res_Block_2 = self._make_layer(BasicBlock, 64, 128, 2, stride=2)
        self.Res_Block_3 = self._make_layer(BasicBlock, 128, 256, 2, stride=2)
        self.Res_Block_4 = self._make_layer(BasicBlock, 256, 512, 2, stride=2)

        self.Deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.Deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.Deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.Deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.Conv_out = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.Tanh(),
        )

        initialize_weights(self)
    def _make_layer(self, block,in_planes, out_planes, num_blocks, stride=1):
        layers = []
        layers.append(block(in_planes, out_planes, stride))
        in_planes=out_planes
        for i in range(1, num_blocks):
            layers.append(block(in_planes, out_planes))
        return nn.Sequential(*layers)

    def forward(self, input):
        N,C,W,H=input.shape[0],input.shape[1],input.shape[2],input.shape[3]

        ##out: (3,256,384)->(512,16,24)
        out_res_1 = self.Res_Block_1(input)
        out_res_2 = self.Res_Block_2(out_res_1)
        out_res_3 = self.Res_Block_3(out_res_2)
        out_res_4 = self.Res_Block_4(out_res_3)

        ##Deocoder:
        # (512,16,24)->(256,32,48)
        out_Deconv1_t0 = self.Deconv_1(out_res_4)
        # (256*2,32,48)->(128,64,96)
        Deconv2_input =  torch.cat((out_Deconv1_t0, out_res_3), 1)
        out_Deconv2_t0 = self.Deconv_2(Deconv2_input)
        # (128*2,64,96)->(64,128,198)
        Deconv3_input =  torch.cat((out_Deconv2_t0, out_res_2), 1)
        out_Deconv3_t0 = self.Deconv_3(Deconv3_input)
        # (64*2,128,198)->(32,256,384)
        #Deconv_4_input=out_Deconv3_t0+out_t0_conv3
        Deconv4_input =  torch.cat((out_Deconv3_t0, out_res_1), 1)
        out_Deconv4_t0 = self.Deconv_4(Deconv4_input)
        out=self.Conv_out(out_Deconv4_t0)
        out=(out+1)/2
        return out

class Discriminator_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, input):
        base_out = self.base_model(input)
        x = self.avg_pool(base_out).squeeze()
        out = self.classifier(x)
        return out

    def load_model(self, model_path):
        print("Loading the params from {} ...".format(model_path))
        load_params = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(load_params['state_dict'], strict=True)
        print("Loading Successful!".format(model_path))

class Classification_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 2),
            #nn.Sigmoid(),
        )

    def forward(self, gen_imgs,img):
        input=torch.cat((gen_imgs,img),dim=1)
        base_out = self.base_model(input)
        x = self.avg_pool(base_out).squeeze()
        out = self.classifier(x)
        return out

    def load_model(self, model_path):
        print("Loading the params from {} ...".format(model_path))
        load_params = torch.load(model_path, map_location=torch.device('cpu'))
        self.load_state_dict(load_params['state_dict'], strict=True)
        print("Loading Successful!".format(model_path))

