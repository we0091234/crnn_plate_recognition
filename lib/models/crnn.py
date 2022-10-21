import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import models

class BidirectionalLSTM(nn.Module):
    # Inputs hidden units Out
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh, n_rnn=2, leakyRelu=False):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, batchNormalization=False):
            nIn = nc if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module('relu{0}'.format(i),
                               nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))

        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d(2, 2))  # 64x16x64
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d(2, 2))  # 128x8x32
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 256x4x16
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2, 2), (2, 1), (0, 1)))  # 512x2x16
        convRelu(6, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))
        self.newCnn=nn.Conv2d(512,nclass,1,1)

    def forward(self, input):

        # conv features
        conv = self.cnn(input)
        conv=self.newCnn(conv)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2) # b *512 * width
        conv = conv.permute(2, 0, 1)  # [w, b, c]
        # output = F.log_softmax(self.rnn(conv), dim=2)
        output = F.log_softmax(conv, dim=2)

        return output

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_crnn(config,export=False,cfg=None,num_class=78):

    # model = CRNN(config.MODEL.IMAGE_SIZE.H, 3, config.MODEL.NUM_CLASSES + 1, config.MODEL.NUM_HIDDEN)
    # model = caffeNetOcr(num_classes=78,export=export)
    model = myNet(num_classes=num_class,export=export,cfg=cfg)
    # model.apply(weights_init)

    return model


myCfg = [32,'M',64,'M',128,'M',256]
# cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256]

class myNet(nn.Module):
    def __init__(self,cfg=None,num_classes=78,export=False):
        super(myNet, self).__init__()
        if cfg is None:
            cfg =[32,32,64,64,'M',128,128,'M',196,196,'M',256,256]
            # cfg =[32,32,'M',64,64,'M',128,128,'M',256,256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        # self.loc =  nn.MaxPool2d((2, 2), (5, 1), (0, 1),ceil_mode=True)
        # self.loc =  nn.AvgPool2d((2, 2), (5, 2), (0, 1),ceil_mode=False)
        self.loc =  nn.MaxPool2d((5, 2), (1, 1),(0,1),ceil_mode=False)
        self.newCnn=nn.Conv2d(256,num_classes,1,1)
        # self.newBn=nn.BatchNorm2d(num_classes)
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d =nn.Conv2d(in_channels, cfg[i], kernel_size=5,stride =1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1,1),stride =1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x=self.loc(x)
        x=self.newCnn(x)
        # x=self.newBn(x)
        if self.export:              #推理模式
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2,1)  # [w, b, c]
            conv =conv.argmax(dim=2)
            return conv
        else:                       #训练模式
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # output = F.log_softmax(self.rnn(conv), dim=2)
            output = F.log_softmax(conv, dim=2)
            return output


myCfg = [32,32,64,64,'M',128,128,'M',256,256]
class myNet1(nn.Module):
    def __init__(self,cfg=None,num_classes=78,export=False):
        super(myNet1, self).__init__()
        if cfg is None:
            cfg = myCfg
        self.feature = self.make_layers(cfg, True)
        self.export = export
        # self.classifier = nn.Linear(cfg[-1], num_classes)
        # self.loc =  nn.MaxPool2d((2, 2), (5, 1), (0, 1),ceil_mode=True)
        self.loc =  nn.MaxPool2d((5, 2), (1, 1),ceil_mode=False)
        self.newCnn=nn.Conv2d(256,num_classes,1,1)
    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d =nn.Conv2d(in_channels, cfg[i], kernel_size=5,stride =1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else :
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(0,1),stride =1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        x=self.loc(x)
        x=self.newCnn(x)
        if self.export:
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2,1)  # [w, b, c]
            conv =conv.argmax(dim=2)
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # output = F.log_softmax(self.rnn(conv), dim=2)
            output = F.log_softmax(conv, dim=2)
            return output

class caffeNetOcr(nn.Module):
    def __init__(self,num_classes=2,export=False):
        super(caffeNetOcr,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=False),
        )

        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=False),
        )
 

        self.conv3=nn.Sequential(
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3,2,ceil_mode=False),
        )

        self.conv4=nn.Sequential(
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,1),(0,1),ceil_mode=False),
        )

        self.conv5=nn.Sequential(
            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d((2,2),(2,1),(0,1),ceil_mode=False),
        )

        self.conv6=nn.Conv2d(512,num_classes,1,1)
        self.export=export
           
        # self.Linear1=nn.Linear(192*7*7,num_classes)

    
    def forward(self,x):
        x = self.conv1(x)
        # x=self.maxPool1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x=self.conv5(x)
        x=self.conv6(x)
        if self.export:
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2,1)  # [w, b, c]
            conv =conv.argmax(dim=2)
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # output = F.log_softmax(self.rnn(conv), dim=2)
            output = F.log_softmax(conv, dim=2)
            return output
            # x=self.conv5(x)
        # x=x.view(x.size(0),-1)
        # x=self.Linear1(x)
      
        # x=self.maxPool2(x)
        return x

class myNetRes50(nn.Module):
    def __init__(self,num_classes=78,export=False):
        super(myNetRes50,self).__init__()
        self.model = models.resnet50(pretrained=True)
        self.avgPool=nn.AvgPool2d((6,1),(1,1))
        self.conv= nn.Conv2d(512,num_classes,1,1)
        self.export = export
        # self.conv1 = nn.Conv2d(3,64,7,2,3)
    def forward(self,x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x =self.model.layer1(x)
        x=self.model.layer2(x)
        x = self.avgPool(x)
        x = self.conv(x)
        if self.export:
            conv = x.squeeze(2) # b *512 * width
            conv = conv.transpose(2,1)  # [w, b, c]
            conv =conv.argmax(dim=2)
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2) # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            # output = F.log_softmax(self.rnn(conv), dim=2)
            output = F.log_softmax(conv, dim=2)
            return output
        
    
        
        

if __name__ == '__main__':
    #  model = CRNN(32, 3, 79,10)
     cfg =[32,'M',64,'M',128,'M',256]
     model = myNet(num_classes=78,export=True,cfg=cfg)
     input = Variable(torch.FloatTensor(1, 3, 48, 168))
    #  torch.save(model.state_dict,"test.pth")
     out = model(input)
     print(out.size())
