import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import *
import torchvision.transforms as transforms
import cvtorchvision.cvtransforms as cvTransforms
import torchvision.datasets as dset
import numpy as np 
import os
import argparse
import time
import cv2
# from myNet import myNet
from colorNet import myNet_ocr_color
# from model.resnet import resnet101
# from dataset.DogCat import DogCat
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
class Net(torch.nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = torch.nn.Sequential(
			torch.nn.Conv2d(3, 32, 5, stride=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(32),
			torch.nn.MaxPool2d(kernel_size=3,stride=2,ceil_mode=True))
		self.conv2 = torch.nn.Sequential(
			torch.nn.Conv2d(32, 64, 3, stride=1, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(64),
			torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
		)
		self.conv3 = torch.nn.Sequential(
			torch.nn.Conv2d(64, 96, 3, stride=1, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(96),
			torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
		)
		self.conv4 = torch.nn.Sequential(
			torch.nn.Conv2d(96, 128, 3, stride=1, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(128),
			torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
		)
		self.conv5 = torch.nn.Sequential(
			torch.nn.Conv2d(128, 192, 3, stride=1, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(192),
			torch.nn.MaxPool2d(kernel_size=3, stride=2,ceil_mode=True)
		)
		self.conv6 = torch.nn.Sequential(
			torch.nn.Conv2d(192, 256, 3, stride=1, padding=1),
			torch.nn.ReLU(inplace=True),
			torch.nn.BatchNorm2d(256),
			torch.nn.AvgPool2d(kernel_size=3, stride=1)
		)
		self.fc1=torch.nn.Linear(256,3)

	def forward(self, x):
		conv1_out = self.conv1(x)
		# print(conv1_out.shape)
		conv2_out = self.conv2(conv1_out)
		# print(conv2_out.shape)
		conv3_out = self.conv3(conv2_out)
		# print(conv3_out.shape)
		conv4_out = self.conv4(conv3_out)
		# print(conv4_out.shape)
		conv5_out = self.conv5(conv4_out)
		# print(conv5_out.shape)
		out=self.conv6(conv5_out)
		# print("out={}",out.shape)
		out = out.view(out.shape[0],-1)
		# print(out.shape)
		# print(out.shape)
		out=self.fc1(out)
		# print(out.shape)
		return out

myCfg = [32, 'M', 64, 'M', 96, 'M', 128, 'M', 192, 'M', 256]

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss

parser=argparse.ArgumentParser()
parser.add_argument('--num_workers',type=int,default=8)
parser.add_argument('--batchSize',type=int,default=256)
parser.add_argument('--nepoch',type=int,default=120)
parser.add_argument('--lr',type=float,default=0.025)
parser.add_argument('--gpu',type=str,default='0')
opt=parser.parse_args()
print(opt)
os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
device=torch.device("cuda")
torch.backends.cudnn.benchmark=True
mean_value=(0.588,0.588,0.588)
std_value=(0.193,0.193,0.193)
transform_train=cvTransforms.Compose([
	cvTransforms.Resize((48,168)),
	# cvTransforms.RandomCrop((128,128)),
	cvTransforms.RandomHorizontalFlip(), #镜像
	cvTransforms.ToTensorNoDiv(), 
	cvTransforms.NormalizeCaffe(mean_value,std_value) 
])

transform_val=cvTransforms.Compose([
	cvTransforms.Resize((48,168)),
	cvTransforms.ToTensorNoDiv(),
	cvTransforms.NormalizeCaffe(mean_value,std_value),
])

# def fix_bn(m):
#     classname = m.__class__.__name__
#     if  classname.find('BatchNorm') != -1:
#         if m.num_features==32:
#            m.eval()


trainset=dset.ImageFolder(r'datasets/palte_color/train',transform=transform_train,loader=cv_imread)
print(trainset[0][0])
valset  =dset.ImageFolder(r'datasets/palte_color/val',transform=transform_val,loader=cv_imread)
print(len(valset))
trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)

# model=myNet(num_classes=5)
cfg =[16,16,32,32,'M',64,64,'M',96,96,'M',128,256]
model = myNet_ocr_color(color_num=5,cfg=cfg)
model.cuda()
optimizer=torch.optim.SGD(model.parameters(),lr=opt.lr,momentum=0.9,weight_decay=5e-4)
# scheduler=StepLR(optimizer,step_size=20)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
# criterion=nn.CrossEntropyLoss()
criterion=CrossEntropyLabelSmooth(5)
criterion.cuda()

def train(epoch):
	print('\nEpoch: %d' % epoch)
	scheduler.step()
	print(scheduler.get_lr())
	model.train()
	# model.apply(fix_bn)

	# time_start = time.time()

	for batch_idx,(img,label) in enumerate(trainloader):
		# time_end = time.time()
		# print('totally cost', time_end - time_start)
		image=Variable(img.cuda())
		label=Variable(label.cuda())
		optimizer.zero_grad()
		_,out=model(image)
		loss=criterion(out,label)
		loss.backward()
		optimizer.step()
		if batch_idx%50==0:
			print("Epoch:%d [%d|%d] loss:%f lr:%s" %(epoch,batch_idx,len(trainloader),loss.mean(),scheduler.get_lr()))
	# exModelName="ckp/epoth_"+str(epoch)+"_model"+".pth"
	# # torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(),exModelName)
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			_,out=model(image)
			_,predicted=torch.max(out.data,1)
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	print("Acc: %f "% ((1.0*correct.numpy())/total))
	exModelName = r"color_model3/" +str(accuracy)+"_"+"epoth_"+ str(epoch) + "_model" + ".pth"
	# torch.save(model.state_dict(),exModelName)
	# torch.save(model.state_dict(), exModelName)
	torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, exModelName)

for epoch in range(opt.nepoch):
	train(epoch)
	val(epoch)

