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
from torch.utils.data import Dataset, DataLoader
# import adabound
# from lr_scheduler import LRScheduler
def cv_imread(path):
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img
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

def fix_bn(m):   #固定bn层中的running_mean running_var  除了颜色识别那层不固定
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        if m.num_features !=5 and m.num_features !=12:     #颜色那个分支的两个bn层输出通道数是5，和12 这两层不固定，其他固定  ，5是颜色的类别数
            m.eval()
            
def train(epoch):
	print('\nEpoch: %d' % epoch)
 
	
	print(scheduler.get_lr())
	model.train()
	model.apply(fix_bn)

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
		if batch_idx % 50 == 0:
			print("Epoch:%d [%d|%d] loss:%f lr:%s" % (epoch, batch_idx, len(trainloader), loss.mean(), scheduler.get_lr()))
	scheduler.step()
 
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
	exModelName = opt.model_path +'/'+str(format(accuracy,".6f"))+"_"+"epoth_"+ str(epoch) + "_model" + ".pth"
	# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': cfg, 'state_dict': model.state_dict()}, exModelName)

if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--weights',type=str,default='saved_model/plate_rec_small.pth')  #车牌识别模型
	parser.add_argument('--train_path',type=str,default='datasets/palte_color/train') #颜色训练集
	parser.add_argument('--val_path',type=str,default='datasets/palte_color/val')    #颜色验证集
	parser.add_argument('--num_color', type=int, default=5)       
	parser.add_argument('--num_workers', type=int, default=8)
	parser.add_argument('--batchSize', type=int, default=256)
	parser.add_argument('--nepoch', type=int, default=120)
	parser.add_argument('--lr', type=float, default=0.0025)
	parser.add_argument('--gpu', type=str, default='0')
	parser.add_argument('--model_path',type=str,default='color_model',help='model_path')
	opt = parser.parse_args()
   
	print(opt)
    
	os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
	device = torch.device("cuda")
	torch.backends.cudnn.benchmark = True
	if not os.path.exists(opt.model_path):
		os.mkdir(opt.model_path)
  
	mean_value=(0.588,0.588,0.588)
	std_value=(0.193,0.193,0.193)

	transform_train = cvTransforms.Compose([
		cvTransforms.Resize((48, 168)),
		cvTransforms.RandomHorizontalFlip(),  
		cvTransforms.ToTensorNoDiv(),  
		cvTransforms.NormalizeCaffe(mean_value,std_value) 
	])

	transform_val = cvTransforms.Compose([
		cvTransforms.Resize((48, 168)),
		cvTransforms.ToTensorNoDiv(),
		cvTransforms.NormalizeCaffe(mean_value,std_value),
	])

	rec_model_Path = opt.weights   #车牌识别模型
	checkPoint = torch.load(rec_model_Path)
	cfg = checkPoint["cfg"]
	print(cfg)
	model = myNet_ocr_color(cfg=cfg,color_num=opt.num_color)
	model_dict = checkPoint['state_dict']
	model.load_state_dict(model_dict,strict=False)   #导入之前训练好的车牌识别模型
 
	trainset=dset.ImageFolder(opt.train_path,transform=transform_train,loader=cv_imread)
	valset  =dset.ImageFolder(opt.val_path,transform=transform_val,loader=cv_imread)
	print(len(valset))
	trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
	valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)	
	model.cuda()
	# name_list =['feature.0.weight','feature.0.bias','feature.1.weight','feature.1.bias','feature.4.weight','feature.4.bias','feature.5.weight','feature.5.bias']    #list中为需要冻结的网络层
	for name, value in model.named_parameters():
		if name not in ['color_classifier.weight','color_classifier.bias','color_bn.weight','color_bn.bias','conv1.weight','conv1.bias','bn1.weight','bn1.bias']: #除了这几个层，其他全部固定
			value.requires_grad = False
	params = filter(lambda p: p.requires_grad, model.parameters())
	optimizer=torch.optim.SGD(params,lr=opt.lr,momentum=0.9,weight_decay=5e-4)
	# scheduler=StepLR(optimizer,step_size=40)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
	# criterion=nn.CrossEntropyLoss()
	criterion=CrossEntropyLabelSmooth(opt.num_color)
	criterion.cuda()
	for epoch in range(opt.nepoch):
		train(epoch)
		val(epoch)

