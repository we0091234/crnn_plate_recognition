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
from colorNet import myNet_ocr
# from MyDateSets import  MyDataSets
import cv2
import torch.nn.functional as F
# import adabound
# from lr_scheduler import LRScheduler
def distillation(y, labels, teacher_scores, temp, alpha):  #知识蒸馏loss
    return nn.KLDivLoss(reduction="batchmean")(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
            temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)
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
        if m.num_features !=5:       
            m.eval()
            
def train(epoch):
        print('\nEpoch: %d' % epoch)
        
        print(scheduler.get_lr())
        model.train()
        model.apply(fix_bn)         #训练时固定车牌识别bn层里面的running_mean  running_var
        # time_start = time.time()
        techermodel.eval()
        for batch_idx,(img,label) in enumerate(trainloader):
                image=Variable(img.cuda())
                label=Variable(label.cuda())
                optimizer.zero_grad()
                _,out=model(image)
                _,outT=techermodel(image)
                loss=distillation(out,label,outT,temp=5.0,alpha=0.7)  #训练loss，包括softmaxloss和知识蒸馏loss
                loss.backward()
                optimizer.step()
                if batch_idx%50==0:
                    print("Epoch:%d [%d|%d] loss:%f" % (epoch, batch_idx, len(trainloader), loss.mean()))
        scheduler.step()
        
def val(epoch):
	print("\nValidation Epoch: %d" %epoch)
	model.eval()
	total=0
	correct=0
	test_loss = 0
	with torch.no_grad():
		for batch_idx,(img,label) in enumerate(valloader):
			image=Variable(img.cuda())
			label=Variable(label.cuda())
			_,out=model(image)
			_,predicted=torch.max(out.data,1)
			test_loss += torch.nn.functional.cross_entropy(out, label, reduction='sum').item()
			total+=image.size(0)
			correct+=predicted.data.eq(label.data).cpu().sum()
	accuracy=1.0*correct.numpy()/total
	test_loss /= len(valloader.dataset)
	print("testAcc: %f testLoss:%f"% ((1.0*correct.numpy())/total,test_loss))
	exModelName = opt.model_path +'/' +str(format(accuracy,'.6f'))+"_"+"epoth_"+ str(epoch) + "_model" + ".pth.tar"
	# torch.save(model.state_dict(),exModelName)
	torch.save({'cfg': Stcfg, 'state_dict': model.state_dict()}, exModelName,_use_new_zipfile_serialization=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t_model',type=str,default='color_model3/0.9934579439252337_epoth_42_model.pth',help='teacher model')#教师模型
    parser.add_argument('--s_model',type=str,default='color_model_fix/0.968224_epoth_89_model.pth.tar',help='stuedent model')#学生模型
    parser.add_argument('--train_path',type=str,default='datasets/palte_color/train') #颜色训练集
    parser.add_argument('--val_path',type=str,default='datasets/palte_color/val')    #颜色验证集
    parser.add_argument('--color_num', type=int, default=5)       #颜色类别数
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--batchSize', type=int, default=256)
    parser.add_argument('--nepoch', type=int, default=120)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--model_path',type=str,default='color_model_pd',help='model_path')  #模型保存的路径
    opt = parser.parse_args()
    print(opt)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    
    #训练的均值方差，和车牌识别保持一致
    mean_value=(0.588,0.588,0.588)
    std_value=(0.193,0.193,0.193)

   #训练transforms
   
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)
    transform_train = cvTransforms.Compose([
        cvTransforms.Resize((48, 168)),
        cvTransforms.RandomHorizontalFlip(),  
        cvTransforms.ToTensorNoDiv(),  
        cvTransforms.NormalizeCaffe(mean_value,std_value) 
    ]) 
    #验证transforms
    transform_val = cvTransforms.Compose([
        cvTransforms.Resize((48, 168)),
        cvTransforms.ToTensorNoDiv(),
        cvTransforms.NormalizeCaffe(mean_value,std_value),
    ])


    modelPath = opt.t_model  #知识蒸馏中的教师模型
    StmodelPath=opt.s_model  #知识蒸馏中的学生模型，也就是之前单独训练的车牌识别模型

    checkPoint = torch.load(modelPath)  #加载教师模型
    cfg = checkPoint["cfg"]
    techermodel =myNet_ocr(cfg=cfg,color_num=opt.color_num)
    techermodel.load_state_dict(checkPoint["state_dict"])
    techermodel.cuda()
    
   
    StcheckPoint = torch.load(StmodelPath)  #加载学生模型
    Stcfg = StcheckPoint["cfg"]
    print(Stcfg)
    model = myNet_ocr(cfg=Stcfg,color_num=opt.color_num)
    model_dict = StcheckPoint['state_dict']
    model.load_state_dict(model_dict)
    model.cuda()
    
    #datasets
    trainset = dset.ImageFolder(opt.train_path, transform=transform_train,loader=cv_imread)
    valset = dset.ImageFolder(opt.val_path, transform=transform_val, loader=cv_imread)
    
    #dataLoader
    trainloader=torch.utils.data.DataLoader(trainset,batch_size=opt.batchSize,shuffle=True,num_workers=opt.num_workers)
    valloader=torch.utils.data.DataLoader(valset,batch_size=opt.batchSize,shuffle=False,num_workers=opt.num_workers)	# model=myNet(num_classes=3)
    
    #车牌识别分支冻结，只训练车牌颜色
    # name_list =['feature.0.weight','feature.0.bias','feature.1.weight','feature.1.bias','feature.4.weight','feature.4.bias','feature.5.weight','feature.5.bias']    #list中为需要冻结的网络层
    for name, value in model.named_parameters():
        if name not in ['color_classifier.weight','color_classifier.bias','color_bn.weight','color_bn.bias']: #除了这几个层，其他全部固定
            value.requires_grad = False

    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer=torch.optim.SGD(params,lr=opt.lr,momentum=0.9,weight_decay=5e-4)
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(opt.nepoch))
    criterion=nn.CrossEntropyLoss()

    criterion.cuda()
    for epoch in range(opt.nepoch):
        train(epoch)
        val(epoch)

