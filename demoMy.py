import numpy as np
import time
import cv2
import torch
from torch.autograd import Variable
import lib.utils.utils as utils
import lib.models.crnn as crnn
import lib.config.alphabets as alphabets
import yaml
from easydict import EasyDict as edict
import argparse
import os
plateName1=alphabets.plateName1
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
def parse_arg():
    parser = argparse.ArgumentParser(description="demo")

    parser.add_argument('--cfg', help='experiment configuration filename', type=str, default='lib/config/360CC_config.yaml')
    parser.add_argument('--image_path', type=str, default='/mnt/Gpan/Mydata/pytorchPorject/myCrnnPlate/01.jpg', help='the path to your image')
    parser.add_argument('--checkpoint', type=str, default='/mnt/Gpan/Mydata/pytorchPorject/myCrnnPlate/output/360CC/crnn/2022-09-27-20-24/checkpoints/checkpoint_61_acc_0.9715.pth',
                        help='the path to your checkpoints')
    # parser.add_argument('--checkpoint', type=str, default='saved_model/best.pth',help='the path to your checkpoints')

    args = parser.parse_args()

    with open(args.cfg, 'r') as f:
        config = yaml.load(f)
        config = edict(config)

    config.DATASET.ALPHABETS = alphabets.plateName

    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)

    return config, args
def decodePlate(preds):
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    return newPreds
def recognition(config, img, model, converter, device):

    # github issues: https://github.com/Sierkinhane/CRNN_Chinese_Characters_Rec/issues/211
    # h, w = img.shape
    # fisrt step: resize the height and width of image to (32, x)
    # img = cv2.resize(img, (0, 0), fx=config.MODEL.IMAGE_SIZE.H / h, fy=48config.MODEL.IMAGE_SIZE.H / h, interpolation=cv2.INTER_CUBIC)

    # # second step: keep the ratio of image's text same with training
    # h, w = img.shape
    # w_cur = int(img.shape[1] / (config.MODEL.IMAGE_SIZE.OW / config.MODEL.IMAGE_SIZE.W))
    # img = cv2.resize(img, (0, 0), fx=w_cur / w, fy=1.0, interpolation=cv2.INTER_CUBIC)
    # img = np.reshape(img, (config.MODEL.IMAGE_SIZE.H, w_cur, 1))

    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - config.DATASET.MEAN) / config.DATASET.STD
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    model.eval()
    preds = model(img)
    preds=preds.view(-1).detach().cpu().numpy()
    # _, preds = preds.max(2)
    # preds = preds.transpose(1, 0).contiguous().view(-1)

    # preds_size = Variable(torch.IntTensor([preds.size(0)]))
    # sim_pred = converter.decode(preds.data, preds_size.data, raw=False)

    # print('results: {0}'.format(sim_pred))
    newPreds=decodePlate(preds)
    plate=""
    for i in newPreds:
        plate+=plateName1[i]
    return plate
    

if __name__ == '__main__':
    testPath = r"/mnt/Gu/trainData/plate/new_git_train/val/"
    config, args = parse_arg()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # device =torch.device('cpu')
    model = crnn.get_crnn(config,export=True).to(device)
    print('loading pretrained model from {0}'.format(args.checkpoint))
    checkpoint = torch.load(args.checkpoint,map_location=device)
    if 'state_dict' in checkpoint.keys():
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    
    fileList =[]
    allFilePath(testPath,fileList)
    right=0
    begin = time.time()
    for imge_path in fileList:
        img_raw = cv2.imread(imge_path)
        img =img_raw
        converter = utils.strLabelConverter(config.DATASET.ALPHABETS)
        plate=recognition(config, img, model, converter, device)
        plate_ori = imge_path.split('/')[-1].split('_')[0]
        # print(plate,"---",plate_ori)
        if(plate==plate_ori):

            right+=1
        else:
            print(plate_ori,"--->",plate,imge_path)
    end=time.time()
    print("sum:%d ,right:%d , accuracy: %f, time: %f"%(len(fileList),right,right/len(fileList),end-begin))
      
