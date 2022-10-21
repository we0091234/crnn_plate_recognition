from plateNet import myNet_ocr
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
import time
import argparse
def cv_imread(path):   #读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)

# plateName="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危险品0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
mean_value,std_value=(0.588,0.193)
def decodePlate(preds):
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    return newPreds

def image_processing(img,device):
    img = cv2.resize(img, (168,48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img

def get_plate_result(img,device,model):
    # img = cv2.imread(image_path)
    input = image_processing(img,device)
    preds = model(input)
    # print(preds)
    preds=preds.view(-1).detach().cpu().numpy()
    newPreds=decodePlate(preds)
    plate=""
    for i in newPreds:
        plate+=plateName[int(i)]
    return plate

def init_model(device,model_path):
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=78,export=True,cfg=cfg)        #export  True 用来推理
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='saved_model/best.pth', help='model.pt path(s)')  
    parser.add_argument('--image_path', type=str, default='images/test.jpg', help='source') 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device =torch.device("cpu")
    opt = parser.parse_args()
    model = init_model(device,opt.model_path)
    if os.path.isfile(opt.image_path): 
        right=0
        begin = time.time()
        img = cv_imread(opt.image_path)
        if img.shape[-1]!=3:
            img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        plate=get_plate_result(img, device,model)
        print(plate)
    else:
            file_list=[]
            allFilePath(opt.image_path,file_list)
            for pic_ in file_list:
                try:
                    pic_name = os.path.basename(pic_)
                    img = cv_imread(pic_)
                    if img.shape[-1]!=3:
                        img = cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
                    plate=get_plate_result(img,device,model)
                    print(plate,pic_name)
                except:
                    print("error")
                

