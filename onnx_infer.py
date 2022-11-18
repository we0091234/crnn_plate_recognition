import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time
from alphabets import plate_chr
def cv_imread(path):   #防止读取中文路径失败
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

# plateName=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
mean_value,std_value=((0.588,0.193))#识别模型均值标准差

def decodePlate(preds):        #识别后处理
    pre=0
    newPreds=[]
    for i in range(len(preds)):
        if preds[i]!=0 and preds[i]!=pre:
            newPreds.append(preds[i])
        pre=preds[i]
    plate=""
    for i in newPreds:
        plate+=plate_chr[int(i)]
    return plate

def rec_pre_precessing(img,size=(48,168)): #识别前处理
    img =cv2.resize(img,(168,48))
    img = img.astype(np.float32)
    img = (img/255-mean_value)/std_value  #归一化 减均值 除标准差
    img = img.transpose(2,0,1)         #h,w,c 转为 c,h,w
    img = img.reshape(1,*img.shape)    #channel,height,width转为batch,channel,height,channel
    return img


def get_plate_result(img,session_rec): #识别后处理
    img =rec_pre_precessing(img)
    y_onnx = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: img})[0]
    index =np.argmax(y_onnx[0],axis=1)
    # print(y_onnx[0])
    plate_no = decodePlate(index)
    return plate_no

def allFilePath(rootPath,allFIleList):  #遍历文件
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_file', type=str, default='saved_model/best.onnx', help='model.pt path(s)')#识别模型
    parser.add_argument('--image_path', type=str, default='images', help='source') 
    parser.add_argument('--img_h', type=int, default=48, help='inference size (pixels)')
    parser.add_argument('--img_w', type=int, default=168, help='inference size (pixels)')
    # parser.add_argument('--output', type=str, default='result1', help='source') 
    opt = parser.parse_args()
    providers =  ['CPUExecutionProvider']
    session_rec = onnxruntime.InferenceSession(opt.onnx_file, providers=providers )
    file_list = []
    right=0
    
    if os.path.isfile(opt.image_path):
        img=cv_imread(opt.image_path)
        if img.shape[-1]==4:
            img =cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
        plate = get_plate_result(img,session_rec)
        print(f"{plate} {opt.image_path}")
    else:
        allFilePath(opt.image_path,file_list)
        for pic_ in file_list:
            img=cv_imread(pic_)
            if img.shape[-1]==4:
                img =cv2.cvtColor(img,cv2.COLOR_BGRA2BGR)
            plate = get_plate_result(img,session_rec)
            print(f"{plate} {pic_}")
