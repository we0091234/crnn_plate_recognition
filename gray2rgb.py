import cv2
import os
import shutil
import numpy as np
def allFilePath(rootPath,allFIleList):
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath,temp)):
            allFIleList.append(os.path.join(rootPath,temp))
        else:
            allFilePath(os.path.join(rootPath,temp),allFIleList)
            
def cv_imread(path):   #读取中文路径的图片
    img=cv2.imdecode(np.fromfile(path,dtype=np.uint8),-1)
    return img

if __name__ == "__main__":
    file_list = []
    root_path = r"/mnt/Gu/trainData/plate/final/git_release/train_huoche"
    allFilePath(root_path,file_list)
    
    for pic_ in  file_list:
        img = cv_imread(pic_)
        try:
            # img_h, img_w ,_= img.shape
            img_shape = img.shape
            if len(img_shape)!=3:
                print(pic_,img.shape)
                img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                cv2.imwrite(pic_,img)
        except:
            print(pic_)
        
    # img = cv2.imread(r"/mnt/Gu/trainData/plate/final/git_release/train_huoche/huoche1/HXD1C0507_450.jpg")
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img.shape)
                