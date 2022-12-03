import cv2
import imageio
import numpy as np
import os
import shutil
import argparse
from alphabets import plate_chr
def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)
def is_str_right(plate_name):
    for str_ in plate_name:
        if str_ not in palteStr:
            return False
    return True
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default="/mnt/Gu/trainData/plate/new_git_train", help='source') 
    parser.add_argument('--label_file', type=str, default='datasets/train.txt', help='model.pt path(s)')  
    
    opt = parser.parse_args()
    rootPath = opt.image_path
    labelFile = opt.label_file
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危险品0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    # palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航深0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
    palteStr=plate_chr
    print(len(palteStr))
    plateDict ={}
    for i in range(len(list(palteStr))):
        plateDict[palteStr[i]]=i
    fp = open(labelFile,"w",encoding="utf-8")
    file =[]
    allFileList(rootPath,file)
    picNum = 0
    for jpgFile in file:
        print(jpgFile)
        jpgName = jpgFile.split(os.sep)[-1]
        name =jpgName.split("_")[0]
        if " " in name:
            continue
        labelStr=" "
        if not is_str_right(name):
            continue
        strList = list(name)
        for  i in range(len(strList)):
            labelStr+=str(plateDict[strList[i]])+" "
        # while i<7:
        #     labelStr+=str(0)+" "
        #     i+=1
        picNum+=1
        # print(jpgFile+labelStr)
        fp.write(jpgFile+labelStr+"\n")
    fp.close()