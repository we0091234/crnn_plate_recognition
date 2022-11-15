import cv2
import imageio
import numpy as np
import os
import shutil
def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)


rootPath = r"D:\trainTemp\carPlate\train"
palteStr=r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民深危险品0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"
labelFile =r"D:\trainTemp\carPlate\0914_train.txt"
plateDict ={}
for i in range(len(list(palteStr))):
    plateDict[palteStr[i]]=i
fp = open(labelFile,"w")
file =[]
allFileList(rootPath,file)
picNum = 0
for jpgFile in file:
    jpgName = jpgFile.split("\\")[-1]
    name =jpgName.split("_")[0]
    labelStr=" "
    strList = list(name)
    # print(jpgFile)
    for  i in range(len(strList)):
         labelStr+=str(plateDict[strList[i]])+" "
    while i<7:
        labelStr+=str(0)+" "
        i+=1
    picNum+=1
    print(jpgFile+labelStr)
    fp.write(jpgFile+labelStr+"\n")
