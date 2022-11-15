import os
import shutil
import numpy as np

def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)

def getPlateName(plate):
    jpgName = plate.split("\\")[-1]
    plateName=jpgName.split("_")[0]
    return plateName,plate
filePath1 = r"E:\carPlate\realrealTest"
filePath2=r"D:\trainTemp\carPlate\train\train_ori"
savePath = r"E:\carPlate\realrealTest\val"

fileList1=[]
filelist2=[]

allFileList(filePath1,fileList1)
allFileList(filePath2,filelist2)

platelist1=[]
platelist2=[]

for temp in filelist2:
    platelist2.append(getPlateName(temp)[0])
for temp in fileList1:
    platelist1.append(getPlateName(temp))
platelist2=list(set(platelist2))
i = 0
print(platelist2)
for temp1 in platelist1:
         if temp1[0] in platelist2:
             print(temp1[0],temp1[1])
             folder = temp1[1].split("\\")[-2]
             folderPath =os.path.join(savePath,folder)
             if not os.path.exists(folderPath):
                 os.mkdir(folderPath)
             picName=temp1[1].split("\\")[-1]
             picPath =os.path.join(folderPath,picName)
             shutil.move(temp1[1],picPath)
             i+=1

print(i)