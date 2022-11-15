import random
# https://albumentations.ai/docs/examples/example/
# https://github.com/albumentations-team/albumentations
import cv2
# from matplotlib import pyplot as plt
import numpy as np
import os

import albumentations as A

def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)


fileList = []
rootPath = r"E:\carPlate\trainAug\1"
savePath =r"E:\carPlate\trainAug\2"
allFileList(rootPath,fileList)
i = 0
for temp in fileList:
    i = i+1
    print(i,temp)
    image = cv2.imdecode(np.fromfile(temp,dtype=np.uint8),-1)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = A.Compose([

        A.Blur(blur_limit=20,p=1),
        # A.MedianBlur(),
        # A.ElasticTransform(p=1),
        # # # A.RandomBrightnessContrast(),
        # A.OpticalDistortion(),
        # A.MedianBlur(blur_limit=8,p=1),
        A.RandomBrightnessContrast(p=1),
        # A.ImageCompression(),
        # A.RGBShift(),
        # A.RandomGamma(),
        # A.VerticalFlip(),
        # A.Rotate()
        # A.GridDistortion(),
        # A.HueSaturationValue(),
    ])
    random.seed(79)
    augmented_image = transform(image=image)['image']
    imageName = temp.split("\\")[-1]
    # cv2.imshow("haha1",image)
    # cv2.imshow("haha2", augmented_image)
    # cv2.waitKey(0)
    savePicPath = os.path.join(savePath,imageName)
    cv2.imencode('.jpg', augmented_image)[1].tofile(savePicPath)

