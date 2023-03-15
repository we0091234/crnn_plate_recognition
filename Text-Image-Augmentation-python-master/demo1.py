# -*- coding:utf-8 -*-
# Author: RubanSeven

import cv2
import imageio
import numpy as np
import os
from augment import distort, stretch, perspective
import argparse

def allFileList(rootfile,allFile):
    folder =os.listdir(rootfile)
    for temp in folder:
        fileName = os.path.join(rootfile,temp)
        if os.path.isfile(fileName):
            allFile.append(fileName)
        else:
            allFileList(fileName,allFile)

def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


if __name__ == '__main__':
    parser =argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='/mnt/Gu/trainData/plate/final/git_release/train_huoche/', help='model.pt path(s)')  
    parser.add_argument('--dst_path', type=str, default='/mnt/Gu/trainData/plate/final/git_release/train_huoche_aug/', help='source') 
    opt = parser.parse_args()
    rootFile = opt.src_path
    saveFile = opt.dst_path
    fileList = []
    allFileList(rootFile, fileList)
    picOunt=0
    for temp in fileList:
        print(picOunt,temp)
        picOunt+=1
        im = cv2.imdecode(np.fromfile(temp,dtype=np.uint8),-1)
        _,_,c = im.shape
        if c!=3:
            continue
    # im = cv2.resize(im, (200, 64))
    #     cv2.imshow("im_CV", im)
        distort_img_list = list()
        stretch_img_list = list()
        perspective_img_list = list()
        for i in range(1):
            try:
                distort_img = distort(im, 8)
                distort_img_list.append(distort_img)
                # cv2.imshow("distort_img", distort_img)

                stretch_img = stretch(distort_img, 8)
                # cv2.imshow("stretch_img", stretch_img)
                stretch_img_list.append(stretch_img)

                name = temp.split(os.sep)[-1].split(".")[0]
                name = name+str(picOunt)+".jpg"
                newPath = os.path.join(saveFile,name)
                perspective_img = perspective(stretch_img)
                # cv2.imshow("perspective_img", perspective_img)
                perspective_img_list.append(perspective_img)
                # cv2.waitKey(1)
                cv2.imencode('.jpg', perspective_img)[1].tofile(newPath)
            except:
                print("transForm error")
    # create_gif(distort_img_list, r'imgs/distort.gif')
    # create_gif(stretch_img_list, r'imgs/stretch.gif')
    # create_gif(perspective_img_list, r'imgs/perspective.gif')
