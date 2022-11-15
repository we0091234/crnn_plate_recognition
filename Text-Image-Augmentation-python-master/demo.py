# -*- coding:utf-8 -*-
# Author: RubanSeven

import cv2
import imageio
import numpy as np
from augment import distort, stretch, perspective


def create_gif(image_list, gif_name, duration=0.1):
    frames = []
    for image in image_list:
        frames.append(image)
    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


if __name__ == '__main__':
    im = cv2.imdecode(np.fromfile(r"H:\@chepai\Text-Image-Augmentation-python-master\imgs\demo.png",dtype=np.uint8),-1)
    # im = cv2.resize(im, (200, 64))
    cv2.imshow("im_CV", im)
    distort_img_list = list()
    stretch_img_list = list()
    perspective_img_list = list()
    for i in range(1):
        distort_img = distort(im, 4)
        distort_img_list.append(distort_img)
        # cv2.imshow("distort_img", distort_img)

        stretch_img = stretch(distort_img, 4)
        # cv2.imshow("stretch_img", stretch_img)
        stretch_img_list.append(stretch_img)

        perspective_img = perspective(stretch_img)
        cv2.imshow("perspective_img", perspective_img)
        perspective_img_list.append(perspective_img)
        cv2.waitKey(1)

    create_gif(distort_img_list, r'imgs/distort.gif')
    create_gif(stretch_img_list, r'imgs/stretch.gif')
    create_gif(perspective_img_list, r'imgs/perspective.gif')
