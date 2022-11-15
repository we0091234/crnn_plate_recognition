# -*- coding:utf-8 -*-
# Author: RubanSeven

# import cv2
import numpy as np
# from transform import get_perspective_transform, warp_perspective
from warp_mls import WarpMLS


def distort(src, segment):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut // 3
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), np.random.randint(thresh)])
    dst_pts.append([img_w - np.random.randint(thresh), img_h - np.random.randint(thresh)])
    dst_pts.append([np.random.randint(thresh), img_h - np.random.randint(thresh)])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        np.random.randint(thresh) - half_thresh])
        dst_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
                        img_h + np.random.randint(thresh) - half_thresh])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def stretch(src, segment):
    img_h, img_w = src.shape[:2]

    cut = img_w // segment
    thresh = cut * 4 // 5
    # thresh = img_h // segment // 3
    # thresh = img_h // 5

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, 0])
    dst_pts.append([img_w, 0])
    dst_pts.append([img_w, img_h])
    dst_pts.append([0, img_h])

    half_thresh = thresh * 0.5

    for cut_idx in np.arange(1, segment, 1):
        move = np.random.randint(thresh) - half_thresh
        src_pts.append([cut * cut_idx, 0])
        src_pts.append([cut * cut_idx, img_h])
        dst_pts.append([cut * cut_idx + move, 0])
        dst_pts.append([cut * cut_idx + move, img_h])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst


def perspective(src):
    img_h, img_w = src.shape[:2]

    thresh = img_h // 2

    src_pts = list()
    dst_pts = list()

    src_pts.append([0, 0])
    src_pts.append([img_w, 0])
    src_pts.append([img_w, img_h])
    src_pts.append([0, img_h])

    dst_pts.append([0, np.random.randint(thresh)])
    dst_pts.append([img_w, np.random.randint(thresh)])
    dst_pts.append([img_w, img_h - np.random.randint(thresh)])
    dst_pts.append([0, img_h - np.random.randint(thresh)])

    trans = WarpMLS(src, src_pts, dst_pts, img_w, img_h)
    dst = trans.generate()

    return dst

# def distort(src, segment):
#     img_h, img_w = src.shape[:2]
#     dst = np.zeros_like(src, dtype=np.uint8)
#
#     cut = img_w // segment
#     thresh = img_h // 8
#
#     src_pts = list()
#     # dst_pts = list()
#
#     src_pts.append([-np.random.randint(thresh), -np.random.randint(thresh)])
#     src_pts.append([-np.random.randint(thresh), img_h + np.random.randint(thresh)])
#
#     # dst_pts.append([0, 0])
#     # dst_pts.append([0, img_h])
#     dst_box = np.array([[0, 0], [0, img_h], [cut, 0], [cut, img_h]], dtype=np.float32)
#
#     half_thresh = thresh * 0.5
#
#     for cut_idx in np.arange(1, segment, 1):
#         src_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
#                         np.random.randint(thresh) - half_thresh])
#         src_pts.append([cut * cut_idx + np.random.randint(thresh) - half_thresh,
#                         img_h + np.random.randint(thresh) - half_thresh])
#
#         # dst_pts.append([cut * i, 0])
#         # dst_pts.append([cut * i, img_h])
#
#         src_box = np.array(src_pts[-4:-2] + src_pts[-2:-1] + src_pts[-1:], dtype=np.float32)
#
#         # mat = cv2.getPerspectiveTransform(src_box, dst_box)
#         # print(mat)
#         # dst[:, cut * (cut_idx - 1):cut * cut_idx] = cv2.warpPerspective(src, mat, (cut, img_h))
#
#         mat = get_perspective_transform(dst_box, src_box)
#         dst[:, cut * (cut_idx - 1):cut * cut_idx] = warp_perspective(src, mat, (cut, img_h))
#         # print(mat)
#
#     src_pts.append([img_w + np.random.randint(thresh) - half_thresh,
#                     np.random.randint(thresh) - half_thresh])
#     src_pts.append([img_w + np.random.randint(thresh) - half_thresh,
#                     img_h + np.random.randint(thresh) - half_thresh])
#     src_box = np.array(src_pts[-4:-2] + src_pts[-2:-1] + src_pts[-1:], dtype=np.float32)
#
#     # mat = cv2.getPerspectiveTransform(src_box, dst_box)
#     # dst[:, cut * (segment - 1):] = cv2.warpPerspective(src, mat, (img_w - cut * (segment - 1), img_h))
#     mat = get_perspective_transform(dst_box, src_box)
#     dst[:, cut * (segment - 1):] = warp_perspective(src, mat, (img_w - cut * (segment - 1), img_h))
#
#     return dst
