import cv2

imgfile = "/home/cxl/myData/pytorchProject/crnn_cn_pt-master/新AU3006_convert0177.jpg"
img = cv2.imread(imgfile)
img1 = cv2.resize(img,(160,32))
cv2.imshow("haha",img1)
cv2.waitKey(0)