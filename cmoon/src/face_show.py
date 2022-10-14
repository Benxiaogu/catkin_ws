#!/usr/bin/env python3
# coding: UTF-8

import cv2
import time

path= '/home/sundawn/图片/photo.jpg'
cv2.namedWindow("Face", cv2.WINDOW_NORMAL)  # 建立图像对象
img = cv2.imread(path)
cv2.imshow("Face", img)
cv2.waitKey(5000)

cv2.destroyAllWindows()